from __future__ import annotations

import logging
import multiprocessing as mp
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.allocation.constraint_optimization_allocation import ConstraintOptimizationAllocationStage
from stream.stages.stage import Stage, StageCallable
from stream.utils import get_unique_nodes
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)


_MP_EVAL_STAGE = None


def _mp_safe_evaluate_candidate(params: dict[str, int]) -> Any | None:
	global _MP_EVAL_STAGE
	stage = _MP_EVAL_STAGE
	if stage is None:
		return None
	return stage._safe_evaluate_candidate(params)


@dataclass(frozen=True)
class HardwareGAParameter:
	"""One hardware-template decision variable for NSGA-II."""

	name: str
	lower: int
	upper: int
	scale: int = 1


class _HardwareTemplateProblem(Problem):
	"""Pymoo problem wrapper that evaluates a template candidate via downstream Stream stages."""

	def __init__(self, stage: "GeneticHardwareStage", parameter_specs: list[HardwareGAParameter]):
		self.stage = stage
		self.parameter_specs = parameter_specs
		self.best_score = float("inf")
		self.best_params: dict[str, int] | None = None
		self.best_scme = None

		super().__init__(
			n_var=len(parameter_specs),
			n_obj=3,  # energy, latency, area
			n_constr=0,
			xl=np.array([spec.lower for spec in parameter_specs], dtype=int),
			xu=np.array([spec.upper for spec in parameter_specs], dtype=int),
			vtype=int,
			elementwise_evaluation=False,
		)

	def _decode_params(self, x: np.ndarray) -> dict[str, int]:
		params: dict[str, int] = {}
		for raw_value, spec in zip(x.tolist(), self.parameter_specs, strict=True):
			params[spec.name] = int(raw_value) * int(spec.scale)
		return params

	def _evaluate(self, X: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any):
		objective_rows: list[list[float]] = []
		params_list = [self._decode_params(np.asarray(x, dtype=int)) for x in X]
		results = self.stage._evaluate_candidates(params_list)

		for params, scme in zip(params_list, results, strict=True):
			if scme is None:
				f = [1e30, 1e30, 1e30]
			else:
				f = [float(scme.energy), float(scme.latency), float(scme.accelerator.area)]

			score = self.stage._score_objectives(f)
			if scme is not None and score < self.best_score:
				self.best_score = score
				self.best_params = params
				self.best_scme = scme

			objective_rows.append(f)

		out["F"] = np.asarray(objective_rows, dtype=float)


class GeneticHardwareStage(Stage):
	"""Optimize template parameters with NSGA-II and pass best candidate to downstream stages.

	Expected kwargs:
	- hardware_template: path to a SoC yaml template.
	- hardware_ga_core_template: path to a core yaml template to optimize.
	- hardware_ga_parameter_specs: list of dicts with keys {name, lower, upper, scale?}.

	Optional kwargs:
	- hardware_ga_generations (default 3)
	- hardware_ga_population (default 8)
	- hardware_ga_seed (default 42)
	- template_params (base SoC template context dict)
	- hardware_ga_core_template_params (base core template context dict, merged with candidate params)
	- hardware_ga_target_core_id (default 0)
	"""

	def __init__(self, list_of_callables: list[StageCallable], **kwargs: Any):
		super().__init__(list_of_callables, **kwargs)
		self.accelerator: Accelerator = kwargs["accelerator"]
		self.workload = kwargs["workload"]
		self.hardware_template: str | None = kwargs.get("hardware_template")
		self.hardware_ga_core_template: str | None = kwargs.get("hardware_ga_core_template")
		self.template_params: dict[str, Any] = dict(kwargs.get("template_params", {}))
		self.hardware_ga_core_template_params: dict[str, Any] = dict(
			kwargs.get("hardware_ga_core_template_params", {})
		)
		self.hardware_ga_target_core_id = int(kwargs.get("hardware_ga_target_core_id", 0))
		self.hardware_ga_stack_index = kwargs.get("hardware_ga_stack_index")
		self.layer_stacks: list[tuple[int, ...]] = list(kwargs.get("layer_stacks", []))
		self.parameter_specs = self._parse_parameter_specs(kwargs.get("hardware_ga_parameter_specs", []))

		self.ga_generations = int(kwargs.get("hardware_ga_generations", 3))
		self.ga_population = int(kwargs.get("hardware_ga_population", 8))
		self.ga_seed = int(kwargs.get("hardware_ga_seed", 42))
		raw_workers = kwargs.get("hardware_ga_workers", os.cpu_count() or 1)
		if raw_workers is None:
			raw_workers = os.cpu_count() or 1
		self.ga_workers = max(1, int(raw_workers))
		self.ga_weight_energy = float(kwargs.get("hardware_ga_weight_energy", 1.0))
		self.ga_weight_latency = float(kwargs.get("hardware_ga_weight_latency", 1.0))
		self.ga_weight_area = float(kwargs.get("hardware_ga_weight_area", 1.0))

	@staticmethod
	def _parse_parameter_specs(raw_specs: Any) -> list[HardwareGAParameter]:
		specs: list[HardwareGAParameter] = []
		if not raw_specs:
			return specs
		for raw in raw_specs:
			specs.append(
				HardwareGAParameter(
					name=str(raw["name"]),
					lower=int(raw["lower"]),
					upper=int(raw["upper"]),
					scale=int(raw.get("scale", 1)),
				)
			)
		return specs

	def run(self):
		# If no GA configuration is provided, keep existing behavior and pass-through.
		if not self.hardware_template or not self.hardware_ga_core_template or not self.parameter_specs:
			logger.info(
				"GeneticHardwareStage disabled (missing SoC template, core template, or parameter specs)."
			)
			kwargs = self.kwargs.copy()
			kwargs["accelerator"] = self.accelerator
			kwargs["workload"] = self.workload
			sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
			yield from sub_stage.run()
			return

		logger.info(
			"Starting hardware NSGA-II: vars=%d, pop=%d, generations=%d, workers=%d",
			len(self.parameter_specs),
			self.ga_population,
			self.ga_generations,
			self.ga_workers,
		)
		problem = _HardwareTemplateProblem(self, self.parameter_specs)
		algorithm = NSGA2(pop_size=self.ga_population)
		minimize(
			problem,
			algorithm,
			termination=("n_gen", self.ga_generations),
			seed=self.ga_seed,
			verbose=False,
		)

		if problem.best_params is None:
			raise RuntimeError("Hardware GA did not produce any valid candidate.")

		logger.info("Best hardware parameters from GA: %s", problem.best_params)
		best_accelerator = self._build_accelerator_from_templates(problem.best_params)

		kwargs = self.kwargs.copy()
		kwargs["accelerator"] = best_accelerator
		kwargs["workload"] = self.workload
		kwargs["selected_hardware_params"] = problem.best_params
		sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
		yield from sub_stage.run()

	def _evaluate_candidates(self, params_list: list[dict[str, int]]) -> list[Any | None]:
		if self.ga_workers <= 1 or len(params_list) <= 1:
			return [self._safe_evaluate_candidate(params) for params in params_list]

		workers = min(self.ga_workers, len(params_list))
		global _MP_EVAL_STAGE
		_MP_EVAL_STAGE = self
		try:
			ctx = mp.get_context("fork")
			with ctx.Pool(processes=workers) as pool:
				return pool.map(_mp_safe_evaluate_candidate, params_list)
		except Exception:
			logger.exception("Multiprocessing candidate evaluation failed; falling back to sequential evaluation.")
			return [self._safe_evaluate_candidate(params) for params in params_list]
		finally:
			_MP_EVAL_STAGE = None

	def _safe_evaluate_candidate(self, params: dict[str, int]) -> Any | None:
		try:
			return self._evaluate_candidate(params)
		except Exception:
			logger.exception("Candidate evaluation failed for params=%s", params)
			return None

	def _score_objectives(self, objectives: list[float]) -> float:
		energy, latency, area = objectives
		return (
			self.ga_weight_energy * float(energy)
			+ self.ga_weight_latency * float(latency)
			+ self.ga_weight_area * float(area)
		)

	def _evaluate_candidate(self, params: dict[str, int]):
		candidate_accelerator = self._build_accelerator_from_templates(params)
		kwargs = self.kwargs.copy()
		kwargs["accelerator"] = candidate_accelerator
		kwargs["workload"] = self.workload

		if not self.list_of_callables:
			raise RuntimeError("GeneticHardwareStage requires downstream stages.")

		zigzag_stage = self.list_of_callables[0]

		with tempfile.TemporaryDirectory(prefix="ga_hw_eval_") as tmp_dir:
			self._override_candidate_paths(kwargs, Path(tmp_dir))
			# During GA exploration, run only ZigZagCoreMappingEstimationStage,
			# not full constraint optimization.
			sub_stage = zigzag_stage([_CollectZigZagStatsLeaf], **kwargs)
			answers = list(sub_stage.run())

		if not answers:
			raise RuntimeError("No result returned by ZigZag candidate evaluation.")

		_, extra_info = answers[0]
		if not isinstance(extra_info, dict):
			raise RuntimeError("Invalid ZigZag candidate evaluation output.")

		cost_lut = extra_info.get("cost_lut")
		workload = extra_info.get("workload")
		accelerator = extra_info.get("accelerator", candidate_accelerator)
		if cost_lut is None or workload is None:
			raise RuntimeError("Missing cost_lut/workload in ZigZag candidate evaluation output.")

		return self._build_proxy_scme_from_cost_lut(
			cost_lut=cost_lut,
			workload=workload,
			accelerator=accelerator,
			layer_stacks=self.layer_stacks,
			target_core_id=self.hardware_ga_target_core_id,
			stack_index=self.hardware_ga_stack_index,
		)

	@staticmethod
	def _extract_steady_state_nodes_per_stack(
		workload: Any,
		layer_stacks: list[tuple[int, ...]],
	) -> dict[tuple[int, ...], set[ComputationNode]]:
		if not layer_stacks:
			return {}

		co_stage = ConstraintOptimizationAllocationStage.__new__(ConstraintOptimizationAllocationStage)
		co_stage.workload = workload
		co_stage.layer_stacks = layer_stacks
		co_stage.ss_to_computes = {}
		co_stage.hashes_per_sink_node = {}
		co_stage.steady_state_hashes = {}
		co_stage.compute_per_sink_node = {}
		co_stage.ss_iterations_per_stack = {}
		co_stage.optimal_allocation_per_stack = {}
		co_stage.nb_macs_per_stack = {}
		co_stage.nb_macs_in_ss_per_stack = {}
		co_stage.ss_mac_percentages_per_stack = {}

		co_stage.extract_steady_state_per_stack()
		return co_stage.ss_to_computes

	@staticmethod
	def _resolve_assigned_core_id(node: Any) -> int | None:
		if hasattr(node, "chosen_core_allocation"):
			chosen = getattr(node, "chosen_core_allocation")
			if chosen is None:
				return None
			if isinstance(chosen, list):
				if len(chosen) == 1:
					return int(chosen[0])
				return None
			return int(chosen)

		possible = getattr(node, "possible_core_allocation", None)
		if isinstance(possible, list) and len(possible) == 1:
			return int(possible[0])
		return None

	@staticmethod
	def _build_proxy_scme_from_cost_lut(
		cost_lut: Any,
		workload: Any,
		accelerator: Any,
		layer_stacks: list[tuple[int, ...]],
		target_core_id: int,
		stack_index: int | None,
	) -> Any:
		"""Aggregate steady-state CT latency/energy for tiles assigned to target core."""
		total_energy = 0.0
		total_latency = 0.0

		ss_nodes_per_stack = GeneticHardwareStage._extract_steady_state_nodes_per_stack(workload, layer_stacks)
		if ss_nodes_per_stack:
			candidate_stacks = list(ss_nodes_per_stack.keys())
			if stack_index is not None:
				if stack_index < 0 or stack_index >= len(candidate_stacks):
					raise IndexError(
						f"hardware_ga_stack_index={stack_index} out of range for {len(candidate_stacks)} stacks."
					)
				candidate_stacks = [candidate_stacks[stack_index]]

			nodes_to_score: list[ComputationNode] = []
			for stack in candidate_stacks:
				nodes_to_score.extend(sorted(ss_nodes_per_stack.get(stack, set())))
		else:
			nodes_to_score = [node for node in get_unique_nodes(workload) if isinstance(node, ComputationNode)]

		for node in nodes_to_score:
			assigned_core_id = GeneticHardwareStage._resolve_assigned_core_id(node)
			if assigned_core_id is not None and assigned_core_id != target_core_id:
				continue

			equal_node = cost_lut.get_equal_node(node) or node
			cores = cost_lut.get_cores(equal_node)
			if not cores:
				raise RuntimeError(f"No ZigZag CME found for node {node}.")

			target_core_energy = float("inf")
			target_core_latency = float("inf")
			found_target_core = False
			for core in cores:
				core_id = int(getattr(core, "id", -1))
				if core_id != target_core_id:
					continue

				found_target_core = True
				cme = cost_lut.get_cme(equal_node, core)
				energy = float(getattr(cme, "energy_total", getattr(cme, "energy", 1e30)))
				latency = float(getattr(cme, "latency_total2", getattr(cme, "latency", 1e30)))
				target_core_energy = min(target_core_energy, energy)
				target_core_latency = min(target_core_latency, latency)

			if not found_target_core:
				continue

			total_energy += target_core_energy
			total_latency += target_core_latency

		return SimpleNamespace(
			energy=total_energy,
			latency=total_latency,
			accelerator=SimpleNamespace(area=float(getattr(accelerator, "area", 1e30))),
		)

	@staticmethod
	def _override_candidate_paths(kwargs: dict[str, Any], root: Path) -> None:
		path_keys = [
			"tiled_workload_path",
			"cost_lut_path",
			"allocations_path",
			"tiled_workload_post_co_path",
			"cost_lut_post_co_path",
		]
		for key in path_keys:
			if key not in kwargs:
				continue

			original = str(kwargs[key])
			base_name = os.path.basename(original.rstrip("/")) or key
			if key == "allocations_path":
				destination = root / "allocations"
				destination.mkdir(parents=True, exist_ok=True)
				kwargs[key] = str(destination) + "/"
			else:
				kwargs[key] = str(root / base_name)

	@staticmethod
	def _find_core_key(cores: dict[Any, Any], target_core_id: int) -> Any | None:
		if target_core_id in cores:
			return target_core_id
		target_str = str(target_core_id)
		if target_str in cores:
			return target_str
		for key in cores:
			try:
				if int(key) == target_core_id:
					return key
			except (TypeError, ValueError):
				continue
		return None

	def _build_accelerator_from_templates(self, candidate_params: dict[str, int]) -> Accelerator:
		assert self.hardware_template is not None
		assert self.hardware_ga_core_template is not None

		soc_context = dict(self.template_params)
		rendered_soc_yaml = self._render_template(self.hardware_template, soc_context)
		soc_data = yaml.safe_load(rendered_soc_yaml)
		if not isinstance(soc_data, dict):
			raise ValueError("Rendered SoC template did not produce a YAML mapping.")

		cores = soc_data.get("cores")
		if not isinstance(cores, dict):
			raise ValueError("Rendered SoC template must contain a 'cores' mapping.")

		core_key = self._find_core_key(cores, self.hardware_ga_target_core_id)
		if core_key is None:
			raise KeyError(f"Target core id {self.hardware_ga_target_core_id} was not found in SoC cores.")

		core_context = dict(self.hardware_ga_core_template_params)
		core_context.update(candidate_params)
		rendered_core_yaml = self._render_template(self.hardware_ga_core_template, core_context)

		soc_template_dir = Path(self.hardware_template).resolve().parent

		with tempfile.TemporaryDirectory(prefix="ga_hw_core_build_") as tmp_dir:
			tmp_path = Path(tmp_dir)
			tmp_cores_dir = tmp_path / "cores"
			tmp_cores_dir.mkdir(parents=True, exist_ok=True)

			normalized_cores: dict[Any, Any] = {}
			for key, value in cores.items():
				local_core_name = f"core_{str(key)}.yaml"
				local_core_path = tmp_cores_dir / local_core_name

				if key == core_key:
					local_core_path.write_text(rendered_core_yaml, encoding="utf-8")
					normalized_cores[key] = f"./cores/{local_core_name}"
					continue

				if isinstance(value, str):
					source_core_path = Path(value)
					if not source_core_path.is_absolute():
						source_core_path = (soc_template_dir / source_core_path).resolve()
					shutil.copyfile(source_core_path, local_core_path)
					normalized_cores[key] = f"./cores/{local_core_name}"
				else:
					normalized_cores[key] = value

			soc_data["cores"] = normalized_cores

			rendered_soc_path = tmp_path / "soc_rendered.yaml"
			rendered_soc_path.write_text(yaml.safe_dump(soc_data, sort_keys=False), encoding="utf-8")

			validator = AcceleratorValidator(soc_data, str(rendered_soc_path))
			accelerator_data = validator.normalized_data
			if not validator.validate():
				raise ValueError("Failed to validate accelerator generated from SoC/core templates.")

			factory = AcceleratorFactory(accelerator_data)
			return factory.create()

	@staticmethod
	def _render_template(template_path: str, context: dict[str, Any]) -> str:
		template_file = Path(template_path)
		env = Environment(loader=FileSystemLoader(str(template_file.parent)))
		template = env.get_template(template_file.name)
		return template.render(**context)


class _CollectZigZagStatsLeaf(Stage):
	"""Leaf stage used internally to retrieve ZigZag cost LUT stats for GA scoring."""

	def is_leaf(self) -> bool:
		return True

	def run(self):
		extra_info = {
			"cost_lut": self.kwargs.get("cost_lut"),
			"workload": self.kwargs.get("workload"),
			"accelerator": self.kwargs.get("accelerator"),
		}
		yield SimpleNamespace(), extra_info
