from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from pymoo.core.problem import Problem
from stream.api import optimize_allocation_co

from utils import generate_run_id, render_template_to_file


PenaltyVectorFactory = Callable[[int], Sequence[float]]
ConstraintFunction = Callable[[dict[str, float | int]], Sequence[float] | float]
ObjectiveExtractor = Callable[[object], Sequence[float]]
ContextEnricher = Callable[[dict[str, float | int]], dict[str, object]]


@dataclass(frozen=True)
class ParameterSpec:
	"""One integer design variable and how it maps to templates."""

	name: str
	lower: int
	upper: int
	scale: int = 1
	target: Literal["hardware", "workload", "both", "none"] = "hardware"
	template_key: str | None = None

	@property
	def key(self) -> str:
		return self.template_key or self.name


def default_objective_extractor(scme: object) -> list[float]:
	return [float(scme.energy), float(scme.latency), float(scme.accelerator.area)]


def default_penalty_vector(n_objectives: int) -> list[float]:
	return [1e30] * n_objectives


def extract_memory_areas(scme: object) -> dict[str, object]:
	"""Extract memory hierarchy data and area metadata from a STREAM result."""
	memory_data: dict[str, object] = {}
	try:
		core = scme.accelerator.get_core(0)
		memory_nodes = list(core.memory_hierarchy._node.keys())
		for i, mem_node in enumerate(memory_nodes):
			mem_instance = mem_node.memory_instance
			mem_dict = mem_instance.__dict__.copy()
			if "ports" in mem_dict:
				mem_dict["ports"] = [str(port) for port in mem_dict["ports"]]
			memory_data[f"memory_{i}"] = mem_dict
	except Exception as exc:  # pragma: no cover - best effort metadata extraction
		memory_data["error"] = str(exc)
	return memory_data


class StreamOptimizationProblem(Problem):
	"""
	Generic multi-objective optimization problem evaluated with STREAM.

	This class generalizes the depfin2 GA flow to arbitrary parameter counts:
	- hardware architecture is rendered from a Jinja2 template yaml
	- workload can be either a fixed file path or rendered from a template
	- a user-provided constraint function can link any parameters together
	"""

	def __init__(
		self,
		*,
		parameter_specs: Sequence[ParameterSpec],
		hardware_template_path: str | Path,
		base_hardware_dir: str | Path,
		mapping_path: str | Path,
		workload_path: str | Path | None = None,
		workload_template_path: str | Path | None = None,
		workload_template_name: str = "generated_workload.onnx",
		workload_fixed_params: dict[str, object] | None = None,
		constraint_fn: ConstraintFunction | None = None,
		n_link_constraints: int = 0,
		layer_stacks: Sequence[Sequence[int]] | None = None,
		experiment_id: str = "stream-ga",
		stream_mode: str = "fused",
		output_root: str | Path = "outputs",
		n_objectives: int = 3,
		objective_extractor: ObjectiveExtractor = default_objective_extractor,
		penalty_factory: PenaltyVectorFactory = default_penalty_vector,
		hardware_context_enricher: ContextEnricher | None = None,
		workload_context_enricher: ContextEnricher | None = None,
		normalize_objectives: bool = False,
		n_processes: int = 1,
	):
		if not parameter_specs:
			raise ValueError("parameter_specs must not be empty.")
		if workload_path is None and workload_template_path is None:
			raise ValueError(
				"Provide either workload_path or workload_template_path for STREAM evaluation."
			)
		if workload_path is not None and workload_template_path is not None:
			raise ValueError(
				"Provide only one of workload_path or workload_template_path, not both."
			)
		if n_link_constraints < 0:
			raise ValueError("n_link_constraints must be >= 0.")

		self.parameter_specs = list(parameter_specs)
		self.hardware_template_path = Path(hardware_template_path)
		self.base_hardware_dir = Path(base_hardware_dir)
		self.mapping_path = Path(mapping_path)
		self.workload_path = None if workload_path is None else Path(workload_path)
		self.workload_template_path = (
			None if workload_template_path is None else Path(workload_template_path)
		)
		self.workload_template_name = workload_template_name
		self.workload_fixed_params = workload_fixed_params or {}
		self.constraint_fn = constraint_fn
		self.n_link_constraints = n_link_constraints
		self.layer_stacks = [tuple(stack) for stack in layer_stacks] if layer_stacks else None
		self.experiment_id = experiment_id
		self.stream_mode = stream_mode
		self.output_root = Path(output_root)
		self.objective_extractor = objective_extractor
		self.penalty_factory = penalty_factory
		self.hardware_context_enricher = hardware_context_enricher
		self.workload_context_enricher = workload_context_enricher
		self.normalize_objectives = normalize_objectives
		self.n_processes = max(1, int(n_processes))

		self.run_uid = generate_run_id()
		self.root_dir = self.output_root / experiment_id / f"run_{self.run_uid}"
		self.root_dir.mkdir(parents=True, exist_ok=True)

		self.cache: dict[tuple[int, ...], list[float]] = {}
		self.evaluations: list[dict[str, object]] = []
		self.last_normalization_bounds: list[tuple[float, float] | None] = [
			None
		] * n_objectives
		self.generation_idx = 0

		super().__init__(
			n_var=len(self.parameter_specs),
			n_obj=n_objectives,
			n_constr=n_link_constraints + 1,
			xl=[spec.lower for spec in self.parameter_specs],
			xu=[spec.upper for spec in self.parameter_specs],
			vtype=int,
			elementwise_evaluation=False,
		)

	def denormalize_f(self, f_values: Sequence[float]) -> list[float]:
		"""Convert normalized objective values back to physical units."""
		denorm: list[float] = []
		for obj_idx, value in enumerate(f_values):
			scalar_value = float(value)
			bounds = self.last_normalization_bounds[obj_idx]
			if scalar_value >= 1e20 or bounds is None:
				denorm.append(scalar_value)
				continue

			obj_min, obj_max = bounds
			if obj_max > obj_min:
				denorm.append(scalar_value * (obj_max - obj_min) + obj_min)
			else:
				denorm.append(obj_min)
		return denorm

	def _decode_parameters(
		self, x: Sequence[int]
	) -> tuple[dict[str, float | int], dict[str, object], dict[str, object]]:
		full_params: dict[str, float | int] = {}
		hardware_context: dict[str, object] = {}
		workload_context: dict[str, object] = dict(self.workload_fixed_params)

		for value, spec in zip(x, self.parameter_specs, strict=True):
			unit_value = int(value)
			real_value = unit_value * int(spec.scale)
			full_params[f"{spec.name}_units"] = unit_value
			full_params[spec.name] = real_value

			if spec.target in ("hardware", "both"):
				hardware_context[spec.key] = real_value
			if spec.target in ("workload", "both"):
				workload_context[spec.key] = real_value

		if self.hardware_context_enricher is not None:
			extra_hardware = self.hardware_context_enricher(full_params)
			if extra_hardware:
				hardware_context.update(extra_hardware)
				full_params.update(extra_hardware)

		if self.workload_context_enricher is not None:
			extra_workload = self.workload_context_enricher(full_params)
			if extra_workload:
				workload_context.update(extra_workload)
				full_params.update(extra_workload)

		return full_params, hardware_context, workload_context

	def _build_variant_hardware_dir(
		self, eval_dir: Path, hardware_context: dict[str, object]
	) -> Path:
		variant_dir = eval_dir / "hardware"
		if variant_dir.exists():
			shutil.rmtree(variant_dir)
		shutil.copytree(self.base_hardware_dir, variant_dir)

		destination = variant_dir / "cores" / "core.yaml"
		render_template_to_file(self.hardware_template_path, destination, hardware_context)
		return variant_dir / "soc.yaml"

	def _resolve_workload(self, eval_dir: Path, workload_context: dict[str, object]) -> Path:
		if self.workload_path is not None:
			return self.workload_path

		workload_dir = eval_dir / "workload"
		workload_dir.mkdir(parents=True, exist_ok=True)
		workload_out = workload_dir / self.workload_template_name
		render_template_to_file(self.workload_template_path, workload_out, workload_context)
		return workload_out

	def _evaluate_single(self, task: tuple[str, np.ndarray]):
		eval_id, x_np = task
		x = [int(v) for v in x_np.tolist()]
		eval_dir = self.root_dir / eval_id
		eval_dir.mkdir(parents=True, exist_ok=True)

		params, hardware_context, workload_context = self._decode_parameters(x)
		key = tuple(x)

		g_link = [0.0] * self.n_link_constraints
		if self.constraint_fn is not None:
			raw_constraints = self.constraint_fn(params)
			if isinstance(raw_constraints, (int, float)):
				g_link = [float(raw_constraints)]
			else:
				g_link = [float(v) for v in raw_constraints]

			if len(g_link) != self.n_link_constraints:
				raise ValueError(
					f"constraint_fn returned {len(g_link)} constraints, expected {self.n_link_constraints}."
				)

		evaluation_failed = False
		memory_areas: dict[str, object] | None = None

		if key in self.cache:
			f = self.cache[key]
			evaluation_failed = all(value >= 1e20 for value in f)
			return f, g_link, params, memory_areas, evaluation_failed

		if any(val > 0 for val in g_link):
			f = list(self.penalty_factory(self.n_obj))
			self.cache[key] = f
			return f, g_link, params, memory_areas, False

		try:
			hardware_soc = self._build_variant_hardware_dir(eval_dir, hardware_context)
			resolved_workload = self._resolve_workload(eval_dir, workload_context)

			rel_path = eval_dir.relative_to(self.output_root)
			scme = optimize_allocation_co(
				hardware=str(hardware_soc),
				workload=str(resolved_workload),
				mapping=str(self.mapping_path),
				mode=self.stream_mode,
				layer_stacks=self.layer_stacks,
				experiment_id=str(rel_path),
				output_path=str(self.output_root),
				skip_if_exists=False,
			)
			f = [float(v) for v in self.objective_extractor(scme)]
			memory_areas = extract_memory_areas(scme)
		except Exception as exc:
			f = list(self.penalty_factory(self.n_obj))
			evaluation_failed = True
			(eval_dir / "error.txt").write_text(str(exc), encoding="utf-8")

		self.cache[key] = f
		return f, g_link, params, memory_areas, evaluation_failed

	def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
		eval_tasks = [
			(f"g{self.generation_idx:03d}_i{i:03d}", X[i])
			for i in range(len(X))
		]

		if self.n_processes > 1:
			with ThreadPoolExecutor(max_workers=self.n_processes) as executor:
				results = list(executor.map(self._evaluate_single, eval_tasks))
		else:
			results = [self._evaluate_single(task) for task in eval_tasks]

		F: list[list[float]] = []
		G: list[list[float]] = []

		for (eval_id, x), (f, g_link, params, memory_areas, evaluation_failed) in zip(
			eval_tasks, results, strict=True
		):
			F.append(f)
			fail_constraint = 1.0 if evaluation_failed else 0.0
			constraints = list(g_link) + [fail_constraint]
			G.append(constraints)

			record: dict[str, object] = {
				"eval_id": eval_id,
				"x": [int(v) for v in np.asarray(x).tolist()],
				"F": [float(v) for v in f],
				"G": [float(v) for v in constraints],
				"params": params,
			}
			if memory_areas is not None:
				record["memory_areas"] = memory_areas
			self.evaluations.append(record)

		self.generation_idx += 1

		F_array = np.asarray(F, dtype=float)

		if self.normalize_objectives:
			self.last_normalization_bounds = [None] * self.n_obj
			for obj_idx in range(self.n_obj):
				values = F_array[:, obj_idx]
				valid_mask = values < 1e20
				valid_values = values[valid_mask]

				if valid_values.size == 0:
					continue

				obj_min = float(np.min(valid_values))
				obj_max = float(np.max(valid_values))
				self.last_normalization_bounds[obj_idx] = (obj_min, obj_max)
				if obj_max > obj_min:
					F_array[valid_mask, obj_idx] = (valid_values - obj_min) / (
						obj_max - obj_min
					)

		valid_mask = np.all(F_array < 1e20, axis=1)
		valid_count = int(np.sum(valid_mask))
		failed_count = len(F_array) - valid_count
		logging.info(
			"Batch evaluated: %s designs, %s valid, %s failed",
			len(F_array),
			valid_count,
			failed_count,
		)

		out["F"] = F_array
		out["G"] = np.asarray(G, dtype=float)
