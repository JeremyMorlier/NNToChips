from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from zigzag.utils import pickle_deepcopy

from stream.stages.generation.tiled_workload_generation import (
    TiledWorkloadGenerationStage as StreamTiledWorkloadGenerationStage,
)
from stream.stages.stage import Stage
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.mapping import TILING_T, TILING_WILDCARD_T
from stream.workload.onnx_workload import ComputationNodeWorkload, ONNXWorkload

from stages.tiling_combination_evaluation import TilingCombinationEvaluationStage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TilingChoiceSpec:
    """One discrete GA variable selecting a tiling option for a specific node and tiling kind."""

    node_id: int
    kind: Literal["intra", "inter"]
    choices: tuple[TILING_T | TILING_WILDCARD_T, ...]


class _TilingGAProblem(Problem):
    """Pymoo problem wrapper for tiled-workload tiling optimization."""

    def __init__(
        self,
        stage: "TilingGeneticOptimizationStage",
        choice_specs: list[TilingChoiceSpec],
    ):
        self.stage = stage
        self.choice_specs = choice_specs
        self.best_score = float("inf")
        self.best_vector: tuple[int, ...] | None = None
        self.best_metrics: dict[str, int] | None = None

        super().__init__(
            n_var=len(choice_specs),
            n_obj=4,  # num_tiles, total_edges, inter_edges, inter_bits
            n_constr=0,
            xl=np.zeros(len(choice_specs), dtype=int),
            xu=np.array([len(spec.choices) - 1 for spec in choice_specs], dtype=int),
            vtype=int,
            elementwise_evaluation=False,
        )

    def _evaluate(self, X: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any):
        objective_rows: list[list[float]] = []

        for x in X:
            vector = tuple(int(v) for v in np.asarray(x, dtype=int).tolist())
            metrics = self.stage.evaluate_vector(vector)
            if metrics is None:
                f = [1e30, 1e30, 1e30, 1e30]
            else:
                f = [
                    float(metrics["num_tiles"]),
                    float(metrics["total_edges"]),
                    float(metrics["inter_edges"]),
                    float(metrics["inter_bits"]),
                ]

            score = self.stage._score(
                {
                    "num_tiles": int(f[0]),
                    "total_edges": int(f[1]),
                    "inter_edges": int(f[2]),
                    "inter_bits": int(f[3]),
                    "total_bits": int(f[3]),
                }
            )
            if metrics is not None and score < self.best_score:
                self.best_score = score
                self.best_vector = vector
                self.best_metrics = metrics

            objective_rows.append(f)

        out["F"] = np.asarray(objective_rows, dtype=float)


class _CollectTiledWorkloadLeaf(Stage):
    """Leaf stage used to extract tiled workload from Stream TiledWorkloadGenerationStage."""

    def is_leaf(self) -> bool:
        return True

    def run(self):
        yield None, {"workload": self.kwargs.get("workload")}


class TilingGeneticOptimizationStage(TilingCombinationEvaluationStage):
    """Optimize intra/inter tiling combinations with NSGA-II and heuristic scoring.

    Inherits parsing/normalization/scoring from TilingCombinationEvaluationStage.

    Extra kwargs:
    - tiling_ga_generations: int (default 6)
    - tiling_ga_population: int (default 24)
    - tiling_ga_seed: int (default 42)
    - tiling_force_inter_wildcard: bool (default True)
      If True, any inter-core factor > 1 is converted to '*' to stay compatible with
      downstream ConstraintOptimizationAllocationStage scheduling assumptions.
    """

    _EVAL_KWARGS = TilingCombinationEvaluationStage._EVAL_KWARGS.union(
        {
            "tiling_ga_generations",
            "tiling_ga_population",
            "tiling_ga_seed",
            "tiling_force_inter_wildcard",
        }
    )

    def __init__(self, list_of_callables, **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.ga_generations = int(kwargs.get("tiling_ga_generations", 6))
        self.ga_population = int(kwargs.get("tiling_ga_population", 24))
        self.ga_seed = int(kwargs.get("tiling_ga_seed", 42))
        self.force_inter_wildcard = bool(kwargs.get("tiling_force_inter_wildcard", True))

        self._choice_specs: list[TilingChoiceSpec] = []
        self._default_candidate: dict[int, dict[str, Any]] = {}
        self._metrics_cache: dict[tuple[int, ...], dict[str, int] | None] = {}
        self._candidate_cache: dict[tuple[int, ...], dict[int, dict[str, Any]]] = {}
        self._vector_eval_ids: dict[tuple[int, ...], int] = {}
        self._next_eval_idx = 0
        self._eval_tmp_dir = "/tmp"

    def run(self):
        self._prepare_search_space(self.workload)

        with tempfile.TemporaryDirectory(prefix="tiling_ga_eval_") as tmp_dir:
            self._eval_tmp_dir = tmp_dir

            if not self._choice_specs:
                logger.info("No GA decision variables found. Falling back to current tilings.")
                candidate = self._clone_candidate(self._default_candidate)
                tiled_workload, scheduling_order = self._build_tiled_workload(
                    candidate, 0, self._eval_tmp_dir
                )
                metrics = self._compute_heuristics(tiled_workload)
                evaluations = [
                    {
                        "candidate_index": 0,
                        "candidate": self._serialize_candidate(candidate),
                        "metrics": metrics,
                        "score": self._score(metrics),
                    }
                ]
                yield from self._forward_best(
                    candidate, tiled_workload, scheduling_order, evaluations
                )
                yield None, None
                return

            logger.info(
                "Starting tiling NSGA-II: vars=%d, pop=%d, generations=%d",
                len(self._choice_specs),
                self.ga_population,
                self.ga_generations,
            )

            problem = _TilingGAProblem(self, self._choice_specs)
            algorithm = NSGA2(pop_size=self.ga_population)
            result = minimize(
                problem,
                algorithm,
                termination=("n_gen", self.ga_generations),
                seed=self.ga_seed,
                verbose=False,
            )

            if problem.best_vector is None:
                raise RuntimeError("Tiling GA did not produce any valid candidate.")

            best_candidate = self._decode_vector(problem.best_vector)
            best_metrics = self.evaluate_vector(problem.best_vector)
            if best_metrics is None:
                raise RuntimeError("Best tiling candidate could not be re-evaluated.")
            best_candidate_idx = self._vector_eval_ids.get(problem.best_vector)
            if best_candidate_idx is None:
                best_candidate_idx = self._next_eval_idx
                self._next_eval_idx += 1
                self._vector_eval_ids[problem.best_vector] = best_candidate_idx

            tiled_workload, scheduling_order = self._build_tiled_workload(
                best_candidate,
                best_candidate_idx,
                self._eval_tmp_dir,
            )

            evaluations = self._build_history(result)
            logger.info(
                "Best tiling GA score %.4f for candidate %s",
                float(problem.best_score),
                self._serialize_candidate(best_candidate),
            )

            yield from self._forward_best(
                best_candidate, tiled_workload, scheduling_order, evaluations
            )
            yield None, None

    def _prepare_search_space(self, workload: ONNXWorkload) -> None:
        self._choice_specs.clear()
        self._default_candidate.clear()
        self._metrics_cache.clear()
        self._candidate_cache.clear()
        self._vector_eval_ids.clear()
        self._next_eval_idx = 0

        for node in workload.topological_sort():
            if not isinstance(node, ComputationNode):
                continue

            default_intra = list(node.intra_core_tiling)
            default_inter = list(node.inter_core_tiling)
            self._default_candidate[node.id] = {
                "intra": default_intra,
                "inter": default_inter,
            }

            intra_raw_options = self._lookup_node_options(self.intra_tiling_options, node.id)
            if intra_raw_options:
                intra_choices = tuple(
                    self._normalize_intra_tiling(node, option) for option in intra_raw_options
                )
                if len(intra_choices) > 1:
                    self._choice_specs.append(
                        TilingChoiceSpec(node_id=node.id, kind="intra", choices=intra_choices)
                    )
                elif len(intra_choices) == 1:
                    self._default_candidate[node.id]["intra"] = list(intra_choices[0])

            inter_raw_options = self._lookup_node_options(self.inter_tiling_options, node.id)
            if inter_raw_options:
                inter_choices = tuple(
                    self._normalize_inter_tiling(node, option) for option in inter_raw_options
                )
                if len(inter_choices) > 1:
                    self._choice_specs.append(
                        TilingChoiceSpec(node_id=node.id, kind="inter", choices=inter_choices)
                    )
                elif len(inter_choices) == 1:
                    self._default_candidate[node.id]["inter"] = list(inter_choices[0])

    def _clone_candidate(self, candidate: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
        cloned: dict[int, dict[str, Any]] = {}
        for node_id, cfg in candidate.items():
            cloned[node_id] = {
                "intra": list(cfg["intra"]),
                "inter": list(cfg["inter"]),
            }
        return cloned

    def _decode_vector(self, vector: tuple[int, ...]) -> dict[int, dict[str, Any]]:
        if vector in self._candidate_cache:
            return self._clone_candidate(self._candidate_cache[vector])

        candidate = self._clone_candidate(self._default_candidate)
        for idx, spec in enumerate(self._choice_specs):
            choice_idx = int(vector[idx])
            selected = spec.choices[choice_idx]
            candidate[spec.node_id][spec.kind] = list(selected)

        self._candidate_cache[vector] = self._clone_candidate(candidate)
        return candidate

    def evaluate_vector(self, vector: tuple[int, ...]) -> dict[str, int] | None:
        if vector in self._metrics_cache:
            return self._metrics_cache[vector]

        try:
            candidate = self._decode_vector(vector)
            candidate_idx = self._vector_eval_ids.get(vector)
            if candidate_idx is None:
                candidate_idx = self._next_eval_idx
                self._next_eval_idx += 1
                self._vector_eval_ids[vector] = candidate_idx
            tiled_workload, _ = self._build_tiled_workload(
                candidate, candidate_idx, self._eval_tmp_dir
            )
            metrics = self._compute_heuristics(tiled_workload)
            self._metrics_cache[vector] = metrics
            return metrics
        except Exception:
            logger.exception("Tiling candidate evaluation failed for vector=%s", vector)
            self._metrics_cache[vector] = None
            return None

    def _build_history(self, result: Any) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        if getattr(result, "X", None) is None:
            return history

        xs = result.X.tolist() if hasattr(result.X, "tolist") else [result.X]
        for idx, x in enumerate(xs):
            vector = tuple(int(v) for v in np.asarray(x, dtype=int).tolist())
            metrics = self.evaluate_vector(vector)
            if metrics is None:
                continue
            candidate = self._decode_vector(vector)
            history.append(
                {
                    "candidate_index": idx,
                    "candidate": self._serialize_candidate(candidate),
                    "metrics": metrics,
                    "score": self._score(metrics),
                }
            )
        return history

    def _forward_best(
        self,
        best_candidate: dict[int, dict[str, Any]],
        best_tiled_workload,
        best_scheduling_order: Any,
        evaluations: list[dict[str, Any]],
    ):
        best_original_workload = self._materialize_candidate_workload(best_candidate)
        kwargs = self.kwargs.copy()
        kwargs["original_workload"] = best_original_workload
        kwargs["workload"] = best_tiled_workload
        kwargs["accelerator"] = self.accelerator
        kwargs["selected_tiling_candidate"] = self._serialize_candidate(best_candidate)
        kwargs["tiling_candidate_evaluations"] = evaluations
        # Avoid propagating any stale scheduling order into downstream stages.
        kwargs.pop("scheduling_order", None)

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for item in sub_stage.run():
            yield item

    def _build_tiled_workload(
        self,
        candidate: dict[int, dict[str, Any]],
        candidate_idx: int,
        tmp_dir: str,
    ) -> tuple[ComputationNodeWorkload, Any]:
        """Build tiled workload with Stream's native TiledWorkloadGenerationStage for consistency downstream."""
        candidate_workload: ONNXWorkload = pickle_deepcopy(self.workload)

        for node in candidate_workload.topological_sort():
            if not isinstance(node, ComputationNode):
                continue
            if node.id not in candidate:
                continue
            node.intra_core_tiling = list(candidate[node.id]["intra"])
            node.inter_core_tiling = list(candidate[node.id]["inter"])

        candidate_tiled_path = f"{tmp_dir}/tiled_workload_candidate_{candidate_idx}.pickle"
        stage_kwargs = self._build_tiled_stage_kwargs(candidate_workload, candidate_tiled_path)
        tiled_stage = StreamTiledWorkloadGenerationStage([_CollectTiledWorkloadLeaf], **stage_kwargs)

        tiled_workload: ComputationNodeWorkload | None = None
        for _, extra in tiled_stage.run():
            if isinstance(extra, dict) and isinstance(extra.get("workload"), ComputationNodeWorkload):
                tiled_workload = extra["workload"]

        if tiled_workload is None:
            raise RuntimeError(f"Failed to retrieve tiled workload for candidate {candidate_idx}.")

        return tiled_workload, None

    def _materialize_candidate_workload(
        self, candidate: dict[int, dict[str, Any]]
    ) -> ONNXWorkload:
        """Return an ONNX workload where node tilings match the provided candidate."""
        candidate_workload: ONNXWorkload = pickle_deepcopy(self.workload)
        for node in candidate_workload.topological_sort():
            if not isinstance(node, ComputationNode):
                continue
            cfg = candidate.get(node.id)
            if cfg is None:
                continue
            node.intra_core_tiling = list(cfg["intra"])
            node.inter_core_tiling = list(cfg["inter"])
        return candidate_workload

    def _normalize_inter_tiling(
        self,
        node: ComputationNode,
        raw_tiling: list[tuple[Any, Any]],
    ) -> TILING_WILDCARD_T | TILING_T:
        """Normalize inter tiling and enforce CO-compatible wildcard form when requested."""
        tiling = super()._normalize_inter_tiling(node, raw_tiling)
        if not self.force_inter_wildcard:
            return tiling

        coerced: TILING_WILDCARD_T = []
        for dim, factor in tiling:
            # Keep explicit 1; convert any real split request to wildcard for CO compatibility.
            if isinstance(factor, int) and factor > 1:
                coerced.append((dim, "*"))
            else:
                coerced.append((dim, factor))
        return coerced
