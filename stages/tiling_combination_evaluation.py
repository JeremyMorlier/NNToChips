import itertools
import logging
import os
import tempfile
from typing import Any

from zigzag.datatypes import LayerDim
from zigzag.utils import pickle_deepcopy

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.mapping import TILING_T, TILING_WILDCARD_T
from stream.workload.onnx_workload import ComputationNodeWorkload, ONNXWorkload

from stages.tiled_workload_generation_optimization import TiledWorkloadGenerationStage

logger = logging.getLogger(__name__)

RawTilingEntry = tuple[LayerDim | str, int | str]
RawTiling = list[RawTilingEntry]


class _CollectTiledWorkloadLeaf(Stage):
    """Leaf stage used to extract the tiled workload emitted by TiledWorkloadGenerationStage."""

    def is_leaf(self) -> bool:
        return True

    def run(self):
        yield None, {
            "workload": self.kwargs.get("workload"),
            "scheduling_order": self.kwargs.get("scheduling_order"),
        }


class TilingCombinationEvaluationStage(Stage):
    """Evaluate multiple intra/inter-core tiling combinations using tiled-workload heuristics.

    Expected kwargs:
    - workload: ONNXWorkload
    - accelerator: Accelerator
    - tiled_workload_path: str

    Optional kwargs:
    - intra_tiling_options: dict[node_id, list[tiling]]
    - inter_tiling_options: dict[node_id, list[tiling]]
      where a tiling is list[(layer_dim, factor)] and layer_dim can be str or LayerDim.
    - tiling_max_combinations: int (default 128)
    - tiling_weight_tiles: float (default 1.0)
    - tiling_weight_edges: float (default 1.0)
    - tiling_weight_inter_edges: float (default 2.0)
    - tiling_weight_inter_bits: float (default 1e-6)
    """

    _EVAL_KWARGS = {
        "intra_tiling_options",
        "inter_tiling_options",
        "tiling_max_combinations",
        "tiling_weight_tiles",
        "tiling_weight_edges",
        "tiling_weight_inter_edges",
        "tiling_weight_inter_bits",
    }

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ONNXWorkload,
        accelerator: Accelerator,
        tiled_workload_path: str,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.tiled_workload_path = tiled_workload_path

        self.intra_tiling_options: dict[int | str, list[RawTiling]] = dict(
            kwargs.get("intra_tiling_options", {})
        )
        self.inter_tiling_options: dict[int | str, list[RawTiling]] = dict(
            kwargs.get("inter_tiling_options", {})
        )

        self.max_combinations = int(kwargs.get("tiling_max_combinations", 128))

        self.weight_tiles = float(kwargs.get("tiling_weight_tiles", 1.0))
        self.weight_edges = float(kwargs.get("tiling_weight_edges", 1.0))
        self.weight_inter_edges = float(kwargs.get("tiling_weight_inter_edges", 2.0))
        self.weight_inter_bits = float(kwargs.get("tiling_weight_inter_bits", 1e-6))

    def run(self):
        candidates = self._enumerate_candidates(self.workload)
        if not candidates:
            raise ValueError("No tiling candidates generated. Check intra/inter tiling options.")

        logger.info("Evaluating %d tiling candidate(s).", len(candidates))

        best_score = float("inf")
        best_tiled_workload: ComputationNodeWorkload | None = None
        best_scheduling_order: Any = None
        best_candidate: dict[int, dict[str, Any]] | None = None
        evaluations: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="tiling_eval_") as tmp_dir:
            for idx, candidate in enumerate(candidates):
                tiled_workload, scheduling_order = self._build_tiled_workload(
                    candidate,
                    idx,
                    tmp_dir,
                )
                metrics = self._compute_heuristics(tiled_workload)
                score = self._score(metrics)
                evaluations.append(
                    {
                        "candidate_index": idx,
                        "candidate": self._serialize_candidate(candidate),
                        "metrics": metrics,
                        "score": score,
                    }
                )

                if score < best_score:
                    best_score = score
                    best_tiled_workload = tiled_workload
                    best_scheduling_order = scheduling_order
                    best_candidate = candidate

        if best_tiled_workload is None or best_candidate is None:
            raise RuntimeError("Failed to evaluate tiling candidates.")

        logger.info(
            "Selected tiling candidate with score %.4f: %s",
            best_score,
            self._serialize_candidate(best_candidate),
        )

        kwargs = self.kwargs.copy()
        kwargs["original_workload"] = pickle_deepcopy(self.workload)
        kwargs["workload"] = best_tiled_workload
        kwargs["accelerator"] = self.accelerator
        kwargs["selected_tiling_candidate"] = self._serialize_candidate(best_candidate)
        kwargs["tiling_candidate_evaluations"] = evaluations

        if "scheduling_order" not in kwargs:
            if best_scheduling_order is not None:
                kwargs["scheduling_order"] = best_scheduling_order
            else:
                kwargs["scheduling_order"] = TiledWorkloadGenerationStage.get_scheduling_order(
                    best_tiled_workload
                )

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()
        yield None, None

    def _lookup_node_options(
        self,
        options: dict[int | str, list[RawTiling]],
        node_id: int,
    ) -> list[RawTiling] | None:
        if node_id in options:
            return options[node_id]
        as_str = str(node_id)
        if as_str in options:
            return options[as_str]
        return None

    def _enumerate_candidates(self, workload: ONNXWorkload) -> list[dict[int, dict[str, Any]]]:
        per_node_choices: list[list[tuple[int, TILING_T, TILING_WILDCARD_T | TILING_T]]] = []

        for node in workload.topological_sort():
            if not isinstance(node, ComputationNode):
                continue

            intra_raw_options = self._lookup_node_options(self.intra_tiling_options, node.id)
            inter_raw_options = self._lookup_node_options(self.inter_tiling_options, node.id)

            intra_options = (
                [self._normalize_intra_tiling(node, option) for option in intra_raw_options]
                if intra_raw_options
                else [list(node.intra_core_tiling)]
            )
            inter_options = (
                [self._normalize_inter_tiling(node, option) for option in inter_raw_options]
                if inter_raw_options
                else [list(node.inter_core_tiling)]
            )

            node_choices: list[tuple[int, TILING_T, TILING_WILDCARD_T | TILING_T]] = []
            for intra, inter in itertools.product(intra_options, inter_options):
                node_choices.append((node.id, intra, inter))
            per_node_choices.append(node_choices)

        if not per_node_choices:
            return []

        all_combinations_iter = itertools.product(*per_node_choices)
        candidates: list[dict[int, dict[str, Any]]] = []

        for idx, combination in enumerate(all_combinations_iter):
            if idx >= self.max_combinations:
                logger.warning(
                    "Tiling combination search truncated to %d candidates.",
                    self.max_combinations,
                )
                break

            candidate: dict[int, dict[str, Any]] = {}
            for node_id, intra, inter in combination:
                candidate[node_id] = {
                    "intra": intra,
                    "inter": inter,
                }
            candidates.append(candidate)

        return candidates

    def _build_tiled_workload(
        self,
        candidate: dict[int, dict[str, Any]],
        candidate_idx: int,
        tmp_dir: str,
    ) -> tuple[ComputationNodeWorkload, Any]:
        candidate_workload: ONNXWorkload = pickle_deepcopy(self.workload)

        for node in candidate_workload.topological_sort():
            if not isinstance(node, ComputationNode):
                continue
            if node.id not in candidate:
                continue
            node.intra_core_tiling = list(candidate[node.id]["intra"])
            node.inter_core_tiling = list(candidate[node.id]["inter"])

        candidate_tiled_path = os.path.join(
            tmp_dir,
            f"tiled_workload_candidate_{candidate_idx}.pickle",
        )

        stage_kwargs = self._build_tiled_stage_kwargs(candidate_workload, candidate_tiled_path)
        tiled_stage = TiledWorkloadGenerationStage([_CollectTiledWorkloadLeaf], **stage_kwargs)

        tiled_workload: ComputationNodeWorkload | None = None
        scheduling_order: Any = None

        for _, extra in tiled_stage.run():
            if isinstance(extra, dict) and isinstance(extra.get("workload"), ComputationNodeWorkload):
                tiled_workload = extra["workload"]
                scheduling_order = extra.get("scheduling_order")

        if tiled_workload is None:
            raise RuntimeError(f"Failed to retrieve tiled workload for candidate {candidate_idx}.")

        return tiled_workload, scheduling_order

    def _build_tiled_stage_kwargs(
        self,
        workload: ONNXWorkload,
        tiled_workload_path: str,
    ) -> dict[str, Any]:
        stage_kwargs: dict[str, Any] = {}
        for key, value in self.kwargs.items():
            if key in self._EVAL_KWARGS:
                continue
            if key in {"workload", "accelerator", "tiled_workload_path"}:
                continue
            stage_kwargs[key] = value

        stage_kwargs["workload"] = workload
        stage_kwargs["accelerator"] = self.accelerator
        stage_kwargs["tiled_workload_path"] = tiled_workload_path
        return stage_kwargs

    def _normalize_intra_tiling(self, node: ComputationNode, raw_tiling: RawTiling) -> TILING_T:
        tiling: TILING_T = []
        for dim_raw, factor_raw in raw_tiling:
            dim = dim_raw if isinstance(dim_raw, LayerDim) else LayerDim(str(dim_raw))
            if dim not in node.layer_dim_sizes:
                continue
            if factor_raw == "all":
                factor = int(node.layer_dim_sizes[dim])
            elif isinstance(factor_raw, int):
                factor = factor_raw
            else:
                raise ValueError(
                    f"Invalid intra-core factor {factor_raw!r} for node {node.id}: expected int or 'all'."
                )
            tiling.append((dim, factor))
        return tiling

    def _normalize_inter_tiling(
        self,
        node: ComputationNode,
        raw_tiling: RawTiling,
    ) -> TILING_WILDCARD_T | TILING_T:
        tiling: TILING_WILDCARD_T = []
        for dim_raw, factor_raw in raw_tiling:
            dim = dim_raw if isinstance(dim_raw, LayerDim) else LayerDim(str(dim_raw))
            if dim not in node.layer_dim_sizes:
                continue
            if factor_raw in {"*", "all"}:
                factor = factor_raw
            elif isinstance(factor_raw, int):
                factor = factor_raw
            else:
                raise ValueError(
                    f"Invalid inter-core factor {factor_raw!r} for node {node.id}: expected int, '*', or 'all'."
                )
            tiling.append((dim, factor))
        return tiling

    def _compute_heuristics(self, tiled_workload: ComputationNodeWorkload) -> dict[str, int]:
        num_tiles = len(tiled_workload.node_list)
        edges = list(tiled_workload.edges(data=True))

        total_edges = len(edges)
        total_bits = 0
        inter_edges = 0
        inter_bits = 0

        for _, _, data in edges:
            bits_raw = data.get("bits", 0) if isinstance(data, dict) else 0
            bits = int(bits_raw)
            total_bits += bits
            if bits > 0:
                inter_edges += 1
                inter_bits += bits

        return {
            "num_tiles": num_tiles,
            "total_edges": total_edges,
            "inter_edges": inter_edges,
            "total_bits": total_bits,
            "inter_bits": inter_bits,
        }

    def _score(self, metrics: dict[str, int]) -> float:
        return (
            self.weight_tiles * float(metrics["num_tiles"])
            + self.weight_edges * float(metrics["total_edges"])
            + self.weight_inter_edges * float(metrics["inter_edges"])
            + self.weight_inter_bits * float(metrics["inter_bits"])
        )

    @staticmethod
    def _serialize_tiling(tiling: TILING_T | TILING_WILDCARD_T) -> list[tuple[str, int | str]]:
        return [(str(dim), factor) for dim, factor in tiling]

    def _serialize_candidate(self, candidate: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
        serialized: dict[int, dict[str, Any]] = {}
        for node_id, cfg in candidate.items():
            serialized[node_id] = {
                "intra": self._serialize_tiling(cfg["intra"]),
                "inter": self._serialize_tiling(cfg["inter"]),
            }
        return serialized
