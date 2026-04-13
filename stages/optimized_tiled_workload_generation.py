import logging
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any, Literal

from zigzag.datatypes import LayerDim

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.generation.tiled_workload_generation import (
    TiledWorkloadGenerationStage,
)
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.stage import StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ONNXWorkload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TilingCandidate:
    inter_tiling: tuple[tuple[LayerDim, int], ...]
    intra_tiling: tuple[tuple[LayerDim, int], ...]


class OptimizedTiledWorkloadGenerationStage(TiledWorkloadGenerationStage):
    """Dependency-aware tiling optimizer built on top of TiledWorkloadGenerationStage.

    This stage only optimizes intra/inter-core tilings.
    It intentionally does not optimize core allocation, which is handled in later stages.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ONNXWorkload,
        accelerator: Accelerator,
        tiled_workload_path: str,
        **kwargs: Any,
    ):
        super().__init__(
            list_of_callables,
            workload=workload,
            accelerator=accelerator,
            tiled_workload_path=tiled_workload_path,
            **kwargs,
        )
        self.optimization_method: Literal["ga", "ilp"] = kwargs.get(
            "optimization_method", "ga"
        )
        self.max_inter_core_factor: int = kwargs.get("max_inter_core_factor", 16)
        self.max_intra_core_factor: int = kwargs.get("max_intra_core_factor", 8)
        self.max_inter_dims_per_layer: int = kwargs.get("max_inter_dims_per_layer", 2)
        self.max_intra_dims_per_layer: int = kwargs.get("max_intra_dims_per_layer", 2)
        self.max_tiling_candidates_per_layer: int = kwargs.get(
            "max_tiling_candidates_per_layer", 256
        )

    def run(self):
        self.optimize_layer_tilings()
        yield from super().run()

    def optimize_layer_tilings(self):
        for node in self.workload.node_list:
            if not isinstance(node, ComputationNode):
                continue

            best_candidate = self._select_best_candidate(node)
            if best_candidate is None:
                logger.warning(
                    "No valid tiling candidate found for %s. Keeping existing tiling.",
                    node,
                )
                continue

            node.inter_core_tiling = list(best_candidate.inter_tiling)
            node.intra_core_tiling = list(best_candidate.intra_tiling)

            logger.info(
                "Optimized tiling for %s -> inter=%s intra=%s (possible_core_allocation unchanged=%s)",
                node,
                node.inter_core_tiling,
                node.intra_core_tiling,
                node.possible_core_allocation,
            )

    def _select_best_candidate(self, node: ComputationNode) -> _TilingCandidate | None:
        candidates = self._build_tiling_candidates(node)
        if not candidates:
            return None

        best_score = float("-inf")
        best_candidate: _TilingCandidate | None = None
        for candidate in candidates:
            score = self._dependency_aware_score(node, candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate

    def _build_tiling_candidates(self, node: ComputationNode) -> list[_TilingCandidate]:
        inter_dims = self._get_inter_dims(node)[: self.max_inter_dims_per_layer]
        intra_dims = self._get_intra_dims(node)[: self.max_intra_dims_per_layer]
        if not inter_dims or not intra_dims:
            return []

        inter_options_per_dim: list[list[tuple[LayerDim, int]]] = []
        for dim in inter_dims:
            factors = self._get_divisors_bounded(
                node.layer_dim_sizes[dim], self.max_inter_core_factor
            )
            factors = sorted(set([1] + factors))
            inter_options_per_dim.append([(dim, f) for f in factors])

        intra_options_per_dim: list[list[tuple[LayerDim, int]]] = []
        for dim in intra_dims:
            factors = self._get_divisors_bounded(
                node.layer_dim_sizes[dim], self.max_intra_core_factor
            )
            factors = sorted(set([1] + factors))
            intra_options_per_dim.append([(dim, f) for f in factors])

        candidates: list[_TilingCandidate] = []
        for inter_combo in product(*inter_options_per_dim):
            inter_tiling = tuple(
                (dim, factor) for dim, factor in inter_combo if factor > 1
            )
            if not inter_tiling:
                inter_tiling = ((inter_dims[0], 1),)

            for intra_combo in product(*intra_options_per_dim):
                intra_tiling = tuple(
                    (dim, factor) for dim, factor in intra_combo if factor > 1
                )
                if not intra_tiling:
                    preferred = self._get_preferred_intra_dim(node)
                    intra_tiling = ((preferred, 1),)

                candidates.append(
                    _TilingCandidate(
                        inter_tiling=inter_tiling, intra_tiling=intra_tiling
                    )
                )
                if len(candidates) >= self.max_tiling_candidates_per_layer:
                    return candidates

        return candidates

    def _dependency_aware_score(
        self, node: ComputationNode, candidate: _TilingCandidate
    ) -> float:
        inter_dims = {dim for dim, _ in candidate.inter_tiling}
        intra_factors = [factor for _, factor in candidate.intra_tiling]
        inter_factors = [factor for _, factor in candidate.inter_tiling]

        # Intra-layer score: prioritize stronger fusion opportunities.
        preferred_intra_dim = self._get_preferred_intra_dim(node)
        preferred_intra_boost = sum(
            1.0
            for dim, factor in candidate.intra_tiling
            if dim == preferred_intra_dim and factor > 1
        )
        intra_score = (
            sum((factor - 1) for factor in intra_factors) + 2.0 * preferred_intra_boost
        )

        # Inter-layer score: reward splitting dimensions that overlap with neighbors.
        neighbor_overlap = 0.0
        for pred in self.workload.predecessors(node):
            if isinstance(pred, ComputationNode):
                neighbor_overlap += len(
                    inter_dims.intersection(set(pred.layer_dim_sizes))
                )
        for succ in self.workload.successors(node):
            if isinstance(succ, ComputationNode):
                neighbor_overlap += len(
                    inter_dims.intersection(set(succ.layer_dim_sizes))
                )

        inter_score = sum((factor - 1) for factor in inter_factors) + neighbor_overlap

        # Penalize too many tiles to avoid exploding graph size.
        inter_tile_count = prod(inter_factors) if inter_factors else 1
        intra_tile_count = prod(intra_factors) if intra_factors else 1
        nb_tiles = inter_tile_count * intra_tile_count
        tile_penalty = 0.05 * nb_tiles

        return intra_score + inter_score - tile_penalty

    def _get_inter_dims(self, node: ComputationNode) -> list[LayerDim]:
        preferred = [
            dim
            for dim in TilingGenerationStage.INTER_CORE_PARTITION_DIM_DEFAULT
            if dim in node.layer_dim_sizes
        ]
        fallback = [
            dim
            for dim in node.layer_dim_sizes
            if node.layer_dim_sizes[dim] > 1 and dim != LayerDim("B")
        ]
        dims = preferred if preferred else fallback
        return [dim for dim in dims if node.layer_dim_sizes[dim] > 0]

    def _get_intra_dims(self, node: ComputationNode) -> list[LayerDim]:
        preferred = self._get_preferred_intra_dim(node)
        remaining = [
            dim
            for dim in node.layer_dim_sizes
            if dim not in (LayerDim("B"), LayerDim("G"), preferred)
        ]
        return [preferred] + remaining

    def _get_preferred_intra_dim(self, node: ComputationNode) -> LayerDim:
        preferred = TilingGenerationStage.FUSION_PARTITION_DIM_DEFAULT[node.type]
        if preferred in node.layer_dim_sizes:
            return preferred
        for dim in node.layer_dim_sizes:
            if dim not in (LayerDim("B"), LayerDim("G")):
                return dim
        return next(iter(node.layer_dim_sizes))

    @staticmethod
    def _get_divisors_bounded(value: int, upper_bound: int) -> list[int]:
        if value <= 0:
            return [1]
        divisors: list[int] = []
        max_k = min(value, upper_bound)
        for k in range(1, max_k + 1):
            if value % k == 0:
                divisors.append(k)
        return divisors
