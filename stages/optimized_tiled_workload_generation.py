import logging
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any, Literal

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
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


class _GraphTilingProblem(Problem):
    """Pymoo problem wrapper for whole-graph tiling optimization."""

    def __init__(
        self,
        stage: "OptimizedTiledWorkloadGenerationStage",
        nodes: list[ComputationNode],
        candidates_per_node: list[list[_TilingCandidate]],
    ):
        self.stage = stage
        self.nodes = nodes
        self.candidates_per_node = candidates_per_node
        self.best_cost = float("inf")
        self.best_assignment: dict[ComputationNode, _TilingCandidate] | None = None
        self.evaluation_count = 0

        option_sizes = [len(candidates) for candidates in candidates_per_node]
        super().__init__(
            n_var=len(option_sizes),
            n_obj=1,
            n_constr=0,
            xl=np.zeros(len(option_sizes), dtype=int),
            xu=np.asarray([size - 1 for size in option_sizes], dtype=int),
            vtype=int,
            elementwise_evaluation=False,
        )

    def _decode_assignment(
        self, x: np.ndarray
    ) -> dict[ComputationNode, _TilingCandidate]:
        assignment: dict[ComputationNode, _TilingCandidate] = {}
        for i, node in enumerate(self.nodes):
            candidate_idx = int(x[i])
            assignment[node] = self.candidates_per_node[i][candidate_idx]
        return assignment

    def _evaluate(self, X: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any):
        costs: list[list[float]] = []
        for x in X:
            assignment = self._decode_assignment(np.asarray(x, dtype=int))
            cost = self.stage._graph_dependency_cost(assignment)
            self.evaluation_count += 1
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_assignment = assignment
            costs.append([float(cost)])
        out["F"] = np.asarray(costs, dtype=float)


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
        self.tiling_ga_generations: int = kwargs.get("tiling_ga_generations", 6)
        self.tiling_ga_population: int = kwargs.get("tiling_ga_population", 16)
        self.tiling_ga_seed: int = kwargs.get("tiling_ga_seed", 0)

    def run(self):
        self.optimize_layer_tilings()
        yield from super().run()

    def optimize_layer_tilings(self):
        if self.optimization_method == "ga":
            self._optimize_layer_tilings_ga_global()
            return

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

    def _optimize_layer_tilings_ga_global(self) -> None:
        nodes: list[ComputationNode] = [
            node for node in self.workload.node_list if isinstance(node, ComputationNode)
        ]
        if not nodes:
            return

        candidates_per_node: list[list[_TilingCandidate]] = []
        optimizable_nodes: list[ComputationNode] = []
        for node in nodes:
            inter_options, intra_options = self._build_tiling_options(node)
            if not inter_options or not intra_options:
                continue
            candidates = self._build_tiling_candidates(inter_options, intra_options)
            if not candidates:
                continue
            optimizable_nodes.append(node)
            candidates_per_node.append(candidates)

        if not optimizable_nodes:
            logger.warning("No optimizable computation nodes found for global GA tiling.")
            return

        problem = _GraphTilingProblem(self, optimizable_nodes, candidates_per_node)
        algorithm = NSGA2(pop_size=self.tiling_ga_population)
        minimize(
            problem,
            algorithm,
            termination=("n_gen", self.tiling_ga_generations),
            seed=self.tiling_ga_seed,
            verbose=False,
        )

        if problem.best_assignment is None:
            logger.warning("Global GA did not produce a valid tiling assignment.")
            return

        for node in optimizable_nodes:
            candidate = problem.best_assignment[node]
            node.inter_core_tiling = list(candidate.inter_tiling)
            node.intra_core_tiling = list(candidate.intra_tiling)
            logger.info(
                "Global-GA optimized tiling for %s -> inter=%s intra=%s (possible_core_allocation unchanged=%s)",
                node,
                node.inter_core_tiling,
                node.intra_core_tiling,
                node.possible_core_allocation,
            )

    def _select_best_candidate(self, node: ComputationNode) -> _TilingCandidate | None:
        inter_options, intra_options = self._build_tiling_options(node)
        if not inter_options or not intra_options:
            return None

        return self._select_best_candidate_exhaustive(
            node, inter_options, intra_options
        )

    def _select_best_candidate_exhaustive(
        self,
        node: ComputationNode,
        inter_options: list[list[tuple[LayerDim, int]]],
        intra_options: list[list[tuple[LayerDim, int]]],
    ) -> _TilingCandidate | None:
        best_cost = float("inf")
        best_candidate: _TilingCandidate | None = None
        for candidate in self._build_tiling_candidates(inter_options, intra_options):
            cost = self._dependency_aware_cost(node, candidate)
            if cost < best_cost:
                best_cost = cost
                best_candidate = candidate
        return best_candidate

    def _build_tiling_options(
        self, node: ComputationNode
    ) -> tuple[list[list[tuple[LayerDim, int]]], list[list[tuple[LayerDim, int]]]]:
        inter_dims = self._get_inter_dims(node)[: self.max_inter_dims_per_layer]
        intra_dims = self._get_intra_dims(node)[: self.max_intra_dims_per_layer]
        if not inter_dims or not intra_dims:
            return [], []

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

        return inter_options_per_dim, intra_options_per_dim

    def _build_tiling_candidates(
        self,
        inter_options_per_dim: list[list[tuple[LayerDim, int]]],
        intra_options_per_dim: list[list[tuple[LayerDim, int]]],
    ) -> list[_TilingCandidate]:
        candidates: list[_TilingCandidate] = []
        for inter_combo in product(*inter_options_per_dim):
            inter_tiling = tuple(
                (dim, factor) for dim, factor in inter_combo if factor > 1
            )
            if not inter_tiling:
                inter_tiling = (inter_options_per_dim[0][0],)

            for intra_combo in product(*intra_options_per_dim):
                intra_tiling = tuple(
                    (dim, factor) for dim, factor in intra_combo if factor > 1
                )
                if not intra_tiling:
                    intra_tiling = (intra_options_per_dim[0][0],)

                candidates.append(
                    _TilingCandidate(
                        inter_tiling=inter_tiling, intra_tiling=intra_tiling
                    )
                )
                if len(candidates) >= self.max_tiling_candidates_per_layer:
                    return candidates

        return candidates

    def _dependency_aware_cost(
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
        for pred in self._iter_predecessors(node):
            if isinstance(pred, ComputationNode):
                neighbor_overlap += len(
                    inter_dims.intersection(set(pred.layer_dim_sizes))
                )
        for succ in self._iter_successors(node):
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

        # Lower cost is better: maximize useful locality/fusion while limiting tile explosion.
        return tile_penalty - (intra_score + inter_score)

    def _graph_dependency_cost(
        self, assignment: dict[ComputationNode, _TilingCandidate]
    ) -> float:
        total_cost = 0.0
        for node, candidate in assignment.items():
            total_cost += self._dependency_aware_cost(node, candidate)

        # Reward consistency of inter-core tiling choices across graph edges.
        coupling_reward = 0.0
        for src, dst in self._iter_edges():
            if not isinstance(src, ComputationNode) or not isinstance(dst, ComputationNode):
                continue
            if src not in assignment or dst not in assignment:
                continue

            src_inter = dict(assignment[src].inter_tiling)
            dst_inter = dict(assignment[dst].inter_tiling)
            shared_dims = set(src_inter).intersection(set(dst_inter))
            if not shared_dims:
                continue

            for dim in shared_dims:
                src_factor = src_inter[dim]
                dst_factor = dst_inter[dim]
                ratio = min(src_factor, dst_factor) / max(src_factor, dst_factor)
                coupling_reward += 1.0 + ratio

        return total_cost - coupling_reward

    def _iter_predecessors(self, node: ComputationNode):
        predecessors = getattr(self.workload, "predecessors", None)
        if predecessors is None:
            return []
        return predecessors(node)

    def _iter_successors(self, node: ComputationNode):
        successors = getattr(self.workload, "successors", None)
        if successors is None:
            return []
        return successors(node)

    def _iter_edges(self):
        edges = getattr(self.workload, "edges", None)
        if edges is None:
            return []
        return edges()

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
