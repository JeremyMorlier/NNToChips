import argparse
import itertools
import json
import logging
import os
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from math import isqrt, prod
from typing import Any, Literal

from onnx import ModelProto
from zigzag.datatypes import LayerDim

from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.tiled_workload_generation import (
    TiledWorkloadGenerationStage,
)
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import (
    ONNXModelParserStage as StreamONNXModelParserStage,
)
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload, ONNXWorkload

_logging_level = logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=_logging_level, format=_logging_format)
logger = logging.getLogger(__name__)

TILING_FACTOR_T = int | Literal["*", "all"]
TILING_SPEC_T = list[tuple[LayerDim, TILING_FACTOR_T]]


@dataclass(frozen=True)
class MappingChoice:
    label: str
    tiling: TILING_SPEC_T | None


class ParsedWorkloadCaptureStage(Stage):
    """Leaf stage used to capture the parsed ONNX workload before exploration."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ONNXWorkload,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload

    def run(self):
        yield None, self.workload

    def is_leaf(self) -> bool:
        return True


class MappingOverrideStage(Stage):
    """Apply mapping overrides directly on parsed workload nodes before tiling generation."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ONNXWorkload,
        intra_choice: MappingChoice,
        inter_choice: MappingChoice,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.intra_choice = intra_choice
        self.inter_choice = inter_choice

    def run(self):
        updated_nodes = 0
        for node in self.workload.node_list:
            if not isinstance(node, ComputationNode):
                continue

            if self.intra_choice.tiling is not None:
                node.intra_core_tiling = deepcopy(self.intra_choice.tiling)
            if self.inter_choice.tiling is not None:
                node.inter_core_tiling = deepcopy(self.inter_choice.tiling)
            updated_nodes += 1

        logger.info(
            "Applied mapping overrides to %d computation nodes (intra=%s, inter=%s).",
            updated_nodes,
            self.intra_choice.label,
            self.inter_choice.label,
        )

        kwargs = self.kwargs.copy()
        kwargs["workload"] = self.workload
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()


class TiledWorkloadInspectionStage(Stage):
    """Leaf stage that inspects and stores tiled-workload graph metrics."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        variant_id: str,
        summary_output_path: str,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.variant_id = variant_id
        self.summary_output_path = summary_output_path

    def run(self):
        summary = self.summarize_workload(self.workload, self.variant_id)
        os.makedirs(os.path.dirname(self.summary_output_path), exist_ok=True)
        with open(self.summary_output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Saved tiled workload summary to %s", self.summary_output_path)
        yield None, summary

    def is_leaf(self) -> bool:
        return True

    @staticmethod
    def summarize_workload(
        workload: ComputationNodeWorkload, variant_id: str
    ) -> dict[str, Any]:
        node_count = len(workload.node_list)
        edge_count = workload.number_of_edges()

        inter_layer_edges = 0
        intra_layer_edges = 0
        edge_bits_sum = 0

        for src, dst, edge_data in workload.edges(data=True):
            if src.id == dst.id:
                intra_layer_edges += 1
            else:
                inter_layer_edges += 1
            edge_bits_sum += int(edge_data.get("bits", 0))

        tiles_per_layer: dict[int, int] = defaultdict(int)
        per_layer_name: dict[int, str] = {}
        per_layer_intra: dict[int, list[list[str | int]]] = {}
        per_layer_inter: dict[int, list[list[str | int | str]]] = {}
        dependency_histogram: dict[int, int] = defaultdict(int)
        memory_cost_histogram_bytes: dict[int, int] = defaultdict(int)
        per_tile_metrics: list[dict[str, Any]] = []

        for node in workload.node_list:
            tiles_per_layer[node.id] += 1
            per_layer_name[node.id] = str(node.name)
            per_layer_intra[node.id] = [
                [str(dim), int(factor)] for dim, factor in node.intra_core_tiling
            ]
            per_layer_inter[node.id] = [
                [str(dim), factor] for dim, factor in node.inter_core_tiling
            ]

            # Number of producer tiles with non-zero data dependency needed to compute this tile.
            required_input_tiles = sum(
                1
                for _, _, edge_data in workload.in_edges(node, data=True)
                if int(edge_data.get("bits", 0)) > 0
            )

            # Assume int8 (1 byte/value): memory bytes = sum of operand element counts.
            operand_element_counts: dict[str, int] = {}
            for layer_operand, tensor in node.operand_tensors.items():
                element_count = prod(
                    upper - lower for (lower, upper) in tensor.loop_ranges
                )
                operand_element_counts[str(layer_operand)] = int(element_count)
            memory_cost_bytes = int(sum(operand_element_counts.values()))

            dependency_histogram[required_input_tiles] += 1
            memory_cost_histogram_bytes[memory_cost_bytes] += 1

            per_tile_metrics.append(
                {
                    "tile": {
                        "layer_id": node.id,
                        "sub_id": node.sub_id,
                        "name": str(node.name),
                    },
                    "required_input_tiles": required_input_tiles,
                    "operand_element_counts": operand_element_counts,
                    "memory_cost_bytes_int8": memory_cost_bytes,
                }
            )

        nb_layers = len(tiles_per_layer)
        max_tiles_per_layer = max(tiles_per_layer.values(), default=0)
        avg_tiles_per_layer = float(node_count / nb_layers) if nb_layers else 0.0

        layer_details = []
        for layer_id in sorted(tiles_per_layer.keys()):
            layer_details.append(
                {
                    "layer_id": layer_id,
                    "layer_name": per_layer_name[layer_id],
                    "tile_count": tiles_per_layer[layer_id],
                    "intra_core_tiling": per_layer_intra[layer_id],
                    "inter_core_tiling": per_layer_inter[layer_id],
                }
            )

        return {
            "variant_id": variant_id,
            "graph": {
                "node_count": node_count,
                "edge_count": edge_count,
                "inter_layer_edges": inter_layer_edges,
                "intra_layer_edges": intra_layer_edges,
                "edge_bits_sum": edge_bits_sum,
            },
            "tiling": {
                "layer_count": nb_layers,
                "max_tiles_per_layer": max_tiles_per_layer,
                "avg_tiles_per_layer": avg_tiles_per_layer,
            },
            "distributions": {
                "required_input_tiles": [
                    {
                        "required_input_tiles": required_input_tiles,
                        "tile_count": count,
                    }
                    for required_input_tiles, count in sorted(
                        dependency_histogram.items()
                    )
                ],
                "memory_cost_bytes_int8": [
                    {
                        "memory_cost_bytes_int8": memory_cost,
                        "tile_count": count,
                    }
                    for memory_cost, count in sorted(
                        memory_cost_histogram_bytes.items()
                    )
                ],
            },
            "tiles": per_tile_metrics,
            "layers": layer_details,
        }


class TiledWorkloadCaptureStage(Stage):
    """Leaf stage that returns the tiled workload graph for downstream evaluation."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload

    def run(self):
        yield None, self.workload

    def is_leaf(self) -> bool:
        return True


def evaluate_tiled_workload_heuristics(
    workload: ComputationNodeWorkload,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute heuristic metrics and weighted score for a tiled workload graph.

    Lower score is better.
    """
    default_weights = {
        "node_count": 1.0,
        "edge_count": 0.25,
        "edge_bits_sum": 1e-6,
        "avg_required_input_tiles": 2.0,
        "max_tiles_per_layer": 1.5,
    }
    active_weights = weights or default_weights

    node_count = len(workload.node_list)
    edge_count = workload.number_of_edges()
    edge_bits_sum = 0

    required_input_tiles_sum = 0
    tiles_per_layer: dict[int, int] = defaultdict(int)

    for node in workload.node_list:
        tiles_per_layer[node.id] += 1
        required_input_tiles_sum += sum(
            1
            for _, _, edge_data in workload.in_edges(node, data=True)
            if int(edge_data.get("bits", 0)) > 0
        )

    for _, _, edge_data in workload.edges(data=True):
        edge_bits_sum += int(edge_data.get("bits", 0))

    avg_required_input_tiles = (
        float(required_input_tiles_sum / node_count) if node_count else 0.0
    )
    max_tiles_per_layer = max(tiles_per_layer.values(), default=0)

    metrics = {
        "node_count": float(node_count),
        "edge_count": float(edge_count),
        "edge_bits_sum": float(edge_bits_sum),
        "avg_required_input_tiles": float(avg_required_input_tiles),
        "max_tiles_per_layer": float(max_tiles_per_layer),
    }
    score = sum(
        active_weights.get(metric, 0.0) * value for metric, value in metrics.items()
    )

    return {
        "score": float(score),
        **metrics,
    }


def evaluate_tiling_combinations_with_heuristics(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    experiment_id: str,
    output_path: str,
    intra_choices: list[MappingChoice],
    inter_choices: list[MappingChoice],
    layer_stacks: list[tuple[int, ...]] | None = None,
    heuristic_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Generate tiled workload graphs for mapping combinations and rank them with heuristics."""
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)

    run_root = os.path.join(output_path, experiment_id)
    os.makedirs(run_root, exist_ok=True)

    ranked_results: list[dict[str, Any]] = []
    for intra_choice, inter_choice in itertools.product(intra_choices, inter_choices):
        variant_id = f"intra-{intra_choice.label}__inter-{inter_choice.label}"
        variant_root = os.path.join(run_root, variant_id)
        os.makedirs(variant_root, exist_ok=True)

        tiled_workload_path = os.path.join(variant_root, "tiled_workload.pickle")

        logger.info("Evaluating tiled-workload variant: %s", variant_id)
        mainstage = MainStage(
            [
                AcceleratorParserStage,
                StreamONNXModelParserStage,
                MappingOverrideStage,
                LayerStacksGenerationStage,
                TilingGenerationStage,
                TiledWorkloadGenerationStage,
                TiledWorkloadCaptureStage,
            ],
            accelerator=hardware,
            workload_path=workload,
            mapping_path=mapping,
            mode=mode,
            layer_stacks=layer_stacks,
            tiled_workload_path=tiled_workload_path,
            intra_choice=intra_choice,
            inter_choice=inter_choice,
        )

        answers = mainstage.run()
        tiled_workload = answers[0][1]
        assert isinstance(tiled_workload, ComputationNodeWorkload)
        evaluation = evaluate_tiled_workload_heuristics(
            tiled_workload, weights=heuristic_weights
        )

        ranked_results.append(
            {
                "variant_id": variant_id,
                "intra_choice": intra_choice.label,
                "inter_choice": inter_choice.label,
                "tiled_workload_path": tiled_workload_path,
                **evaluation,
            }
        )

    ranked_results.sort(key=lambda item: float(item["score"]))

    ranking_path = os.path.join(run_root, "heuristic_ranking.json")
    with open(ranking_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_id": experiment_id,
                "heuristic_weights": heuristic_weights
                or {
                    "node_count": 1.0,
                    "edge_count": 0.25,
                    "edge_bits_sum": 1e-6,
                    "avg_required_input_tiles": 2.0,
                    "max_tiles_per_layer": 1.5,
                },
                "results": ranked_results,
            },
            handle,
            indent=2,
        )
    logger.info("Saved heuristic ranking to %s", ranking_path)
    return ranked_results


def _sanity_check_inputs(
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    output_path: str,
):
    assert os.path.exists(hardware), f"Hardware file {hardware} does not exist"
    assert isinstance(workload, ModelProto) or os.path.exists(workload), (
        f"Workload file {workload} does not exist"
    )
    assert os.path.exists(mapping), f"Mapping file {mapping} does not exist"
    assert mode in ["lbl", "fused"], "Mode must be either 'lbl' or 'fused'"
    os.makedirs(output_path, exist_ok=True)


def parse_factor(raw_factor: str) -> TILING_FACTOR_T:
    factor = raw_factor.strip()
    if factor in ("*", "all"):
        return factor
    return int(factor)


def parse_tiling_choice(choice: str) -> MappingChoice:
    raw = choice.strip()
    lower = raw.lower()

    if lower == "keep":
        return MappingChoice(label="keep", tiling=None)
    if lower in ("none", "empty"):
        return MappingChoice(label="none", tiling=[])

    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not entries:
        raise ValueError(f"Invalid tiling choice '{choice}'")

    tiling: TILING_SPEC_T = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                f"Invalid tiling entry '{entry}'. Expected format 'DIM:FACTOR' (example: OY:2 or K:*)."
            )
        dim_str, factor_str = entry.split(":", maxsplit=1)
        dim = LayerDim(dim_str.strip())
        factor = parse_factor(factor_str)
        tiling.append((dim, factor))

    label = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    return MappingChoice(label=label or "custom", tiling=tiling)


def parse_layer_stacks(raw_layer_stacks: str | None) -> list[tuple[int, ...]] | None:
    if raw_layer_stacks is None:
        return None
    parsed = json.loads(raw_layer_stacks)
    if not isinstance(parsed, list):
        raise ValueError("--layer-stacks must be a JSON list.")

    normalized: list[tuple[int, ...]] = []
    for stack in parsed:
        if not isinstance(stack, list):
            raise ValueError("Each layer stack must be a JSON list of integers.")
        normalized.append(tuple(int(layer_id) for layer_id in stack))
    return normalized


def get_positive_divisors(number: int) -> list[int]:
    divisors = set()
    for value in range(1, isqrt(number) + 1):
        if number % value == 0:
            divisors.add(value)
            divisors.add(number // value)
    return sorted(divisors)


def load_parsed_workload(hardware: str, workload: str, mapping: str) -> ONNXWorkload:
    mainstage = MainStage(
        [
            AcceleratorParserStage,
            StreamONNXModelParserStage,
            ParsedWorkloadCaptureStage,
        ],
        accelerator=hardware,
        workload_path=workload,
        mapping_path=mapping,
    )
    answers = mainstage.run()
    parsed_workload = answers[0][1]
    assert isinstance(parsed_workload, ONNXWorkload)
    return parsed_workload


def enumerate_all_intra_choices(parsed_workload: ONNXWorkload) -> list[MappingChoice]:
    """Build all single-dimension intra-core tiling choices seen in the parsed workload.

    For each LayerDim in each computation node, every integer divisor >= 2 of the dimension size
    is added as an exploration option.
    """
    factors_per_dim: dict[LayerDim, set[int]] = defaultdict(set)
    for node in parsed_workload.node_list:
        if not isinstance(node, ComputationNode):
            continue
        for dim, size in node.layer_dim_sizes.items():
            if size <= 1:
                continue
            for factor in get_positive_divisors(size):
                if factor >= 2:
                    factors_per_dim[dim].add(factor)

    choices: list[MappingChoice] = [MappingChoice(label="keep", tiling=None)]
    for dim in sorted(factors_per_dim.keys(), key=lambda item: str(item)):
        sorted_factors = sorted(factors_per_dim[dim])
        for factor in sorted_factors:
            choices.append(
                MappingChoice(
                    label=f"{dim}_{factor}",
                    tiling=[(dim, factor)],
                )
            )
    return choices


def deduplicate_choices(choices: list[MappingChoice]) -> list[MappingChoice]:
    unique: dict[str, MappingChoice] = {}
    for choice in choices:
        unique[choice.label] = choice
    return list(unique.values())


def explore_tiled_workload_mapping_choices(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    experiment_id: str,
    output_path: str,
    intra_choices: list[MappingChoice],
    inter_choices: list[MappingChoice],
    layer_stacks: list[tuple[int, ...]] | None = None,
) -> list[dict[str, Any]]:
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)

    run_root = os.path.join(output_path, experiment_id)
    os.makedirs(run_root, exist_ok=True)

    sweep_results: list[dict[str, Any]] = []

    for intra_choice, inter_choice in itertools.product(intra_choices, inter_choices):
        variant_id = f"intra-{intra_choice.label}__inter-{inter_choice.label}"
        variant_root = os.path.join(run_root, variant_id)
        os.makedirs(variant_root, exist_ok=True)

        tiled_workload_path = os.path.join(variant_root, "tiled_workload.pickle")
        summary_output_path = os.path.join(variant_root, "tiled_workload_summary.json")

        logger.info("Running tiled-workload exploration variant: %s", variant_id)
        mainstage = MainStage(
            [
                AcceleratorParserStage,
                StreamONNXModelParserStage,
                MappingOverrideStage,
                LayerStacksGenerationStage,
                TilingGenerationStage,
                TiledWorkloadGenerationStage,
                TiledWorkloadInspectionStage,
            ],
            accelerator=hardware,
            workload_path=workload,
            mapping_path=mapping,
            mode=mode,
            layer_stacks=layer_stacks,
            tiled_workload_path=tiled_workload_path,
            intra_choice=intra_choice,
            inter_choice=inter_choice,
            variant_id=variant_id,
            summary_output_path=summary_output_path,
        )

        answers = mainstage.run()
        summary = answers[0][1]

        sweep_results.append(
            {
                "variant_id": variant_id,
                "intra_choice": intra_choice.label,
                "inter_choice": inter_choice.label,
                "summary_path": summary_output_path,
                "node_count": summary["graph"]["node_count"],
                "edge_count": summary["graph"]["edge_count"],
                "avg_tiles_per_layer": summary["tiling"]["avg_tiles_per_layer"],
                "max_tiles_per_layer": summary["tiling"]["max_tiles_per_layer"],
            }
        )

    sweep_index_path = os.path.join(run_root, "sweep_index.json")
    with open(sweep_index_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"experiment_id": experiment_id, "results": sweep_results}, handle, indent=2
        )

    logger.info("Saved sweep index to %s", sweep_index_path)
    return sweep_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore tiled workload graph changes when varying intra/inter layer mapping choices."
    )
    parser.add_argument("--hardware", required=True, help="Path to hardware YAML.")
    parser.add_argument("--workload", required=True, help="Path to ONNX workload.")
    parser.add_argument("--mapping", required=True, help="Path to mapping YAML.")
    parser.add_argument("--mode", choices=["lbl", "fused"], default="fused")
    parser.add_argument("--experiment-id", default="tiled-workload-mapping-sweep")
    parser.add_argument("--output-path", default="outputs")
    parser.add_argument(
        "--intra-option",
        action="append",
        dest="intra_options",
        default=None,
        help=(
            "Repeatable. Options: keep, none, all, or DIM:FACTOR[,DIM:FACTOR...] "
            "(example: OY:2 or K:all). Use 'all' to iterate through all discovered intra tiling choices."
        ),
    )
    parser.add_argument(
        "--inter-option",
        action="append",
        dest="inter_options",
        default=None,
        help="Repeatable. Options: keep, none, or DIM:FACTOR[,DIM:FACTOR...] (example: K:*).",
    )
    parser.add_argument(
        "--layer-stacks",
        default=None,
        help="Optional JSON list of layer stacks, e.g. '[[0,1,2],[3],[4,5]]'. If omitted, default stage behavior is used.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    intra_options = args.intra_options or ["keep"]
    inter_options = args.inter_options or ["keep"]

    use_all_intra_options = any(
        option.strip().lower() == "all" for option in intra_options
    )
    if use_all_intra_options:
        parsed_workload = load_parsed_workload(
            hardware=args.hardware,
            workload=args.workload,
            mapping=args.mapping,
        )
        auto_intra_choices = enumerate_all_intra_choices(parsed_workload)
        explicit_choices = [
            parse_tiling_choice(option)
            for option in intra_options
            if option.strip().lower() != "all"
        ]
        intra_choices = deduplicate_choices(auto_intra_choices + explicit_choices)
        logger.info(
            "Auto-generated %d intra-layer tiling choice(s).", len(intra_choices)
        )
    else:
        intra_choices = [parse_tiling_choice(option) for option in intra_options]

    inter_choices = [parse_tiling_choice(option) for option in inter_options]
    layer_stacks = parse_layer_stacks(args.layer_stacks)

    results = explore_tiled_workload_mapping_choices(
        hardware=args.hardware,
        workload=args.workload,
        mapping=args.mapping,
        mode=args.mode,
        experiment_id=args.experiment_id,
        output_path=args.output_path,
        intra_choices=intra_choices,
        inter_choices=inter_choices,
        layer_stacks=layer_stacks,
    )

    logger.info("Completed %d variant(s).", len(results))


if __name__ == "__main__":
    main()
