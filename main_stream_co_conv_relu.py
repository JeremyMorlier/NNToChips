import argparse
import copy
import csv
import itertools
import logging
import math
import random
from pathlib import Path

import onnx
from onnx.shape_inference import infer_shapes_path
import torch
import yaml

logger = logging.getLogger(__name__)

DEFAULT_MAPPING = [
    {"name": "default", "core_allocation": [0], "intra_core_tiling": ["D, all"]},
    {"name": "Conv", "core_allocation": [0], "intra_core_tiling": ["OY, all"]},
    {"name": "Relu", "core_allocation": [0], "intra_core_tiling": ["D, all"]},
]

CONV_DIMS = {"B": 1, "K": 8, "G": 1, "OX": 96, "C": 8, "FX": 3, "OY": 96, "FY": 3}
RELU_DIMS = {"B": 1, "H": 8, "D": 96, "K": 96}
CONV_INTERMEDIATE_DIMS = ("B", "K", "OX", "OY")
RELU_INTERMEDIATE_DIMS = ("B", "H", "D", "K")
CONV_EXPLORE_DIMS = ("K", "OX", "OY")
RELU_EXPLORE_DIMS = ("H", "D", "K")


class ConvReluModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


def export_conv_relu_onnx(workload_path: Path) -> int:
    model = ConvReluModel().eval()
    x = torch.randn(1, 8, 96, 96)
    workload_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            x,
            str(workload_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            x,
            str(workload_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
        )
    infer_shapes_path(str(workload_path), str(workload_path))

    node_count = len(onnx.load(str(workload_path)).graph.node)
    logger.info(
        "Exported Conv+ReLU ONNX to %s with %d node(s)", workload_path, node_count
    )
    return node_count


def write_default_mapping(mapping_path: Path) -> None:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(DEFAULT_MAPPING, f, sort_keys=False)


def load_mapping(mapping_path: Path) -> list[dict]:
    with mapping_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"Mapping file must contain a list of entries: {mapping_path}")
    return data


def update_mapping_tilings(
    base_mapping: list[dict], conv_tiling: dict[str, int], relu_tiling: dict[str, int]
) -> list[dict]:
    mapping = copy.deepcopy(base_mapping)
    found_conv = False
    found_relu = False

    conv_tiling_yaml = [
        f"{dim}, {factor}" for dim, factor in conv_tiling.items() if factor > 1
    ]
    relu_tiling_yaml = [
        f"{dim}, {factor}" for dim, factor in relu_tiling.items() if factor > 1
    ]

    for entry in mapping:
        name = str(entry.get("name", ""))
        if name == "Conv":
            entry["intra_core_tiling"] = conv_tiling_yaml
            found_conv = True
        elif name == "Relu":
            entry["intra_core_tiling"] = relu_tiling_yaml
            found_relu = True

    if not found_conv or not found_relu:
        raise ValueError(
            "Mapping must contain both 'Conv' and 'Relu' entries for this exploration."
        )

    return mapping


def prime_factors(n: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    x = n
    d = 2
    while d * d <= x:
        while x % d == 0:
            factors[d] = factors.get(d, 0) + 1
            x //= d
        d += 1
    if x > 1:
        factors[x] = factors.get(x, 0) + 1
    return factors


def divisors_from_prime_factors(n: int) -> list[int]:
    if n <= 1:
        return [1]
    factors = prime_factors(n)
    divisors = [1]
    for prime, exponent in factors.items():
        expanded = []
        for divisor in divisors:
            for p in range(exponent + 1):
                expanded.append(divisor * (prime**p))
        divisors = expanded
    return sorted(set(divisors))


def generate_tiling_combinations(
    dim_sizes: dict[str, int], explore_dims: tuple[str, ...] | None = None
) -> list[dict[str, int]]:
    if explore_dims is None:
        ordered_dims = [dim for dim, size in dim_sizes.items() if size > 1]
    else:
        ordered_dims = [dim for dim in explore_dims if dim_sizes.get(dim, 1) > 1]
    choices = [divisors_from_prime_factors(dim_sizes[dim]) for dim in ordered_dims]
    combinations: list[dict[str, int]] = []
    for factors in itertools.product(*choices):
        combinations.append(
            {dim: factor for dim, factor in zip(ordered_dims, factors, strict=True)}
        )
    return combinations


def number_of_tiles(tiling: dict[str, int], dims: tuple[str, ...]) -> int:
    return math.prod(tiling.get(dim, 1) for dim in dims)


def product_of_dims(dim_sizes: dict[str, int], dims: tuple[str, ...]) -> int:
    return math.prod(dim_sizes[dim] for dim in dims)


def format_tiling(tiling: dict[str, int]) -> str:
    parts = [f"{dim}={factor}" for dim, factor in tiling.items() if factor > 1]
    return ",".join(parts) if parts else "no-split"


def enumerate_valid_tiling_pairs(
) -> list[tuple[dict[str, int], dict[str, int], int, int]]:
    conv_combos = generate_tiling_combinations(
        CONV_DIMS, explore_dims=CONV_EXPLORE_DIMS
    )
    relu_combos = generate_tiling_combinations(
        RELU_DIMS, explore_dims=RELU_EXPLORE_DIMS
    )

    conv_by_tiles: dict[int, list[dict[str, int]]] = {}
    for combo in conv_combos:
        tiles = number_of_tiles(combo, CONV_INTERMEDIATE_DIMS)
        conv_by_tiles.setdefault(tiles, []).append(combo)

    relu_by_tiles: dict[int, list[dict[str, int]]] = {}
    for combo in relu_combos:
        tiles = number_of_tiles(combo, RELU_INTERMEDIATE_DIMS)
        relu_by_tiles.setdefault(tiles, []).append(combo)

    intermediate_size = product_of_dims(CONV_DIMS, CONV_INTERMEDIATE_DIMS)
    valid_pairs: list[tuple[dict[str, int], dict[str, int], int, int]] = []

    for tiles in sorted(set(conv_by_tiles) & set(relu_by_tiles)):
        if intermediate_size % tiles != 0:
            continue
        tile_size = intermediate_size // tiles
        for conv_combo in conv_by_tiles[tiles]:
            for relu_combo in relu_by_tiles[tiles]:
                valid_pairs.append((conv_combo, relu_combo, tiles, tile_size))

    return valid_pairs


def sample_valid_pairs(
    valid_pairs: list[tuple[dict[str, int], dict[str, int], int, int]],
    sample_size: int,
    sample_seed: int,
) -> list[tuple[dict[str, int], dict[str, int], int, int]]:
    if sample_size < 0:
        return valid_pairs
    if sample_size == 0:
        return []
    if sample_size >= len(valid_pairs):
        return valid_pairs

    rng = random.Random(sample_seed)
    sampled_indices = sorted(rng.sample(range(len(valid_pairs)), sample_size))
    return [valid_pairs[i] for i in sampled_indices]


def run_stream(
    hardware_path: Path,
    mapping_path: Path,
    workload_path: Path,
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_root: Path,
) -> tuple[float, float]:
    from stream.api import optimize_allocation_co

    scme = optimize_allocation_co(
        hardware=str(hardware_path),
        workload=str(workload_path),
        mapping=str(mapping_path),
        mode="fused",
        layer_stacks=layer_stacks,
        experiment_id=experiment_id,
        output_path=str(output_root),
        skip_if_exists=False,
    )
    return float(scme.latency), float(scme.energy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run STREAM on a Torch Conv+ReLU exported to ONNX with single-core mapping and "
            "prime-factor intra-core-tiling exploration."
        )
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=Path("inputs/stream/hardware/tpu_like_quad_core.yaml"),
        help="Path to hardware YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/main_stream_co_conv_relu"),
        help="Directory where workload, mappings, and STREAM outputs are written.",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=None,
        help=(
            "Optional base mapping YAML path. If omitted, an editable mapping is created at "
            "<output-root>/mapping.yaml."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of pairs to sample from all valid pairs. Use -1 for all.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed used for reproducible pair sampling.",
    )
    parser.add_argument(
        "--single-split-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run only one split case and reuse it as the reference for all sampled mappings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only enumerate valid pairs and write per-case mappings, without running STREAM.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    workload_path = output_root / "conv_relu.onnx"
    node_count = export_conv_relu_onnx(workload_path)
    if node_count <= 0:
        raise ValueError("Exported ONNX graph has no nodes.")

    if args.mapping_path is None:
        mapping_path = output_root / "mapping.yaml"
        if not mapping_path.exists():
            write_default_mapping(mapping_path)
            logger.info("Created editable base mapping file at %s", mapping_path)
        else:
            logger.info("Using existing editable base mapping file at %s", mapping_path)
    else:
        mapping_path = args.mapping_path
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file does not exist: {mapping_path}")
        logger.info("Using user base mapping file at %s", mapping_path)

    base_mapping = load_mapping(mapping_path)
    fused_layer_stacks = [tuple(range(node_count))]
    split_layer_stacks = [(i,) for i in range(node_count)]

    all_valid_pairs = enumerate_valid_tiling_pairs()
    if not all_valid_pairs:
        raise RuntimeError("No valid Conv/ReLU prime-factor tiling pairs found.")
    valid_pairs = sample_valid_pairs(
        all_valid_pairs, sample_size=args.sample_size, sample_seed=args.sample_seed
    )
    if not valid_pairs:
        raise RuntimeError("Sampling produced zero valid Conv/ReLU tiling pairs.")

    logger.info(
        "Found %d total valid pair(s); sampled %d pair(s) using seed=%d.",
        len(all_valid_pairs),
        len(valid_pairs),
        args.sample_seed,
    )

    exploration_dir = output_root / "exploration"
    mappings_dir = exploration_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exploration_dir / "results.csv"

    split_reference: tuple[float, float] | None = None
    split_reference_case_id: str | None = None
    if not args.dry_run and args.single_split_reference:
        ref_conv_tiling, ref_relu_tiling, _, _ = valid_pairs[0]
        ref_mapping = update_mapping_tilings(base_mapping, ref_conv_tiling, ref_relu_tiling)
        ref_mapping_path = mappings_dir / "split_reference.yaml"
        with ref_mapping_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(ref_mapping, f, sort_keys=False)
        split_reference_case_id = "split_reference"
        split_reference = run_stream(
            hardware_path=args.hardware,
            mapping_path=ref_mapping_path,
            workload_path=workload_path,
            layer_stacks=split_layer_stacks,
            experiment_id="split_reference",
            output_root=output_root,
        )

    rows: list[dict[str, object]] = []
    for idx, (conv_tiling, relu_tiling, n_tiles, tile_size) in enumerate(valid_pairs):
        try:
            case_id = f"pair_{idx:04d}"
            case_mapping = update_mapping_tilings(
                base_mapping, conv_tiling, relu_tiling
            )
            case_mapping_path = mappings_dir / f"{case_id}.yaml"
            with case_mapping_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(case_mapping, f, sort_keys=False)

            row: dict[str, object] = {
                "case_id": case_id,
                "n_tiles": n_tiles,
                "tile_size": tile_size,
                "conv_tiling": format_tiling(conv_tiling),
                "relu_tiling": format_tiling(relu_tiling),
                "mapping_path": str(case_mapping_path),
            }

            if not args.dry_run:
                fused_latency, fused_energy = run_stream(
                    hardware_path=args.hardware,
                    mapping_path=case_mapping_path,
                    workload_path=workload_path,
                    layer_stacks=fused_layer_stacks,
                    experiment_id=f"{case_id}_fused",
                    output_root=output_root,
                )
                if split_reference is not None:
                    split_latency, split_energy = split_reference
                else:
                    split_latency, split_energy = run_stream(
                        hardware_path=args.hardware,
                        mapping_path=case_mapping_path,
                        workload_path=workload_path,
                        layer_stacks=split_layer_stacks,
                        experiment_id=f"{case_id}_split",
                        output_root=output_root,
                    )
                row.update(
                    {
                        "fused_latency": fused_latency,
                        "split_latency": split_latency,
                        "latency_gain_split_minus_fused": split_latency - fused_latency,
                        "fused_energy": fused_energy,
                        "split_energy": split_energy,
                        "energy_gain_split_minus_fused": split_energy - fused_energy,
                        "split_reference_case_id": split_reference_case_id,
                    }
                )

            rows.append(row)
        except Exception as e:
            logger.error("Error processing case %s: %s", case_id, e, exc_info=True)

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("base_mapping:", mapping_path)
    print("results_csv:", csv_path)
    print("pairs_evaluated:", len(rows))
    if args.dry_run:
        print("dry_run: true")
    else:
        best_latency = min(rows, key=lambda r: float(r["fused_latency"]))  # type: ignore[arg-type]
        best_gain = max(rows, key=lambda r: float(r["latency_gain_split_minus_fused"]))  # type: ignore[arg-type]
        print(
            "best_fused_latency_case:",
            best_latency["case_id"],
            best_latency["fused_latency"],
        )
        print(
            "best_fusion_latency_gain_case:",
            best_gain["case_id"],
            best_gain["latency_gain_split_minus_fused"],
        )


if __name__ == "__main__":
    main()
