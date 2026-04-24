import argparse
import logging
from pathlib import Path

import onnx
from onnx.shape_inference import infer_shapes_path
import torch

from stream.api import optimize_allocation_co

logger = logging.getLogger(__name__)

DEFAULT_MAPPING = [
    {"name": "default", "core_allocation": [0]},
    {"name": "Conv", "core_allocation": [0], "intra_core_tiling": ["OY, all"]},
    {"name": "Relu", "core_allocation": [0], "intra_core_tiling": ["D, all"]},
]


class ConvReluModel(torch.nn.Module):
    def __init__(self):
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

    def forward(self, x):
        return self.relu(self.conv(x))


def generate_onnx_workload(workload_path: str) -> int:
    """Export Conv+ReLU to ONNX and infer shapes."""
    model = ConvReluModel().eval()
    # 1 * 8 * 96 * 96 = 589,824 bits = 73,728 bytes = 72 KB input tensor size
    x = torch.randn(1, 8, 96, 96)
    workload_file = Path(workload_path)
    workload_file.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        x,
        str(workload_file),
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamo=False,
    )
    infer_shapes_path(str(workload_file), str(workload_file))

    nb_layers = len(onnx.load(str(workload_file)).graph.node)
    logger.info(
        "Generated ONNX workload at %s with %d graph node(s).", workload_file, nb_layers
    )
    return nb_layers


def write_hardware_yaml_files(config_dir: Path, top_sram_size_bits: int) -> Path:
    soc_path = config_dir / "soc.yaml"
    core_path = config_dir / "edge_tpu_one.yaml"
    offchip_path = config_dir / "offchip.yaml"

    core_path.write_text(
        f"""name: edge_tpu_one
type: compute

memories:
  rf_128B:
    size: 1024
    r_cost: 0.095
    w_cost: 0.095
    area: 0
    latency: 1
    auto_cost_extraction: false
    operands: [I2]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 8
        bandwidth_max: 8
        allocation:
          - I2, tl
      - name: w_port_1
        type: write
        bandwidth_min: 8
        bandwidth_max: 8
        allocation:
          - I2, fh
    served_dimensions: []

  rf_2B:
    size: 16
    r_cost: 0.021
    w_cost: 0.021
    area: 0
    latency: 1
    operands: [O]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 16
        bandwidth_max: 16
        allocation:
          - O, tl
      - name: r_port_2
        type: read
        bandwidth_min: 16
        bandwidth_max: 16
        allocation:
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 16
        bandwidth_max: 16
        allocation:
          - O, fh
      - name: w_port_2
        type: write
        bandwidth_min: 16
        bandwidth_max: 16
        allocation:
          - O, fl
    served_dimensions: [D2]

  sram_top:
    size: {top_sram_size_bits}
    r_cost: 416.16
    w_cost: 378.4
    area: 0
    latency: 1
    operands: [I1, I2, O]
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 64
        bandwidth_max: 2048
        allocation:
          - I1, tl
          - I2, tl
          - O, tl
          - O, th
      - name: w_port_1
        type: write
        bandwidth_min: 64
        bandwidth_max: 2048
        allocation:
          - I1, fh
          - I2, fh
          - O, fh
          - O, fl
    served_dimensions: [D1, D2]

operational_array:
  unit_energy: 0.04
  unit_area: 1
  dimensions: [D1, D2]
  sizes: [32, 32]
""",
        encoding="utf-8",
    )

    offchip_path.write_text(
        """name: offchip
type: memory

memories:
  dram:
    size: 10000000000
    r_cost: 1000
    w_cost: 1000
    area: 0
    latency: 1
    operands: [I1, I2, O]
    ports:
      - name: rw_port_1
        type: read_write
        bandwidth_min: 64
        bandwidth_max: 64
        allocation:
          - I1, fh
          - I1, tl
          - I2, fh
          - I2, tl
          - O, fh
          - O, tl
          - O, fl
          - O, th
    served_dimensions: [D1, D2]

operational_array:
  unit_energy: 0
  unit_area: 0
  dimensions: [D1, D2]
  sizes: [0, 0]
""",
        encoding="utf-8",
    )

    soc_path.write_text(
        """name: edge_tpu_one_soc

cores:
  0: ./edge_tpu_one.yaml
  1: ./offchip.yaml

offchip_core_id: 1
unit_energy_cost: 0

core_connectivity:
  - type: link
    cores: [0, 1]
    bandwidth: 64
""",
        encoding="utf-8",
    )

    return soc_path


def write_default_mapping(mapping_path: Path) -> None:
    lines: list[str] = []
    for entry in DEFAULT_MAPPING:
        lines.append(f"- name: {entry['name']}")
        core_allocation = ", ".join(str(c) for c in entry["core_allocation"])
        lines.append(f"  core_allocation: [{core_allocation}]")
        lines.append("")
    mapping_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_case(
    hardware_path: Path,
    mapping_path: Path,
    workload_path: Path,
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_root: Path,
) -> tuple[float, float]:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep top memory size for Conv+ReLU on single EdgeTPU-like core and compare fusion usefulness."
    )
    parser.add_argument(
        "--top-mem-bits",
        type=int,
        nargs="+",
        default=[
            7000,
            # int(131072 / 16),
            # int(131072 / 8),
            # int(131072 / 4),
            # int(131072 / 2),
            # 131072,
            # 262144,
            # 1048576,
            # 4194304,
            # 16777216,
        ],
        help="Top SRAM sizes in bits to sweep.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/conv_relu_fusion_sweep"),
        help="Directory where workload/configs/results are written.",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=None,
        help=(
            "Optional path to a mapping YAML to use. "
            "If omitted, a default editable mapping is created at <output-root>/mapping.yaml."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    workload_path = output_root / "conv_relu.onnx"
    nb_layers = generate_onnx_workload(str(workload_path))

    if nb_layers <= 0:
        raise ValueError("Exported ONNX graph has no nodes.")

    fused_layer_stacks = [tuple(range(nb_layers))]
    split_layer_stacks = [(i,) for i in range(nb_layers)]

    if args.mapping_path is None:
        mapping_path = output_root / "mapping.yaml"
        if not mapping_path.exists():
            write_default_mapping(mapping_path)
            logger.info("Created editable mapping file at %s", mapping_path)
        else:
            logger.info("Using existing editable mapping file at %s", mapping_path)
    else:
        mapping_path = args.mapping_path
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file does not exist: {mapping_path}")
        logger.info("Using user mapping file at %s", mapping_path)

    rows = []
    for top_mem_bits in args.top_mem_bits:
        config_dir = output_root / f"config_topmem_{top_mem_bits}"
        config_dir.mkdir(parents=True, exist_ok=True)
        hardware_path = write_hardware_yaml_files(config_dir, top_mem_bits)

        fused_latency, fused_energy = run_case(
            hardware_path=hardware_path,
            mapping_path=mapping_path,
            workload_path=workload_path,
            layer_stacks=fused_layer_stacks,
            experiment_id=f"fused_topmem_{top_mem_bits}",
            output_root=output_root,
        )
        split_latency, split_energy = run_case(
            hardware_path=hardware_path,
            mapping_path=mapping_path,
            workload_path=workload_path,
            layer_stacks=split_layer_stacks,
            experiment_id=f"split_topmem_{top_mem_bits}",
            output_root=output_root,
        )

        latency_gain = split_latency - fused_latency
        energy_gain = split_energy - fused_energy
        rows.append(
            (
                top_mem_bits,
                fused_latency,
                split_latency,
                latency_gain,
                fused_energy,
                split_energy,
                energy_gain,
            )
        )

    print(
        "top_mem_bits | fused_lat | split_lat | lat_gain(split-fused) | fused_en | split_en | en_gain(split-fused)"
    )
    for row in rows:
        print(
            f"{row[0]:>12} | {row[1]:>9.2f} | {row[2]:>9.2f} | {row[3]:>21.2f} | "
            f"{row[4]:>8.2f} | {row[5]:>8.2f} | {row[6]:>20.2f}"
        )

    print("\nInterpretation:")
    print("If lat_gain(split-fused) > 0, fusion improves latency for that memory size.")
    print("If en_gain(split-fused) > 0, fusion improves energy for that memory size.")


if __name__ == "__main__":
    main()
