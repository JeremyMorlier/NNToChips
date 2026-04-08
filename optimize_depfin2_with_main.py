import argparse
import json
import logging
from pathlib import Path
from textwrap import dedent

from main import optimize_single_hardware_co


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test depfin2 optimization flow through main.optimize_single_hardware_co."
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="depfin2-main-ga-test",
        help="Experiment folder under outputs/.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Output root directory.",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=4,
        help="Hardware GA population size inside GeneticHardwareStage.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=2,
        help="Hardware GA generations inside GeneticHardwareStage.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_depfin2_layer_stacks() -> list[tuple[int, ...]]:
    # Same default split currently used for depfin2 fsrcnn in this repo.
    return [tuple(range(0, 12)), tuple(range(12, 22))] + [(i,) for i in range(22, 49)]


def prepare_local_hardware_assets(run_dir: Path, depfin2_hw_dir: Path) -> None:
    """Create local relative core links for strict accelerator path validation."""
    cores_dir = run_dir / "cores"
    cores_dir.mkdir(parents=True, exist_ok=True)

    src_core = depfin2_hw_dir / "cores" / "core.yaml"
    src_offchip = depfin2_hw_dir / "cores" / "offchip.yaml"

    dst_core = cores_dir / "core.yaml"
    dst_offchip = cores_dir / "offchip.yaml"

    if dst_core.exists() or dst_core.is_symlink():
        dst_core.unlink()
    if dst_offchip.exists() or dst_offchip.is_symlink():
        dst_offchip.unlink()

    dst_core.symlink_to(src_core.resolve())
    dst_offchip.symlink_to(src_offchip.resolve())


def create_soc_template(template_path: Path) -> None:
    template_content = dedent(
        """
        name: {{ soc_name | default('depfin2_main_test_soc', true) }}
        cores:
          0: ./cores/core.yaml
          1: ./cores/offchip.yaml
        offchip_core_id: 1
        unit_energy_cost: 0
        core_connectivity:
          - type: bus
            cores: [0, 1]
            bandwidth: {{ bus_bw | default(64, true) | float }}
        """
    ).lstrip()
    template_path.write_text(template_content, encoding="utf-8")


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    repo_root = Path(__file__).resolve().parent
    depfin2_dir = repo_root / "inputs" / "depfin2"
    depfin2_hw_dir = depfin2_dir / "hardware"
    workload_path = depfin2_dir / "workload" / "fsrcnn.onnx"

    run_dir = Path(args.output_root) / args.experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Local template used only for this test flow.
    prepare_local_hardware_assets(run_dir, depfin2_hw_dir)
    hw_template_path = run_dir / "soc_template_for_main.yaml"
    create_soc_template(hw_template_path)

    template_params = {
        "soc_name": "depfin2_main_test_soc",
        "bus_bw": 64,
    }

    # Current integration with optimize_single_hardware_co supports hardware GA
    # over template placeholders present in the top-level accelerator template.
    hardware_ga_parameter_specs = [
        {"name": "bus_bw", "lower": 32, "upper": 256, "scale": 1},
    ]

    logging.info("Starting optimize_single_hardware_co test flow")
    scme = optimize_single_hardware_co(
        hardware_template=str(hw_template_path),
        workload=str(workload_path),
        mode="fused",
        layer_stacks=build_depfin2_layer_stacks(),
        experiment_id=args.experiment_id,
        output_path=str(Path(args.output_root)),
        skip_if_exists=False,
        template_params=template_params,
        hardware_ga_parameter_specs=hardware_ga_parameter_specs,
        hardware_ga_generations=args.generations,
        hardware_ga_population=args.pop_size,
        hardware_ga_seed=args.seed,
    )

    summary = {
        "experiment_id": args.experiment_id,
        "latency": float(scme.latency),
        "energy": float(scme.energy),
        "area": float(scme.accelerator.area),
        "hardware_template": str(hw_template_path),
        "workload": str(workload_path),
        "ga_population": args.pop_size,
        "ga_generations": args.generations,
        "ga_seed": args.seed,
    }

    summary_path = run_dir / "main_ga_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logging.info("Finished. Summary saved to %s", summary_path)
    logging.info("Latency=%s Energy=%s Area=%s", summary["latency"], summary["energy"], summary["area"])


if __name__ == "__main__":
    main()
