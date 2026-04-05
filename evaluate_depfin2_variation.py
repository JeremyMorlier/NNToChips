import argparse
import json
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from stream.api import optimize_allocation_co


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one depfin2 core architecture variant."
    )
    parser.add_argument(
        "--template",
        type=str,
        default="inputs/depfin2/hardware/cores/core_template.yaml.j2",
        help="Jinja2 template for depfin2 compute core.",
    )
    parser.add_argument(
        "--base-hardware-dir",
        type=str,
        default="inputs/depfin2/hardware",
        help="Directory containing soc.yaml and cores/offchip.yaml.",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="inputs/depfin2/workload/fsrcnn.onnx",
        help="Path to workload ONNX file.",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="inputs/depfin2/mapping/mapping.yaml",
        help="Path to mapping YAML file.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="depfin2-variant",
        help="Experiment output folder name under outputs/.",
    )

    parser.add_argument("--d1-size", type=int, default=128)
    parser.add_argument("--d2-size", type=int, default=16)
    parser.add_argument(
        "--rf-1b-i-size",
        type=int,
        default=1,
        help="Memory size units for rf_1B_I. Applied size = units * 8.",
    )
    parser.add_argument(
        "--rf-1b-w-size",
        type=int,
        default=1,
        help="Memory size units for rf_1B_W. Applied size = units * 8.",
    )
    parser.add_argument(
        "--rf-4b-size",
        type=int,
        default=2,
        help="Memory size units for rf_4B. Applied size = units * 8.",
    )
    parser.add_argument(
        "--l1-w-size",
        type=int,
        default=524288,
        help="Memory size units for l1_w. Applied size = units * 8.",
    )
    parser.add_argument(
        "--l1-act-size",
        type=int,
        default=1048576,
        help="Memory size units for l1_act. Applied size = units * 8.",
    )

    parser.add_argument("--rf-1b-i-bw", type=int, default=8)
    parser.add_argument("--rf-1b-w-bw", type=int, default=8)
    parser.add_argument("--rf-4b-bw", type=int, default=16)
    parser.add_argument("--l1-w-bw", type=int, default=128)
    parser.add_argument("--l1-act-bw-min", type=int, default=64)
    parser.add_argument("--l1-act-bw-max", type=int, default=1024)

    return parser.parse_args()


def render_template_to_file(template_path: Path, output_path: Path, context: dict):
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)
    rendered = template.render(**context)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)


def build_variant_hardware_dir(args) -> Path:
    base_dir = Path(args.base_hardware_dir)
    variant_dir = Path("outputs") / args.experiment_id / "hardware"
    variant_cores_dir = variant_dir / "cores"
    variant_cores_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(base_dir / "soc.yaml", variant_dir / "soc.yaml")
    shutil.copyfile(
        base_dir / "cores" / "offchip.yaml", variant_cores_dir / "offchip.yaml"
    )

    rf_1b_i_size = args.rf_1b_i_size * 8
    rf_1b_w_size = args.rf_1b_w_size * 8
    rf_4b_size = args.rf_4b_size * 8
    l1_w_size = args.l1_w_size * 8
    l1_act_size = args.l1_act_size * 8

    context = {
        "d1_size": args.d1_size,
        "d2_size": args.d2_size,
        "rf_1b_i_size": rf_1b_i_size,
        "rf_1b_w_size": rf_1b_w_size,
        "rf_4b_size": rf_4b_size,
        "l1_w_size": l1_w_size,
        "l1_act_size": l1_act_size,
        "rf_1b_i_bw": args.rf_1b_i_bw,
        "rf_1b_w_bw": args.rf_1b_w_bw,
        "rf_4b_bw": args.rf_4b_bw,
        "l1_w_bw": args.l1_w_bw,
        "l1_act_bw_min": args.l1_act_bw_min,
        "l1_act_bw_max": args.l1_act_bw_max,
    }

    render_template_to_file(
        Path(args.template),
        variant_cores_dir / "core.yaml",
        context,
    )
    return variant_dir


def evaluate_variant(args):
    memory_values = [
        args.rf_1b_i_size,
        args.rf_1b_w_size,
        args.rf_4b_size,
        args.l1_w_size,
        args.l1_act_size,
    ]
    if any(v <= 0 for v in memory_values):
        raise ValueError("All memory size unit values must be positive.")

    if args.l1_act_bw_max < args.l1_act_bw_min:
        raise ValueError("l1_act_bw_max must be >= l1_act_bw_min.")

    hardware_dir = build_variant_hardware_dir(args)
    layer_stacks = [tuple(range(0, 12)), tuple(range(12, 22))] + [
        (i,) for i in range(22, 49)
    ]

    scme = optimize_allocation_co(
        hardware=str(hardware_dir / "soc.yaml"),
        workload=args.workload,
        mapping=args.mapping,
        mode="fused",
        layer_stacks=layer_stacks,
        experiment_id=args.experiment_id,
        output_path="outputs",
        skip_if_exists=False,
    )

    summary = {
        "experiment_id": args.experiment_id,
        "latency": scme.latency,
        "energy": scme.energy,
        "area": scme.accelerator.area,
        "params": {
            "d1_size": args.d1_size,
            "d2_size": args.d2_size,
            "rf_1b_i_size_units": args.rf_1b_i_size,
            "rf_1b_w_size_units": args.rf_1b_w_size,
            "rf_4b_size_units": args.rf_4b_size,
            "l1_w_size_units": args.l1_w_size,
            "l1_act_size_units": args.l1_act_size,
            "rf_1b_i_size_applied": args.rf_1b_i_size * 8,
            "rf_1b_w_size_applied": args.rf_1b_w_size * 8,
            "rf_4b_size_applied": args.rf_4b_size * 8,
            "l1_w_size_applied": args.l1_w_size * 8,
            "l1_act_size_applied": args.l1_act_size * 8,
            "rf_1b_i_bw": args.rf_1b_i_bw,
            "rf_1b_w_bw": args.rf_1b_w_bw,
            "rf_4b_bw": args.rf_4b_bw,
            "l1_w_bw": args.l1_w_bw,
            "l1_act_bw_min": args.l1_act_bw_min,
            "l1_act_bw_max": args.l1_act_bw_max,
        },
    }

    summary_path = Path("outputs") / args.experiment_id / "variant_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    evaluate_variant(args)


if __name__ == "__main__":
    main()
