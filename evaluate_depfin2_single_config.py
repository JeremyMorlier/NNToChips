import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from stream.api import optimize_allocation_co


def generate_run_id():
    """Generate run ID in format YYYYMMDD_HHMM."""
    return datetime.now().strftime("%Y%m%d_%H%M")


BASELINE_X = [128, 16, 1, 1, 2, 524288, 1048576, 128, 64, 1024]
XL = [16, 4, 1, 1, 2, 32768, 65536, 64, 64, 64]
XU = [256, 64, 64, 64, 256, 2097152, 4194304, 2048, 1024, 2048]


def parse_x_values(x_text: str):
    values = [int(v.strip()) for v in x_text.split(",") if v.strip()]
    if len(values) != 10:
        raise ValueError("Expected exactly 10 comma-separated integers for --x.")
    return values


def validate_x(x):
    if len(x) != 10:
        raise ValueError("Configuration vector must have length 10.")

    for i, value in enumerate(x):
        if value < XL[i] or value > XU[i]:
            raise ValueError(f"x[{i}]={value} out of bounds [{XL[i]}, {XU[i]}].")

    if x[9] < x[8]:
        raise ValueError(
            f"Invalid bandwidth constraint: l1_act_bw_max ({x[9]}) < l1_act_bw_min ({x[8]})."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one depfin2 configuration specified as ILP/GA design vector x."
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
        default="depfin2-single-eval",
        help="Base output folder under outputs/.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run identifier. If omitted, a random id is generated.",
    )
    parser.add_argument(
        "--x",
        type=str,
        default=",".join(str(v) for v in BASELINE_X),
        help=(
            "Comma-separated design vector: "
            "d1,d2,rf1i_units,rf1w_units,rf4_units,l1w_units,l1act_units,"
            "l1w_bw,l1act_bw_min,l1act_bw_max"
        ),
    )
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


def params_from_x(x):
    rf_1b_i_units = int(x[2])
    rf_1b_w_units = int(x[3])
    rf_4b_units = int(x[4])
    l1_w_units = int(x[5])
    l1_act_units = int(x[6])

    return {
        "d1_size": int(x[0]),
        "d2_size": int(x[1]),
        "rf_1b_i_size": rf_1b_i_units * 8,
        "rf_1b_w_size": rf_1b_w_units * 8,
        "rf_4b_size": rf_4b_units * 8,
        "l1_w_size": l1_w_units * 8,
        "l1_act_size": l1_act_units * 8,
        "rf_1b_i_bw": rf_1b_i_units * 8,
        "rf_1b_w_bw": rf_1b_w_units * 8,
        "rf_4b_bw": rf_4b_units * 8,
        "l1_w_bw": int(x[7]),
        "l1_act_bw_min": int(x[8]),
        "l1_act_bw_max": int(x[9]),
        "rf_1b_i_size_units": rf_1b_i_units,
        "rf_1b_w_size_units": rf_1b_w_units,
        "rf_4b_size_units": rf_4b_units,
        "l1_w_size_units": l1_w_units,
        "l1_act_size_units": l1_act_units,
    }


def build_variant_hardware_dir(
    template_path: Path,
    base_hardware_dir: Path,
    out_dir: Path,
    params: dict,
):
    variant_dir = out_dir / "hardware"
    variant_cores_dir = variant_dir / "cores"
    variant_cores_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(base_hardware_dir / "soc.yaml", variant_dir / "soc.yaml")
    shutil.copyfile(
        base_hardware_dir / "cores" / "offchip.yaml", variant_cores_dir / "offchip.yaml"
    )
    render_template_to_file(template_path, variant_cores_dir / "core.yaml", params)


def evaluate_single_config(args):
    x = parse_x_values(args.x)
    validate_x(x)
    params = params_from_x(x)

    run_uid = args.run_id.strip() if args.run_id.strip() else generate_run_id()
    out_dir = Path("outputs") / args.experiment_id / f"run_{run_uid}"

    build_variant_hardware_dir(
        template_path=Path(args.template),
        base_hardware_dir=Path(args.base_hardware_dir),
        out_dir=out_dir,
        params=params,
    )

    layer_stacks = [tuple(range(0, 12)), tuple(range(12, 22))] + [
        (i,) for i in range(22, 49)
    ]

    # Use the eval folder as the STREAM output location to keep hardware and outputs together
    stream_output_dir = out_dir
    scme = optimize_allocation_co(
        hardware=str(stream_output_dir / "hardware" / "soc.yaml"),
        workload=args.workload,
        mapping=args.mapping,
        mode="fused",
        layer_stacks=layer_stacks,
        experiment_id=f"{args.experiment_id}/run_{run_uid}",
        output_path="outputs",
        skip_if_exists=False,
    )

    summary = {
        "experiment_id": args.experiment_id,
        "run_uid": run_uid,
        "x": x,
        "params": params,
        "energy": float(scme.energy),
        "latency": float(scme.latency),
        "area": float(scme.accelerator.area),
    }

    summary_path = out_dir / "single_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")


def main():
    args = parse_args()
    evaluate_single_config(args)


if __name__ == "__main__":
    main()
