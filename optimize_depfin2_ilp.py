import argparse
import json
import shutil
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from gurobipy import GRB, Model, quicksum
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from stream.api import optimize_allocation_co


def generate_run_id():
    """Generate run ID in format YYYYMMDD_HHMM."""
    return datetime.now().strftime("%Y%m%d_%H%M")


# Same baseline and design space as optimize_depfin2_ga.py
BASELINE_X = [128, 16, 1, 1, 2, 524288, 1048576, 128, 64, 1024]
XL = [16, 4, 1, 1, 2, 32768, 65536, 64, 64, 64]
XU = [256, 64, 64, 64, 256, 2097152, 4194304, 2048, 1024, 2048]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "ILP-based depfin2 architecture search using Gurobi with linear surrogates "
            "for energy/latency/area."
        )
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
        default="depfin2-ilp",
        help="Base experiment folder under outputs/.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sample generation."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=24,
        help="Number of STREAM evaluations used to fit linear surrogates.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Number of worker processes for sample evaluation.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="1,1,1;2,1,1;1,2,1;1,1,2;3,1,1;1,3,1;1,1,3",
        help=(
            "Semicolon-separated weight triples for energy,latency,area. "
            "Example: '1,1,1;2,1,1;1,2,1'."
        ),
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Gurobi time limit per ILP solve (seconds).",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.0,
        help="Gurobi relative MIP gap.",
    )
    return parser.parse_args()


def parse_weight_sweep(weights_text: str):
    weights = []
    for part in weights_text.split(";"):
        part = part.strip()
        if not part:
            continue
        elems = [float(v.strip()) for v in part.split(",")]
        if len(elems) != 3:
            raise ValueError(f"Invalid weight tuple '{part}', expected 3 values.")
        total = sum(elems)
        if total <= 0:
            raise ValueError(f"Invalid weight tuple '{part}', sum must be > 0.")
        weights.append(tuple(v / total for v in elems))
    if not weights:
        raise ValueError("At least one weight tuple is required.")
    return weights


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


def evaluate_design(task):
    (
        x,
        idx,
        template,
        base_hardware_dir,
        workload,
        mapping,
        experiment_id,
        run_uid,
        output_root,
    ) = task

    x = [int(v) for v in x]
    params = params_from_x(x)
    eval_id = f"eval_{idx:05d}"
    eval_dir = output_root / eval_id

    result = {
        "x": x,
        "params": params,
        "evaluation_id": eval_id,
        "success": False,
    }

    if params["l1_act_bw_max"] < params["l1_act_bw_min"]:
        result["error"] = "l1_act_bw_max < l1_act_bw_min"
        result["F"] = [1e30, 1e30, 1e30]
        return result

    try:
        build_variant_hardware_dir(
            template_path=Path(template),
            base_hardware_dir=Path(base_hardware_dir),
            out_dir=eval_dir,
            params=params,
        )

        layer_stacks = [tuple(range(0, 12)), tuple(range(12, 22))] + [
            (i,) for i in range(22, 49)
        ]
        # Use a relative path from outputs to ensure STREAM outputs to the same folder as hardware
        # Extract relative path from the eval_dir to construct the experiment_id correctly
        rel_path = eval_dir.relative_to("outputs")
        scme = optimize_allocation_co(
            hardware=str(eval_dir / "hardware" / "soc.yaml"),
            workload=workload,
            mapping=mapping,
            mode="fused",
            layer_stacks=layer_stacks,
            experiment_id=str(rel_path),
            output_path="outputs",
            skip_if_exists=False,
        )

        result["success"] = True
        result["F"] = [
            float(scme.energy),
            float(scme.latency),
            float(scme.accelerator.area),
        ]
    except Exception as exc:
        result["error"] = str(exc)
        result["F"] = [1e30, 1e30, 1e30]

    return result


def sample_designs(num_samples, seed):
    rng = np.random.default_rng(seed)

    designs = [BASELINE_X.copy()]
    seen = {tuple(BASELINE_X)}

    while len(designs) < num_samples:
        x = [int(rng.integers(XL[i], XU[i] + 1)) for i in range(10)]
        if x[9] < x[8]:
            continue
        key = tuple(x)
        if key in seen:
            continue
        seen.add(key)
        designs.append(x)

    return designs


def fit_linear_surrogate(valid_evals):
    # Linear model: y = b0 + b1*x1 + ... + b10*x10
    X = np.array([ev["x"] for ev in valid_evals], dtype=float)
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    coeffs = {}
    stats = {}

    names = ["energy", "latency", "area"]
    for i, name in enumerate(names):
        y = np.array([ev["F"][i] for ev in valid_evals], dtype=float)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta

        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        coeffs[name] = beta.tolist()
        stats[name] = {
            "r2": r2,
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "mean": float(np.mean(y)),
        }

    return coeffs, stats


def solve_ilp_with_surrogate(coeffs, stats, weights, time_limit, mip_gap):
    model = Model("depfin2_ilp")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap

    x_vars = [
        model.addVar(vtype=GRB.INTEGER, lb=XL[i], ub=XU[i], name=f"x{i}")
        for i in range(10)
    ]

    # Architecture validity constraint
    model.addConstr(x_vars[8] <= x_vars[9], name="bw_min_leq_bw_max")

    e_hat = model.addVar(vtype=GRB.CONTINUOUS, name="energy_hat")
    l_hat = model.addVar(vtype=GRB.CONTINUOUS, name="latency_hat")
    a_hat = model.addVar(vtype=GRB.CONTINUOUS, name="area_hat")

    beta_e = coeffs["energy"]
    beta_l = coeffs["latency"]
    beta_a = coeffs["area"]

    model.addConstr(
        e_hat == beta_e[0] + quicksum(beta_e[i + 1] * x_vars[i] for i in range(10)),
        name="energy_surrogate",
    )
    model.addConstr(
        l_hat == beta_l[0] + quicksum(beta_l[i + 1] * x_vars[i] for i in range(10)),
        name="latency_surrogate",
    )
    model.addConstr(
        a_hat == beta_a[0] + quicksum(beta_a[i + 1] * x_vars[i] for i in range(10)),
        name="area_surrogate",
    )

    e_den = max(stats["energy"]["max"] - stats["energy"]["min"], 1e-9)
    l_den = max(stats["latency"]["max"] - stats["latency"]["min"], 1e-9)
    a_den = max(stats["area"]["max"] - stats["area"]["min"], 1e-9)

    w_e, w_l, w_a = weights
    objective = (
        w_e * (e_hat - stats["energy"]["min"]) / e_den
        + w_l * (l_hat - stats["latency"]["min"]) / l_den
        + w_a * (a_hat - stats["area"]["min"]) / a_den
    )
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"ILP solve failed with status {model.Status}")

    x_sol = [int(round(v.X)) for v in x_vars]
    pred = {
        "energy": float(e_hat.X),
        "latency": float(l_hat.X),
        "area": float(a_hat.X),
    }
    return x_sol, pred, int(model.Status), float(model.ObjVal)


def main():
    args = parse_args()
    weight_sweep = parse_weight_sweep(args.weights)

    run_uid = generate_run_id()
    out_dir = Path("outputs") / args.experiment_id / f"run_{run_uid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled_designs = sample_designs(args.num_samples, args.seed)
    tasks = [
        (
            x,
            idx,
            args.template,
            args.base_hardware_dir,
            args.workload,
            args.mapping,
            args.experiment_id,
            run_uid,
            out_dir / "samples",
        )
        for idx, x in enumerate(sampled_designs)
    ]

    with Pool(processes=args.processes) as pool:
        sampled_results = pool.map(evaluate_design, tasks)

    valid = [r for r in sampled_results if r["success"]]
    if len(valid) < 12:
        raise RuntimeError(
            "Not enough valid STREAM evaluations to fit a robust surrogate. "
            f"Valid={len(valid)}. Increase --num-samples or inspect failures."
        )

    coeffs, stats = fit_linear_surrogate(valid)

    ilp_solutions = []
    for w in weight_sweep:
        x_star, pred, status, obj_val = solve_ilp_with_surrogate(
            coeffs=coeffs,
            stats=stats,
            weights=w,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
        )
        ilp_solutions.append(
            {
                "weights": list(w),
                "x": x_star,
                "predicted": pred,
                "solver_status": status,
                "objective_value": obj_val,
            }
        )

    # Evaluate unique ILP solutions with true STREAM objectives to build a Pareto-like set.
    unique_x = []
    seen = set()
    for sol in ilp_solutions:
        key = tuple(sol["x"])
        if key not in seen:
            seen.add(key)
            unique_x.append(sol["x"])

    eval_tasks = [
        (
            x,
            idx,
            args.template,
            args.base_hardware_dir,
            args.workload,
            args.mapping,
            args.experiment_id,
            run_uid,
            out_dir / "ilp_candidates",
        )
        for idx, x in enumerate(unique_x)
    ]
    with Pool(processes=args.processes) as pool:
        true_candidate_results = pool.map(evaluate_design, eval_tasks)

    true_by_x = {tuple(r["x"]): r for r in true_candidate_results}
    pareto_front = []
    for sol in ilp_solutions:
        true_eval = true_by_x.get(tuple(sol["x"]))
        if true_eval and true_eval["success"]:
            pareto_front.append(
                {
                    "weights": sol["weights"],
                    "x": sol["x"],
                    "params": params_from_x(sol["x"]),
                    "predicted": sol["predicted"],
                    "energy": true_eval["F"][0],
                    "latency": true_eval["F"][1],
                    "area": true_eval["F"][2],
                }
            )

    summary = {
        "experiment_id": args.experiment_id,
        "run_uid": run_uid,
        "seed": args.seed,
        "baseline_x": BASELINE_X,
        "num_samples": args.num_samples,
        "num_valid_samples": len(valid),
        "weight_sweep": [list(w) for w in weight_sweep],
        "surrogate_stats": stats,
        "surrogate_coefficients": coeffs,
        "sampled_evaluations": sampled_results,
        "ilp_solutions": ilp_solutions,
        "pareto_front": pareto_front,
    }

    summary_path = out_dir / "ilp_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Run UID: {run_uid}")
    print(f"Valid sample evaluations: {len(valid)}/{len(sampled_results)}")
    print(f"ILP solutions generated: {len(ilp_solutions)}")
    print(f"Pareto-like solutions evaluated: {len(pareto_front)}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
