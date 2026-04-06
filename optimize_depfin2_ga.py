import argparse
import json
import logging
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize

from optimization_problem import ParameterSpec, StreamOptimizationProblem


BASELINE_X = [
    128,  # d1_size
    16,  # d2_size
    1,  # rf_1b_i_units (8 bytes / 8)
    1,  # rf_1b_w_units (8 bytes / 8)
    2,  # rf_4b_units (16 bytes / 8)
    524288,  # l1_w_units (4194304 bytes / 8)
    1048576,  # l1_act_units (8388608 bytes / 8)
    128,  # l1_w_bw
    64,  # l1_act_bw_min
    1024,  # l1_act_bw_max
]


class BaselineInitialSampling(Sampling):
    """
    Custom sampling that seeds 5% of the initial population with the baseline configuration.
    """

    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def _do(self, problem, n_samples, **kwargs):
        # Seed 5% of initial population with baseline configuration
        n_baseline = max(1, int(np.ceil(0.05 * n_samples)))
        X = np.zeros((n_samples, problem.n_var), dtype=int)

        # Fill first n_baseline samples with baseline
        for i in range(n_baseline):
            X[i] = BASELINE_X

        # Fill remaining samples with random values within bounds
        for i in range(n_baseline, n_samples):
            for j in range(problem.n_var):
                X[i, j] = np.random.randint(problem.xl[j], problem.xu[j] + 1)

        print("Initial population X:")
        print(X)

        return X


def parse_args():
    parser = argparse.ArgumentParser(
        description="NSGA-II search for depfin2 architecture (energy, latency, area)."
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
        default="depfin2-ga",
        help="Base experiment folder under outputs/.",
    )
    parser.add_argument(
        "--pop-size", type=int, default=16, help="NSGA-II population size."
    )
    parser.add_argument(
        "--generations", type=int, default=8, help="Number of NSGA-II generations."
    )
    parser.add_argument(
        "--processes", type=int, default=4, help="Number of parallel worker processes."
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_depfin2_parameter_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="d1_size", lower=16, upper=256),
        ParameterSpec(name="d2_size", lower=4, upper=64),
        ParameterSpec(name="rf_1b_i_units", lower=1, upper=64),
        ParameterSpec(name="rf_1b_w_units", lower=1, upper=64),
        ParameterSpec(name="rf_4b_units", lower=2, upper=256),
        ParameterSpec(name="l1_w_units", lower=32768, upper=2097152),
        ParameterSpec(name="l1_act_units", lower=65536, upper=4194304),
        ParameterSpec(name="l1_w_bw", lower=64, upper=2048),
        ParameterSpec(name="l1_act_bw_min", lower=64, upper=1024),
        ParameterSpec(name="l1_act_bw_max", lower=64, upper=2048),
    ]


def depfin2_constraint_fn(params: dict[str, float | int]) -> list[float]:
    return [float(params["l1_act_bw_min"]) - float(params["l1_act_bw_max"])]


def depfin2_hardware_context_enricher(
    params: dict[str, float | int],
) -> dict[str, int]:
    rf_1b_i_units = int(params["rf_1b_i_units"])
    rf_1b_w_units = int(params["rf_1b_w_units"])
    rf_4b_units = int(params["rf_4b_units"])
    l1_w_units = int(params["l1_w_units"])
    l1_act_units = int(params["l1_act_units"])

    return {
        "rf_1b_i_size": rf_1b_i_units * 8,
        "rf_1b_w_size": rf_1b_w_units * 8,
        "rf_4b_size": rf_4b_units * 8,
        "l1_w_size": l1_w_units * 8,
        "l1_act_size": l1_act_units * 8,
        "rf_1b_i_bw": rf_1b_i_units * 8,
        "rf_1b_w_bw": rf_1b_w_units * 8,
        "rf_4b_bw": rf_4b_units * 8,
    }


def depfin2_params_from_x(problem: StreamOptimizationProblem, x) -> dict[str, int]:
    x_values = [int(v) for v in x]
    params, _, _ = problem._decode_parameters(x_values)
    return {k: int(v) for k, v in params.items() if isinstance(v, (int, np.integer))}


def main():
    args = parse_args()

    # Setup main logging to console
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Starting NSGA-II optimization for depfin2 architecture")
    logging.info(
        f"Configuration: pop_size={args.pop_size}, generations={args.generations}, processes={args.processes}"
    )
    logging.info(f"Template: {args.template}")
    logging.info(f"Workload: {args.workload}")
    logging.info("Baseline enabled: 5% of initial population")

    layer_stacks = [tuple(range(0, 12)), tuple(range(12, 22))] + [
        (i,) for i in range(22, 49)
    ]

    problem = StreamOptimizationProblem(
        parameter_specs=build_depfin2_parameter_specs(),
        hardware_template_path=args.template,
        base_hardware_dir=args.base_hardware_dir,
        mapping_path=args.mapping,
        workload_path=args.workload,
        constraint_fn=depfin2_constraint_fn,
        n_link_constraints=1,
        layer_stacks=layer_stacks,
        experiment_id=args.experiment_id,
        stream_mode="fused",
        output_root="outputs",
        n_objectives=3,
        hardware_context_enricher=depfin2_hardware_context_enricher,
        normalize_objectives=True,
        n_processes=args.processes,
    )
    logging.info(
        f"Problem initialized: run_id={problem.run_uid}, {problem.n_var} variables, {problem.n_obj} objectives, {problem.n_constr} constraints"
    )

    # Use baseline-seeded sampling for initial population
    sampling = BaselineInitialSampling(problem)

    algorithm = NSGA2(pop_size=args.pop_size, sampling=sampling)

    logging.info("Starting optimization with NSGA-II...")
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", args.generations),
        seed=args.seed,
        verbose=True,
    )

    logging.info(
        f"Optimization completed. Total evaluations: {len(problem.evaluations)}"
    )

    out_dir = Path("outputs") / args.experiment_id / f"run_{problem.run_uid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.X is not None:
        pareto = []
        xs = result.X.tolist() if hasattr(result.X, "tolist") else [result.X]
        fs = result.F.tolist() if hasattr(result.F, "tolist") else [result.F]
        logging.info(f"Extracting Pareto front with {len(xs)} solutions")
        for x, f in zip(xs, fs):
            params = depfin2_params_from_x(problem, x)
            denorm_f = problem.denormalize_f(f)
            pareto.append(
                {
                    "x": [int(v) for v in x],
                    "params": params,
                    "energy": float(denorm_f[0]),
                    "latency": float(denorm_f[1]),
                    "area": float(denorm_f[2]),
                }
            )

        # Log pareto front statistics
        if pareto:
            energies = [p["energy"] for p in pareto]
            latencies = [p["latency"] for p in pareto]
            areas = [p["area"] for p in pareto]
            logging.info(
                f"Pareto front energy range: {min(energies):.2f} - {max(energies):.2f} pJ"
            )
            logging.info(
                f"Pareto front latency range: {min(latencies):.0f} - {max(latencies):.0f} cycles"
            )
            logging.info(
                f"Pareto front area range: {min(areas):.4f} - {max(areas):.4f} mm²"
            )
    else:
        pareto = []

    summary = {
        "experiment_id": args.experiment_id,
        "run_uid": problem.run_uid,
        "population": args.pop_size,
        "generations": args.generations,
        "seed": args.seed,
        "baseline_x": BASELINE_X,
        "baseline_description": "Depfin2 baseline from inputs/depfin2/hardware/cores/core.yaml",
        "num_evaluations": len(problem.evaluations),
        "pareto_front": pareto,
        "all_evaluations": problem.evaluations,
    }

    summary_path = out_dir / "ga_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logging.info(f"Run UID: {problem.run_uid}")
    logging.info(f"Evaluations: {len(problem.evaluations)}")
    logging.info(f"Pareto points: {len(pareto)}")
    logging.info(f"Summary saved to: {summary_path}")
    logging.info("Optimization finished successfully")


if __name__ == "__main__":
    main()
