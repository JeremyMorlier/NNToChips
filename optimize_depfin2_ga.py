import argparse
import json
import logging
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from stream.api import optimize_allocation_co
from utils import generate_run_id, render_template_to_file


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


def extract_memory_areas(scme):
    """
    Extract memory instance data (including areas) from SCME object.

    Returns a dict with memory hierarchy info:
    {
        'memory_0': {'name': ..., 'size': ..., 'area': ..., ...},
        'memory_1': {...},
        ...
    }
    """
    memory_data = {}
    try:
        core = scme.accelerator.get_core(0)
        memory_nodes = list(core.memory_hierarchy._node.keys())

        for i, mem_node in enumerate(memory_nodes):
            mem_instance = mem_node.memory_instance
            mem_dict = mem_instance.__dict__.copy()

            # Convert non-serializable objects to strings
            if "ports" in mem_dict:
                mem_dict["ports"] = [str(p) for p in mem_dict["ports"]]

            memory_data[f"memory_{i}"] = mem_dict
    except Exception as e:
        memory_data["error"] = str(e)

    return memory_data


class Depfin2ArchProblem(Problem):
    def __init__(self, args, n_processes=1):
        self.args = args
        self.n_processes = n_processes
        self.run_uid = generate_run_id()
        self.root_dir = Path("outputs") / args.experiment_id / f"run_{self.run_uid}"
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # Integer design variables.
        # x = [d1, d2, rf1i_units, rf1w_units, rf4_units, l1w_units, l1act_units, l1w_bw, l1act_bw_min, l1act_bw_max]
        # Applied memory size equals units * 8.
        xl = [16, 4, 1, 1, 2, 32768, 65536, 64, 64, 64]
        xu = [256, 64, 64, 64, 256, 2097152, 4194304, 2048, 1024, 2048]

        super().__init__(
            n_var=10,
            n_obj=3,
            n_constr=2,
            xl=xl,
            xu=xu,
            vtype=int,
            elementwise_evaluation=False,
        )
        self.cache = {}
        self.evaluations = []
        self.last_normalization_bounds = [None, None, None]
        self.generation_idx = 0

    def denormalize_f(self, f_values):
        """Convert normalized objective values back to physical units."""
        denorm = []
        for obj_idx, value in enumerate(f_values):
            value = float(value)
            bounds = self.last_normalization_bounds[obj_idx]
            if value >= 1e20 or bounds is None:
                denorm.append(value)
                continue

            obj_min, obj_max = bounds
            if obj_max > obj_min:
                denorm.append(value * (obj_max - obj_min) + obj_min)
            else:
                denorm.append(obj_min)

        return denorm

    def _params_from_x(self, x):
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

    def _single_eval(self, task):
        """Evaluate a single individual. Used for parallel evaluation."""
        eval_id, x = task
        x = [int(v) for v in x]
        eval_dir = self.root_dir / eval_id

        # Store logs together with each STREAM evaluation artifacts.
        log_folder = eval_dir / "logs"
        log_folder.mkdir(parents=True, exist_ok=True)

        # Configure the root logger to capture ALL logging from this process and all modules underneath
        root_logger = logging.getLogger()

        # Remove existing handlers from root logger to avoid duplicates
        root_logger.handlers = []

        # Set root logger to DEBUG level to allow ALL messages through to handlers
        root_logger.setLevel(logging.DEBUG)

        # Handler for ERROR level and above - goes to error log file
        error_handler = logging.FileHandler(log_folder / "error.log")
        error_handler.setLevel(logging.ERROR)

        # Handler for WARNING level and above - goes to warning log file (captures WARNING, ERROR, CRITICAL)
        warning_handler = logging.FileHandler(log_folder / "warning.log")
        warning_handler.setLevel(logging.WARNING)

        # Handler for INFO level and above - goes to info log file (captures INFO, WARNING, ERROR, CRITICAL)
        info_handler = logging.FileHandler(log_folder / "info.log")
        info_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        error_handler.setFormatter(formatter)
        warning_handler.setFormatter(formatter)
        info_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(warning_handler)
        root_logger.addHandler(info_handler)

        params = self._params_from_x(x)
        key = tuple(x)

        logging.info(f"{eval_id}: x={x}")

        g = [params["l1_act_bw_min"] - params["l1_act_bw_max"]]
        evaluation_failed = False  # Track optimization failure

        if key in self.cache:
            f = self.cache[key]
            # Check if this was a failed evaluation
            evaluation_failed = f == [1e30, 1e30, 1e30]
            if evaluation_failed:
                logging.warning(f"{eval_id}: Cached failed evaluation")
            else:
                logging.info(
                    f"{eval_id}: Using cached result - energy={f[0]:.2f}, latency={f[1]:.0f}, area={f[2]:.4f}"
                )
            return f, g, None, None, evaluation_failed

        # Penalize invalid architecture before calling stream.
        if params["l1_act_bw_max"] < params["l1_act_bw_min"]:
            f = [1e30, 1e30, 1e30]
            self.cache[key] = f
            logging.warning(
                f"{eval_id}: Constraint violation - l1_act_bw_max ({params['l1_act_bw_max']}) < l1_act_bw_min ({params['l1_act_bw_min']})"
            )
            return (
                f,
                g,
                None,
                None,
                False,
            )  # Constraint violation, not optimization failure

        logging.info(f"{eval_id}: Building variant hardware directory...")
        logging.info(
            f"{eval_id}: Parameters - d1={params['d1_size']}, d2={params['d2_size']}, "
            f"l1_w_bw={params['l1_w_bw']}, l1_act_bw={params['l1_act_bw_min']}-{params['l1_act_bw_max']}"
        )

        memory_areas = None
        try:
            build_variant_hardware_dir(
                template_path=Path(self.args.template),
                base_hardware_dir=Path(self.args.base_hardware_dir),
                out_dir=eval_dir,
                params=params,
            )
            logging.info(f"{eval_id}: Hardware directory built successfully")

            layer_stacks = [tuple(range(0, 12)), tuple(range(12, 22))] + [
                (i,) for i in range(22, 49)
            ]

            logging.info(f"{eval_id}: Launching STREAM evaluation...")
            # Use a relative path from outputs to ensure STREAM outputs to the same folder as hardware
            rel_path = eval_dir.relative_to("outputs")
            scme = optimize_allocation_co(
                hardware=str(eval_dir / "hardware" / "soc.yaml"),
                workload=self.args.workload,
                mapping=self.args.mapping,
                mode="fused",
                layer_stacks=layer_stacks,
                experiment_id=str(rel_path),
                output_path="outputs",
                skip_if_exists=False,
            )
            f = [float(scme.energy), float(scme.latency), float(scme.accelerator.area)]

            # Extract memory instance areas
            memory_areas = extract_memory_areas(scme)

            logging.info(
                f"{eval_id}: Success - energy={f[0]:.2f}pJ, latency={f[1]:.0f}cycles, area={f[2]:.4f}mm²"
            )

        except Exception as exc:
            # Keep search robust: failed evaluations are dominated by valid designs.
            f = [1e30, 1e30, 1e30]
            evaluation_failed = True
            error_path = eval_dir / "error.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(str(exc))
            logging.error(f"{eval_id}: Evaluation failed with exception: {exc}")

        self.cache[key] = f
        logging.info(f"{eval_id}: Evaluation complete - x={x}, f={f}")
        return f, g, params, memory_areas, evaluation_failed

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate a batch of individuals using multiprocessing."""
        eval_tasks = []
        for individual_idx, x in enumerate(X):
            eval_id = f"g{self.generation_idx:03d}_i{individual_idx:03d}"
            eval_tasks.append((eval_id, x))

        with Pool(processes=self.n_processes) as pool:
            results = pool.map(self._single_eval, eval_tasks)

        F = []
        G = []
        for (eval_id, x), (f, g, params, memory_areas, evaluation_failed) in zip(
            eval_tasks, results
        ):
            F.append(f)
            # Add second constraint: 0 if evaluation succeeded, 1 if failed
            # (constraint g <= 0 is satisfied)
            g_constraint = 1.0 if evaluation_failed else 0.0
            constraints = g + [g_constraint]
            G.append(constraints)

            # Record evaluation
            eval_record = {
                "eval_id": eval_id,
                "x": [int(v) for v in x],
                "F": f,
            }
            if params:
                eval_record["params"] = params
            if memory_areas:
                eval_record["memory_areas"] = memory_areas
            self.evaluations.append(eval_record)

        self.generation_idx += 1

        F_array = np.array(F)

        # Log batch summary
        valid_evals = np.sum(np.all(F_array < 1e20, axis=1))
        failed_evals = len(F_array) - valid_evals
        logging.info(
            f"Batch evaluated: {len(F_array)} designs, {valid_evals} valid, {failed_evals} failed"
        )

        # Log objective statistics for valid evaluations
        if valid_evals > 0:
            valid_mask = np.all(F_array < 1e20, axis=1)
            valid_F = F_array[valid_mask]
            logging.info("Objective statistics (valid evals only):")
            logging.info(
                f"  Energy: min={valid_F[:, 0].min():.2f}, max={valid_F[:, 0].max():.2f}, mean={valid_F[:, 0].mean():.2f} pJ"
            )
            logging.info(
                f"  Latency: min={valid_F[:, 1].min():.0f}, max={valid_F[:, 1].max():.0f}, mean={valid_F[:, 1].mean():.0f} cycles"
            )
            logging.info(
                f"  Area: min={valid_F[:, 2].min():.4f}, max={valid_F[:, 2].max():.4f}, mean={valid_F[:, 2].mean():.4f} mm²"
            )

        # Min-max normalization for each objective
        # Normalize to [0, 1] range using (x - min) / (max - min)
        self.last_normalization_bounds = [None, None, None]
        for obj_idx in range(3):  # 3 objectives: energy, latency, area
            obj_values = F_array[:, obj_idx]
            # Filter out penalty values (1e30) for min/max calculation
            valid_values = obj_values[obj_values < 1e20]

            if len(valid_values) > 0:
                obj_min = np.min(valid_values)
                obj_max = np.max(valid_values)
                self.last_normalization_bounds[obj_idx] = (
                    float(obj_min),
                    float(obj_max),
                )

                if obj_max > obj_min:
                    # Normalize valid values
                    F_array[obj_values < 1e20, obj_idx] = (
                        obj_values[obj_values < 1e20] - obj_min
                    ) / (obj_max - obj_min)
                # If min == max, values stay as 0 (or keep original if all same)

        out["F"] = F_array
        out["G"] = np.array(G)


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

    problem = Depfin2ArchProblem(args, n_processes=args.processes)
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
            params = problem._params_from_x(x)
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
