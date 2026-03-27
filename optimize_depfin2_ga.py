import argparse
import json
import shutil
from multiprocessing import Pool
from pathlib import Path
from uuid import uuid4

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from stream.api import optimize_allocation_co


# Baseline configuration from inputs/depfin2/hardware/cores/core.yaml
# Converted to design variables: [d1, d2, rf1i_units, rf1w_units, rf4_units, l1w_units, l1act_units, l1w_bw, l1act_bw_min, l1act_bw_max]
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
        self.run_uid = uuid4().hex[:8]
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
            n_constr=1,
            xl=xl,
            xu=xu,
            vtype=int,
            elementwise_evaluation=False,
        )
        self.cache = {}
        self.evaluations = []

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

    def _single_eval(self, x):
        """Evaluate a single individual. Used for parallel evaluation."""
        params = self._params_from_x(x)
        key = tuple(int(v) for v in x)

        g = [params["l1_act_bw_min"] - params["l1_act_bw_max"]]

        if key in self.cache:
            f = self.cache[key]
            return f, g, None, None

        # Penalize invalid architecture before calling stream.
        if params["l1_act_bw_max"] < params["l1_act_bw_min"]:
            f = [1e30, 1e30, 1e30]
            self.cache[key] = f
            return f, g, None, None

        eval_id = f"eval_{len(self.evaluations):05d}"
        experiment_id = f"{self.args.experiment_id}/{self.run_uid}/{eval_id}"
        eval_dir = self.root_dir / eval_id

        memory_areas = None
        try:
            build_variant_hardware_dir(
                template_path=Path(self.args.template),
                base_hardware_dir=Path(self.args.base_hardware_dir),
                out_dir=eval_dir,
                params=params,
            )

            layer_stacks = [tuple(range(0, 12)), tuple(range(12, 22))] + [
                (i,) for i in range(22, 49)
            ]

            scme = optimize_allocation_co(
                hardware=str(eval_dir / "hardware" / "soc.yaml"),
                workload=self.args.workload,
                mapping=self.args.mapping,
                mode="fused",
                layer_stacks=layer_stacks,
                experiment_id=experiment_id,
                output_path="outputs",
                skip_if_exists=False,
            )
            f = [float(scme.energy), float(scme.latency), float(scme.accelerator.area)]

            # Extract memory instance areas
            memory_areas = extract_memory_areas(scme)

        except Exception as exc:
            # Keep search robust: failed evaluations are dominated by valid designs.
            f = [1e30, 1e30, 1e30]
            error_path = eval_dir / "error.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(str(exc))

        self.cache[key] = f
        return f, g, params, memory_areas

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate a batch of individuals using multiprocessing."""
        with Pool(processes=self.n_processes) as pool:
            results = pool.map(self._single_eval, X)

        F = []
        G = []
        for x, (f, g, params, memory_areas) in zip(X, results):
            F.append(f)
            G.append(g)

            # Record evaluation
            eval_record = {
                "x": [int(v) for v in x],
                "F": f,
            }
            if params:
                eval_record["params"] = params
            if memory_areas:
                eval_record["memory_areas"] = memory_areas
            self.evaluations.append(eval_record)

        F_array = np.array(F)

        # Min-max normalization for each objective
        # Normalize to [0, 1] range using (x - min) / (max - min)
        for obj_idx in range(3):  # 3 objectives: energy, latency, area
            obj_values = F_array[:, obj_idx]
            # Filter out penalty values (1e30) for min/max calculation
            valid_values = obj_values[obj_values < 1e20]

            if len(valid_values) > 0:
                obj_min = np.min(valid_values)
                obj_max = np.max(valid_values)

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

    problem = Depfin2ArchProblem(args, n_processes=args.processes)
    algorithm = NSGA2(pop_size=args.pop_size)

    # Use baseline-seeded sampling for initial population
    sampling = BaselineInitialSampling(problem)

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", args.generations),
        seed=args.seed,
        verbose=True,
        sampling=sampling,
    )

    out_dir = Path("outputs") / args.experiment_id / f"run_{problem.run_uid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.X is not None:
        pareto = []
        xs = result.X.tolist() if hasattr(result.X, "tolist") else [result.X]
        fs = result.F.tolist() if hasattr(result.F, "tolist") else [result.F]
        for x, f in zip(xs, fs):
            params = problem._params_from_x(x)
            pareto.append(
                {
                    "x": [int(v) for v in x],
                    "params": params,
                    "energy": float(f[0]),
                    "latency": float(f[1]),
                    "area": float(f[2]),
                }
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

    print(f"Run UID: {problem.run_uid}")
    print(f"Evaluations: {len(problem.evaluations)}")
    print(f"Pareto points: {len(pareto)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
