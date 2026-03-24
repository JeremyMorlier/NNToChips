import argparse
import json
import shutil
from pathlib import Path
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from stream.api import optimize_allocation_co


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


class Depfin2ArchProblem(ElementwiseProblem):
    def __init__(self, args):
        self.args = args
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

    def _evaluate(self, x, out, *args, **kwargs):
        params = self._params_from_x(x)
        key = tuple(int(v) for v in x)

        g = [params["l1_act_bw_min"] - params["l1_act_bw_max"]]

        if key in self.cache:
            f = self.cache[key]
            out["F"] = f
            out["G"] = g
            return

        # Penalize invalid architecture before calling stream.
        if params["l1_act_bw_max"] < params["l1_act_bw_min"]:
            f = [1e30, 1e30, 1e30]
            self.cache[key] = f
            out["F"] = f
            out["G"] = g
            return

        eval_id = f"eval_{len(self.evaluations):05d}"
        experiment_id = f"{self.args.experiment_id}/{self.run_uid}/{eval_id}"
        eval_dir = self.root_dir / eval_id

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

        except Exception as exc:
            # Keep search robust: failed evaluations are dominated by valid designs.
            f = [1e30, 1e30, 1e30]
            error_path = eval_dir / "error.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(str(exc))

        self.cache[key] = f
        self.evaluations.append({"x": [int(v) for v in x], "params": params, "F": f})
        out["F"] = f
        out["G"] = g


def main():
    args = parse_args()

    problem = Depfin2ArchProblem(args)
    algorithm = NSGA2(pop_size=args.pop_size)

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", args.generations),
        seed=args.seed,
        verbose=True,
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
