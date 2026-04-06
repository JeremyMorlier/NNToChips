"""
Universal Dataflow Accelerator Architecture Optimization using Genetic Algorithm
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from onnx.shape_inference import infer_shapes_path

from optimization_problem import ParameterSpec, StreamOptimizationProblem
from udc import array_to_hardware
from utils import render_template_to_file, save_history_csv, save_pareto_results_csv


class MLP_model(nn.Module):
    """Multi-Layer Perceptron model for testing."""

    def __init__(self, dim=4096, hidden_dim=11008):
        super(MLP_model, self).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.sigmoid(self.linear2(x))
        return self.linear1(x) * gate


def generate_hardware_from_udc(array_desc, output_path, name="udc_core"):
    """
    Generate hardware YAML using UDC template.

    Args:
        array_desc: List of hardware parameters
        output_path: Path to save the generated hardware file
        name: Name of the hardware core
    """
    hardware_yaml = array_to_hardware(array_desc, name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(hardware_yaml)

    return output_path


def generate_top_core_with_offchip(output_hardware_dir, offchip_source_path):
    """
    Generate a top-level hardware YAML containing the UDC core and offchip core.

    It creates:
      - <output_hardware_dir>/cores/udc_core.yaml   (already generated separately)
      - <output_hardware_dir>/cores/offchip.yaml    (copied from source)
            - <output_hardware_dir>/core.yaml             (top-level accelerator)
    """
    output_hardware_dir = Path(output_hardware_dir)
    cores_dir = output_hardware_dir / "cores"
    cores_dir.mkdir(parents=True, exist_ok=True)

    offchip_source = Path(offchip_source_path)
    if not offchip_source.exists():
        raise FileNotFoundError(
            f"offchip core file not found at: {offchip_source_path}"
        )

    shutil.copy(offchip_source, cores_dir / "offchip.yaml")

    top_core_desc = {
        "name": "udc_top_core",
        "cores": {
            0: "./cores/udc_core.yaml",
            1: "./cores/offchip.yaml",
        },
        "offchip_core_id": 1,
        "unit_energy_cost": 0,
        "core_connectivity": [
            {
                "type": "link",
                "cores": [0, 1],
                "bandwidth": 64,
            }
        ],
    }

    top_core_path = output_hardware_dir / "core.yaml"
    with open(top_core_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(top_core_desc, f, sort_keys=False)

    return str(top_core_path)


def render_mapping_from_template(
    template_path,
    output_path,
    linear1_k_tile=4,
    linear1_c_tile="all",
    linear2_k_tile=4,
    linear2_c_tile="all",
    sigmoid_h_tile=4,
    mul_h_tile=4,
):
    """
    Render mapping configuration from Jinja2 template.

    Args:
        template_path: Path to mapping template file
        output_path: Path to save rendered mapping
        linear1_k_tile: K dimension tile size for linear1 layer
        linear1_c_tile: C dimension tile size for linear1 layer
        linear2_k_tile: K dimension tile size for linear2 layer
        linear2_c_tile: C dimension tile size for linear2 layer
        sigmoid_h_tile: H dimension tile size for sigmoid layer
        mul_h_tile: H dimension tile size for multiplication layer
    """
    return render_template_to_file(
        template_path,
        output_path,
        {
            "linear1_k_tile": linear1_k_tile,
            "linear1_c_tile": linear1_c_tile,
            "linear2_k_tile": linear2_k_tile,
            "linear2_c_tile": linear2_c_tile,
            "sigmoid_h_tile": sigmoid_h_tile,
            "mul_h_tile": mul_h_tile,
        },
    )


class UDCArchitectureOptimizationProblem(StreamOptimizationProblem):
    """
    Pymoo Problem definition for UDC architecture optimization.

    Decision variables (21 encoded variables):
    [0]  d1: Dimension 1 of operational array
    [1]  d2: Dimension 2 of operational array
    [2]  sram_size: Main SRAM size
    [3]  sram_min_b: Main SRAM minimum bandwidth
    [4]  sram_max_b_selector: Encoded selector for main SRAM maximum bandwidth
    [5]  d2_sram_size: D2 SRAM size
    [6]  d2_sram_min_b: D2 SRAM minimum bandwidth
    [7]  d2_sram_max_b_selector: Encoded selector for D2 SRAM maximum bandwidth
    [8]  d1_sram_size: D1 SRAM size
    [9]  d1_sram_min_b: D1 SRAM minimum bandwidth
    [10] d1_sram_max_b_selector: Encoded selector for D1 SRAM maximum bandwidth
    [11] rf_I1_size: RF I1 size
    [12] rf_I1_bw: RF I1 bandwidth
    [13] rf_I2_size: RF I2 size
    [14] rf_I2_bw: RF I2 bandwidth
    [15] rf_O_size: RF O size
    [16] rf_O_bw: RF O bandwidth
    [17] linear1_k_tile: Linear1 layer K dimension tile size
    [18] linear2_k_tile: Linear2 layer K dimension tile size
    [19] sigmoid_h_tile: Sigmoid layer H dimension tile size
    [20] mul_h_tile: Multiplication layer H dimension tile size

    The three `*_max_b_selector` variables are decoded so that:
        max_bandwidth >= min_bandwidth
    is always satisfied by construction.
    """

    SRAM_MAX_BW = 512.0
    LOCAL_SRAM_MAX_BW = 256.0

    def __init__(
        self,
        workload_path,
        mapping_template_path,
        base_experiment_id,
        output_path,
        mode,
        layer_stacks,
        processes,
        offchip_source_path="inputs/test/hardware/cores/offchip.yaml",
    ):
        self.mapping_template_path = mapping_template_path
        self.offchip_source_path = offchip_source_path

        super().__init__(
            parameter_specs=self._parameter_specs(),
            hardware_template_path=mapping_template_path,
            base_hardware_dir=output_path,
            mapping_path=mapping_template_path,
            workload_path=workload_path,
            constraint_fn=self._constraint_fn,
            n_link_constraints=6,
            layer_stacks=layer_stacks,
            experiment_id=base_experiment_id,
            stream_mode=mode,
            output_root=output_path,
            n_objectives=2,
            objective_extractor=self._objective_extractor,
            penalty_factory=lambda n: [1e30] * n,
            hardware_context_enricher=self._hardware_context_enricher,
            normalize_objectives=False,
            n_processes=processes,
        )

    @staticmethod
    def _parameter_specs() -> list[ParameterSpec]:
        return [
            ParameterSpec(name="d1", lower=1, upper=16),
            ParameterSpec(name="d2", lower=1, upper=16),
            ParameterSpec(name="sram_size", lower=65536, upper=67108864),
            ParameterSpec(name="sram_min_b", lower=1, upper=512),
            ParameterSpec(name="sram_max_b_selector", lower=0, upper=511),
            ParameterSpec(name="d2_sram_size", lower=4096, upper=1048576),
            ParameterSpec(name="d2_sram_min_b", lower=1, upper=256),
            ParameterSpec(name="d2_sram_max_b_selector", lower=0, upper=255),
            ParameterSpec(name="d1_sram_size", lower=4096, upper=1048576),
            ParameterSpec(name="d1_sram_min_b", lower=1, upper=256),
            ParameterSpec(name="d1_sram_max_b_selector", lower=0, upper=255),
            ParameterSpec(name="rf_I1_size", lower=8, upper=256),
            ParameterSpec(name="rf_I1_bw", lower=1, upper=128),
            ParameterSpec(name="rf_I2_size", lower=8, upper=256),
            ParameterSpec(name="rf_I2_bw", lower=1, upper=128),
            ParameterSpec(name="rf_O_size", lower=8, upper=256),
            ParameterSpec(name="rf_O_bw", lower=1, upper=128),
            ParameterSpec(name="linear1_k_tile", lower=1, upper=128),
            ParameterSpec(name="linear2_k_tile", lower=1, upper=128),
            ParameterSpec(name="sigmoid_h_tile", lower=1, upper=128),
            ParameterSpec(name="mul_h_tile", lower=1, upper=128),
        ]

    @staticmethod
    def _decode_max_bandwidth(min_bw, selector, absolute_upper):
        """Decode an encoded selector into a max bandwidth >= min bandwidth."""
        min_bw_arr = np.asarray(min_bw, dtype=np.float64)
        selector_arr = np.asarray(selector, dtype=np.float64)
        selector_upper = max(1.0, absolute_upper - 1.0)
        remaining_bw = np.maximum(0.0, absolute_upper - min_bw_arr)
        scaled_gap = np.rint((selector_arr / selector_upper) * remaining_bw)
        return min_bw_arr + scaled_gap

    def _hardware_context_enricher(self, params: dict[str, float | int]) -> dict[str, object]:
        sram_max_b = int(
            self._decode_max_bandwidth(
                params["sram_min_b"], params["sram_max_b_selector"], self.SRAM_MAX_BW
            )
        )
        d2_sram_max_b = int(
            self._decode_max_bandwidth(
                params["d2_sram_min_b"],
                params["d2_sram_max_b_selector"],
                self.LOCAL_SRAM_MAX_BW,
            )
        )
        d1_sram_max_b = int(
            self._decode_max_bandwidth(
                params["d1_sram_min_b"],
                params["d1_sram_max_b_selector"],
                self.LOCAL_SRAM_MAX_BW,
            )
        )

        array_desc = [
            int(params["d1"]),
            int(params["d2"]),
            int(params["sram_size"]),
            int(params["sram_min_b"]),
            sram_max_b,
            int(params["d2_sram_size"]),
            int(params["d2_sram_min_b"]),
            d2_sram_max_b,
            int(params["d1_sram_size"]),
            int(params["d1_sram_min_b"]),
            d1_sram_max_b,
            int(params["rf_I1_size"]),
            int(params["rf_I1_bw"]),
            int(params["rf_I2_size"]),
            int(params["rf_I2_bw"]),
            int(params["rf_O_size"]),
            int(params["rf_O_bw"]),
        ]

        return {
            "sram_max_b": sram_max_b,
            "d2_sram_max_b": d2_sram_max_b,
            "d1_sram_max_b": d1_sram_max_b,
            "array_desc": array_desc,
        }

    @staticmethod
    def _constraint_fn(params: dict[str, float | int]) -> list[float]:
        return [
            float(int(params["sram_size"]) % 8),
            float(int(params["d2_sram_size"]) % 8),
            float(int(params["d1_sram_size"]) % 8),
            float(int(params["rf_I1_size"]) % 8),
            float(int(params["rf_I2_size"]) % 8),
            float(int(params["rf_O_size"]) % 8),
        ]

    @staticmethod
    def _objective_extractor(scme: object) -> list[float]:
        return [float(scme.latency), float(scme.energy)]

    def _build_variant_hardware_dir(
        self, eval_dir: Path, hardware_context: dict[str, object]
    ) -> Path:
        hardware_dir = eval_dir / "hardware"
        hardware_core_path = hardware_dir / "cores" / "udc_core.yaml"
        generate_hardware_from_udc(
            hardware_context["array_desc"],
            hardware_core_path,
            name="udc_core",
        )
        top_hardware_path = generate_top_core_with_offchip(
            output_hardware_dir=hardware_dir,
            offchip_source_path=self.offchip_source_path,
        )
        return Path(top_hardware_path)

    def _resolve_mapping(self, eval_dir: Path, params: dict[str, float | int]) -> Path:
        mapping_path = eval_dir / "mapping" / "mapping.yaml"
        render_mapping_from_template(
            self.mapping_template_path,
            mapping_path,
            linear1_k_tile=int(params["linear1_k_tile"]),
            linear1_c_tile="all",
            linear2_k_tile=int(params["linear2_k_tile"]),
            linear2_c_tile="all",
            sigmoid_h_tile=int(params["sigmoid_h_tile"]),
            mul_h_tile=int(params["mul_h_tile"]),
        )
        return mapping_path

    def decode_design_variables(self, x):
        """Decode encoded decision variables into actual hardware parameters."""
        decoded = np.array(x, dtype=np.float64, copy=True)

        if decoded.ndim == 1:
            decoded[4] = self._decode_max_bandwidth(
                decoded[3], decoded[4], self.SRAM_MAX_BW
            )
            decoded[7] = self._decode_max_bandwidth(
                decoded[6], decoded[7], self.LOCAL_SRAM_MAX_BW
            )
            decoded[10] = self._decode_max_bandwidth(
                decoded[9], decoded[10], self.LOCAL_SRAM_MAX_BW
            )
            return decoded

        decoded[:, 4] = self._decode_max_bandwidth(
            decoded[:, 3], decoded[:, 4], self.SRAM_MAX_BW
        )
        decoded[:, 7] = self._decode_max_bandwidth(
            decoded[:, 6], decoded[:, 7], self.LOCAL_SRAM_MAX_BW
        )
        decoded[:, 10] = self._decode_max_bandwidth(
            decoded[:, 9], decoded[:, 10], self.LOCAL_SRAM_MAX_BW
        )
        return decoded

    def _evaluate_single(self, task):
        result = super()._evaluate_single(task)
        eval_id, _ = task
        f, _, _, _, evaluation_failed = result
        if evaluation_failed:
            logging.warning("%s: STREAM evaluation failed, applying penalty.", eval_id)
        else:
            logging.info(
                "%s: latency=%.0f cycles, energy=%.2f pJ",
                eval_id,
                f[0],
                f[1],
            )
        return result


def optimize_udc_architecture(
    workload_path,
    mapping_template_path,
    experiment_id,
    output_path,
    mode="fused",
    layer_stacks=None,
    n_gen=10,
    pop_size=20,
    seed=1,
    n_jobs=1,
):
    """
    Optimize UDC architecture using genetic algorithm.

    Args:
        workload_path: Path to ONNX workload
        mapping_template_path: Path to mapping template file
        experiment_id: Experiment identifier
        output_path: Output directory
        mode: STREAM optimization mode
        layer_stacks: Layer stacking configuration
        n_gen: Number of generations
        pop_size: Population size
        seed: Random seed
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (pareto_x, pareto_f) - Pareto front solutions
    """
    if layer_stacks is None:
        layer_stacks = [(0, 1), (2,), (3,)]

    problem = UDCArchitectureOptimizationProblem(
        workload_path=workload_path,
        mapping_template_path=mapping_template_path,
        base_experiment_id=experiment_id,
        output_path=output_path,
        mode=mode,
        layer_stacks=layer_stacks,
        processes=n_jobs,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=BinomialCrossover(n_offsprings=2, prob=0.9),
        mutation=PM(prob=0.2, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=False,
        save_history=True,
    )

    # Extract Pareto front
    pareto_x = res.X
    pareto_f = res.F
    pareto_g = res.G if hasattr(res, "G") else None

    if pareto_x is None or pareto_f is None:
        raise RuntimeError("GA did not find any valid architecture configuration.")

    pareto_x = problem.decode_design_variables(pareto_x)

    logging.info("=" * 80)
    logging.info("PARETO FRONT SUMMARY")
    logging.info("=" * 80)
    logging.info("Number of solutions: %s", pareto_x.shape[0])
    logging.info("Decision variables shape: %s", pareto_x.shape)
    logging.info("Objectives shape: %s", pareto_f.shape)
    logging.info("=" * 80)

    # Save results
    ga_dir = Path(output_path) / experiment_id / "ga"
    ga_dir.mkdir(parents=True, exist_ok=True)

    decision_headers = [
        "d1",
        "d2",
        "sram_size",
        "sram_min_b",
        "sram_max_b",
        "d2_sram_size",
        "d2_sram_min_b",
        "d2_sram_max_b",
        "d1_sram_size",
        "d1_sram_min_b",
        "d1_sram_max_b",
        "rf_I1_size",
        "rf_I1_bw",
        "rf_I2_size",
        "rf_I2_bw",
        "rf_O_size",
        "rf_O_bw",
        "linear1_k_tile",
        "linear2_k_tile",
        "sigmoid_h_tile",
        "mul_h_tile",
    ]

    constraint_headers = [
        "g_sram_size_mod8",
        "g_d2_sram_size_mod8",
        "g_d1_sram_size_mod8",
        "g_rf_I1_size_mod8",
        "g_rf_I2_size_mod8",
        "g_rf_O_size_mod8",
    ]

    # Save Pareto front
    save_pareto_results_csv(
        ga_dir / "pareto_front.csv",
        pareto_x,
        pareto_f,
        pareto_g,
        decision_headers,
        constraint_headers,
    )

    # Save full history
    if hasattr(res, "history"):
        save_history_csv(
            ga_dir / "history.csv",
            res.history,
            decision_headers,
            constraint_headers,
            decoder=problem.decode_design_variables,
        )

    logging.info("[GA] Results saved to %s", ga_dir)
    return pareto_x, pareto_f


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize UDC architecture using genetic algorithm"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/UDC_GA",
        help="Output directory path",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default="udc_arch_opt",
        help="Experiment identifier",
    )
    parser.add_argument("--dim", type=int, default=1024, help="MLP input dimension")
    parser.add_argument(
        "--hidden_dim_factor",
        type=int,
        default=4,
        help="Hidden dimension as multiple of dim",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument(
        "--n_gen", type=int, default=10, help="Number of GA generations"
    )
    parser.add_argument("--pop_size", type=int, default=20, help="GA population size")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--mapping_template_path",
        type=str,
        default="inputs/test/mapping/mapping_template.jinja2",
        help="Path to mapping template file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fused",
        choices=["fused", "layer-by-layer"],
        help="STREAM optimization mode",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Create MLP model and export to ONNX
    dim = args.dim
    hidden_dim = dim * args.hidden_dim_factor
    model = MLP_model(dim=dim, hidden_dim=hidden_dim)

    onnx_path = f"{args.output_path}/{args.experiment_id}/model.onnx"
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Exporting MLP model (dim=%s, hidden_dim=%s) to ONNX...",
        dim,
        hidden_dim,
    )
    dummy_input = torch.randn(1, dim)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        dynamo=False,
    )

    # Infer shapes
    infer_shapes_path(onnx_path, onnx_path)
    logging.info("ONNX model saved to %s", onnx_path)

    # Define layer stacks for fused mode
    layer_stacks = [(0, 1), (2,), (3,)]

    # Run GA optimization
    logging.info("=" * 80)
    logging.info("STARTING GENETIC ALGORITHM OPTIMIZATION")
    logging.info("=" * 80)
    logging.info("Generations: %s", args.n_gen)
    logging.info("Population size: %s", args.pop_size)
    logging.info("Parallel jobs: %s", args.n_jobs)
    logging.info("Mode: %s", args.mode)
    logging.info("=" * 80)

    best_x, best_f = optimize_udc_architecture(
        workload_path=onnx_path,
        mapping_template_path=args.mapping_template_path,
        experiment_id=args.experiment_id,
        output_path=args.output_path,
        mode=args.mode,
        layer_stacks=layer_stacks,
        n_gen=args.n_gen,
        pop_size=args.pop_size,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    logging.info("=" * 80)
    logging.info("OPTIMIZATION COMPLETE")
    logging.info("=" * 80)
    logging.info("Results saved to: %s/%s/ga/", args.output_path, args.experiment_id)
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
