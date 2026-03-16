"""
Universal Dataflow Accelerator Architecture Optimization using Genetic Algorithm
"""

import argparse
import csv
import shutil
from multiprocessing import Pool
from os import getpid
from pathlib import Path
from uuid import uuid4
import numpy as np
from torch import nn
import torch
from jinja2 import Template
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.optimize import minimize
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from onnx.shape_inference import infer_shapes_path
import yaml

from stream.api import optimize_allocation_co
from udc import array_to_hardware


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
    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())

    rendered_mapping = template.render(
        linear1_k_tile=linear1_k_tile,
        linear1_c_tile=linear1_c_tile,
        linear2_k_tile=linear2_k_tile,
        linear2_c_tile=linear2_c_tile,
        sigmoid_h_tile=sigmoid_h_tile,
        mul_h_tile=mul_h_tile,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_mapping)

    return output_path


def run_stream_with_udc_hardware(
    array_desc,
    workload_path,
    mapping_template_path,
    mode,
    layer_stacks,
    experiment_id,
    output_path,
    linear1_k_tile=4,
    linear2_k_tile=4,
    sigmoid_h_tile=4,
    mul_h_tile=4,
    offchip_source_path="inputs/test/hardware/cores/offchip.yaml",
):
    """
    Run STREAM optimization with UDC-generated hardware.

    Args:
        array_desc: Hardware parameters for UDC template
        workload_path: Path to ONNX workload
        mapping_template_path: Path to mapping template file
        mode: STREAM optimization mode
        layer_stacks: Layer stacking configuration
        experiment_id: Unique experiment identifier
        output_path: Output directory path
        linear1_k_tile: K tile size for linear1
        linear2_k_tile: K tile size for linear2
        sigmoid_h_tile: H tile size for sigmoid
        mul_h_tile: H tile size for mul
        offchip_source_path: Source path for offchip core YAML

    Returns:
        Tuple of (scme, latency, energy)
    """
    penalty = 1e30
    try:
        # Generate hardware configuration from UDC template
        hardware_dir = f"{output_path}/{experiment_id}/hardware"
        hardware_core_path = f"{hardware_dir}/cores/udc_core.yaml"
        generate_hardware_from_udc(array_desc, hardware_core_path, name="udc_core")

        # Generate top-level core that includes UDC + offchip
        top_hardware_path = generate_top_core_with_offchip(
            output_hardware_dir=hardware_dir,
            offchip_source_path=offchip_source_path,
        )

        # Generate mapping configuration from template
        mapping_path = f"{output_path}/{experiment_id}/mapping/mapping.yaml"
        render_mapping_from_template(
            mapping_template_path,
            mapping_path,
            linear1_k_tile=linear1_k_tile,
            linear1_c_tile="all",
            linear2_k_tile=linear2_k_tile,
            linear2_c_tile="all",
            sigmoid_h_tile=sigmoid_h_tile,
            mul_h_tile=mul_h_tile,
        )

        # Run STREAM optimization
        scme = optimize_allocation_co(
            hardware=top_hardware_path,
            workload=workload_path,
            mapping=mapping_path,
            mode=mode,
            layer_stacks=layer_stacks,
            experiment_id=experiment_id,
            output_path=output_path,
            skip_if_exists=False,
        )

        print(f"{experiment_id}: latency={scme.latency}, energy={scme.energy}")
        return scme, scme.latency, scme.energy

    except Exception as e:
        print(f"[run_stream_with_udc_hardware] Failed for {experiment_id}: {e}")
        return None, penalty, penalty


class UDCArchitectureOptimizationProblem(Problem):
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
    ):
        # Define bounds for each variable
        xl = np.array(
            [
                1,  # d1_min
                1,  # d2_min
                65536,  # sram_size_min (64KB)
                1,  # sram_min_b_min
                0,  # sram_max_b_selector_min
                4096,  # d2_sram_size_min (4KB)
                1,  # d2_sram_min_b_min
                0,  # d2_sram_max_b_selector_min
                4096,  # d1_sram_size_min (4KB)
                1,  # d1_sram_min_b_min
                0,  # d1_sram_max_b_selector_min
                8,  # rf_I1_size_min
                1,  # rf_I1_bw_min
                8,  # rf_I2_size_min
                1,  # rf_I2_bw_min
                8,  # rf_O_size_min
                1,  # rf_O_bw_min
                1,  # linear1_k_tile_min
                1,  # linear2_k_tile_min
                1,  # sigmoid_h_tile_min
                1,  # mul_h_tile_min
            ]
        )

        xu = np.array(
            [
                16,  # d1_max
                16,  # d2_max
                67108864,  # sram_size_max (64MB)
                512,  # sram_min_b_max
                511,  # sram_max_b_selector_max
                1048576,  # d2_sram_size_max (1MB)
                256,  # d2_sram_min_b_max
                255,  # d2_sram_max_b_selector_max
                1048576,  # d1_sram_size_max (1MB)
                256,  # d1_sram_min_b_max
                255,  # d1_sram_max_b_selector_max
                256,  # rf_I1_size_max (1/4KB)
                128,  # rf_I1_bw_max
                256,  # rf_I2_size_max (1/4KB)
                128,  # rf_I2_bw_max
                256,  # rf_O_size_max (1/4KB)
                128,  # rf_O_bw_max
                128,  # linear1_k_tile_max
                128,  # linear2_k_tile_max
                128,  # sigmoid_h_tile_max
                128,  # mul_h_tile_max
            ]
        )

        super().__init__(
            n_var=21,
            n_obj=2,  # latency and energy
            n_ieq_constr=6,  # memory-alignment constraints
            xl=xl,
            xu=xu,
            vtype=int,
            elementwise_evaluation=False,
        )

        self.workload_path = workload_path
        self.mapping_template_path = mapping_template_path
        self.base_experiment_id = base_experiment_id
        self.output_path = output_path
        self.mode = mode
        self.layer_stacks = layer_stacks
        self.processes = processes

    @staticmethod
    def _decode_max_bandwidth(min_bw, selector, absolute_upper):
        """Decode an encoded selector into a max bandwidth >= min bandwidth."""
        min_bw_arr = np.asarray(min_bw, dtype=np.float64)
        selector_arr = np.asarray(selector, dtype=np.float64)
        selector_upper = max(1.0, absolute_upper - 1.0)
        remaining_bw = np.maximum(0.0, absolute_upper - min_bw_arr)
        scaled_gap = np.rint((selector_arr / selector_upper) * remaining_bw)
        return min_bw_arr + scaled_gap

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

    def single_stream_eval(self, x):
        """Evaluate a single architecture configuration."""
        penalty = 1e30
        decoded_x = self.decode_design_variables(x)

        # First 17 variables are hardware parameters
        array_desc = [int(v) for v in decoded_x[:17]]

        # Last 4 variables are tiling parameters
        linear1_k_tile = int(decoded_x[17])
        linear2_k_tile = int(decoded_x[18])
        sigmoid_h_tile = int(decoded_x[19])
        mul_h_tile = int(decoded_x[20])

        experiment_id = (
            f"{self.base_experiment_id}_ga_eval_{getpid()}_{uuid4().hex[:8]}"
        )

        try:
            _, latency, energy = run_stream_with_udc_hardware(
                array_desc=array_desc,
                workload_path=self.workload_path,
                mapping_template_path=self.mapping_template_path,
                mode=self.mode,
                layer_stacks=self.layer_stacks,
                experiment_id=experiment_id,
                output_path=self.output_path,
                linear1_k_tile=linear1_k_tile,
                linear2_k_tile=linear2_k_tile,
                sigmoid_h_tile=sigmoid_h_tile,
                mul_h_tile=mul_h_tile,
            )
            objective = float(latency), float(energy)
        except Exception as e:
            print(f"[single_stream_eval] Exception: {e}")
            objective = penalty, penalty

        return objective

    @staticmethod
    def _compute_simple_constraints_scalar(x):
        """Compute simple scalar constraints for one individual (G <= 0)."""
        sram_size = float(x[2])
        d2_sram_size = float(x[5])
        d1_sram_size = float(x[8])
        rf_I1_size = float(x[11])
        rf_I2_size = float(x[13])
        rf_O_size = float(x[15])

        # Compute / memory / area constraints
        # compute = d1 * d2
        # g_compute = compute - 131072.0

        # total_memory = (
        #     sram_size
        #     + d2_sram_size
        #     + d1_sram_size
        #     + rf_I1_size * d1 * d2
        #     + rf_I2_size * d1 * d2
        #     + rf_O_size * d1
        # )
        # g_memory = total_memory - (8 * 1024 * 1024 * 8)

        # total_area_um2 = compute * 400.0 + total_memory * 0.022
        # g_area = total_area_um2 - 500_000.0

        # Memory alignment constraints: all memory sizes are multiples of 8
        g_sram_size_mod8 = sram_size % 8.0
        g_d2_sram_size_mod8 = d2_sram_size % 8.0
        g_d1_sram_size_mod8 = d1_sram_size % 8.0
        g_rf_I1_size_mod8 = rf_I1_size % 8.0
        g_rf_I2_size_mod8 = rf_I2_size % 8.0
        g_rf_O_size_mod8 = rf_O_size % 8.0

        return [
            g_sram_size_mod8,
            g_d2_sram_size_mod8,
            g_d1_sram_size_mod8,
            g_rf_I1_size_mod8,
            g_rf_I2_size_mod8,
            g_rf_O_size_mod8,
        ]

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate population and compute constraints."""
        decoded_x = self.decode_design_variables(x)

        # Extract variables for constraint computation
        sram_size = decoded_x[:, 2].astype(np.float64)
        d2_sram_size = decoded_x[:, 5].astype(np.float64)
        d1_sram_size = decoded_x[:, 8].astype(np.float64)
        rf_I1_size = decoded_x[:, 11].astype(np.float64)
        rf_I2_size = decoded_x[:, 13].astype(np.float64)
        rf_O_size = decoded_x[:, 15].astype(np.float64)

        # ============================================================
        # CONSTRAINT DEFINITIONS (G <= 0)
        # ============================================================

        # Example optional constraints kept here for future extension:
        # compute = d1 * d2
        # g_compute = compute - 131072.0  # 128K MACs

        # total_memory = (
        #     sram_size
        #     + d2_sram_size
        #     + d1_sram_size
        #     + rf_I1_size * d1 * d2
        #     + rf_I2_size * d1 * d2
        #     + rf_O_size * d1
        # )
        # g_memory = total_memory - (8 * 1024 * 1024 * 8)  # 8MB in bits

        # sram_bitcell_area_um2 = 0.022
        # mac_area_um2 = 400.0
        # total_area_um2 = compute * mac_area_um2 + total_memory * sram_bitcell_area_um2
        # area_budget_um2 = 500_000.0  # 0.5 mm^2
        # g_area = total_area_um2 - area_budget_um2

        # Memory alignment constraints: every memory size must be a multiple of 8
        # For integer variables, remainder is >= 0. Enforcing remainder <= 0 forces remainder == 0.
        g_sram_size_mod8 = np.mod(sram_size, 8.0)
        g_d2_sram_size_mod8 = np.mod(d2_sram_size, 8.0)
        g_d1_sram_size_mod8 = np.mod(d1_sram_size, 8.0)
        g_rf_I1_size_mod8 = np.mod(rf_I1_size, 8.0)
        g_rf_I2_size_mod8 = np.mod(rf_I2_size, 8.0)
        g_rf_O_size_mod8 = np.mod(rf_O_size, 8.0)

        # Build constraint matrix first to identify feasible candidates.
        g_matrix = np.column_stack(
            [
                g_sram_size_mod8,
                g_d2_sram_size_mod8,
                g_d1_sram_size_mod8,
                g_rf_I1_size_mod8,
                g_rf_I2_size_mod8,
                g_rf_O_size_mod8,
            ]
        )

        # Evaluate only feasible candidates (G <= 0) and penalize the rest.
        penalty = 1e30
        feasible_mask = np.all(g_matrix <= 0.0, axis=1)
        f_matrix = np.full((x.shape[0], 2), penalty, dtype=np.float64)

        feasible_indices = np.where(feasible_mask)[0]
        if feasible_indices.size > 0:
            feasible_individuals = [x[i] for i in feasible_indices]
            with Pool(processes=self.processes) as pool:
                feasible_results = pool.map(
                    self.single_stream_eval, feasible_individuals
                )
            f_matrix[feasible_indices] = np.asarray(feasible_results, dtype=np.float64)

        out["F"] = f_matrix

        # ============================================================
        # SPACE FOR ADDITIONAL CONSTRAINTS
        # ============================================================
        # Add your custom constraints here following the pattern:
        # g_custom = expression - threshold
        # where the constraint is: expression <= threshold

        # Example: Ensure RF sizes are proportional to array dimensions
        # g_rf_proportion = (rf_I1_size + rf_I2_size) - (d1 * d2 * 16)

        # Example: Bandwidth utilization constraint
        # min_bw_utilization = 0.5
        # g_bw_util = min_bw_utilization - (sram_min_b / sram_max_b)

        # ============================================================

        # Stack all constraints
        out["G"] = g_matrix


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

    print("\n" + "=" * 80)
    print("PARETO FRONT SUMMARY")
    print("=" * 80)
    print(f"Number of solutions: {pareto_x.shape[0]}")
    print(f"Decision variables shape: {pareto_x.shape}")
    print(f"Objectives shape: {pareto_f.shape}")
    print("=" * 80 + "\n")

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
    save_pareto_results(
        ga_dir / "pareto_front.csv",
        pareto_x,
        pareto_f,
        pareto_g,
        decision_headers,
        constraint_headers,
    )

    # Save full history
    if hasattr(res, "history"):
        save_full_history(
            ga_dir / "history.csv",
            res.history,
            decision_headers,
            constraint_headers,
            decoder=problem.decode_design_variables,
        )

    print(f"[GA] Results saved to {ga_dir}")
    return pareto_x, pareto_f


def save_pareto_results(
    filepath, pareto_x, pareto_f, pareto_g, decision_headers, constraint_headers
):
    """Save Pareto front results to CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "pareto_rank",
                *decision_headers,
                "latency",
                "energy",
                *constraint_headers,
                "cv",
                "feasible",
            ]
        )

        # Handle single or multiple solutions
        if pareto_x.ndim == 1:
            pareto_x = pareto_x.reshape(1, -1)
            pareto_f = pareto_f.reshape(1, -1) if pareto_f.ndim == 1 else pareto_f
            if pareto_g is not None:
                pareto_g = pareto_g.reshape(1, -1) if pareto_g.ndim == 1 else pareto_g

        for i, (ind_x, ind_f) in enumerate(zip(pareto_x, pareto_f, strict=False)):
            ind_f_flat = ind_f.flatten() if ind_f.ndim > 1 else ind_f
            latency_value = float(ind_f_flat[0])
            energy_value = float(ind_f_flat[1])

            # Compute constraint violation
            if pareto_g is not None:
                ind_g = pareto_g[i]
                ind_g_flat = ind_g.flatten() if ind_g.ndim > 1 else ind_g
                cv = sum(max(0.0, float(g)) for g in ind_g_flat)
                constraint_values = [float(g) for g in ind_g_flat]
            else:
                cv = 0.0
                constraint_values = [0.0] * len(constraint_headers)

            feasible = int(cv <= 1e-6)

            writer.writerow(
                [
                    i,
                    *ind_x.tolist(),
                    latency_value,
                    energy_value,
                    *constraint_values,
                    cv,
                    feasible,
                ]
            )

    print(f"[GA] Saved Pareto front to {filepath}")


def save_full_history(
    filepath, history, decision_headers, constraint_headers, decoder=None
):
    """Save full GA history to CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "generation",
                "individual",
                *decision_headers,
                "latency",
                "energy",
                *constraint_headers,
                "cv",
                "feasible",
            ]
        )

        for gen_idx, run in enumerate(history):
            pop = run.pop
            for ind_idx, individual in enumerate(pop):
                decoded_x = (
                    decoder(individual.X) if decoder is not None else individual.X
                )
                ind_x = np.asarray(decoded_x).tolist()
                ind_f_raw = individual.F
                ind_g_raw = individual.G if hasattr(individual, "G") else None

                ind_f_flat = (
                    ind_f_raw.flatten()
                    if getattr(ind_f_raw, "ndim", 1) > 1
                    else ind_f_raw
                )

                latency_value = float(ind_f_flat[0])
                energy_value = float(ind_f_flat[1])

                if ind_g_raw is not None:
                    ind_g_flat = (
                        ind_g_raw.flatten()
                        if getattr(ind_g_raw, "ndim", 1) > 1
                        else ind_g_raw
                    )
                    cv = sum(max(0.0, float(g)) for g in ind_g_flat)
                    constraint_values = [float(g) for g in ind_g_flat]
                else:
                    cv = 0.0
                    constraint_values = [0.0] * len(constraint_headers)

                feasible = int(cv <= 1e-6)

                writer.writerow(
                    [
                        gen_idx,
                        ind_idx,
                        *ind_x,
                        latency_value,
                        energy_value,
                        *constraint_values,
                        cv,
                        feasible,
                    ]
                )

    print(f"[GA] Saved full history to {filepath}")


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

    # Create MLP model and export to ONNX
    dim = args.dim
    hidden_dim = dim * args.hidden_dim_factor
    model = MLP_model(dim=dim, hidden_dim=hidden_dim)

    onnx_path = f"{args.output_path}/{args.experiment_id}/model.onnx"
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting MLP model (dim={dim}, hidden_dim={hidden_dim}) to ONNX...")
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
    print(f"ONNX model saved to {onnx_path}")

    # Define layer stacks for fused mode
    layer_stacks = [(0, 1), (2,), (3,)]

    # Run GA optimization
    print("\n" + "=" * 80)
    print("STARTING GENETIC ALGORITHM OPTIMIZATION")
    print("=" * 80)
    print(f"Generations: {args.n_gen}")
    print(f"Population size: {args.pop_size}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Mode: {args.mode}")
    print("=" * 80 + "\n")

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

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_path}/{args.experiment_id}/ga/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
