import argparse
import csv
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

from stream.api import optimize_allocation_co
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json
from onnx.shape_inference import infer_shapes_path


class MLP_model(nn.Module):
    def __init__(self, dim=4096, hidden_dim=11008):
        super(MLP_model, self).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.sigmoid(self.linear2(x))
        return self.linear1(x) * gate


def run_stream(
    accelerator,
    workload_path,
    mapping_path,
    mode,
    layer_stacks,
    experiment_id,
    output_path,
):
    penalty = 1e30
    try:
        # Run stream optimization
        scme = optimize_allocation_co(
            hardware=accelerator,
            workload=workload_path,
            mapping=mapping_path,
            mode=mode,
            layer_stacks=layer_stacks,
            experiment_id=experiment_id,
            output_path=output_path,
            skip_if_exists=False,
        )
        print(experiment_id, scme.latency, scme.energy)
        # #####################CostModelEvaluationLUT LOAD#############################
        # cost_lut_path = f"{output_path}/{experiment_id}/cost_lut_post_co.pickle"
        # cost_lut = CostModelEvaluationLUT(cost_lut_path)
        # #############################################################################
        # #########################PLOTTING PATHS##############################
        # memory_fig_path = f"{output_path}/{experiment_id}/memory.png"
        # json_path = f"{output_path}/{experiment_id}/scme.json"
        # #####################################################################

        # #####################CostModelEvaluationLUT LOAD#############################
        # cost_lut_path = f"{output_path}/{experiment_id}/cost_lut_post_co.pickle"
        # cost_lut = CostModelEvaluationLUT(cost_lut_path)
        # # Plotting memory usage of best SCME
        # plot_memory_usage(scme, (0,), (100,), fig_path=memory_fig_path, show_dram=True)

        # # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
        # convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)

        return scme, scme.latency, scme.energy
    except Exception as e:
        print(f"[run_stream] Failed for {experiment_id}: {e}")
        return None, penalty, penalty


def render_hardware_from_template(
    template_path,
    output_path,
    rf_I2_size=1024,
    rf_o_size=16,
    sram_size=2097152,
    d1_size=128,
    d2_size=128,
):
    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())

    rendered_hardware = template.render(
        rf_I2_size=rf_I2_size,
        rf_o_size=rf_o_size,
        sram_size=sram_size,
        d1_size=d1_size,
        d2_size=d2_size,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_hardware)


def copy_hardware_files(
    base_hardware_path,
    output_hardware_dir,
):
    """Copy tpu_like_quad_core.yaml and necessary core files to output directory."""
    import shutil

    output_hardware_dir = Path(output_hardware_dir)
    output_hardware_dir.mkdir(parents=True, exist_ok=True)

    # Copy the main tpu_like_quad_core.yaml
    shutil.copy(
        f"{base_hardware_path}/tpu_like_quad_core.yaml",
        output_hardware_dir / "tpu_like_quad_core.yaml",
    )

    # Create cores subdirectory
    cores_dir = output_hardware_dir / "cores"
    cores_dir.mkdir(parents=True, exist_ok=True)

    # Copy offchip.yaml
    shutil.copy(f"{base_hardware_path}/cores/offchip.yaml", cores_dir / "offchip.yaml")

    # Copy other core files that are referenced
    for core_file in ["pooling.yaml", "simd.yaml"]:
        src = Path(f"{base_hardware_path}/cores/{core_file}")
        if src.exists():
            shutil.copy(src, cores_dir / core_file)


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


def generate_mapping_and_run_stream(
    accelerator,
    workload_path,
    experiment_id,
    output_path,
    mode,
    layer_stacks,
    mapping_template_path="inputs/test/mapping/mapping_template.jinja2",
    hardware_template_path="inputs/test/hardware/cores/tpu_like_template.jinja2",
    base_hardware_path="inputs/test/hardware",
    linear1_k_tile=4,
    linear1_c_tile=1,
    linear2_k_tile=4,
    linear2_c_tile=1,
    sigmoid_h_tile=4,
    mul_h_tile=4,
    rf_I2_size=1024,
    rf_o_size=16,
    sram_size=2097152,
    d1_size=128,
    d2_size=128,
):
    # Setup hardware directory and copy base files
    hardware_dir = f"{output_path}/{experiment_id}/hardware"
    copy_hardware_files(base_hardware_path, hardware_dir)

    # Render hardware core from template
    hardware_core_path = f"{hardware_dir}/cores/tpu_like.yaml"
    render_hardware_from_template(
        hardware_template_path,
        hardware_core_path,
        rf_I2_size=rf_I2_size,
        rf_o_size=rf_o_size,
        sram_size=sram_size,
        d1_size=d1_size,
        d2_size=d2_size,
    )

    # Render mapping from template
    mapping_path = f"{output_path}/{experiment_id}/mapping/mapping.yaml"
    render_mapping_from_template(
        mapping_template_path,
        mapping_path,
        linear1_k_tile=linear1_k_tile,
        linear1_c_tile=linear1_c_tile,
        linear2_k_tile=linear2_k_tile,
        linear2_c_tile=linear2_c_tile,
        sigmoid_h_tile=sigmoid_h_tile,
        mul_h_tile=mul_h_tile,
    )

    # Use the generated hardware configuration
    accelerator_path = f"{hardware_dir}/tpu_like_quad_core.yaml"

    return run_stream(
        accelerator_path,
        workload_path,
        mapping_path,
        mode,
        layer_stacks,
        experiment_id,
        output_path,
    )


class MappingOptimizationProblem(Problem):
    def __init__(
        self,
        accelerator,
        workload_path,
        base_experiment_id,
        output_path,
        mode,
        layer_stacks,
        mapping_template_path,
        hardware_template_path,
        base_hardware_path,
        processes,
    ):
        # x = [linear2_k_tile, sigmoid_h_tile, rf_I2_size, rf_o_size, sram_size, d1_size, d2_size]
        super().__init__(
            n_var=7,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.array([1, 1, 256, 16, 1_048_576, 8, 8]),
            xu=np.array(
                [64, 64, 262_144, 8_192, 67_108_864, 512, 512]
            ),  # D1, D2, and RF sizes are powers of 2, but we relax that for the GA and round in the repair step
            vtype=int,
            elementwise_evaluation=False,
        )
        self.accelerator = accelerator
        self.workload_path = workload_path
        self.base_experiment_id = base_experiment_id
        self.output_path = output_path
        self.mode = mode
        self.layer_stacks = layer_stacks
        self.mapping_template_path = mapping_template_path
        self.hardware_template_path = hardware_template_path
        self.base_hardware_path = base_hardware_path
        self.processes = processes

    def single_stream_eval(self, x):
        (
            linear2_k_tile,
            sigmoid_h_tile,
            rf_I2_size,
            rf_o_size,
            sram_size,
            d1_size,
            d2_size,
        ) = [int(v) for v in x]

        experiment_id = (
            f"{self.base_experiment_id}_ga_eval_{getpid()}_{uuid4().hex[:8]}"
        )

        try:
            _, latency, energy = generate_mapping_and_run_stream(
                accelerator=self.accelerator,
                workload_path=self.workload_path,
                experiment_id=experiment_id,
                output_path=self.output_path,
                mode=self.mode,
                layer_stacks=self.layer_stacks,
                mapping_template_path=self.mapping_template_path,
                linear2_k_tile=linear2_k_tile,
                sigmoid_h_tile=sigmoid_h_tile,
                hardware_template_path=self.hardware_template_path,
                base_hardware_path=self.base_hardware_path,
                rf_I2_size=rf_I2_size,
                rf_o_size=rf_o_size,
                sram_size=sram_size,
                d1_size=d1_size,
                d2_size=d2_size,
            )
            objective = float(latency), float(energy)
        except Exception:
            objective = 1e30, 1e30

        return objective

    def _evaluate(self, x, out, *args, **kwargs):
        with Pool(processes=self.processes) as pool:
            r = pool.map(self.single_stream_eval, [individual for individual in x])

        out["F"] = np.array(r)

        # Constraints in pymoo are defined as G <= 0.
        # 1) Compute bound: D1 * D2 <= 128
        # 2) Memory bound: rf_I2_size*D1*D2 + rf_o_size*D1 + sram_size <= 1024*1024*8
        # 3) Physical-area bound using rough 12nm estimates:
        #    - SRAM bitcell area ~= 0.022 um^2 / bit
        #    - INT8 MAC area    ~= 400   um^2 / MAC
        #    Total area ~= (#MACs * mac_area) + (memory_bits * bitcell_area)
        #    Constrain to <= 0.30 mm^2 (= 300_000 um^2)
        rf_i2 = x[:, 2].astype(np.float64)
        rf_o = x[:, 3].astype(np.float64)
        sram = x[:, 4].astype(np.float64)
        d1 = x[:, 5].astype(np.float64)
        d2 = x[:, 6].astype(np.float64)

        compute = d1 * d2
        memory = rf_i2 * d1 * d2 + rf_o * d1 + sram
        sram_bitcell_area_um2 = 0.022
        mac_area_um2 = 400.0
        total_area_um2 = compute * mac_area_um2 + memory * sram_bitcell_area_um2

        g_compute = compute - 1024.0
        g_memory = memory - float(1024 * 1024 * 8)
        area_budget_um2 = 300_000.0
        g_area = total_area_um2 - area_budget_um2

        out["G"] = np.column_stack([g_compute])


def optimize_mapping_with_pymoo(
    accelerator,
    workload_path,
    experiment_id,
    output_path,
    mode,
    layer_stacks,
    mapping_template_path="inputs/test/mapping/mapping_template.jinja2",
    hardware_template_path="inputs/test/hardware/cores/tpu_like_template.jinja2",
    base_hardware_path="inputs/test/hardware",
    n_gen=8,
    pop_size=12,
    seed=1,
    n_jobs=1,
):
    problem = MappingOptimizationProblem(
        accelerator=accelerator,
        workload_path=workload_path,
        base_experiment_id=experiment_id,
        output_path=output_path,
        mode=mode,
        layer_stacks=layer_stacks,
        mapping_template_path=mapping_template_path,
        hardware_template_path=hardware_template_path,
        base_hardware_path=base_hardware_path,
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
        ("n_gen", n_gen),  # Number of generations
        seed=1,
        verbose=True,
        save_history=True,
    )

    # For multi-objective optimization, res.X and res.F contain the Pareto front
    pareto_x = res.X
    pareto_f = res.F
    pareto_g = res.G if hasattr(res, "G") else None

    # if res.X is not None:

    if pareto_x is None or pareto_f is None:
        raise RuntimeError("GA did not find any valid mapping configuration.")
    print("Pareto front:")
    print(f"X shape: {pareto_x.shape}")
    print(f"F shape: {pareto_f.shape}")

    if res.X is None:
        raise RuntimeError("GA did not find any valid mapping configuration.")

    ga_dir = Path(output_path) / experiment_id / "ga"
    ga_dir.mkdir(parents=True, exist_ok=True)

    decision_headers = [
        "linear2_k_tile",
        "sigmoid_h_tile",
        "rf_I2_size",
        "rf_o_size",
        "sram_size",
        "d1_size",
        "d2_size",
    ]

    # Save Pareto front configurations only
    result_csv_path = ga_dir / "result.csv"
    with open(result_csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "pareto_rank",
                *decision_headers,
                "latency",
                "energy",
                "g_compute",
                "g_memory",
                "g_area",
                "cv",
                "feasible",
            ]
        )

        # Handle both single solution and multiple solutions in Pareto front
        if pareto_x.ndim == 1:
            # Single solution
            pareto_x = pareto_x.reshape(1, -1)
            pareto_f = pareto_f.reshape(1, -1) if pareto_f.ndim == 1 else pareto_f
            if pareto_g is not None:
                pareto_g = pareto_g.reshape(1, -1) if pareto_g.ndim == 1 else pareto_g

        if pareto_g is None:
            # Fallback in case constraints are not exposed on result object
            rf_i2 = pareto_x[:, 2].astype(np.float64)
            rf_o = pareto_x[:, 3].astype(np.float64)
            sram = pareto_x[:, 4].astype(np.float64)
            d1 = pareto_x[:, 5].astype(np.float64)
            d2 = pareto_x[:, 6].astype(np.float64)

            compute = d1 * d2
            memory = rf_i2 * d1 * d2 + rf_o * d1 + sram
            total_area_um2 = compute * 400.0 + memory * 0.022

            g_compute = compute - 128.0
            g_memory = memory - float(1024 * 1024 * 8)
            g_area = total_area_um2 - 300_000.0
            pareto_g = np.column_stack([g_compute, g_memory, g_area])

        for i, (ind_x, ind_f, ind_g) in enumerate(
            zip(pareto_x, pareto_f, pareto_g, strict=True)
        ):
            ind_f_flat = ind_f.flatten() if ind_f.ndim > 1 else ind_f
            # ind_g_flat = ind_g.flatten() if ind_g.ndim > 1 else ind_g
            latency_value = float(ind_f_flat[0])
            energy_value = float(ind_f_flat[1])
            # g_compute_value = float(ind_g_flat[0])
            # g_memory_value = float(ind_g_flat[1])
            # g_area_value = float(ind_g_flat[2])
            # cv = (
            #     max(0.0, g_compute_value)
            #     + max(0.0, g_memory_value)
            #     + max(0.0, g_area_value)
            # )
            # feasible = int(cv <= 0.0)
            writer.writerow(
                [
                    i,
                    *ind_x.tolist(),
                    latency_value,
                    energy_value,
                    # g_compute_value,
                    # g_memory_value,
                    # g_area_value,
                    # cv,
                    # feasible,
                ]
            )

    # Save full generation history
    history_csv_path = ga_dir / "history.csv"
    with open(history_csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "generation",
                "individual",
                *decision_headers,
                "latency",
                "energy",
                "g_compute",
                "g_memory",
                "g_area",
                "cv",
                "feasible",
            ]
        )
        for gen_idx, run in enumerate(res.history):
            pop = run.pop
            for ind_idx, individual in enumerate(pop):
                ind_x = individual.X.tolist()
                ind_f_raw = individual.F
                ind_g_raw = individual.G

                ind_f_flat = (
                    ind_f_raw.flatten()
                    if getattr(ind_f_raw, "ndim", 1) > 1
                    else ind_f_raw
                )
                ind_g_flat = (
                    ind_g_raw.flatten()
                    if getattr(ind_g_raw, "ndim", 1) > 1
                    else ind_g_raw
                )

                latency_value = float(ind_f_flat[0])
                energy_value = float(ind_f_flat[1])
                g_compute_value = float(ind_g_flat[0])
                # g_memory_value = float(ind_g_flat[1])
                # g_area_value = float(ind_g_flat[2])
                # cv = (
                #     max(0.0, g_compute_value)
                #     + max(0.0, g_memory_value)
                #     + max(0.0, g_area_value)
                # )
                cv = max(
                    0.0, g_compute_value
                )  # + max(0.0, g_memory_value) + max(0.0, g_area_value)
                feasible = int(cv <= 0.0)

                writer.writerow(
                    [
                        gen_idx,
                        ind_idx,
                        *ind_x,
                        latency_value,
                        energy_value,
                        g_compute_value,
                        # g_memory_value,
                        # g_area_value,
                        # cv,
                        feasible,
                    ]
                )

    print(f"[GA] Saved GA result CSV: {result_csv_path}")
    print(f"[GA] Saved GA history CSV: {history_csv_path}")
    return pareto_x, pareto_f


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_gen", type=int, default=4)
    parser.add_argument("--pop_size", type=int, default=10)
    args = parser.parse_args()
    return args


def main(args):

    experiment_id = "mlp_test"
    accelerator = "inputs/test/hardware/tpu_like_quad_core.yaml"
    dim = 1024
    hidden_dim = dim * 4
    model = MLP_model(dim=dim, hidden_dim=hidden_dim)
    onnx_path = f"{args.output_path}/{experiment_id}/model.onnx"
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, dim)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        dynamo=False,
    )
    # infer shape
    infer_shapes_path(onnx_path, onnx_path)

    mode = "fused"
    layer_stacks = [(0, 1), (2,), (3,)]

    best_x, best_f = optimize_mapping_with_pymoo(
        accelerator=accelerator,
        workload_path=onnx_path,
        experiment_id=experiment_id,
        output_path=args.output_path,
        mode=mode,
        layer_stacks=layer_stacks,
        mapping_template_path="inputs/test/mapping/mapping_template.jinja2",
        n_gen=args.n_gen,
        pop_size=args.pop_size,
        seed=1,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    args = args_parse()
    main(args)
