import logging as _logging
from pathlib import Path

import torch
import yaml
from onnx.shape_inference import infer_shapes_path
from torch import nn

from stream.api import optimize_allocation_co
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json

_logging_level = _logging.INFO
_logging_format = (
    "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)


class MLPModel(nn.Module):
    """Simple MLP model used to generate the test UDC workload."""

    def __init__(self, dim=1024, hidden_dim=4096):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.linear1(x)
        # gate = self.sigmoid(self.linear2(x))
        # return self.linear1(x) * gate


def generate_model(workload_path, dim=1024, hidden_dim_factor=4, batch_size=1):
    """Generate the ONNX workload used by STREAM."""
    hidden_dim = dim * hidden_dim_factor
    model = MLPModel(dim=dim, hidden_dim=hidden_dim)
    workload_path = Path(workload_path)
    workload_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating ONNX model | path=%s  dim=%s  hidden_dim=%s",
        workload_path,
        dim,
        hidden_dim,
    )

    dummy_input = torch.randn(batch_size, dim)
    torch.onnx.export(
        model,
        dummy_input,
        str(workload_path),
        input_names=["input"],
        dynamo=False,
    )
    infer_shapes_path(str(workload_path), str(workload_path))
    logger.info("Generated ONNX workload at %s", workload_path)


def update_rf_memory_sizes(
    core_yaml_path, rf_i1_size=None, rf_i2_size=None, rf_o_size=None
):
    """Update RF memory sizes in a UDC core YAML file."""
    with open(core_yaml_path, "r", encoding="utf-8") as f:
        core_yaml = yaml.safe_load(f)

    memories = core_yaml.get("memories", {})

    if rf_i1_size is not None and "rf_I1" in memories:
        memories["rf_I1"]["size"] = int(rf_i1_size)
    if rf_i2_size is not None and "rf_I2" in memories:
        memories["rf_I2"]["size"] = int(rf_i2_size)
    if rf_o_size is not None and "rf_O" in memories:
        memories["rf_O"]["size"] = int(rf_o_size)

    with open(core_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(core_yaml, f, sort_keys=False)
    logger.info(
        "Updated RF sizes in %s: rf_I1=%s  rf_I2=%s  rf_O=%s",
        core_yaml_path,
        rf_i1_size,
        rf_i2_size,
        rf_o_size,
    )


def update_operational_array_sizes(core_yaml_path, d1_size=None, d2_size=None):
    """Update operational array sizes (D1/D2) in a UDC core YAML file."""
    with open(core_yaml_path, "r", encoding="utf-8") as f:
        core_yaml = yaml.safe_load(f)

    op_array = core_yaml.get("operational_array", {})
    dimensions = op_array.get("dimensions", [])
    sizes = op_array.get("sizes", [])

    if len(dimensions) != len(sizes):
        raise ValueError(
            f"Invalid operational_array format in {core_yaml_path}: "
            f"dimensions and sizes length mismatch"
        )

    dim_to_idx = {dim: i for i, dim in enumerate(dimensions)}

    if d1_size is not None:
        if "D1" not in dim_to_idx:
            raise KeyError(
                f"D1 not found in operational_array.dimensions of {core_yaml_path}"
            )
        sizes[dim_to_idx["D1"]] = int(d1_size)

    if d2_size is not None:
        if "D2" not in dim_to_idx:
            raise KeyError(
                f"D2 not found in operational_array.dimensions of {core_yaml_path}"
            )
        sizes[dim_to_idx["D2"]] = int(d2_size)

    with open(core_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(core_yaml, f, sort_keys=False)
    logger.info(
        "Updated operational array sizes in %s: D1=%s  D2=%s",
        core_yaml_path,
        d1_size,
        d2_size,
    )


def run_stream(
    accelerator,
    workload_path,
    mapping_path,
    mode,
    layer_stacks,
    output_path="outputs",
    experiment_id="test",
):
    """Run STREAM constraint optimization and return the SCME."""
    logger.info(
        "Starting STREAM | experiment=%s  mode=%s  accelerator=%s  workload=%s",
        experiment_id,
        mode,
        accelerator,
        workload_path,
    )

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

    memory_fig_path = f"{output_path}/{experiment_id}/memory.png"
    json_path = f"{output_path}/{experiment_id}/scme.json"
    cost_lut_path = f"{output_path}/{experiment_id}/cost_lut_post_co.pickle"

    cost_lut = CostModelEvaluationLUT(cost_lut_path)

    plot_memory_usage(
        scme,
        section_start_percent=(0,),
        percent_shown=(100,),
        fig_path=memory_fig_path,
        show_dram=True,
    )

    convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)
    logger.info(
        "STREAM done | experiment=%s  latency=%s  energy=%s",
        experiment_id,
        scme.latency,
        scme.energy,
    )

    return scme


def evaluate_rf_size(rf_i1_size, rf_i2_size, rf_o_size, d1_size=None, d2_size=None):
    """Helper function to evaluate a specific RF/op-array size configuration."""
    core_yaml_path = Path("inputs/test_udc/hardware/cores/udc_core.yaml")
    update_rf_memory_sizes(
        core_yaml_path,
        rf_i1_size=rf_i1_size,
        rf_i2_size=rf_i2_size,
        rf_o_size=rf_o_size,
    )
    update_operational_array_sizes(core_yaml_path, d1_size=d1_size, d2_size=d2_size)

    d1_label = d1_size if d1_size is not None else "keep"
    d2_label = d2_size if d2_size is not None else "keep"
    experiment_id = (
        f"test_udc_rf{rf_i1_size}_{rf_i2_size}_{rf_o_size}_d{d1_label}_{d2_label}"
    )
    latency, energy = None, None
    try:
        scme = run_stream(
            accelerator="inputs/test_udc/hardware/core.yaml",
            workload_path="inputs/test_udc/model.onnx",
            mapping_path="inputs/test_udc/mapping/mapping.yaml",
            mode="lbl",
            layer_stacks=[tuple(range(0, 12)), tuple(range(12, 22))]
            + list((i,) for i in range(22, 49)),
            output_path="outputs/test_udc/",
            experiment_id=experiment_id,
        )
        latency, energy = scme.latency, scme.energy
    except Exception as e:
        logger.error(
            f"RF_i1={rf_i1_size} RF_i2={rf_i2_size} RF_o={rf_o_size} "
            f"D1={d1_size} D2={d2_size} FAILED: {e}",
            exc_info=True,
        )

    return latency, energy


def main():
    ############################################INPUTS############################################
    workload_path = "inputs/test_udc/model.onnx"
    dim = 1024
    hidden_dim_factor = 4
    ##############################################################################################

    ########################################MODEL GENERATION######################################
    generate_model(
        workload_path=workload_path,
        dim=dim,
        hidden_dim_factor=hidden_dim_factor,
        batch_size=1,
    )
    ##############################################################################################

    ##############################RF SIZE SWEEP###################################################
    # rf_sizes = [2**i for i in range(4, 16)]  # 1, 2, 4, ..., 2048
    # results = []

    # for rf_size in rf_sizes:
    #     latency, energy = evaluate_rf_size(
    #         rf_i1_size=rf_size, rf_i2_size=rf_size, rf_o_size=rf_size
    #     )

    #     results.append(
    #         (rf_size, latency, energy, latency is not None and energy is not None)
    #     )
    # logger.info("SWEEP SUMMARY")
    # logger.info("%10s  %20s  %20s  %5s", "rf_size", "latency", "energy", "ok")
    # for rf_size, latency, energy, ok in results:
    #     lat_str = f"{latency:.4e}" if latency is not None else "N/A"
    #     eng_str = f"{energy:.4e}" if energy is not None else "N/A"
    #     logger.info("%10d  %20s  %20s  %5s", rf_size, lat_str, eng_str, ok)

    rf_i1_size = 256
    rf_i2_size = 256 * 2
    rf_o_size = 256 * 4

    results = []
    # latency, energy = evaluate_rf_size(
    #     rf_i1_size=256,
    #     rf_i2_size=256 * 2,
    #     rf_o_size=256 * 4,
    #     d1_size=9,
    #     d2_size=16,
    # )

    logger.info("COMBINED SWEEP (D1=1..16, D2=1..16)")
    for d1_size in range(1, 17):
        for d2_size in range(1, 17):
            latency, energy = evaluate_rf_size(
                rf_i1_size=rf_i1_size,
                rf_i2_size=rf_i2_size,
                rf_o_size=rf_o_size,
                d1_size=d1_size,
                d2_size=d2_size,
            )
            results.append(
                (
                    d1_size,
                    d2_size,
                    latency,
                    energy,
                    latency is not None and energy is not None,
                )
            )

    logger.info("COMBINED SWEEP SUMMARY")
    logger.info("%6s  %6s  %20s  %20s  %5s", "D1", "D2", "latency", "energy", "ok")
    for d1_size, d2_size, latency, energy, ok in results:
        lat_str = f"{latency:.4e}" if latency is not None else "N/A"
        eng_str = f"{energy:.4e}" if energy is not None else "N/A"
        logger.info("%6d  %6d  %20s  %20s  %5s", d1_size, d2_size, lat_str, eng_str, ok)


if __name__ == "__main__":
    main()
