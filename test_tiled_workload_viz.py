import logging as _logging
from pathlib import Path

import gurobipy as gp
import onnx
import torch
import yaml
from onnx.shape_inference import infer_shapes_path
from torch import nn
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import pickle_save

from stream.stages.allocation.constraint_optimization_allocation import (
    ConstraintOptimizationAllocationStage,
)
from stream.stages.estimation.zigzag_core_mapping_estimation import (
    ZigZagCoreMappingEstimationStage,
)
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.tiled_workload_generation import (
    TiledWorkloadGenerationStage,
)
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import (
    ONNXModelParserStage as StreamONNXModelParserStage,
)
from stream.stages.set_fixed_allocation_performance import (
    SetFixedAllocationPerformanceStage,
)
from stream.stages.stage import MainStage
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json

_logging_level = _logging.INFO
_logging_format = (
    "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)


class TinyResidualConvNet(nn.Module):
    """2-conv residual block: Conv->ReLU->Conv + skip -> ReLU."""

    def __init__(self, channels: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(channels * 8 * 8, channels * 8 * 64)
        self.linear2 = nn.Linear(channels * 8 * 64, channels * 8 * 8)

    def forward(self, x):
        residual = x
        # x = self.conv1(torch.relu(x))
        x = self.linear2(torch.relu(self.linear1(x)))
        # x = torch.relu(self.conv1(x))
        # x = self.conv2(x)
        # # x = x + residual
        # # residual = x
        # x = torch.relu(x)
        # x = torch.relu(self.conv3(x))
        # x = self.conv4(x)
        # # x = x + residual
        # x = torch.relu(x)
        return x


def generate_onnx_workload(workload_path: str) -> int:
    """Export a very small residual ConvNet to ONNX and infer shapes."""
    model = TinyResidualConvNet(channels=4).eval()
    dummy_input = torch.randn(1, 4 * 8 * 8)
    workload_file = Path(workload_path)
    workload_file.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(workload_file),
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamo=False,
    )
    infer_shapes_path(str(workload_file), str(workload_file))

    nb_layers = len(onnx.load(str(workload_file)).graph.node)
    logger.info(
        "Generated ONNX workload at %s with %d graph node(s).", workload_file, nb_layers
    )
    return nb_layers


def create_mapping_file(
    base_mapping_path: str,
    output_mapping_path: str,
    mapping_updates: dict[str, dict[str, object]] | None = None,
) -> str:
    """Create a mapping YAML and optionally override entries from Python.

    mapping_updates example:
    {
        "default": {"core_allocation": [0]},
        "Conv": {"core_allocation": [0, 1], "inter_core_tiling": ["K, *"]},
        "Add": {"core_allocation": [5]},
    }
    """
    with open(base_mapping_path, "r", encoding="utf-8") as f:
        mapping_data = yaml.safe_load(f)

    if mapping_updates:
        for entry in mapping_data:
            name = entry.get("name")
            if name in mapping_updates:
                entry.update(mapping_updates[name])

    output_file = Path(output_mapping_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(mapping_data, f, sort_keys=False)

    logger.info("Generated mapping file at %s.", output_file)
    return str(output_file)


def sanity_check_gurobi_license():
    try:
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        model.optimize()
    except gp.GurobiError as exc:
        if exc.errno == gp.GRB.Error.NO_LICENSE:
            msg = (
                "No valid Gurobi license found. Get an academic WLS license at "
                "https://www.gurobi.com/academia/academic-program-and-licenses/"
            )
        else:
            msg = f"Unexpected Gurobi error: {exc.message}"
        raise ValueError(msg) from exc


def run_constraint_optimization_with_prime_sweep(
    *,
    hardware: str,
    workload: str,
    mapping: str,
    mode: str,
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_path: str,
):
    sanity_check_gurobi_license()
    experiment_dir = Path(output_path) / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    tiled_workload_path = str(experiment_dir / "tiled_workload.pickle")
    cost_lut_path = str(experiment_dir / "cost_lut.pickle")
    allocations_path = str(experiment_dir / "waco")
    tiled_workload_post_co_path = str(experiment_dir / "tiled_workload_post_co.pickle")
    cost_lut_post_co_path = str(experiment_dir / "cost_lut_post_co.pickle")
    scme_path = str(experiment_dir / "scme.pickle")
    prime_tiling_plots_dir = str(experiment_dir / "prime_intra_core_tiling_plots")

    mainstage = MainStage(
        [
            AcceleratorParserStage,
            StreamONNXModelParserStage,
            LayerStacksGenerationStage,
            TilingGenerationStage,
            TiledWorkloadGenerationStage,
            ZigZagCoreMappingEstimationStage,
            SetFixedAllocationPerformanceStage,
            ConstraintOptimizationAllocationStage,
        ],
        accelerator=hardware,
        workload_path=workload,
        mapping_path=mapping,
        loma_lpf_limit=6,
        mode=mode,
        layer_stacks=layer_stacks,
        tiled_workload_path=tiled_workload_path,
        cost_lut_path=cost_lut_path,
        allocations_path=allocations_path,
        tiled_workload_post_co_path=tiled_workload_post_co_path,
        cost_lut_post_co_path=cost_lut_post_co_path,
        temporal_mapping_type=TemporalMappingType.UNEVEN,
        operands_to_prefetch=[],
        sweep_prime_intra_core_tiling=True,
        prime_tiling_plots_dir=prime_tiling_plots_dir,
    )

    answers = mainstage.run()
    scme = answers[0][0]
    pickle_save(scme, scme_path)  # type: ignore[arg-type]
    return scme, prime_tiling_plots_dir, cost_lut_post_co_path


def main():
    ############################################INPUTS############################################
    accelerator = "inputs/stream/hardware/tpu_like_quad_core.yaml"
    workload_path = "inputs/test_tiled_workload_viz/workload/tiny_residual_2conv.onnx"
    base_mapping_path = "inputs/stream/mapping/tpu_like_quad_core_viz.yaml"
    mode = "fused"
    output_path = "outputs"
    mapping_updates = {
        "Conv": {
            "core_allocation": [0],
            "intra_core_tiling": ["OY, 8"],
            "inter_core_tiling": ["K, 1"],
        },
        "Relu": {
            "core_allocation": [0],
            "intra_core_tiling": ["D, 8"],
            "inter_core_tiling": ["H, 1"],
        },
        # Edit this dict in Python to change mapping behavior.
        # Example:
        # "Conv": {"core_allocation": [0, 1]},
        # "Add": {"core_allocation": [5]},
    }
    ##############################################################################################

    nb_layers = generate_onnx_workload(workload_path)
    layer_stacks = [tuple(range(nb_layers))]
    mapping_path = create_mapping_file(
        base_mapping_path=base_mapping_path,
        output_mapping_path="inputs/test_tiled_workload_viz/mapping/mapping.yaml",
        mapping_updates=mapping_updates,
    )

    ################################PARSING###############################
    hw_name = Path(accelerator).stem
    wl_name = Path(workload_path).stem
    experiment_id = f"{hw_name}-{wl_name}-{mode}-constraint_optimization"
    ######################################################################

    scme, prime_tiling_plots_dir, cost_lut_path = (
        run_constraint_optimization_with_prime_sweep(
            hardware=accelerator,
            workload=workload_path,
            mapping=mapping_path,
            mode=mode,
            layer_stacks=layer_stacks,
            experiment_id=experiment_id,
            output_path=output_path,
        )
    )

    #########################PLOTTING PATHS##############################
    memory_fig_path = f"{output_path}/{experiment_id}/memory.png"
    json_path = f"{output_path}/{experiment_id}/scme.json"
    #####################################################################

    #####################CostModelEvaluationLUT LOAD#############################
    cost_lut = CostModelEvaluationLUT(cost_lut_path)
    #############################################################################

    prime_plot_count = len(list(Path(prime_tiling_plots_dir).glob("*.png")))
    if prime_plot_count == 0:
        raise RuntimeError(
            f"No prime intra-core tiling plots were generated in {prime_tiling_plots_dir}"
        )
    logger.info(
        "Generated %d prime intra-core tiling plot(s) in %s.",
        prime_plot_count,
        prime_tiling_plots_dir,
    )

    plot_memory_usage(
        scme, section_start_percent=(0,), percent_shown=(100,), fig_path=memory_fig_path
    )
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)
    print(scme.latency, scme.energy)


if __name__ == "__main__":
    main()
