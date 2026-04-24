# Write a pytorch model with one gemm and export it to ONNX

import shutil
import torch
import torch.nn as nn
import argparse
import yaml
from stream.api import optimize_allocation_co
from pathlib import Path
from onnx import shape_inference

from stream.stream.utils import CostModelEvaluationLUT
from stream.stream.visualization.memory_usage import plot_memory_usage
from stream.stream.visualization.perfetto import convert_scme_to_perfetto_json

from torchinfo import summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GEMM model")
    parser.add_argument(
        "--output", type=str, default="gemm_model.onnx", help="Output ONNX file name"
    )
    return parser.parse_args()


class ConvConvModel(nn.Module):
    def __init__(self):
        super(ConvConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        print("Conv1 Flops", self.conv1.weight.numel() * h * w * 2)
        print("Conv2 Flops", self.conv2.weight.numel() * h * w * 2)
        x = self.conv1(x)
        b, c, h, w = x.size()
        print("Relu1 Flops", b * c * h * w)
        x = torch.relu(x)
        x = self.conv2(x)
        b, c, h, w = x.size()
        print("Relu2 Flops", b * c * h * w)
        x = torch.relu(x)
        return x


def copy_core_files(hardware_yaml_path, output_dir):
    """
    Parse the hardware YAML file and copy all referenced core files
    to the output directory while maintaining the directory structure.

    Args:
        hardware_yaml_path: Path to the hardware YAML file
        output_dir: Directory where files should be copied to
    """
    hardware_path = Path(hardware_yaml_path)
    output_path = Path(output_dir)

    # Read the hardware YAML file
    with open(hardware_path, "r") as f:
        hardware_config = yaml.safe_load(f)

    # Extract core file references
    if "cores" in hardware_config:
        cores = hardware_config["cores"]

        for core_id, core_file in cores.items():
            # Resolve the core file path relative to the hardware YAML file
            core_file_path = hardware_path.parent / core_file

            if core_file_path.exists():
                # Create the corresponding output path
                # Keep the directory structure relative to the hardware file's directory
                relative_path = Path(core_file)
                output_file_path = output_path / relative_path

                # Create parent directories if needed
                output_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy the file
                shutil.copyfile(core_file_path, output_file_path)
                print(f"Copied core file: {core_file} -> {output_file_path}")
            else:
                print(f"Warning: Core file not found: {core_file_path}")


def run_stream(
    accelerator,
    workload_path,
    mapping_path,
    mode="fused",
    layer_stacks=None,
    experiment_id=None,
    output_path="outputs",
):
    # create folder for experiment outputs

    Path(f"{output_path}/{experiment_id}").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/{experiment_id}/hardware/").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/{experiment_id}/mapping/").mkdir(parents=True, exist_ok=True)
    # Copy input files to output folder for record-keeping
    shutil.copyfile(workload_path, f"{output_path}/{experiment_id}/workload.onnx")
    shutil.copyfile(mapping_path, f"{output_path}/{experiment_id}/mapping/mapping.yaml")
    shutil.copyfile(
        accelerator, f"{output_path}/{experiment_id}/hardware/hardware.yaml"
    )

    # Copy all referenced core files while maintaining directory structure
    copy_core_files(accelerator, f"{output_path}/{experiment_id}/hardware/")

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
    ############PLOTTING#############
    plot_full_schedule = True
    draw_dependencies = True
    plot_data_transfer = True
    section_start_percent = (0,)
    percent_shown = (100,)
    #################################

    #########################PLOTTING PATHS##############################
    timeline_fig_path_plotly = f"{output_path}/{experiment_id}/schedule.html"
    memory_fig_path = f"{output_path}/{experiment_id}/memory.png"
    json_path = f"{output_path}/{experiment_id}/scme.json"
    #####################################################################

    #####################CostModelEvaluationLUT LOAD#############################
    cost_lut_path = f"{output_path}/{experiment_id}/cost_lut_post_co.pickle"
    cost_lut = CostModelEvaluationLUT(cost_lut_path)
    #############################################################################

    # Plotting memory usage of best SCME
    plot_memory_usage(
        scme, section_start_percent, percent_shown, fig_path=memory_fig_path
    )

    # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)

    return scme, scme.latency, scme.energy


def main(args):
    model = ConvConvModel()
    # Export the model to ONNX

    onnx_path = f"{args.output}model.onnx"
    dummy_input = torch.randn(1, 3, 32, 32)
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy_input, onnx_path, dynamo=False)
    # infer shape
    shape_inference.infer_shapes_path(onnx_path, onnx_path)

    mode = "fused"
    layer_stacks = [tuple(range(0, 3))]
    workload_path = args.output

    id = "1"
    # Experiment 1
    accelerator = "inputs/convconv1/hardware/soc.yaml"
    mapping_path = "inputs/convconv1/mapping/mapping.yaml"
    experiment_id_1 = f"convconv1_test_{id}"

    scme1, scme1.latency, scme1.energy = run_stream(
        accelerator,
        onnx_path,
        mapping_path,
        mode,
        layer_stacks,
        experiment_id_1,
        # output_path=args.output,
    )

    id = "2"
    # Experiment 2
    accelerator = "inputs/convconv2/hardware/soc.yaml"
    mapping_path = "inputs/convconv2/mapping/mapping.yaml"
    experiment_id_2 = f"convconv2_test_{id}"
    scme2, scme2.latency, scme2.energy = run_stream(
        accelerator,
        onnx_path,
        mapping_path,
        mode,
        layer_stacks,
        experiment_id_2,
        # output_path=args.output,
    )
    # # Experiment 2
    # accelerator = "inputs/small_gemm/hardware/tpu_like_quad_core.yaml"
    # mapping_path = "inputs/small_gemm/mapping/tpu_like_quad_core.yaml"
    # experiment_id_2 = f"small_gemm_test_{id}"
    # scme2, scme2.latency, scme2.energy = run_stream(
    #     accelerator, workload_path, mapping_path, mode, layer_stacks, experiment_id_2
    # )
    # # Experiment 3
    # accelerator = "inputs/small_gemm_full_connect/hardware/tpu_like_quad_core.yaml"
    # mapping_path = "inputs/small_gemm_full_connect/mapping/tpu_like_quad_core.yaml"
    # experiment_id_3 = f"small_gemm_full_connect_test_{id}"

    # scme3, scme3.latency, scme3.energy = run_stream(
    #     accelerator, workload_path, mapping_path, mode, layer_stacks, experiment_id_3
    # )

    print(f"Experiment {experiment_id_1} results:")
    print(f"Latency: {scme1.latency}, Energy: {scme1.energy}")
    print(f"Experiment {experiment_id_2} results:")
    print(f"Latency: {scme2.latency}, Energy: {scme2.energy}")
    # print(f"Experiment {experiment_id_3} results:")
    # print(f"Latency: {scme3.latency}, Energy: {scme3.energy}")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
