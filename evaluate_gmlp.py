# Write a pytorch model with one gemm and export it to ONNX

import shutil
import torch
import torch.nn as nn
import argparse
import yaml
from stream.api import optimize_allocation_co
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GEMM model")
    parser.add_argument(
        "--output", type=str, default="gemm_model.onnx", help="Output ONNX file name"
    )
    return parser.parse_args()


class GatedMLPModel(nn.Module):
    def __init__(self):
        super(GatedMLPModel, self).__init__()
        self.linear1 = nn.Linear(4096, 11008)
        self.linear2 = nn.Linear(4096, 11008)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.sigmoid(self.linear2(x))
        return self.linear1(x) * gate


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
):
    # create folder for experiment outputs

    Path(f"outputs/{experiment_id}").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/{experiment_id}/hardware/").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/{experiment_id}/mapping/").mkdir(parents=True, exist_ok=True)
    # Copy input files to output folder for record-keeping
    shutil.copyfile(workload_path, f"outputs/{experiment_id}/workload.onnx")
    shutil.copyfile(mapping_path, f"outputs/{experiment_id}/mapping/mapping.yaml")
    shutil.copyfile(accelerator, f"outputs/{experiment_id}/hardware/hardware.yaml")

    # Copy all referenced core files while maintaining directory structure
    copy_core_files(accelerator, f"outputs/{experiment_id}/hardware/")

    scme = optimize_allocation_co(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        experiment_id=experiment_id,
        output_path="outputs",
        skip_if_exists=False,
    )
    print(experiment_id, scme.latency, scme.energy)

    return scme, scme.latency, scme.energy


def main(args):
    model = GemmModel()
    # Export the model to ONNX

    dummy_input = torch.randn(1, 128, 4096)
    torch.onnx.export(model, dummy_input, args.output, opset_version=11)

    mode = "fused"
    layer_stacks = [tuple(range(0, 1))]
    workload_path = args.output

    id = "3"
    # Experiment 1
    accelerator = "inputs/big_gemm/hardware/tpu_like_quad_core.yaml"
    mapping_path = "inputs/big_gemm/mapping/tpu_like_quad_core.yaml"
    experiment_id_1 = f"big_gemm_test_{id}"

    scme1, scme1.latency, scme1.energy = run_stream(
        accelerator, workload_path, mapping_path, mode, layer_stacks, experiment_id_1
    )

    # Experiment 2
    accelerator = "inputs/small_gemm/hardware/tpu_like_quad_core.yaml"
    mapping_path = "inputs/small_gemm/mapping/tpu_like_quad_core.yaml"
    experiment_id_2 = f"small_gemm_test_{id}"
    scme2, scme2.latency, scme2.energy = run_stream(
        accelerator, workload_path, mapping_path, mode, layer_stacks, experiment_id_2
    )
    # Experiment 3
    accelerator = "inputs/small_gemm_full_connect/hardware/tpu_like_quad_core.yaml"
    mapping_path = "inputs/small_gemm_full_connect/mapping/tpu_like_quad_core.yaml"
    experiment_id_3 = f"small_gemm_full_connect_test_{id}"

    scme3, scme3.latency, scme3.energy = run_stream(
        accelerator, workload_path, mapping_path, mode, layer_stacks, experiment_id_3
    )

    print(f"Experiment {experiment_id_1} results:")
    print(f"Latency: {scme1.latency}, Energy: {scme1.energy}")
    print(f"Experiment {experiment_id_2} results:")
    print(f"Latency: {scme2.latency}, Energy: {scme2.energy}")
    print(f"Experiment {experiment_id_3} results:")
    print(f"Latency: {scme3.latency}, Energy: {scme3.energy}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
