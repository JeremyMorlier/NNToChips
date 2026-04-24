# Parametric MLP Layer with Llama2 Dimensions
# Llama2 uses a feed-forward network with two linear layers and SiLU activation

import shutil
import torch
import torch.nn as nn
import argparse
import yaml
from stream.api import optimize_allocation_co
from pathlib import Path
from onnx import shape_inference
import onnx
from onnxsim import simplify


def parse_args():
    parser = argparse.ArgumentParser(description="Generate parametric MLP layer")
    parser.add_argument(
        "--output", type=str, default="outputs/mlp_model/", help="Output directory"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=4096,
        help="Hidden dimension (default: Llama2 4096)",
    )
    parser.add_argument(
        "--intermediate_dim",
        type=int,
        default=11008,
        help="Intermediate dimension (default: Llama2 11008)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=32,
        help="Number of attention heads (for context)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sequence length (default: Llama2 2048)",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--export_onnx", action="store_true", help="Export to ONNX format"
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Path to hardware YAML file for stream evaluation",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="Path to mapping YAML file for stream evaluation",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run stream evaluation with hardware and mapping files",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default="mlp_eval",
        help="Experiment ID for stream evaluation",
    )
    return parser.parse_args()


class LlamaMLPLayer(nn.Module):
    """
    Parametric MLP Layer matching Llama2 architecture.

    The MLP in Llama2 consists of:
    - Linear projection from hidden_dim to intermediate_dim
    - SiLU activation
    - Linear projection from intermediate_dim back to hidden_dim

    This forms a feed-forward network (FFN) block.
    """

    def __init__(self, hidden_dim=4096, intermediate_dim=11008):
        """
        Args:
            hidden_dim: Dimension of the hidden state (default: 4096 for Llama2-7B)
            intermediate_dim: Dimension of the intermediate layer (default: 11008 for Llama2-7B)
        """
        super(LlamaMLPLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # Gate projection: hidden_dim -> intermediate_dim
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim)

        # Up projection: hidden_dim -> intermediate_dim
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim)

        # Down projection: intermediate_dim -> hidden_dim
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim)

        # SiLU activation function
        self.act_fn = nn.SiLU()

    def forward(self, x):
        """
        Forward pass implementing the Llama2 MLP:
        output = down_proj(act_fn(gate_proj(x)) * up_proj(x))

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Gate branch: project and apply activation
        gate = self.act_fn(self.gate_proj(x))

        # Up branch: project without activation
        up = self.up_proj(x)

        # Element-wise multiplication (gating)
        combined = gate * up

        # Down projection back to hidden dimension
        output = self.down_proj(combined)

        return output


class ParametricMLPModel(nn.Module):
    """
    Complete MLP model for ONNX export and evaluation.
    Can be used as a standalone workload or as part of a larger transformer.
    """

    def __init__(self, hidden_dim=4096, intermediate_dim=11008):
        super(ParametricMLPModel, self).__init__()
        self.mlp = LlamaMLPLayer(hidden_dim, intermediate_dim)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

    def forward(self, x):
        return self.mlp(x)


def export_to_onnx(model, output_dir, hidden_dim=4096, seq_len=2048, batch_size=1):
    """
    Export the MLP model to ONNX format.

    Args:
        model: PyTorch model to export
        output_dir: Directory to save ONNX file
        hidden_dim: Hidden dimension for dummy input
        seq_len: Sequence length for dummy input
        batch_size: Batch size for dummy input
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create dummy input matching Llama2 typical shape: (batch_size, seq_len, hidden_dim)
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim)

    # Export to ONNX
    onnx_path = Path(output_dir) / "mlp_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        verbose=False,
    )

    # Run shape inference for better optimization
    shape_inference.infer_shapes_path(str(onnx_path), str(onnx_path))

    # Load and simplify the ONNX model
    model_onnx = onnx.load(str(onnx_path))
    simplified_model, check = simplify(model_onnx)

    if not check:
        print("Warning: ONNX simplification check failed, but model saved anyway")
    onnx.save(simplified_model, str(onnx_path))

    return onnx_path


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
    hardware_path,
    workload_path,
    mapping_path,
    experiment_id="mlp_eval",
    mode="fused",
    layer_stacks=None,
):
    """
    Run stream optimization on the MLP model.

    Args:
        hardware_path: Path to hardware YAML file
        workload_path: Path to ONNX workload file
        mapping_path: Path to mapping YAML file
        experiment_id: Experiment ID for outputs
        mode: Optimization mode ('fused' or other)
        layer_stacks: Layer stack configuration

    Returns:
        SCME object with latency and energy results
    """
    # Create output directories
    Path(f"outputs/{experiment_id}").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/{experiment_id}/hardware/").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/{experiment_id}/mapping/").mkdir(parents=True, exist_ok=True)

    # Copy input files to output folder for record-keeping
    shutil.copyfile(workload_path, f"outputs/{experiment_id}/workload.onnx")
    shutil.copyfile(mapping_path, f"outputs/{experiment_id}/mapping/mapping.yaml")
    shutil.copyfile(hardware_path, f"outputs/{experiment_id}/hardware/hardware.yaml")

    # Copy all referenced core files while maintaining directory structure
    copy_core_files(hardware_path, f"outputs/{experiment_id}/hardware/")

    print(f"\nRunning stream optimization for experiment: {experiment_id}")
    print(f"Hardware: {hardware_path}")
    print(f"Workload: {workload_path}")
    print(f"Mapping: {mapping_path}")

    # Run stream optimization
    scme = optimize_allocation_co(
        hardware=hardware_path,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        experiment_id=experiment_id,
        output_path="outputs",
        skip_if_exists=False,
    )

    # Print results
    print(f"\n" + "=" * 60)
    print(f"Stream Evaluation Results for {experiment_id}")
    print("=" * 60)
    print(f"Latency: {scme.latency} cycles")
    print(f"Energy: {scme.energy} pJ")
    print("=" * 60 + "\n")

    return scme


def main():
    args = parse_args()

    # Create model
    model = ParametricMLPModel(
        hidden_dim=args.hidden_dim, intermediate_dim=args.intermediate_dim
    )
    test_input = torch.randn(args.batch_size, args.seq_len, args.hidden_dim)
    with torch.no_grad():
        output = model(test_input)

    # Export to ONNX if requested
    onnx_path = None
    if args.export_onnx:
        onnx_path = export_to_onnx(
            model,
            args.output,
            hidden_dim=args.hidden_dim,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )

    # Run stream evaluation if requested
    if args.evaluate:
        if not args.hardware or not args.mapping:
            print("Error: --hardware and --mapping are required for evaluation.")
            print(
                "Usage: python MLP_GA.py --export_onnx --evaluate --hardware <path> --mapping <path>"
            )
            return

        if not onnx_path:
            print("Error: ONNX export is required for stream evaluation.")
            print(
                "Usage: python MLP_GA.py --export_onnx --evaluate --hardware <path> --mapping <path>"
            )
            return

        # Run stream optimization
        scme = run_stream(
            hardware_path=args.hardware,
            workload_path=onnx_path,
            mapping_path=args.mapping,
            experiment_id=args.experiment_id,
            mode="fused",
            layer_stacks=[tuple(range(0, 9))],
        )


if __name__ == "__main__":
    main()
