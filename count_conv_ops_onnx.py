#!/usr/bin/env python3
"""Count convolution operations in an ONNX model.

Formula used per Conv node:
  MACs = N * Cout * (Π Oi) * (Cin/group) * (Π Ki)
where:
  - N is batch size
  - Cout is number of output channels
  - Oi are output spatial dimensions
  - Cin is input channels
  - group is the Conv group attribute
  - Ki are kernel spatial dimensions

If you want FLOPs (multiply + add), this script reports:
  FLOPs = 2 * MACs
(plus optional bias adds: + N * Cout * (Π Oi)).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import onnx
from onnx import shape_inference


def _get_group(node: onnx.NodeProto) -> int:
    for attr in node.attribute:
        if attr.name == "group":
            return int(attr.i)
    return 1


def _dims_from_vi(vi: onnx.ValueInfoProto) -> list[int] | None:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return None

    dims: list[int] = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            # Unknown symbolic/unspecified dim -> cannot compute exact ops.
            return None
    return dims


def _collect_initializer_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for init in model.graph.initializer:
        shapes[init.name] = [int(x) for x in init.dims]
    return shapes


def _collect_value_shapes(model: onnx.ModelProto) -> dict[str, list[int] | None]:
    shapes: dict[str, list[int] | None] = {}
    all_vis = (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    )
    for vi in all_vis:
        shapes[vi.name] = _dims_from_vi(vi)
    return shapes


def count_conv_ops(
    onnx_path: Path, include_bias_add: bool = False
) -> tuple[list[dict], int, int]:
    model = onnx.load(str(onnx_path))
    inferred = shape_inference.infer_shapes(model)

    init_shapes = _collect_initializer_shapes(inferred)
    value_shapes = _collect_value_shapes(inferred)

    per_layer: list[dict] = []
    total_macs = 0
    total_flops = 0

    for idx, node in enumerate(inferred.graph.node):
        if node.op_type != "Conv":
            continue

        node_name = node.name or f"Conv_{idx}"

        if len(node.input) < 2 or len(node.output) < 1:
            per_layer.append({"name": node_name, "error": "Invalid Conv node IO."})
            continue

        weight_name = node.input[1]
        output_name = node.output[0]

        w_shape = init_shapes.get(weight_name)
        o_shape = value_shapes.get(output_name)

        if w_shape is None:
            per_layer.append(
                {
                    "name": node_name,
                    "error": f"Weight tensor '{weight_name}' is not an initializer or shape is missing.",
                }
            )
            continue

        if o_shape is None:
            per_layer.append(
                {
                    "name": node_name,
                    "error": f"Output tensor '{output_name}' has unknown/incomplete shape.",
                }
            )
            continue

        if len(w_shape) < 3 or len(o_shape) < 3:
            per_layer.append(
                {
                    "name": node_name,
                    "error": f"Unexpected rank: weight={w_shape}, output={o_shape}",
                }
            )
            continue

        # ONNX Conv weight shape: [Cout, Cin/group, K1, K2, ...]
        n = o_shape[0]
        cout = o_shape[1]
        out_spatial = o_shape[2:]

        cin_per_group = w_shape[1]
        kernel_spatial = w_shape[2:]
        group = _get_group(node)

        if group <= 0:
            per_layer.append({"name": node_name, "error": f"Invalid group={group}"})
            continue

        out_elems = math.prod(out_spatial)
        kernel_elems = math.prod(kernel_spatial)

        macs = n * cout * out_elems * cin_per_group * kernel_elems
        bias_adds = (
            n * cout * out_elems if include_bias_add and len(node.input) >= 3 else 0
        )
        flops = 2 * macs + bias_adds

        per_layer.append(
            {
                "name": node_name,
                "type": node.op_type,
                "group": group,
                "weight_shape": w_shape,
                "output_shape": o_shape,
                "macs": macs,
                "flops": flops,
            }
        )

        total_macs += macs
        total_flops += flops

    return per_layer, total_macs, total_flops


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count Conv operations in an ONNX model"
    )
    parser.add_argument("onnx_file", type=Path, help="Path to ONNX file")
    parser.add_argument(
        "--include-bias-add",
        action="store_true",
        help="Include bias additions in FLOP count when Conv has bias input",
    )
    args = parser.parse_args()

    layers, total_macs, total_flops = count_conv_ops(
        args.onnx_file, include_bias_add=args.include_bias_add
    )

    print("Formula per Conv layer:")
    print("  MACs = N * Cout * (Π Oi) * (Cin/group) * (Π Ki)")
    print("  FLOPs = 2 * MACs" + (" + bias_adds" if args.include_bias_add else ""))
    print()

    if not layers:
        print("No Conv nodes found.")
        return

    print("Per-layer counts:")
    for layer in layers:
        if "error" in layer:
            print(f"- {layer['name']}: ERROR -> {layer['error']}")
            continue

        print(
            f"- {layer['name']}: "
            f"weight={layer['weight_shape']}, output={layer['output_shape']}, "
            f"group={layer['group']}, MACs={layer['macs']:,}, FLOPs={layer['flops']:,}"
        )

    print()
    print(f"Total Conv MACs : {total_macs:,}")
    print(f"Total Conv FLOPs: {total_flops:,}")


if __name__ == "__main__":
    main()
