from types import SimpleNamespace

from stages.onnxparser import ONNXModelParserStageNoMapping
from stream.stages.stage import Stage


class FakeAccelerator:
    def __init__(self, core_ids: list[int], offchip_core_id: int | None = None):
        self.cores = SimpleNamespace(node_list=[SimpleNamespace(id=core_id) for core_id in core_ids])
        self.offchip_core_id = offchip_core_id

    def get_spatial_mapping_from_core(self, core_allocation):
        raise ValueError("No spatial mapping configured.")


class CaptureStage(Stage):
    def __init__(self, list_of_callables, **kwargs):
        super().__init__(list_of_callables, **kwargs)

    def is_leaf(self) -> bool:
        return True

    def run(self):
        yield self.kwargs["workload"], self.kwargs["onnx_model"]


def _run_parser(workload_path: str):
    stage = ONNXModelParserStageNoMapping(
        [CaptureStage],
        workload_path=workload_path,
        accelerator=FakeAccelerator([0, 1, 2, 3]),
    )
    return list(stage.run())[0]


def test_onnx_parser_no_mapping_parses_fsrcnn():
    workload, onnx_model = _run_parser("stream/stream/inputs/examples/workload/fsrcnn.onnx")

    assert onnx_model.graph.name
    assert workload.number_of_nodes() == 8
    assert workload.number_of_edges() == 7
    assert [node.name for node in workload.node_list[:3]] == [
        "custom_added_Conv1",
        "custom_added_Conv2",
        "custom_added_Conv3",
    ]
    assert [node.name for node in workload.node_list[-3:]] == [
        "custom_added_Conv6",
        "custom_added_Conv7",
        "custom_added_Conv8",
    ]


def test_onnx_parser_no_mapping_parses_resnet18():
    workload, onnx_model = _run_parser("stream/stream/inputs/examples/workload/resnet18.onnx")

    assert onnx_model.graph.name
    assert workload.number_of_nodes() == 49
    assert workload.number_of_edges() == 56
    assert [node.name for node in workload.node_list[:3]] == [
        "/conv1/Conv",
        "/relu/Relu",
        "/maxpool/MaxPool",
    ]
    assert [node.name for node in workload.node_list[-3:]] == [
        "/avgpool/GlobalAveragePool",
        "/Flatten",
        "/fc/Gemm",
    ]


def test_no_mapping_parser_equivalent_to_standard_parser_with_mapping():
	"""Verify that ONNXModelParserStageNoMapping produces deterministic workloads
	with the same structure regardless of whether an explicit mapping is used."""
	from stream.parser.mapping_factory import MappingFactory
	from stream.parser.onnx.model import ONNXModelParser

	workload_path = "stream/stream/inputs/examples/workload/fsrcnn.onnx"

	# Parse with no-mapping stage using auto-generated default mapping
	workload_nomapping, onnx_nomapping = _run_parser(workload_path)

	# Parse with direct ONNXModelParser using explicit default mapping
	accel = FakeAccelerator([0, 1, 2, 3])
	mapping = MappingFactory([{
		'name': 'default',
		'core_allocation': [0, 1, 2, 3],
		'inter_core_tiling': [],
		'layer_dimension_names': [],
		'intra_core_tiling': [],
		'spatial_mapping': None,
	}]).create()
	parser = ONNXModelParser(workload_path, mapping, accel)
	parser.run()
	workload_direct = parser.workload
	onnx_direct = parser.onnx_model

	# Verify both produce the same ONNX model
	assert onnx_direct.graph.name == onnx_nomapping.graph.name

	# Verify both produce equivalent workload structures
	assert workload_direct.number_of_nodes() == workload_nomapping.number_of_nodes()
	assert workload_direct.number_of_edges() == workload_nomapping.number_of_edges()

	# Verify node identities match
	direct_names = [n.name for n in workload_direct.node_list]
	nomapping_names = [n.name for n in workload_nomapping.node_list]
	assert direct_names == nomapping_names

	# Verify node types match
	direct_types = [n.type for n in workload_direct.node_list]
	nomapping_types = [n.type for n in workload_nomapping.node_list]
	assert direct_types == nomapping_types