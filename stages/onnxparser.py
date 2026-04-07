import logging
from typing import Any

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.mapping_factory import MappingFactory
from stream.parser.onnx.model import ONNXModelParser
from stream.stages.stage import Stage, StageCallable
from stream.workload.mapping import InterCoreMappingAttributes

logger = logging.getLogger(__name__)
def _get_default_core_allocation(accelerator: Accelerator) -> list[int]:
	"""Return all compute cores as default allocation (excluding offchip core)."""
	core_ids = sorted(core.id for core in accelerator.cores.node_list)
	if accelerator.offchip_core_id is not None:
		core_ids = [core_id for core_id in core_ids if core_id != accelerator.offchip_core_id]
	if not core_ids:
		raise ValueError("No compute cores found in accelerator to build default ONNX mapping.")
	return core_ids


def _build_default_all_mappings(accelerator: Accelerator) -> dict[str, InterCoreMappingAttributes]:
	"""Build default mapping data equivalent to a minimal mapping yaml file."""
	mapping_data: list[dict[str, Any]] = [
		{
			"name": "default",
			"core_allocation": _get_default_core_allocation(accelerator),
			"inter_core_tiling": [],
			"layer_dimension_names": [],
			"intra_core_tiling": [],
			"spatial_mapping": None,
		}
	]
	return MappingFactory(mapping_data).create()


class ONNXModelParserStageNoMapping(Stage):
	"""Parse ONNX into a workload without requiring an external mapping file."""

	def __init__(
		self,
		list_of_callables: list[StageCallable],
		*,
		workload_path: str,
		accelerator: Accelerator,
		**kwargs: Any,
	):
		super().__init__(list_of_callables, **kwargs)
		self.workload_path = workload_path
		self.accelerator = accelerator

	def run(self):
		all_mappings = _build_default_all_mappings(self.accelerator)

		onnx_model_parser = ONNXModelParser(self.workload_path, all_mappings, self.accelerator)
		onnx_model_parser.run()
		onnx_model = onnx_model_parser.onnx_model
		workload = onnx_model_parser.workload

		self.kwargs["accelerator"] = self.accelerator
		self.kwargs["all_mappings"] = all_mappings

		sub_stage = self.list_of_callables[0](
			self.list_of_callables[1:],
			onnx_model=onnx_model,
			workload=workload,
			**self.kwargs,
		)
		yield from sub_stage.run()

