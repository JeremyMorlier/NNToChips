from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import onnx
import pytest

from stages.optimized_tiled_workload_generation import (
    OptimizedTiledWorkloadGenerationStage,
)
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import MainStage, Stage
from stream.workload.computation.computation_node import ComputationNode


DEPFIN2_DIR = Path(__file__).parent.parent / "inputs" / "depfin2"
DEPFIN2_SOC_PATH = DEPFIN2_DIR / "hardware" / "soc.yaml"
DEPFIN2_WORKLOAD_PATH = DEPFIN2_DIR / "workload" / "fsrcnn.onnx"
DEPFIN2_MAPPING_PATH = DEPFIN2_DIR / "mapping" / "mapping.yaml"


class CaptureStage(Stage):
    def __init__(self, list_of_callables, **kwargs):
        super().__init__(list_of_callables, **kwargs)

    def is_leaf(self) -> bool:
        return True

    def run(self):
        yield (
            self.kwargs["workload"],
            {
                "original_workload": self.kwargs["original_workload"],
                "accelerator": self.kwargs["accelerator"],
            },
        )


def _build_single_stack_for_full_graph(workload_path: Path) -> list[tuple[int, ...]]:
    nb_layers = len(onnx.load(str(workload_path)).graph.node)
    if nb_layers <= 0:
        raise ValueError("Workload has no nodes to evaluate.")
    return [tuple(range(nb_layers))]


@pytest.mark.parametrize("optimization_method", ["ga", "ilp"])
def test_optimized_tiled_workload_generation_stage_on_fsrcnn(optimization_method: str):
    layer_stacks = _build_single_stack_for_full_graph(DEPFIN2_WORKLOAD_PATH)

    with TemporaryDirectory() as temp_output_root:
        tiled_workload_path = (
            Path(temp_output_root) / f"tiled_workload_{optimization_method}.pickle"
        )

        mainstage = MainStage(
            [
                AcceleratorParserStage,
                ONNXModelParserStage,
                LayerStacksGenerationStage,
                TilingGenerationStage,
                OptimizedTiledWorkloadGenerationStage,
                CaptureStage,
            ],
            accelerator=str(DEPFIN2_SOC_PATH),
            workload_path=str(DEPFIN2_WORKLOAD_PATH),
            mapping_path=str(DEPFIN2_MAPPING_PATH),
            mode="fused",
            layer_stacks=layer_stacks,
            tiled_workload_path=str(tiled_workload_path),
            optimization_method=optimization_method,
            max_inter_core_factor=4,
            max_intra_core_factor=2,
            tile_alloc_ga_generations=2,
            tile_alloc_ga_population=6,
            tile_alloc_random_seed=0,
        )
        answers = mainstage.run()

        non_empty_answers = [
            (workload, extra) for workload, extra in answers if workload is not None
        ]
        assert len(non_empty_answers) == 1

        tiled_workload, extra = non_empty_answers[0]
        original_workload = extra["original_workload"]

        assert tiled_workload is not None
        assert tiled_workload.number_of_nodes() > 0
        assert tiled_workload.number_of_edges() > 0
        assert tiled_workload_path.exists()

        computation_nodes = [
            node
            for node in original_workload.node_list
            if isinstance(node, ComputationNode)
        ]
        assert computation_nodes
        assert all(node.inter_core_tiling for node in computation_nodes)
        assert all(node.possible_core_allocation for node in computation_nodes)


def test_optimized_tiled_workload_generation_optimizes_single_core_nodes(monkeypatch):
    """A node with a single possible core allocation must still get tiling optimization."""
    import stages.optimized_tiled_workload_generation as optimized_stage_module

    class FakeComputationNode:
        def __init__(self):
            self.id = 7
            self.name = "fake_conv"
            self.type = "conv"
            self.layer_dim_sizes = {
                optimized_stage_module.LayerDim("K"): 8,
                optimized_stage_module.LayerDim("OY"): 4,
            }
            self.possible_core_allocation = [0]
            self.inter_core_tiling = []
            self.intra_core_tiling = []

        def __str__(self):
            return self.name

    class FakeAccelerator:
        def __init__(self):
            self.offchip_core_id = 99
            self.cores = SimpleNamespace(
                node_list=[SimpleNamespace(id=0), SimpleNamespace(id=1)]
            )

    class FakeWorkload:
        def __init__(self, node):
            self.node_list = [node]

    fake_node = FakeComputationNode()
    fake_workload = FakeWorkload(fake_node)
    fake_accelerator = FakeAccelerator()

    monkeypatch.setattr(optimized_stage_module, "ComputationNode", FakeComputationNode)

    stage = optimized_stage_module.OptimizedTiledWorkloadGenerationStage(
        [CaptureStage],
        workload=fake_workload,
        accelerator=fake_accelerator,
        tiled_workload_path="/tmp/fake_tiled_workload.pickle",
    )

    called = {"count": 0}

    def _fake_select_best_candidate(node):
        called["count"] += 1
        return optimized_stage_module._TilingCandidate(
            inter_tiling=((optimized_stage_module.LayerDim("K"), 2),),
            intra_tiling=((optimized_stage_module.LayerDim("OY"), 2),),
        )

    monkeypatch.setattr(stage, "_select_best_candidate", _fake_select_best_candidate)

    stage.optimize_layer_tilings()

    assert called["count"] == 1
    assert fake_node.inter_core_tiling == [(optimized_stage_module.LayerDim("K"), 2)]
    assert fake_node.intra_core_tiling == [(optimized_stage_module.LayerDim("OY"), 2)]
    assert fake_node.possible_core_allocation == [0]
