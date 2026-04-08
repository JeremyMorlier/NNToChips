from pathlib import Path
from tempfile import TemporaryDirectory
import onnx
import pytest

from stages.accelerator_template_parser import AcceleratorTemplateParserStage


DEPFIN2_DIR = Path(__file__).parent.parent / "inputs" / "depfin2"
DEPFIN2_SOC_PATH = DEPFIN2_DIR / "hardware" / "soc.yaml"
DEPFIN2_CORE_TEMPLATE_PATH = DEPFIN2_DIR / "hardware" / "cores" / "core_template.j2"
DEPFIN2_WORKLOAD_PATH = DEPFIN2_DIR / "workload" / "fsrcnn.onnx"
DEPFIN2_MAPPING_PATH = DEPFIN2_DIR / "mapping" / "mapping.yaml"


def _build_single_stack_for_full_graph(workload_path: Path) -> list[tuple[int, ...]]:
    """Build one layer stack that spans the full ONNX graph."""
    nb_layers = len(onnx.load(str(workload_path)).graph.node)
    if nb_layers <= 0:
        raise ValueError("Workload has no nodes to evaluate.")
    return [tuple(range(nb_layers))]


def test_depfin2_template_parser_plugged_into_optimize_allocation_co(monkeypatch):
    """Run full Stream CO flow for depfin2 while using AcceleratorTemplateParserStage."""
    import stream.api as stream_api

    # Plug the custom template accelerator parser stage into the standard API pipeline.
    monkeypatch.setattr(stream_api, "AcceleratorParserStage", AcceleratorTemplateParserStage)

    # This is a full evaluation test; skip if the environment cannot run Gurobi.
    try:
        stream_api._sanity_check_gurobi_license()
    except Exception as exc:
        pytest.skip(f"Skipping full Stream integration test (Gurobi unavailable): {exc}")

    layer_stacks = _build_single_stack_for_full_graph(DEPFIN2_WORKLOAD_PATH)

    with TemporaryDirectory() as temp_output_root:
        experiment_id = "test_depfin2_template_parser_integration"
        scme = stream_api.optimize_allocation_co(
            hardware=str(DEPFIN2_SOC_PATH),
            workload=str(DEPFIN2_WORKLOAD_PATH),
            mapping=str(DEPFIN2_MAPPING_PATH),
            mode="fused",
            layer_stacks=layer_stacks,
            experiment_id=experiment_id,
            output_path=temp_output_root,
            skip_if_exists=False,
        )

        assert scme is not None
        assert scme.latency > 0
        assert scme.energy > 0
        assert scme.accelerator is not None

        run_dir = Path(temp_output_root) / experiment_id
        assert (run_dir / "scme.pickle").exists()
        assert (run_dir / "cost_lut_post_co.pickle").exists()


def test_optimize_single_hardware_co_full_with_genetic_stage():
    """Run full optimize_single_hardware_co including GeneticHardwareStage on depfin2 inputs."""
    import main as main_api

    # Full run requires a valid Gurobi license.
    try:
        main_api._sanity_check_gurobi_license()
    except Exception as exc:
        pytest.skip(f"Skipping full optimize_single_hardware_co GA test (Gurobi unavailable): {exc}")

    layer_stacks = _build_single_stack_for_full_graph(DEPFIN2_WORKLOAD_PATH)

    with TemporaryDirectory() as temp_output_root:
        result = main_api.optimize_single_hardware_co(
            hardware_template=str(DEPFIN2_SOC_PATH),
            workload=str(DEPFIN2_WORKLOAD_PATH),
            mode="fused",
            layer_stacks=layer_stacks,
            experiment_id="test_optimize_single_hardware_co_pipeline",
            output_path=temp_output_root,
            skip_if_exists=False,
            # Keep GA tiny to make integration test runtime practical.
            hardware_ga_parameter_specs=[
                {"name": "d1_size", "lower": 64, "upper": 128, "scale": 1},
            ],
            hardware_ga_core_template=str(DEPFIN2_CORE_TEMPLATE_PATH),
            hardware_ga_core_template_params={
                "rf_1b_i_size": 8,
                "rf_1b_i_bw": 8,
                "rf_1b_w_size": 8,
                "rf_1b_w_bw": 8,
                "rf_4b_size": 16,
                "rf_4b_bw": 16,
                "l1_w_size": 4_194_304,
                "l1_w_bw": 128,
                "l1_act_size": 8_388_608,
                "l1_act_bw_min": 64,
                "l1_act_bw_max": 1024,
                "d1_size": 128,
                "d2_size": 16,
            },
            hardware_ga_target_core_id=0,
            hardware_ga_generations=1,
            hardware_ga_population=2,
            hardware_ga_seed=0,
            template_params={},
        )

    assert result is not None
    assert result.latency > 0
    assert result.energy > 0
    assert result.accelerator is not None
