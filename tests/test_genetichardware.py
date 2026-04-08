from types import SimpleNamespace

import numpy as np

from stages.genetichardware import (
    GeneticHardwareStage,
    HardwareGAParameter,
    _HardwareTemplateProblem,
)
from stream.stages.stage import Stage


class RecorderLeafStage(Stage):
    """Leaf stage used to capture forwarded kwargs from parent stages."""

    last_kwargs = None

    def is_leaf(self) -> bool:
        return True

    def run(self):
        RecorderLeafStage.last_kwargs = self.kwargs
        yield self.kwargs.get("scme", SimpleNamespace()), None


class FakeEvalStage:
    """Minimal object exposing the methods expected by _HardwareTemplateProblem."""

    def _evaluate_candidate(self, params: dict[str, int]):
        x = params["p"]
        # Lower is better for all three objectives.
        return SimpleNamespace(
            energy=float(x),
            latency=float(x) + 1.0,
            accelerator=SimpleNamespace(area=float(x) + 2.0),
        )

    def _evaluate_candidates(self, params_list: list[dict[str, int]]):
        return [self._evaluate_candidate(params) for params in params_list]

    @staticmethod
    def _score_objectives(objectives: list[float]) -> float:
        return float(sum(objectives))


def test_parse_parameter_specs():
    raw_specs = [
        {"name": "d1_size", "lower": 8, "upper": 128, "scale": 2},
        {"name": "d2_size", "lower": 4, "upper": 64},
    ]

    specs = GeneticHardwareStage._parse_parameter_specs(raw_specs)

    assert len(specs) == 2
    assert specs[0] == HardwareGAParameter(name="d1_size", lower=8, upper=128, scale=2)
    assert specs[1] == HardwareGAParameter(name="d2_size", lower=4, upper=64, scale=1)


def test_run_passthrough_when_ga_not_configured():
    original_accelerator = SimpleNamespace(name="orig_accel")
    workload = object()

    stage = GeneticHardwareStage(
        [RecorderLeafStage],
        accelerator=original_accelerator,
        workload=workload,
        # No hardware_template and no hardware_ga_parameter_specs => pass-through mode
    )

    results = list(stage.run())

    assert len(results) == 1
    assert RecorderLeafStage.last_kwargs["accelerator"] is original_accelerator
    assert RecorderLeafStage.last_kwargs["workload"] is workload


def test_override_candidate_paths():
    kwargs = {
        "tiled_workload_path": "/tmp/original/tiled_workload.pickle",
        "cost_lut_path": "/tmp/original/cost_lut.pickle",
        "allocations_path": "/tmp/original/waco/",
        "tiled_workload_post_co_path": "/tmp/original/tiled_workload_post_co.pickle",
        "cost_lut_post_co_path": "/tmp/original/cost_lut_post_co.pickle",
    }

    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        GeneticHardwareStage._override_candidate_paths(kwargs, Path(temp_dir))

        assert kwargs["tiled_workload_path"].startswith(temp_dir)
        assert kwargs["cost_lut_path"].startswith(temp_dir)
        assert kwargs["allocations_path"].startswith(temp_dir)
        assert kwargs["allocations_path"].endswith("/")
        assert kwargs["tiled_workload_post_co_path"].startswith(temp_dir)
        assert kwargs["cost_lut_post_co_path"].startswith(temp_dir)


def test_hardware_template_problem_tracks_best_candidate():
    parameter_specs = [HardwareGAParameter(name="p", lower=1, upper=5, scale=1)]
    problem = _HardwareTemplateProblem(FakeEvalStage(), parameter_specs)

    out = {}
    problem._evaluate(np.array([[4], [2], [1]], dtype=int), out)

    assert out["F"].shape == (3, 3)
    assert problem.best_params == {"p": 1}


def test_run_ga_mode_forwards_best_params(monkeypatch):
    original_accelerator = SimpleNamespace(name="seed_accel")
    workload = object()
    best_accelerator = SimpleNamespace(name="best_accel")

    stage = GeneticHardwareStage(
        [RecorderLeafStage],
        accelerator=original_accelerator,
        workload=workload,
        hardware_template="/tmp/fake_template.yaml",
        hardware_ga_core_template="/tmp/fake_core_template.yaml",
        hardware_ga_parameter_specs=[{"name": "d1_size", "lower": 1, "upper": 4}],
        hardware_ga_population=4,
        hardware_ga_generations=2,
        hardware_ga_seed=7,
    )

    # Avoid running real optimization; directly set a best candidate.
    def fake_minimize(problem, algorithm, termination, seed, verbose):
        problem.best_params = {"d1_size": 3}
        return SimpleNamespace()

    monkeypatch.setattr("stages.genetichardware.minimize", fake_minimize)
    monkeypatch.setattr(stage, "_build_accelerator_from_templates", lambda params: best_accelerator)

    results = list(stage.run())

    assert len(results) == 1
    assert RecorderLeafStage.last_kwargs["accelerator"] is best_accelerator
    assert RecorderLeafStage.last_kwargs["selected_hardware_params"] == {"d1_size": 3}
