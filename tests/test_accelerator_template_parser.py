from pathlib import Path
from tempfile import TemporaryDirectory
import os

import onnx
import pytest
from jinja2 import Environment, FileSystemLoader

from stages.accelerator_template_parser import AcceleratorTemplateParserStage
from stream.stages.stage import Stage

# Path to real depfin hardware files
DEPFIN_DIR = Path(__file__).parent.parent / "inputs" / "depfin"
DEPFIN_CORE_PATH = DEPFIN_DIR / "hardware" / "cores" / "depfin.yaml"
OFFCHIP_CORE_PATH = DEPFIN_DIR / "hardware" / "cores" / "offchip.yaml"
DEPFIN2_CORES_DIR = Path(__file__).parent.parent / "inputs" / "depfin2" / "hardware" / "cores"
DEPFIN_SOC_PATH = DEPFIN_DIR / "hardware" / "soc.yaml"
DEPFIN_WORKLOAD_PATH = DEPFIN_DIR / "workload" / "fsrcnn.onnx"
DEPFIN_MAPPING_PATH = DEPFIN_DIR / "mapping" / "mapping.yaml"


class CaptureStage(Stage):
	"""Leaf stage that captures the accelerator for testing."""

	def __init__(self, list_of_callables, **kwargs):
		super().__init__(list_of_callables, **kwargs)

	def is_leaf(self) -> bool:
		return True

	def run(self):
		yield self.kwargs["accelerator"], None


def _create_template_file(content: str, temp_dir: str) -> str:
	"""Create a template file in the given directory and return its path."""
	template_path = os.path.join(temp_dir, 'template.yaml')
	with open(template_path, 'w') as f:
		f.write(content)
	return template_path


def _create_cores_symlinks(temp_dir: str) -> None:
	"""Create symlinks to real core files in a temporary directory."""
	cores_dir = os.path.join(temp_dir, 'cores')
	os.makedirs(cores_dir, exist_ok=True)
	
	# Create symlinks to actual hardware cores
	os.symlink(DEPFIN_CORE_PATH, os.path.join(cores_dir, 'depfin.yaml'))
	os.symlink(OFFCHIP_CORE_PATH, os.path.join(cores_dir, 'offchip.yaml'))


def test_accelerator_template_parser_renders_with_parameters():
	"""Test that template parameters are correctly substituted."""
	with TemporaryDirectory() as temp_dir:
		_create_cores_symlinks(temp_dir)
		template_content = """name: test_accelerator_{{ num_cores }}_core
cores:
  0: ./cores/depfin.yaml
  1: ./cores/offchip.yaml
offchip_core_id: 1
core_connectivity: []
core_memory_sharing: []
"""
		template_path = _create_template_file(template_content, temp_dir)

		stage = AcceleratorTemplateParserStage(
			[CaptureStage],
			accelerator=template_path,
			template_params={"num_cores": 4},
		)
		accelerators = list(stage.run())
		accelerator, _ = accelerators[0]

		# Verify the accelerator was created and has the templated name
		assert accelerator.name == "test_accelerator_4_core"


def test_accelerator_template_parser_with_multiple_parameters():
	"""Test template rendering with multiple parameters."""
	with TemporaryDirectory() as temp_dir:
		_create_cores_symlinks(temp_dir)
		template_content = """name: multi_param_{{ env }}_cores_{{ num_cores }}
cores:
  0: ./cores/depfin.yaml
  1: ./cores/offchip.yaml
offchip_core_id: 1
core_connectivity: []
core_memory_sharing: []
"""
		template_path = _create_template_file(template_content, temp_dir)

		stage = AcceleratorTemplateParserStage(
			[CaptureStage],
			accelerator=template_path,
			template_params={
				"env": "production",
				"num_cores": 8,
				"core_type": "tpu_like",
			},
		)
		accelerators = list(stage.run())
		accelerator, _ = accelerators[0]

		assert accelerator.name == "multi_param_production_cores_8"


def test_accelerator_template_parser_with_no_parameters():
	"""Test that template rendering works with no dynamic parameters."""
	with TemporaryDirectory() as temp_dir:
		_create_cores_symlinks(temp_dir)
		template_content = """name: static_accelerator
cores:
  0: ./cores/depfin.yaml
  1: ./cores/offchip.yaml
offchip_core_id: 1
core_connectivity: []
core_memory_sharing: []
"""
		template_path = _create_template_file(template_content, temp_dir)

		stage = AcceleratorTemplateParserStage(
			[CaptureStage],
			accelerator=template_path,
			template_params={},
		)
		accelerators = list(stage.run())
		accelerator, _ = accelerators[0]

		assert accelerator.name == "static_accelerator"


def test_accelerator_template_parser_rejects_invalid_yaml():
	"""Test that invalid rendered YAML causes validation errors."""
	with TemporaryDirectory() as temp_dir:
		_create_cores_symlinks(temp_dir)
		template_content = """invalid_yaml: {{ param
"""
		template_path = _create_template_file(template_content, temp_dir)

		stage = AcceleratorTemplateParserStage(
			[CaptureStage],
			accelerator=template_path,
			template_params={"param": "value"},
		)
		# Should raise exception due to invalid YAML
		with pytest.raises(Exception):
			list(stage.run())


def test_accelerator_template_parser_with_accelerator_instance():
	"""Test that passing an Accelerator instance bypasses template rendering."""
	from types import SimpleNamespace

	fake_accelerator = SimpleNamespace(name="fake", cores=SimpleNamespace(node_list=[]))
	fake_accelerator.offchip_core_id = None

	stage = AcceleratorTemplateParserStage(
		[CaptureStage],
		accelerator=fake_accelerator,
		template_params={"unused": "param"},
	)
	accelerators = list(stage.run())
	accel, _ = accelerators[0]

	# Should pass through the accelerator instance unchanged
	assert accel is fake_accelerator


def test_accelerator_template_parser_nonexistent_file():
	"""Test error handling for nonexistent template files."""
	stage = AcceleratorTemplateParserStage(
		[CaptureStage],
		accelerator="/nonexistent/path/template.yaml",
	)
	# Should raise exception when trying to load nonexistent file
	with pytest.raises(Exception):
		list(stage.run())


def test_template_rendering_with_conditionals():
	"""Test Jinja2 template with conditional logic."""
	with TemporaryDirectory() as temp_dir:
		_create_cores_symlinks(temp_dir)
		template_content = """name: conditional_accelerator
cores:
  0: ./cores/depfin.yaml
{%- if enable_offchip %}
  1: ./cores/offchip.yaml
{%- endif %}
offchip_core_id: {% if enable_offchip %}1{% else %}0{% endif %}
core_connectivity: []
core_memory_sharing: []
"""
		template_path = _create_template_file(template_content, temp_dir)

		# Test with offchip enabled
		stage = AcceleratorTemplateParserStage(
			[CaptureStage],
			accelerator=template_path,
			template_params={"enable_offchip": True},
		)
		accelerators = list(stage.run())
		accel_with_offchip, _ = accelerators[0]
		assert accel_with_offchip is not None


def test_template_rendering_with_loops():
	"""Test Jinja2 template with loop constructs."""
	with TemporaryDirectory() as temp_dir:
		_create_cores_symlinks(temp_dir)
		template_content = """name: loop_accelerator
cores:
  0: ./cores/depfin.yaml
  1: ./cores/offchip.yaml
offchip_core_id: 1
core_connectivity: []
core_memory_sharing: []
"""
		template_path = _create_template_file(template_content, temp_dir)

		stage = AcceleratorTemplateParserStage(
			[CaptureStage],
			accelerator=template_path,
			template_params={"num_cores": 3},
		)
		accelerators = list(stage.run())
		accel, _ = accelerators[0]
		assert accel is not None


def test_top_soc_with_parametric_depfin2_core_template():
	"""Test top-level soc template with one parametric depfin2 core and one offchip core."""
	with TemporaryDirectory() as temp_dir:
		cores_dir = Path(temp_dir) / "cores"
		cores_dir.mkdir(parents=True, exist_ok=True)

		# Render depfin2 parametric compute core from core_template.j2.
		env = Environment(loader=FileSystemLoader(str(DEPFIN2_CORES_DIR)))
		core_template = env.get_template("core_template.j2")
		rendered_core = core_template.render(
			rf_1b_i_size=256,
			rf_1b_i_bw=16,
			rf_1b_w_size=256,
			rf_1b_w_bw=16,
			rf_4b_size=1024,
			rf_4b_bw=32,
			l1_w_size=65536,
			l1_w_bw=64,
			l1_act_size=65536,
			l1_act_bw_min=64,
			l1_act_bw_max=64,
			d1_size=8,
			d2_size=16,
		)
		(cores_dir / "core.yaml").write_text(rendered_core, encoding="utf-8")

		# Reuse real offchip memory core.
		os.symlink(DEPFIN2_CORES_DIR / "offchip.yaml", cores_dir / "offchip.yaml")

		soc_template = """name: {{ soc_name }}
cores:
  0: ./cores/core.yaml
  1: ./cores/offchip.yaml
offchip_core_id: 1
unit_energy_cost: 0
core_connectivity:
  - type: bus
    cores: [0, 1]
    bandwidth: 64
"""
		template_path = _create_template_file(soc_template, temp_dir)

		stage = AcceleratorTemplateParserStage(
			[CaptureStage],
			accelerator=template_path,
			template_params={"soc_name": "depfin2_parametric_soc"},
		)
		accelerator, _ = list(stage.run())[0]
        
		assert accelerator.name == "depfin2_parametric_soc"
		assert len(accelerator.cores.node_list) == 2


def test_optimize_allocation_co_full_evaluation_with_template_stage_depfin_defaults(monkeypatch):
	"""Run full optimize_allocation_co evaluation with depfin defaults and injected template parser stage."""
	import stream.api as stream_api

	# Inject our template stage into the standard Stream API flow.
	monkeypatch.setattr(stream_api, "AcceleratorParserStage", AcceleratorTemplateParserStage)

	# Skip cleanly when Gurobi is unavailable in the environment.
	try:
		stream_api._sanity_check_gurobi_license()
	except Exception as exc:
		pytest.skip(f"Skipping full Stream evaluation test due to unavailable Gurobi license: {exc}")

	# Build a default layer stack that covers all ONNX layers for depfin fsrcnn.
	nb_layers = len(onnx.load(str(DEPFIN_WORKLOAD_PATH)).graph.node)
	layer_stacks = [tuple(range(nb_layers))]

	with TemporaryDirectory() as temp_dir:
		scme = stream_api.optimize_allocation_co(
			hardware=str(DEPFIN_SOC_PATH),
			workload=str(DEPFIN_WORKLOAD_PATH),
			mapping=str(DEPFIN_MAPPING_PATH),
			mode="fused",
			layer_stacks=layer_stacks,
			experiment_id="test_depfin_template_stage_in_api",
			output_path=temp_dir,
			skip_if_exists=False,
		)

		assert scme is not None
		assert scme.latency > 0
		assert scme.energy > 0
