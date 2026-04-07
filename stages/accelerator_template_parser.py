import logging
from typing import Any

from jinja2 import Environment, FileSystemLoader

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class AcceleratorTemplateParserStage(Stage):
	"""Parse an accelerator from a Jinja2-templated YAML file with tunable parameters.
	
	Allows reading a template file with placeholder parameters that can be substituted
	during the pipeline execution, enabling dynamic hardware configuration without
	needing separate YAML files for each parameter combination.
	"""

	def __init__(
		self,
		list_of_callables: list[StageCallable],
		*,
		accelerator: str | Accelerator,
		template_params: dict[str, Any] | None = None,
		**kwargs: Any,
	):
		"""
		Args:
			list_of_callables: Subsequent stages in the pipeline.
			accelerator: Path to template YAML file or Accelerator instance.
					   If string, must end with .yaml and will be treated as template.
			template_params: Dictionary of parameters to substitute in the template.
					   Only used if accelerator is a string path.
			**kwargs: Passed to parent Stage.
		"""
		super().__init__(list_of_callables, **kwargs)
		self.accelerator = accelerator
		self.template_params = template_params or {}

	def run(self):
		if isinstance(self.accelerator, Accelerator):
			accelerator = self.accelerator
		else:
			if isinstance(self.accelerator, str):
				assert self.accelerator.split(".")[-1] == "yaml", "Expected a yaml file as accelerator input"
				accelerator = self.parse_accelerator_from_template(self.accelerator, self.template_params)
			else:
				# Assume it's an Accelerator-compatible object (e.g., from a test mock)
				accelerator = self.accelerator

		sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, **self.kwargs)
		yield from sub_stage.run()

	def parse_accelerator_from_template(
		self, template_path: str, params: dict[str, Any]
	) -> Accelerator:
		"""Load and render a Jinja2 template, then parse the result as accelerator config.
		
		Args:
			template_path: Path to the template YAML file.
			params: Dictionary of parameters to provide to the template.
		
		Returns:
			Accelerator instance created from the rendered template.
		"""
		# Render template
		rendered_yaml = self._render_template(template_path, params)

		# Parse YAML
		accelerator_data = self._parse_yaml_from_string(rendered_yaml)

		# Validate
		validator = AcceleratorValidator(accelerator_data, template_path)
		accelerator_data = validator.normalized_data
		validate_success = validator.validate()
		if not validate_success:
			raise ValueError("Failed to validate accelerator from rendered template.")

		# Create and return
		factory = AcceleratorFactory(accelerator_data)
		return factory.create()

	@staticmethod
	def _render_template(template_path: str, params: dict[str, Any]) -> str:
		"""Render a Jinja2 template file with given parameters.
		
		Args:
			template_path: Path to template file relative to current working directory.
			params: Parameters to pass to Jinja2 for template rendering.
		
		Returns:
			Rendered template content as string.
		"""
		from pathlib import Path

		template_file = Path(template_path)
		template_dir = template_file.parent
		template_name = template_file.name

		env = Environment(loader=FileSystemLoader(str(template_dir)))
		template = env.get_template(template_name)
		return template.render(**params)

	@staticmethod
	def _parse_yaml_from_string(yaml_string: str) -> Any:
		"""Parse YAML from a string (as opposed to a file).
		
		Args:
			yaml_string: YAML content as string.
		
		Returns:
			Parsed YAML data structure.
		"""
		import yaml

		return yaml.safe_load(yaml_string)
