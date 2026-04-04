import logging as _logging
import re

from stream.api import optimize_allocation_co
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json

_logging_level = _logging.INFO
_logging_format = (
    "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)

############################################INPUTS############################################
accelerator = "inputs/depfin/hardware/soc.yaml"
accelerator = "inputs/depfin2/hardware/soc.yaml"
workload_path = "inputs/depfin2/workload/fsrcnn.onnx"
mapping_path = "inputs/depfin2/mapping/mapping.yaml"
mode = "fused"
layer_stacks = [tuple(range(0, 12)), tuple(range(12, 22))] + list(
    (i,) for i in range(22, 49)
)
##############################################################################################

################################PARSING###############################
experiment_id = "depfin-eval"
######################################################################

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

############PLOTTING#############
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#################################

#########################PLOTTING PATHS##############################
timeline_fig_path_plotly = f"outputs/{experiment_id}/schedule.html"
memory_fig_path = f"outputs/{experiment_id}/memory.png"
json_path = f"outputs/{experiment_id}/scme.json"
#####################################################################

#####################CostModelEvaluationLUT LOAD#############################
cost_lut_path = f"outputs/{experiment_id}/cost_lut_post_co.pickle"
cost_lut = CostModelEvaluationLUT(cost_lut_path)
#############################################################################

# Plotting memory usage of best SCME
plot_memory_usage(
    scme, section_start_percent, percent_shown, fig_path=memory_fig_path, show_dram=True
)

# Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)
print(scme)
print(scme.latency, scme.energy, scme.accelerator.area)
print(
    list(scme.accelerator.get_core(0).memory_hierarchy._node.keys())[
        0
    ].memory_instance.__dict__
)
print(
    list(scme.accelerator.get_core(0).memory_hierarchy._node.keys())[
        1
    ].memory_instance.__dict__
)
print(
    list(scme.accelerator.get_core(0).memory_hierarchy._node.keys())[
        2
    ].memory_instance.__dict__
)
print(
    list(scme.accelerator.get_core(0).memory_hierarchy._node.keys())[
        3
    ].memory_instance.__dict__
)
print(
    list(scme.accelerator.get_core(0).memory_hierarchy._node.keys())[
        4
    ].memory_instance.__dict__
)
