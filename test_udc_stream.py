import logging as _logging
import re
from pathlib import Path

import yaml

from stream.api import optimize_allocation_co
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json

_logging_level = _logging.INFO
_logging_format = (
    "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)


def update_rf_memory_sizes(
    core_yaml_path, rf_i1_size=None, rf_i2_size=None, rf_o_size=None
):
    """Update RF memory sizes in a UDC core YAML file."""
    with open(core_yaml_path, "r", encoding="utf-8") as f:
        core_yaml = yaml.safe_load(f)

    memories = core_yaml.get("memories", {})

    if rf_i1_size is not None and "rf_I1" in memories:
        memories["rf_I1"]["size"] = int(rf_i1_size)
    if rf_i2_size is not None and "rf_I2" in memories:
        memories["rf_I2"]["size"] = int(rf_i2_size)
    if rf_o_size is not None and "rf_O" in memories:
        memories["rf_O"]["size"] = int(rf_o_size)

    with open(core_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(core_yaml, f, sort_keys=False)


############################################INPUTS############################################
accelerator = "inputs/test_udc/hardware/core.yaml"
workload_path = "inputs/test_udc/model.onnx"
mapping_path = "inputs/test_udc/mapping/mapping.yaml"
mode = "lbl"
# Set to an integer to update RF sizes before running, or keep as None to preserve existing sizes.
rf_i1_size = None
rf_i2_size = None
rf_o_size = None
layer_stacks = [tuple(range(0, 12)), tuple(range(12, 22))] + list(
    (i,) for i in range(22, 49)
)
##############################################################################################

#############################OPTIONAL RF SIZE UPDATE##########################################
if any(v is not None for v in (rf_i1_size, rf_i2_size, rf_o_size)):
    core_yaml_path = Path(accelerator).parent / "cores" / "udc_core.yaml"
    update_rf_memory_sizes(
        core_yaml_path,
        rf_i1_size=rf_i1_size,
        rf_i2_size=rf_i2_size,
        rf_o_size=rf_o_size,
    )
##############################################################################################

################################PARSING###############################
hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = f"{hw_name}-{wl_name}-{mode}-constraint_optimization"
######################################################################

scme = optimize_allocation_co(
    hardware=accelerator,
    workload=workload_path,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=layer_stacks,
    experiment_id=experiment_id,
    output_path="outputs",
    skip_if_exists=True,
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
print(scme.latency, scme.energy)
