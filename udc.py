"""
Universal Dataflow Accelerator Designer
"""

import yaml


def array_to_hardware(array_desc: list, name: str):
    """Convert an array to a Stream/Zigzag hardware description object."""

    (
        d1,
        d2,
        sram_size,
        sram_min_b,
        sram_max_b,
        d2_sram_size,
        d2_sram_min_b,
        d2_sram_max_b,
        d1_sram_size,
        d1_sram_min_b,
        d1_sram_max_b,
        rf_I1_size,
        rf_I1_bw,
        rf_I2_size,
        rf_I2_bw,
        rf_O_size,
        rf_O_bw,
    ) = array_desc
    dict = {
        "name": name,
        "type": "compute",
        "memories": {
            "rf_O": {
                "size": rf_O_size,
                "r_cost": 0.021,
                "w_cost": 0.021,
                # "auto_cost_extraction": True,
                "area": 0,
                "latency": 1,
                "operands": ["O"],
                "served_dimensions": [],
                "ports": [
                    {
                        "name": "r_port_1",
                        "type": "read",
                        "bandwidth_min": rf_O_bw,
                        "bandwidth_max": rf_O_bw,
                        "allocation": ["O, tl"],
                    },
                    {
                        "name": "r_port_2",
                        "type": "read",
                        "bandwidth_min": rf_O_bw,
                        "bandwidth_max": rf_O_bw,
                        "allocation": ["O, th"],
                    },
                    {
                        "name": "w_port_1",
                        "type": "write",
                        "bandwidth_min": rf_O_bw,
                        "bandwidth_max": rf_O_bw,
                        "allocation": ["O, fh"],
                    },
                    {
                        "name": "w_port_2",
                        "type": "write",
                        "bandwidth_min": rf_O_bw,
                        "bandwidth_max": rf_O_bw,
                        "allocation": ["O, fl"],
                    },
                ],
            },
            "rf_I1": {
                "size": rf_I1_size,
                "r_cost": 0.095,
                "w_cost": 0.095,
                # "auto_cost_extraction": True,
                "area": 0,
                "latency": 1,
                "operands": ["I1"],
                "served_dimensions": [],
                "ports": [
                    {
                        "name": "r_port_1",
                        "type": "read",
                        "bandwidth_min": rf_I1_bw,
                        "bandwidth_max": rf_I1_bw,
                        "allocation": ["I1, tl"],
                    },
                    {
                        "name": "w_port_1",
                        "type": "write",
                        "bandwidth_min": rf_I1_bw,
                        "bandwidth_max": rf_I1_bw,
                        "allocation": ["I1, fh"],
                    },
                ],
            },
            "rf_I2": {
                "size": rf_I2_size,
                "r_cost": 0.095,
                "w_cost": 0.095,
                # "auto_cost_extraction": True,
                "area": 0,
                "latency": 1,
                "operands": ["I2"],
                "served_dimensions": [],
                "ports": [
                    {
                        "name": "r_port_1",
                        "type": "read",
                        "bandwidth_min": rf_I2_bw,
                        "bandwidth_max": rf_I2_bw,
                        "allocation": ["I2, tl"],
                    },
                    {
                        "name": "w_port_1",
                        "type": "write",
                        "bandwidth_min": rf_I2_bw,
                        "bandwidth_max": rf_I2_bw,
                        "allocation": ["I2, fh"],
                    },
                ],
            },
            "d1_sram": {
                "size": d1_sram_size,
                "r_cost": 0,
                "w_cost": 0,
                # "auto_cost_extraction": True,
                "area": 0,
                "latency": 1,
                "operands": ["I1", "I2", "O"],
                "served_dimensions": ["D1"],
                "ports": [
                    {
                        "name": "r_port_1",
                        "type": "read",
                        "bandwidth_min": d1_sram_min_b,
                        "bandwidth_max": d1_sram_max_b,
                        "allocation": [
                            "I1, tl",
                            "I2, tl",
                            "O, tl",
                            "O, th",
                        ],
                    },
                    {
                        "name": "w_port_1",
                        "type": "write",
                        "bandwidth_min": d1_sram_min_b,
                        "bandwidth_max": d1_sram_max_b,
                        "allocation": [
                            "I1, fh",
                            "I2, fh",
                            "O, fh",
                            "O, fl",
                        ],
                    },
                ],
            },
            "d2_sram": {
                "size": d2_sram_size,
                "r_cost": 0,
                "w_cost": 0,
                # "auto_cost_extraction": True,
                "area": 0,
                "latency": 1,
                "operands": ["I1", "I2", "O"],
                "served_dimensions": ["D2"],
                "ports": [
                    {
                        "name": "r_port_1",
                        "type": "read",
                        "bandwidth_min": d2_sram_min_b,
                        "bandwidth_max": d2_sram_max_b,
                        "allocation": [
                            "I1, tl",
                            "I2, tl",
                            "O, tl",
                            "O, th",
                        ],
                    },
                    {
                        "name": "w_port_1",
                        "type": "write",
                        "bandwidth_min": d2_sram_min_b,
                        "bandwidth_max": d2_sram_max_b,
                        "allocation": [
                            "I1, fh",
                            "I2, fh",
                            "O, fh",
                            "O, fl",
                        ],
                    },
                ],
            },
            "sram": {
                "size": sram_size,
                "r_cost": 0,
                "w_cost": 0,
                # "auto_cost_extraction": True,
                "area": 0,
                "latency": 1,
                "operands": ["I1", "I2", "O"],
                "served_dimensions": ["D1", "D2"],
                "ports": [
                    {
                        "name": "r_port_1",
                        "type": "read",
                        "bandwidth_min": sram_min_b,
                        "bandwidth_max": sram_max_b,
                        "allocation": [
                            "I1, tl",
                            "I2, tl",
                            "O, tl",
                            "O, th",
                        ],
                    },
                    {
                        "name": "w_port_1",
                        "type": "write",
                        "bandwidth_min": sram_min_b,
                        "bandwidth_max": sram_max_b,
                        "allocation": [
                            "I1, fh",
                            "I2, fh",
                            "O, fh",
                            "O, fl",
                        ],
                    },
                ],
            },
        },
        "operational_array": {
            "unit_energy": 0.04,  # pJ
            "unit_area": 1,  # unit
            "dimensions": ["D1", "D2"],
            "sizes": [d1, d2],
        },
    }

    # to yaml
    yaml_str = yaml.dump(dict, sort_keys=False)
    return yaml_str
