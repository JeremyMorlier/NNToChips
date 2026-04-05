import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined


def mkdir_recursive(directory: str) -> None:
    """
    Recursively create directories if they don't exist.

    Args:
        directory: Path to the directory to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def ensure_dir_exists(file_path: str) -> None:
    """
    Create parent directories for a file path if they don't exist.

    Args:
        file_path: Path to the file
    """
    directory = os.path.dirname(file_path)
    if directory:
        mkdir_recursive(directory)


def generate_run_id() -> str:
    """Generate a compact run identifier in format YYYYMMDD_HHMM."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def render_template_to_file(template_path: Path | str, output_path: Path | str, context: dict):
    """Render a Jinja2 template to a file and return the output path."""
    template_path = Path(template_path)
    output_path = Path(output_path)

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)
    rendered = template.render(**context)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return str(output_path)


def _flatten_constraint_values(values) -> list[float]:
    array_values = np.asarray(values, dtype=np.float64)
    return array_values.reshape(-1).astype(float).tolist()


def _constraint_violation(values) -> tuple[float, list[float]]:
    flattened = _flatten_constraint_values(values)
    return sum(max(0.0, float(value)) for value in flattened), flattened


def save_pareto_results_csv(
    filepath,
    pareto_x,
    pareto_f,
    pareto_g,
    decision_headers,
    constraint_headers,
    objective_headers=("latency", "energy"),
):
    """Save a Pareto front to CSV using a consistent layout."""
    pareto_x = np.asarray(pareto_x)
    pareto_f = np.asarray(pareto_f)
    pareto_g = None if pareto_g is None else np.asarray(pareto_g)

    if pareto_x.ndim == 1:
        pareto_x = pareto_x.reshape(1, -1)
    if pareto_f.ndim == 1:
        pareto_f = pareto_f.reshape(1, -1)
    if pareto_g is not None and pareto_g.ndim == 1:
        pareto_g = pareto_g.reshape(1, -1)

    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "pareto_rank",
                *decision_headers,
                *objective_headers,
                *constraint_headers,
                "cv",
                "feasible",
            ]
        )

        for index, (ind_x, ind_f) in enumerate(zip(pareto_x, pareto_f, strict=False)):
            ind_f_flat = np.asarray(ind_f).reshape(-1)
            objective_values = [float(value) for value in ind_f_flat[: len(objective_headers)]]

            if pareto_g is not None:
                ind_g = pareto_g[index]
                cv, constraint_values = _constraint_violation(ind_g)
            else:
                cv = 0.0
                constraint_values = [0.0] * len(constraint_headers)

            feasible = int(cv <= 1e-6)
            writer.writerow(
                [
                    index,
                    *np.asarray(ind_x).reshape(-1).tolist(),
                    *objective_values,
                    *constraint_values,
                    cv,
                    feasible,
                ]
            )


def save_history_csv(
    filepath,
    history,
    decision_headers,
    constraint_headers,
    decoder=None,
    objective_headers=("latency", "energy"),
):
    """Save the full GA history to CSV using a consistent layout."""
    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "generation",
                "individual",
                *decision_headers,
                *objective_headers,
                *constraint_headers,
                "cv",
                "feasible",
            ]
        )

        for generation_index, run in enumerate(history):
            for individual_index, individual in enumerate(run.pop):
                decoded_x = decoder(individual.X) if decoder is not None else individual.X
                ind_x = np.asarray(decoded_x).reshape(-1).tolist()
                ind_f = np.asarray(individual.F).reshape(-1)
                ind_g = getattr(individual, "G", None)

                if ind_g is not None:
                    cv, constraint_values = _constraint_violation(ind_g)
                else:
                    cv = 0.0
                    constraint_values = [0.0] * len(constraint_headers)

                feasible = int(cv <= 1e-6)
                writer.writerow(
                    [
                        generation_index,
                        individual_index,
                        *ind_x,
                        *[float(value) for value in ind_f[: len(objective_headers)]],
                        *constraint_values,
                        cv,
                        feasible,
                    ]
                )
