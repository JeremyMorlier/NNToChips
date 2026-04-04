import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute an overall Pareto front from GA population evaluations only "
            "(uses all_evaluations, ignores stored pareto_front fields)."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="outputs/depfin2-ga",
        help="Directory containing run_* folders with ga_summary.json files.",
    )
    parser.add_argument(
        "--summary-glob",
        type=str,
        default="run_*/ga_summary.json",
        help="Glob pattern, relative to --experiment-dir, to discover summary files.",
    )
    parser.add_argument(
        "--penalty-threshold",
        type=float,
        default=1e20,
        help="Discard evaluations with any objective >= this threshold.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="overall_pareto_from_populations.json",
        help="Output JSON filename written under --experiment-dir.",
    )
    return parser.parse_args()


def dominates(a, b):
    """True if objective vector a dominates b (minimization)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def find_nondominated(records):
    n = len(records)
    is_dominated = [False] * n

    for i in range(n):
        if is_dominated[i]:
            continue
        fi = records[i]["F"]
        for j in range(n):
            if i == j or is_dominated[i]:
                continue
            fj = records[j]["F"]
            if dominates(fj, fi):
                is_dominated[i] = True

    return [records[i] for i in range(n) if not is_dominated[i]]


def load_population_evaluations(summary_path, penalty_threshold):
    data = json.loads(summary_path.read_text())

    run_uid = data.get("run_uid", summary_path.parent.name)
    all_evaluations = data.get("all_evaluations", [])

    extracted = []
    for idx, ev in enumerate(all_evaluations):
        f = ev.get("F")
        if not isinstance(f, list) or len(f) < 3:
            continue

        f3 = [float(f[0]), float(f[1]), float(f[2])]
        if any(v >= penalty_threshold for v in f3):
            continue

        extracted.append(
            {
                "source_summary": str(summary_path),
                "run_uid": run_uid,
                "evaluation_index": idx,
                "eval_id": ev.get("eval_id"),
                "x": ev.get("x"),
                "params": ev.get("params"),
                "F": f3,
                "energy": f3[0],
                "latency": f3[1],
                "area": f3[2],
            }
        )

    return extracted


def main():
    args = parse_args()

    experiment_dir = Path(args.experiment_dir)
    summary_paths = sorted(experiment_dir.glob(args.summary_glob))
    if not summary_paths:
        raise FileNotFoundError(
            f"No GA summary files found in {experiment_dir} with pattern {args.summary_glob}."
        )

    all_records = []
    for summary_path in summary_paths:
        all_records.extend(
            load_population_evaluations(summary_path, args.penalty_threshold)
        )

    if not all_records:
        raise RuntimeError("No valid evaluations found in all_evaluations entries.")

    pareto_records = find_nondominated(all_records)
    pareto_records.sort(key=lambda r: (r["energy"], r["latency"], r["area"]))

    output = {
        "experiment_dir": str(experiment_dir),
        "summary_pattern": args.summary_glob,
        "num_summary_files": len(summary_paths),
        "num_population_evaluations": len(all_records),
        "num_pareto_points": len(pareto_records),
        "note": "Pareto computed from all_evaluations only; stored pareto_front fields are ignored.",
        "pareto_front": pareto_records,
    }

    output_path = experiment_dir / args.output
    output_path.write_text(json.dumps(output, indent=2))

    print(f"Read summaries: {len(summary_paths)}")
    print(f"Valid population evaluations: {len(all_records)}")
    print(f"Overall Pareto points: {len(pareto_records)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
