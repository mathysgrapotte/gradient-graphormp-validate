#!/usr/bin/env python3
"""Score a drug-discovery submission with the official Polaris grader.

Vendored verbatim from BioML-bench's per-task ``grade.py`` (Polaris tasks):
it re-loads the benchmark and calls ``benchmark.evaluate`` so the score is
identical to the paper's leaderboard. The single scalar is written to
``result.json`` under ``result.objective`` — the field Stimulus optimises.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import polaris as po


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--benchmark-id", required=True)
    p.add_argument("--main-metric", required=True)
    p.add_argument("--objective-mode", required=True, choices=("minimize", "maximize"))
    p.add_argument("--task-type", required=True, choices=("regression", "classification"))
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def grade(submission: pd.DataFrame, benchmark_id: str, main_metric: str, task_type: str) -> float:
    """BioML-bench Polaris grader: evaluate the prediction column, return main metric.

    Classification benchmarks expose metrics that need labels (accuracy, f1, mcc)
    and metrics that need probabilities (pr_auc, roc_auc), so both are passed; the
    submission column holds P(positive) and labels are thresholded at 0.5.
    """
    benchmark = po.load_benchmark(benchmark_id)
    predictions = submission.sort_values("id")[submission.columns[1]].tolist()
    if task_type == "classification":
        labels = [int(p >= 0.5) for p in predictions]
        results = benchmark.evaluate(y_pred=labels, y_prob=predictions)
    else:
        results = benchmark.evaluate(predictions)
    table = results.results
    row = table[table["Metric"] == main_metric]
    if row.empty:
        raise ValueError(f"main metric {main_metric!r} not in results: {list(table['Metric'])}")
    return float(row["Score"].iloc[0])


def main() -> None:
    """Grade the submission and write the Stimulus-facing result.json."""
    args = parse_args()
    submission = pd.read_csv(args.submission)
    score = grade(submission, args.benchmark_id, args.main_metric, args.task_type)
    if not math.isfinite(score):
        raise ValueError(f"objective is not finite: {score}")

    payload = {
        "status": "success",
        "result": {
            "objective": score,
            "objective_key": args.main_metric,
            "objective_mode": args.objective_mode,
            "objective_split": "test",
        },
        "metrics": {args.main_metric: score, "n_test": int(len(submission))},
        "task": args.benchmark_id,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"{args.benchmark_id} {args.main_metric}={score:.6f} ({args.objective_mode})")


if __name__ == "__main__":
    main()
