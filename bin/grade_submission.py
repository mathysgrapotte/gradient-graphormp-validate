#!/usr/bin/env python3
"""Score a drug-discovery submission against pinned, private answers.

The metric definitions are vendored from BioML-bench's per-task Polaris grader,
but the hidden test labels are NO LONGER re-downloaded from Polaris Hub at
runtime: they are read from the immutable ``answers.csv`` that the prepare stage
staged on a grader-only channel. The model stage never receives this file, so
the answers are unreachable from ``train_predict.py`` by construction.

The single scalar is written to ``result.json`` under ``result.objective`` — the
field Stimulus optimises.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--answers", required=True, type=Path)
    p.add_argument("--benchmark-id", required=True)
    p.add_argument("--main-metric", required=True)
    p.add_argument("--objective-mode", required=True, choices=("minimize", "maximize"))
    p.add_argument("--task-type", required=True, choices=("regression", "classification"))
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, main_metric: str,
                   task_type: str) -> float:
    """Compute the requested Polaris/TDC metric on aligned (y_true, y_pred).

    Mirrors the metric set BioML-bench's Polaris grader exposes. The submission
    column holds the continuous prediction (regression) or P(positive)
    (classification); labels are thresholded at 0.5 for label-based metrics.

    Raises:
        ValueError: if ``main_metric`` is not a known metric for the task type.
    """
    if task_type == "regression":
        if main_metric == "mean_absolute_error":
            return float(mean_absolute_error(y_true, y_pred))
        if main_metric == "mean_squared_error":
            return float(mean_squared_error(y_true, y_pred))
        if main_metric == "r2":
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            return 1.0 - ss_res / ss_tot
        if main_metric == "pearsonr":
            return float(pearsonr(y_true, y_pred)[0])
        if main_metric == "spearmanr":
            return float(spearmanr(y_true, y_pred)[0])
        raise ValueError(f"unknown regression metric {main_metric!r}")
    labels = (y_pred >= 0.5).astype(int)
    if main_metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    if main_metric == "pr_auc":
        return float(average_precision_score(y_true, y_pred))
    if main_metric == "accuracy":
        return float(np.mean(labels == y_true))
    raise ValueError(f"unknown classification metric {main_metric!r}")


def grade(submission: pd.DataFrame, answers: pd.DataFrame, main_metric: str,
          task_type: str) -> float:
    """Align submission to answers on ``id`` and return the main metric.

    Raises:
        ValueError: if ids do not align between submission and answers.
    """
    sub = submission.sort_values("id").reset_index(drop=True)
    ans = answers.sort_values("id").reset_index(drop=True)
    if len(sub) != len(ans) or not (sub["id"].to_numpy() == ans["id"].to_numpy()).all():
        raise ValueError("submission ids do not align with answers ids")
    y_pred = sub[sub.columns[1]].to_numpy(dtype=float)
    y_true = ans[ans.columns[1]].to_numpy(dtype=float)
    return compute_metric(y_true, y_pred, main_metric, task_type)


def main() -> None:
    """Grade the submission and write the Stimulus-facing result.json."""
    args = parse_args()
    submission = pd.read_csv(args.submission)
    answers = pd.read_csv(args.answers)
    score = grade(submission, answers, args.main_metric, args.task_type)
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
