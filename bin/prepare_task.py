#!/usr/bin/env python3
"""Materialise a Polaris/TDC drug-discovery task into the BioML-bench file contract.

Mirrors ``biomlbench`` ``prepare.py`` for Polaris tasks, but self-contained:
it loads the benchmark directly from Polaris Hub (anonymous) and writes the
public split the model trains on. Test labels stay hidden inside Polaris and
are scored later by ``grade_submission.py`` via ``benchmark.evaluate`` — the
exact grader BioML-bench uses, so the objective is leaderboard-comparable.

Outputs (under ``--public-dir``):
  - train.csv            molecule_col, target_col
  - test_features.csv    id, molecule_col   (test order is fixed by Polaris)
  - sample_submission.csv id, target_col    (the submission schema)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polaris as po


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-id", required=True, help="e.g. tdcommons/caco2-wang")
    p.add_argument("--public-dir", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    """Download the benchmark and write the public training contract."""
    args = parse_args()
    public = args.public_dir
    public.mkdir(parents=True, exist_ok=True)

    benchmark = po.load_benchmark(args.benchmark_id)
    molecule_col = list(benchmark.input_cols)[0]
    target_col = list(benchmark.target_cols)[0]

    train, test = benchmark.get_train_test_split()

    train_x = [train[i][0] for i in range(len(train))]
    train_y = [train[i][1] for i in range(len(train))]
    pd.DataFrame({molecule_col: train_x, target_col: train_y}).to_csv(
        public / "train.csv", index=False
    )

    test_x = [test[i] for i in range(len(test))]
    pd.DataFrame({"id": range(len(test_x)), molecule_col: test_x}).to_csv(
        public / "test_features.csv", index=False
    )

    pd.DataFrame({"id": range(len(test_x)), target_col: [0.0] * len(test_x)}).to_csv(
        public / "sample_submission.csv", index=False
    )

    print(f"prepared {args.benchmark_id}: {len(train_x)} train, {len(test_x)} test")
    print(f"molecule_col={molecule_col} target_col={target_col}")


if __name__ == "__main__":
    main()
