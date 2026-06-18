#!/usr/bin/env python3
"""Stage and validate a pre-prepared, immutable drug-discovery task dataset.

The task data is materialised ONCE off-pipeline (``biomlbench prepare`` +
``scripts/prepare_and_upload_data.py``) and pinned to an immutable, versioned
S3 prefix. This script no longer touches Polaris Hub and no longer computes the
train/test split at runtime: it only locates, validates and stages the already
pinned artefacts so the contract is identical on every run.

The dataset prefix (mounted via ``--data-dir``) holds::

    public/train.csv             molecule_col, target_col   (TRAIN labels only)
    public/test_features.csv     id, molecule_col           (NO labels)
    public/sample_submission.csv id, target_col             (dummy values)
    private/answers.csv          id, target_col             (TRUE test labels)

This script copies the public inputs into ``--public-dir`` (the only thing the
model stage ever sees) and the private answers to ``--answers`` (the only thing
the grade stage ever sees). It fails loudly if anything is missing so a bad data
mount can never silently produce a degenerate score.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REQUIRED_PUBLIC = (
    "train.csv",
    "test_features.csv",
    "sample_submission.csv",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Pinned dataset prefix (holds public/ and private/answers.csv)")
    p.add_argument("--public-dir", required=True, type=Path,
                   help="Destination dir for the staged public inputs (model-visible)")
    p.add_argument("--answers", required=True, type=Path,
                   help="Destination path for the staged private answers.csv (grader-only)")
    return p.parse_args()


def locate_prepared(data_dir: Path) -> tuple[Path, Path]:
    """Return (public_dir, answers_csv) inside a pinned dataset tree.

    Accepts the dataset-version root, its ``prepared/`` subdir, or a dir that
    already holds ``public/`` and ``private/answers.csv`` directly.

    Raises:
        FileNotFoundError: if a public dir or answers.csv cannot be found.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"--data-dir is not a directory: {data_dir}")

    candidates = [data_dir, data_dir / "prepared"]
    for base in candidates:
        public = base / "public"
        answers = base / "private" / "answers.csv"
        if public.is_dir() and answers.is_file():
            return public, answers
        # Some prepared trees place answers.csv next to public/.
        answers_flat = base / "answers.csv"
        if public.is_dir() and answers_flat.is_file():
            return public, answers_flat

    raise FileNotFoundError(
        f"could not find public/ + private/answers.csv under {data_dir} "
        f"(looked in {[str(c) for c in candidates]}); "
        "stage the dataset with scripts/prepare_and_upload_data.py first"
    )


def validate_public(public: Path) -> None:
    """Check every required public input exists.

    Raises:
        FileNotFoundError: if any required file is missing.
    """
    missing = [name for name in REQUIRED_PUBLIC if not (public / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"pinned public dir {public} is missing required files: {missing}"
        )


def main() -> None:
    """Locate, validate and stage the pinned task artefacts."""
    args = parse_args()

    src_public, src_answers = locate_prepared(args.data_dir)
    validate_public(src_public)

    args.public_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_PUBLIC:
        shutil.copy(src_public / name, args.public_dir / name)

    shutil.copy(src_answers, args.answers)

    print(f"staged public inputs -> {args.public_dir} ({REQUIRED_PUBLIC})")
    print(f"staged private answers -> {args.answers}")


if __name__ == "__main__":
    main()
