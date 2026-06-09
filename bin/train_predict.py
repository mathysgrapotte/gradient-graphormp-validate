#!/usr/bin/env python3
"""Train a baseline molecular-property model and write a BioML-bench submission.

This is the deliberately-simple *base* model the Stimulus/Gradient loop starts
from and is expected to improve on: fixed-radius Morgan fingerprints fed to a
classical scikit-learn estimator. Every knob that changes the fit is exposed as
a CLI flag so Stimulus can tune it as a search-space dimension.

Regression  -> predicts the continuous target.
Classification -> predicts P(positive class) so PR-AUC / ROC-AUC graders get y_prob.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge


def parse_args() -> argparse.Namespace:
    """Parse CLI flags. Tunable knobs map 1:1 onto the Stimulus search space."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--test-features", required=True, type=Path)
    p.add_argument("--sample-submission", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model-meta", required=True, type=Path)
    p.add_argument("--task-type", required=True, choices=("regression", "classification"))
    p.add_argument("--model-family", default="ridge",
                   choices=("ridge", "random_forest", "hist_gradient_boosting", "logistic"))
    p.add_argument("--morgan-radius", type=int, default=2)
    p.add_argument("--morgan-bits", type=int, default=1024)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--logistic-c", type=float, default=1.0)
    p.add_argument("--rf-n-estimators", type=int, default=300)
    p.add_argument("--hgb-learning-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def featurize(smiles: list[str], radius: int, n_bits: int) -> np.ndarray:
    """Morgan/ECFP bit-vector features; unparsable SMILES become all-zero rows."""
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.int8))
        else:
            rows.append(np.frombuffer(generator.GetFingerprint(mol).ToBitString().encode(),
                                      "u1") - ord("0"))
    return np.vstack(rows).astype(np.float32)


def build_estimator(args: argparse.Namespace):
    """Instantiate the chosen estimator for the chosen task type."""
    if args.task_type == "regression":
        if args.model_family == "random_forest":
            return RandomForestRegressor(n_estimators=args.rf_n_estimators,
                                         random_state=args.seed, n_jobs=-1)
        if args.model_family == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(learning_rate=args.hgb_learning_rate,
                                                 random_state=args.seed)
        return Ridge(alpha=args.ridge_alpha, random_state=args.seed)
    if args.model_family == "random_forest":
        return RandomForestClassifier(n_estimators=args.rf_n_estimators,
                                      random_state=args.seed, n_jobs=-1)
    if args.model_family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(learning_rate=args.hgb_learning_rate,
                                              random_state=args.seed)
    return LogisticRegression(C=args.logistic_c, max_iter=1000, random_state=args.seed)


def main() -> None:
    """Fit on train, predict the test split, write submission.csv in sample schema."""
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test_features)
    sample = pd.read_csv(args.sample_submission)

    molecule_col = train.columns[0]
    target_col = train.columns[1]
    pred_col = sample.columns[1]

    x_train = featurize(train[molecule_col].tolist(), args.morgan_radius, args.morgan_bits)
    x_test = featurize(test[molecule_col].tolist(), args.morgan_radius, args.morgan_bits)
    y_train = train[target_col].to_numpy()

    estimator = build_estimator(args)
    estimator.fit(x_train, y_train)

    if args.task_type == "classification":
        predictions = estimator.predict_proba(x_test)[:, 1]
    else:
        predictions = estimator.predict(x_test)

    pd.DataFrame({"id": test["id"], pred_col: predictions}).to_csv(args.out, index=False)

    args.model_meta.write_text(json.dumps({
        "model_family": args.model_family,
        "task_type": args.task_type,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(x_train.shape[1]),
        "y_pred_mean": float(np.mean(predictions)),
        "y_pred_std": float(np.std(predictions)),
    }, indent=2))
    print(f"trained {args.model_family} ({args.task_type}); "
          f"pred mean={np.mean(predictions):.4f} std={np.std(predictions):.4f}")


if __name__ == "__main__":
    main()
