#!/usr/bin/env python3
"""Fine-tune a PCQM4Mv2-pretrained Graphormer for the BioML-bench drug task.

This is an ORTHOGONAL arm to the classical Morgan/sklearn baseline: a large
graph transformer (microsoft/Graphormer, ported to HuggingFace as
``clefourrier/graphormer-base-pcqm4mv2``) that was *pretrained* on ~3.7M
PCQM4Mv2 molecules to predict the DFT HOMO-LUMO gap, then fine-tuned here on the
~2,500-molecule hepatic-clearance target. Pretraining is the key difference from
the earlier from-scratch graph transformer that stalled near pearsonr ~0.65.

The pretrained checkpoint (weights) and the Cython spatial-encoding extension are
baked into the trial container at build time, so this script does no network I/O.

Every knob that changes the fit is a CLI flag mapped 1:1 onto the Stimulus search
space: the fine-tuning regime (frozen featuriser / full end-to-end / linear-probe
then fine-tune), the prediction head, gradual layer freezing, optimisation knobs,
the seed-averaged ensemble size, and optional engineered-descriptor fusion.

Regression only: predicts the continuous target. A cheap Morgan+Ridge prediction
is always computed as a safety net so a degenerate fine-tune still yields a
finite, ranked submission (the grader rejects non-finite/constant predictions).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transformers import GraphormerForGraphClassification
from transformers.models.graphormer.collating_graphormer import (
    GraphormerDataCollator,
    preprocess_item,
)

# Compact, robust RDKit 2D descriptor block for the optional fusion branch.
_DESCRIPTORS = (
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.FractionCSP3,
    Descriptors.NumAromaticRings, Descriptors.NumAliphaticRings,
    Descriptors.RingCount, Descriptors.HeavyAtomCount,
    Descriptors.NHOHCount, Descriptors.NOCount, Descriptors.qed,
    Descriptors.MolMR, Descriptors.BalabanJ, Descriptors.BertzCT,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags. Tunable knobs map 1:1 onto the Stimulus search space."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--test-features", required=True, type=Path)
    p.add_argument("--sample-submission", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model-meta", required=True, type=Path)
    p.add_argument("--task-type", required=True, choices=("regression",))
    p.add_argument("--ckpt-dir", default=os.environ.get(
        "GRAPHORMER_CKPT_DIR", "clefourrier/graphormer-base-pcqm4mv2"))
    p.add_argument("--finetune-mode", default="lpft",
                   choices=("frozen", "full", "lpft"))
    p.add_argument("--head", default="mlp", choices=("linear", "mlp"))
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--ensemble", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="Freeze the bottom N of 12 transformer layers (full/lpft).")
    p.add_argument("--fuse-descriptors", default="off", choices=("off", "on"))
    p.add_argument("--morgan-radius", type=int, default=2)
    p.add_argument("--morgan-bits", type=int, default=1024)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def mol_to_graph(smiles: str) -> dict:
    """Build an OGB-style integer-feature graph dict from a SMILES string.

    Unparsable SMILES degrade to a single-node graph so the batch never breaks.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return {"edge_index": np.zeros((2, 1), dtype=np.int64),
                "edge_attr": np.zeros((1, 3), dtype=np.int64),
                "node_feat": np.zeros((1, 9), dtype=np.int64),
                "num_nodes": 1, "y": [0.0]}
    node_feat = [[
        a.GetAtomicNum(), int(a.GetChiralTag()), a.GetDegree(),
        a.GetFormalCharge() + 5, a.GetTotalNumHs(),
        a.GetNumRadicalElectrons(), int(a.GetHybridization()),
        int(a.GetIsAromatic()), int(a.IsInRing()),
    ] for a in mol.GetAtoms()]
    edges, edge_feat = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        feat = [int(b.GetBondType()), int(b.GetStereo()), int(b.GetIsConjugated())]
        edges += [[i, j], [j, i]]
        edge_feat += [feat, feat]
    if not edges:
        edges, edge_feat = [[0, 0]], [[0, 0, 0]]
    return {
        "edge_index": np.array(edges, dtype=np.int64).T,
        "edge_attr": np.array(edge_feat, dtype=np.int64),
        "node_feat": np.array(node_feat, dtype=np.int64),
        "num_nodes": mol.GetNumAtoms(),
        "y": [0.0],
    }


def morgan_matrix(smiles: list[str], radius: int, n_bits: int) -> np.ndarray:
    """Morgan/ECFP bit-vector features; unparsable SMILES become all-zero rows."""
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
        else:
            bits = np.frombuffer(generator.GetFingerprint(mol).ToBitString().encode(),
                                 "u1") - ord("0")
            rows.append(bits.astype(np.float32))
    return np.vstack(rows).astype(np.float32)


def descriptor_matrix(smiles: list[str]) -> np.ndarray:
    """RDKit 2D descriptor block; non-finite values are zeroed (imputed later)."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            rows.append(np.zeros(len(_DESCRIPTORS), dtype=np.float32))
            continue
        vals = []
        for fn in _DESCRIPTORS:
            try:
                vals.append(float(fn(mol)))
            except Exception:
                vals.append(0.0)
        rows.append(np.asarray(vals, dtype=np.float32))
    mat = np.vstack(rows).astype(np.float32)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


class GraphormerRegressor(torch.nn.Module):
    """Pretrained Graphormer encoder + a small regression head on the graph token.

    The graph-level representation is the first ("[graph]") token of the encoder's
    last hidden state (matching HuggingFace's own classification head). Optional
    static descriptor features are concatenated before the head.
    """

    def __init__(self, encoder: torch.nn.Module, embedding_dim: int,
                 extra_dim: int, head_type: str, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        in_dim = embedding_dim + extra_dim
        if head_type == "linear":
            self.head = torch.nn.Sequential(
                torch.nn.Dropout(dropout), torch.nn.Linear(in_dim, 1))
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Dropout(dropout), torch.nn.Linear(in_dim, 256),
                torch.nn.GELU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(256, 1))

    def embed(self, batch: dict) -> torch.Tensor:
        """Return the (B, embedding_dim) graph-token representation."""
        out = self.encoder(
            batch["input_nodes"], batch["input_edges"], batch["attn_bias"],
            batch["in_degree"], batch["out_degree"], batch["spatial_pos"],
            batch["attn_edge_type"], return_dict=True)
        return out["last_hidden_state"][:, 0, :]

    def forward(self, batch: dict, extra: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embed(batch)
        if extra is not None:
            emb = torch.cat([emb, extra], dim=1)
        return self.head(emb).squeeze(-1)


def collate(items: list, indices: np.ndarray, collator: GraphormerDataCollator) -> dict:
    """Collate the preprocessed graph items at ``indices`` into a tensor batch."""
    return collator([items[i] for i in indices])


def freeze_bottom_layers(encoder: torch.nn.Module, n_freeze: int) -> None:
    """Freeze the input embeddings and the bottom ``n_freeze`` transformer layers."""
    if n_freeze <= 0:
        return
    ge = encoder.graph_encoder
    for module in (ge.graph_node_feature, ge.graph_attn_bias, ge.emb_layer_norm):
        for param in module.parameters():
            param.requires_grad_(False)
    for layer in list(ge.layers)[:n_freeze]:
        for param in layer.parameters():
            param.requires_grad_(False)


def compute_embeddings(model: GraphormerRegressor, items: list, batch_size: int,
                       collator: GraphormerDataCollator) -> torch.Tensor:
    """One no-grad pass: (N, embedding_dim) graph embeddings for all items."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            idx = np.arange(start, min(start + batch_size, len(items)))
            out.append(model.embed(collate(items, idx, collator)))
    return torch.cat(out, dim=0)


def train_head_on_features(head: torch.nn.Module, feats: torch.Tensor,
                           y: torch.Tensor, epochs: int, lr: float,
                           weight_decay: float, batch_size: int,
                           rng: np.random.Generator) -> None:
    """Train just the head on fixed (frozen) features — the linear-probe phase."""
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    n = feats.shape[0]
    head.train()
    for _ in range(max(1, epochs)):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            opt.zero_grad()
            pred = head(feats[idx]).squeeze(-1)
            loss = loss_fn(pred, y[idx])
            loss.backward()
            opt.step()


def train_end_to_end(model: GraphormerRegressor, items: list, extra: torch.Tensor | None,
                     y: torch.Tensor, epochs: int, lr: float, weight_decay: float,
                     batch_size: int, collator: GraphormerDataCollator,
                     rng: np.random.Generator) -> None:
    """Fine-tune all unfrozen parameters end-to-end on the graph batches."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    n = len(items)
    model.train()
    for _ in range(max(1, epochs)):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch = collate(items, idx, collator)
            ex = extra[idx] if extra is not None else None
            opt.zero_grad()
            pred = model(batch, ex)
            loss = loss_fn(pred, y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()


def predict(model: GraphormerRegressor, items: list, extra: torch.Tensor | None,
            batch_size: int, collator: GraphormerDataCollator) -> np.ndarray:
    """Predict standardised targets for all items (no grad)."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            idx = np.arange(start, min(start + batch_size, len(items)))
            ex = extra[idx] if extra is not None else None
            out.append(model(collate(items, idx, collator), ex).cpu().numpy())
    return np.concatenate(out)


def fit_one_member(args: argparse.Namespace, train_items: list, test_items: list,
                   extra_train: torch.Tensor | None, extra_test: torch.Tensor | None,
                   extra_dim: int, y_std: torch.Tensor, member_seed: int) -> np.ndarray:
    """Train one ensemble member and return its standardised test predictions."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)
    collator = GraphormerDataCollator()

    base = GraphormerForGraphClassification.from_pretrained(
        args.ckpt_dir, num_classes=1, ignore_mismatched_sizes=True)
    model = GraphormerRegressor(base.encoder, base.config.embedding_dim,
                                extra_dim, args.head, args.head_dropout)

    if args.finetune_mode == "frozen":
        for param in model.encoder.parameters():
            param.requires_grad_(False)
        feats = compute_embeddings(model, train_items, args.batch_size, collator)
        if extra_train is not None:
            feats = torch.cat([feats, extra_train], dim=1)
        train_head_on_features(model.head, feats, y_std, max(args.epochs, 10),
                               args.lr, args.weight_decay, args.batch_size, rng)
        return predict(model, test_items, extra_test, args.batch_size, collator)

    if args.finetune_mode == "lpft":
        # Phase 1: linear-probe the head on frozen features (LP-FT, Kumar 2022).
        for param in model.encoder.parameters():
            param.requires_grad_(False)
        feats = compute_embeddings(model, train_items, args.batch_size, collator)
        if extra_train is not None:
            feats = torch.cat([feats, extra_train], dim=1)
        train_head_on_features(model.head, feats, y_std, min(5, max(1, args.epochs)),
                               args.lr, args.weight_decay, args.batch_size, rng)
        for param in model.encoder.parameters():
            param.requires_grad_(True)

    # Phase 2 (lpft) / single phase (full): fine-tune end-to-end.
    freeze_bottom_layers(model.encoder, args.freeze_layers)
    train_end_to_end(model, train_items, extra_train, y_std, args.epochs, args.lr,
                     args.weight_decay, args.batch_size, collator, rng)
    return predict(model, test_items, extra_test, args.batch_size, collator)


def safety_net(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
               seed: int) -> np.ndarray:
    """Cheap Morgan+Ridge fallback so a degenerate fine-tune still ranks molecules."""
    ridge = Ridge(alpha=10.0, random_state=seed)
    ridge.fit(x_train, y_train)
    return ridge.predict(x_test)


def main() -> None:
    """Fine-tune the pretrained Graphormer ensemble and write submission.csv."""
    args = parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test_features)
    sample = pd.read_csv(args.sample_submission)
    molecule_col, target_col = train.columns[0], train.columns[1]
    pred_col = sample.columns[1]

    train_smiles = train[molecule_col].astype(str).tolist()
    test_smiles = test[molecule_col].astype(str).tolist()
    y_train = train[target_col].to_numpy(dtype=np.float64)

    # Standardise the target on train only (fine-tuning stability).
    y_mean, y_scale = float(np.mean(y_train)), float(np.std(y_train) or 1.0)
    y_std = torch.tensor((y_train - y_mean) / y_scale, dtype=torch.float32)

    train_items = [preprocess_item(mol_to_graph(s)) for s in train_smiles]
    test_items = [preprocess_item(mol_to_graph(s)) for s in test_smiles]

    # Optional engineered-descriptor fusion branch (Morgan bits + RDKit 2D),
    # standardised on train only and concatenated before the head.
    extra_train = extra_test = None
    extra_dim = 0
    if args.fuse_descriptors == "on":
        fp_tr = morgan_matrix(train_smiles, args.morgan_radius, args.morgan_bits)
        fp_te = morgan_matrix(test_smiles, args.morgan_radius, args.morgan_bits)
        ds_tr, ds_te = descriptor_matrix(train_smiles), descriptor_matrix(test_smiles)
        scaler = StandardScaler().fit(ds_tr)
        ds_tr, ds_te = scaler.transform(ds_tr), scaler.transform(ds_te)
        fused_tr = np.hstack([fp_tr, ds_tr]).astype(np.float32)
        fused_te = np.hstack([fp_te, ds_te]).astype(np.float32)
        extra_train = torch.tensor(fused_tr, dtype=torch.float32)
        extra_test = torch.tensor(fused_te, dtype=torch.float32)
        extra_dim = fused_tr.shape[1]

    # Seed-averaged ensemble (the most reliable lever on this small, noisy target).
    # A member that fails is skipped rather than crashing the unattended trial.
    preds = []
    for k in range(max(1, args.ensemble)):
        try:
            member = fit_one_member(args, train_items, test_items, extra_train,
                                    extra_test, extra_dim, y_std, args.seed + k)
            if np.all(np.isfinite(member)):
                preds.append(member)
        except Exception as exc:  # noqa: BLE001 - keep the trial alive, fall back below
            print(f"ensemble member {k} failed: {exc}")

    # Safety net: if no member survived, or predictions are non-finite / near-constant,
    # fall back to a Morgan+Ridge fit so the grader always receives a finite, ranked
    # submission (the grader rejects non-finite or zero-variance predictions).
    used_fallback = False
    if preds:
        predictions = np.mean(np.vstack(preds), axis=0) * y_scale + y_mean
    else:
        predictions = np.zeros(len(test_smiles))
    if (not preds) or (not np.all(np.isfinite(predictions))) or float(np.std(predictions)) < 1e-6:
        x_tr = morgan_matrix(train_smiles, args.morgan_radius, args.morgan_bits)
        x_te = morgan_matrix(test_smiles, args.morgan_radius, args.morgan_bits)
        predictions = safety_net(x_tr, y_train, x_te, args.seed)
        used_fallback = True

    pd.DataFrame({"id": test["id"], pred_col: predictions}).to_csv(args.out, index=False)
    args.model_meta.write_text(json.dumps({
        "model_family": "graphormer",
        "checkpoint": args.ckpt_dir,
        "finetune_mode": args.finetune_mode,
        "head": args.head,
        "ensemble": int(args.ensemble),
        "epochs": int(args.epochs),
        "freeze_layers": int(args.freeze_layers),
        "fuse_descriptors": args.fuse_descriptors,
        "extra_dim": int(extra_dim),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "y_pred_mean": float(np.mean(predictions)),
        "y_pred_std": float(np.std(predictions)),
        "used_safety_net": used_fallback,
    }, indent=2))
    print(f"graphormer[{args.finetune_mode}] ensemble={args.ensemble} "
          f"fuse={args.fuse_descriptors} pred mean={np.mean(predictions):.4f} "
          f"std={np.std(predictions):.4f} fallback={used_fallback}")


if __name__ == "__main__":
    main()
