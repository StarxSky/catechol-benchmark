from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from catechol_gnn.baselines import build_gbdt, inverse_variance_ensemble
from catechol_gnn.config import DataConfig
from catechol_gnn.data import make_loro_splits, make_loso_splits
from catechol_gnn.deepmodel import DeepModel


def build_feature_matrix(df: pd.DataFrame, cfg: DataConfig) -> np.ndarray:
    # Tabular baseline uses numeric conditions + provided descriptor columns
    # Expect precomputed features in the CSV; user should align columns.
    # Required numeric columns:
    cols = [cfg.temperature, cfg.residence_time, cfg.percent_b]

    # All remaining numeric columns are treated as descriptors (Spange + ACS PCA + DRFP)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc_cols = [c for c in numeric_cols if c not in cols + [cfg.y_sm, cfg.y_p2, cfg.y_p3]]

    if not desc_cols:
        raise ValueError("No descriptor columns found. Add Spange/ACS/DRFP columns to CSV.")

    return df[cols + desc_cols].to_numpy(dtype=np.float32)


def build_targets(df: pd.DataFrame, cfg: DataConfig) -> np.ndarray:
    return df[[cfg.y_sm, cfg.y_p2, cfg.y_p3]].to_numpy(dtype=np.float32)


def train_deepmodel(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> DeepModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepModel(in_dim=X_train.shape[1]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=128, shuffle=False)

    best_val = float("inf")
    best_state = None
    patience = 50
    no_improve = 0

    for _ in range(400):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(torch.mean((pred - yb) ** 2).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_deepmodel(model: DeepModel, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.tensor(X[i : i + 512]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.vstack(preds)


def mse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((pred - y) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GBDT/DeepModel/Ensemble baselines")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--split", choices=["loso", "loro"], required=True)
    args = parser.parse_args()

    cfg = DataConfig()
    df = pd.read_csv(args.csv)

    if args.split == "loso":
        splits = make_loso_splits(df, cfg)
    else:
        splits = make_loro_splits(df, cfg)

    scores_gbdt: List[float] = []
    scores_deep: List[float] = []
    scores_ens: List[float] = []

    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        train_idx = np.array(train_idx)
        rng = np.random.default_rng(42)
        rng.shuffle(train_idx)
        val_size = max(1, int(0.1 * len(train_idx)))
        val_idx = train_idx[:val_size]
        train_idx = train_idx[val_size:]

        X = build_feature_matrix(df, cfg)
        y = build_targets(df, cfg)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # GBDT
        gbdt = build_gbdt()
        gbdt.fit(X_train, y_train)
        pred_gbdt = gbdt.predict(X_test)

        # DeepModel
        deep = train_deepmodel(X_train, y_train, X_val, y_val)
        pred_deep = eval_deepmodel(deep, X_test)

        # Ensemble
        pred_ens = inverse_variance_ensemble(pred_gbdt, pred_deep)

        mse_gbdt = mse(pred_gbdt, y_test)
        mse_deep = mse(pred_deep, y_test)
        mse_ens = mse(pred_ens, y_test)

        scores_gbdt.append(mse_gbdt)
        scores_deep.append(mse_deep)
        scores_ens.append(mse_ens)

        print(f"split {i}/{len(splits)} gbdt={mse_gbdt:.6f} deep={mse_deep:.6f} ens={mse_ens:.6f}")

    print(f"gbdt mean={np.mean(scores_gbdt):.6f} std={np.std(scores_gbdt):.6f}")
    print(f"deep mean={np.mean(scores_deep):.6f} std={np.std(scores_deep):.6f}")
    print(f"ens  mean={np.mean(scores_ens):.6f} std={np.std(scores_ens):.6f}")


if __name__ == "__main__":
    main()
