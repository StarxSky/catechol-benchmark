from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from catechol_gnn.config import DataConfig, GNNConfig
from catechol_gnn.data import CatecholDataset, collate_fn, make_loro_splits, make_loso_splits
from catechol_gnn.model import CatecholGNN


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def train_one_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    data_cfg: DataConfig,
    model_cfg: GNNConfig,
    drfp: np.ndarray | None,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    train_idx = np.array(train_idx)

    # Validation split (10% of train) for early stopping
    val_size = max(1, int(0.1 * len(train_idx)))
    rng.shuffle(train_idx)
    val_idx = train_idx[:val_size]
    train_idx = train_idx[val_size:]

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    train_dataset = CatecholDataset(train_df, data_cfg, drfp=drfp[train_idx] if drfp is not None else None)
    val_dataset = CatecholDataset(val_df, data_cfg, drfp=drfp[val_idx] if drfp is not None else None)
    test_dataset = CatecholDataset(test_df, data_cfg, drfp=drfp[test_idx] if drfp is not None else None)

    train_loader = DataLoader(
        train_dataset, batch_size=model_cfg.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=model_cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=model_cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # Infer atom feature dimension from one batch
    sample = next(iter(train_loader))
    atom_feature_dim = sample[0].x.shape[-1]

    device = torch.device(model_cfg.device if torch.cuda.is_available() else "cpu")

    model = CatecholGNN(
        atom_feature_dim=atom_feature_dim,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_gat_layers,
        num_heads=model_cfg.num_heads,
        drfp_dim=model_cfg.drfp_dim,
        dropout=model_cfg.dropout,
        head_dropout=model_cfg.head_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=model_cfg.lr_plateau_factor, patience=model_cfg.lr_plateau_patience
    )

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(model_cfg.max_epochs):
        model.train()
        for batch in train_loader:
            batch = [b.to(device) if hasattr(b, "to") else b for b in batch]
            sm, p2, p3, solvent_a, solvent_b, percent_b, temperature, residence_time, drfp_batch, y = batch

            pred = model(
                sm,
                p2,
                p3,
                solvent_a,
                solvent_b,
                percent_b,
                temperature,
                residence_time,
                drfp_batch,
            )
            loss = mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg.grad_clip_norm)
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) if hasattr(b, "to") else b for b in batch]
                sm, p2, p3, solvent_a, solvent_b, percent_b, temperature, residence_time, drfp_batch, y = batch
                pred = model(
                    sm,
                    p2,
                    p3,
                    solvent_a,
                    solvent_b,
                    percent_b,
                    temperature,
                    residence_time,
                    drfp_batch,
                )
                val_losses.append(mse_loss(pred, y).item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= model_cfg.early_stop_patience:
            break

    # Test
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            batch = [b.to(device) if hasattr(b, "to") else b for b in batch]
            sm, p2, p3, solvent_a, solvent_b, percent_b, temperature, residence_time, drfp_batch, y = batch
            pred = model(
                sm,
                p2,
                p3,
                solvent_a,
                solvent_b,
                percent_b,
                temperature,
                residence_time,
                drfp_batch,
            )
            test_losses.append(mse_loss(pred, y).item())

    return float(np.mean(test_losses))


def load_drfp(path: str) -> np.ndarray | None:
    if not path:
        return None
    return np.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN per 2512.19530v1")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--split", choices=["loso", "loro"], required=True)
    parser.add_argument("--drfp", default="", help="Optional path to DRFP .npy (n_samples x 2048)")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    data_cfg = DataConfig()
    model_cfg = GNNConfig(device=args.device)

    df = pd.read_csv(args.csv)
    drfp = load_drfp(args.drfp)

    if args.split == "loso":
        splits = make_loso_splits(df, data_cfg)
    else:
        splits = make_loro_splits(df, data_cfg)

    scores: List[float] = []
    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        mse = train_one_split(df, train_idx, test_idx, data_cfg, model_cfg, drfp)
        scores.append(mse)
        print(f"split {i}/{len(splits)} mse={mse:.6f}")

    print(f"mean mse={np.mean(scores):.6f} std={np.std(scores):.6f}")


if __name__ == "__main__":
    main()
