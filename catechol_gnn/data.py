from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from rdkit import Chem
except Exception as exc:  # pragma: no cover - handled at runtime
    raise ImportError("RDKit is required for SMILES parsing.") from exc

try:
    from torch_geometric.data import Data, Batch
except Exception as exc:  # pragma: no cover - handled at runtime
    raise ImportError("torch_geometric is required for graph batching.") from exc

from .config import DataConfig
from .model import GraphBatch


def atom_features(atom: Chem.Atom) -> List[float]:
    # Minimal atom features (paper does not specify; keep compact and deterministic)
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
    ]


def mol_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def batch_graphs(graphs: List[Data]) -> GraphBatch:
    batch = Batch.from_data_list(graphs)
    return GraphBatch(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)


class CatecholDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DataConfig, drfp: np.ndarray | None = None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.drfp = drfp

        self._solvent_cache: Dict[str, Data] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _get_graph(self, smiles: str, cache: Dict[str, Data] | None = None) -> Data:
        if cache is not None:
            if smiles not in cache:
                cache[smiles] = mol_to_graph(smiles)
            return cache[smiles]
        return mol_to_graph(smiles)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        cfg = self.cfg

        sm = self._get_graph(row[cfg.smiles_sm])
        p2 = self._get_graph(row[cfg.smiles_p2])
        p3 = self._get_graph(row[cfg.smiles_p3])

        # Solvent graph caching per paper
        solvent_a = self._get_graph(row[cfg.smiles_solvent_a], cache=self._solvent_cache)
        solvent_b_smiles = row.get(cfg.smiles_solvent_b, "")

        if isinstance(solvent_b_smiles, str) and solvent_b_smiles.strip():
            solvent_b = self._get_graph(solvent_b_smiles, cache=self._solvent_cache)
        else:
            # Pure solvent: treat as single molecular graph
            solvent_b = solvent_a

        percent_b = float(row[cfg.percent_b])
        temperature = float(row[cfg.temperature])
        residence_time = float(row[cfg.residence_time])

        if self.drfp is None:
            drfp = np.zeros((1, 2048), dtype=np.float32)
        else:
            drfp = self.drfp[idx : idx + 1]

        y = np.array([row[cfg.y_sm], row[cfg.y_p2], row[cfg.y_p3]], dtype=np.float32)

        return {
            "sm": sm,
            "p2": p2,
            "p3": p3,
            "solvent_a": solvent_a,
            "solvent_b": solvent_b,
            "percent_b": percent_b,
            "temperature": temperature,
            "residence_time": residence_time,
            "drfp": drfp.astype(np.float32),
            "y": y,
        }


def collate_fn(batch: List[dict]):
    sm = batch_graphs([b["sm"] for b in batch])
    p2 = batch_graphs([b["p2"] for b in batch])
    p3 = batch_graphs([b["p3"] for b in batch])
    solvent_a = batch_graphs([b["solvent_a"] for b in batch])
    solvent_b = batch_graphs([b["solvent_b"] for b in batch])

    percent_b = torch.tensor([b["percent_b"] for b in batch], dtype=torch.float32)
    temperature = torch.tensor([b["temperature"] for b in batch], dtype=torch.float32)
    residence_time = torch.tensor([b["residence_time"] for b in batch], dtype=torch.float32)
    drfp = torch.tensor(np.vstack([b["drfp"] for b in batch]), dtype=torch.float32)
    y = torch.tensor(np.vstack([b["y"] for b in batch]), dtype=torch.float32)

    return sm, p2, p3, solvent_a, solvent_b, percent_b, temperature, residence_time, drfp, y


def make_loso_splits(df: pd.DataFrame, cfg: DataConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Leave-One-Solvent-Out (LOSO) for pure solvents.
    Expects `solvent_a_id` to identify the solvent.
    """
    splits = []
    solvent_ids = df[cfg.solvent_a_id].unique()
    for sid in solvent_ids:
        test_idx = df.index[df[cfg.solvent_a_id] == sid].to_numpy()
        train_idx = df.index[df[cfg.solvent_a_id] != sid].to_numpy()
        splits.append((train_idx, test_idx))
    return splits


def make_loro_splits(df: pd.DataFrame, cfg: DataConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Leave-One-Ramp-Out (LORO) for mixtures.
    Expects `ramp_id` to identify a single mixture ramp.
    """
    splits = []
    ramp_ids = df[cfg.ramp_id].unique()
    for rid in ramp_ids:
        test_idx = df.index[df[cfg.ramp_id] == rid].to_numpy()
        train_idx = df.index[df[cfg.ramp_id] != rid].to_numpy()
        splits.append((train_idx, test_idx))
    return splits
