from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
except Exception as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "torch_geometric is required for the GNN model. Install via PyTorch Geometric."
    ) from exc


@dataclass
class GraphBatch:
    # A batched molecular graph from PyG
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor


class MoleculeEncoder(nn.Module):
    """
    Encoder used for molecular graphs.

    Paper-aligned settings:
    - Linear projection to D=256
    - 4 stacked GAT layers with 8 heads and residual connections
    - Global mean+max pooling
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        head_dim = hidden_dim // num_heads
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                )
            )

        self.dropout = dropout

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
        x = self.input_proj(x)

        for gat in self.gat_layers:
            x_res = x
            x = gat(x, edge_index)
            x = x + x_res  # residual connection per paper
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global mean + max pooling
        mean_pooled = global_mean_pool(x, batch_index)
        max_pooled = global_max_pool(x, batch_index)
        return torch.cat([mean_pooled, max_pooled], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CatecholGNN(nn.Module):
    """
    GNN architecture from 2512.19530v1:
    - Four molecular graphs: SM, P2, P3, Solvent (A/B)
    - GAT(4 layers, 8 heads), global mean+max pooling
    - Learned mixture encoding: e_mix = MLP([eA; eB; %B; T; time])
    - DRFP (2048-dim) concatenated with all embeddings and numeric conditions
    - Final MLP head -> 3 outputs with sigmoid
    """

    def __init__(
        self,
        atom_feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        drfp_dim: int = 2048,
        dropout: float = 0.15,
        head_dropout: float = 0.075,
    ):
        super().__init__()

        # Separate input projections for reactants/products and solvents (per paper)
        self.reactant_encoder = MoleculeEncoder(
            in_dim=atom_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.solvent_encoder = MoleculeEncoder(
            in_dim=atom_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # e_mol after pooling is 2 * hidden_dim
        e_dim = 2 * hidden_dim

        # Learned mixture encoding (Eq. 3)
        self.mixture_mlp = MLP(
            in_dim=e_dim * 2 + 3,  # eA, eB, %B, T, time
            hidden_dim=e_dim,
            out_dim=e_dim,
            dropout=dropout,
        )

        # Final MLP head (Eq. 4)
        final_in_dim = e_dim * 6 + drfp_dim + 3
        self.head = nn.Sequential(
            nn.Linear(final_in_dim, e_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(e_dim, 3),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sm: GraphBatch,
        p2: GraphBatch,
        p3: GraphBatch,
        solvent_a: GraphBatch,
        solvent_b: GraphBatch,
        percent_b: torch.Tensor,
        temperature: torch.Tensor,
        residence_time: torch.Tensor,
        drfp: torch.Tensor,
    ) -> torch.Tensor:
        # Encode molecules
        e_sm = self.reactant_encoder(sm)
        e_p2 = self.reactant_encoder(p2)
        e_p3 = self.reactant_encoder(p3)
        e_a = self.solvent_encoder(solvent_a)
        e_b = self.solvent_encoder(solvent_b)

        # Learned mixture encoding
        mix_input = torch.cat(
            [e_a, e_b, percent_b.unsqueeze(-1), temperature.unsqueeze(-1), residence_time.unsqueeze(-1)],
            dim=-1,
        )
        e_mix = self.mixture_mlp(mix_input)

        # Final prediction
        final_input = torch.cat(
            [
                e_sm,
                e_p2,
                e_p3,
                e_a,
                e_b,
                e_mix,
                drfp,
                temperature.unsqueeze(-1),
                residence_time.unsqueeze(-1),
                percent_b.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.head(final_input)
