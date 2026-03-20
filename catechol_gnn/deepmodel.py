from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w1(x) * F.silu(self.w2(x)))


class DeepModel(nn.Module):
    """
    Transformer-enhanced SwiGLU MLP baseline per paper:
    - Input projection to 384
    - Single multi-head self-attention (8 heads)
    - 4 residual SwiGLU blocks
    - 2-layer MLP head -> 3 outputs
    """

    def __init__(self, in_dim: int, hidden_dim: int = 384, heads: int = 8, dropout: float = 0.15):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([SwiGLU(hidden_dim, dropout) for _ in range(4)])
        self.block_ln = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(4)])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.075),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Treat feature vector as sequence length 1
        x = self.in_proj(x)
        x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_ln(x + attn_out)
        x = x.squeeze(1)

        for blk, ln in zip(self.blocks, self.block_ln):
            x = ln(x + blk(x))

        return torch.sigmoid(self.head(x))
