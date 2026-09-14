"""
Self-attention: how each position decides what to look at.

Maps to your notes -> Transformer Internals -> Self-attention:
    queries, keys, values, projection matrices, scaled dot product,
    scaling by sqrt(d), softmax weights, weighted sum of values,
    causal mask, attention scores, attention weight matrix

The idea: every position produces a QUERY ("what am I looking for?"), and
every position (including itself) offers a KEY ("what do I contain?") and
a VALUE ("what do I actually hand over if you attend to me?"). A
position's new representation is a weighted sum of everyone's VALUES,
weighted by how well its QUERY matches their KEYS.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Single-head scaled dot-product self-attention, from raw matrix
    multiplies -- no nn.MultiheadAttention. model/block.py combines
    several of these into multi-head attention."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        # Three separate projection matrices: same input x, three
        # different questions asked of it.
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask: position i may only attend to positions <= i.
        # A lower-triangular matrix of 1s, computed once and reused.
        # register_buffer: moves with the model, never trained.
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask)

        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """x: (batch, seq_len, n_embd) -> (batch, seq_len, head_size)"""
        B, T, C = x.shape

        q = self.query(x)   # (B, T, head_size) -- "what am I looking for"
        k = self.key(x)     # (B, T, head_size) -- "what do I contain"
        v = self.value(x)   # (B, T, head_size) -- "what do I hand over"

        # Attention scores: how well does each query match each key?
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        scores = q @ k.transpose(-2, -1)

        # Scale by sqrt(head_size): unscaled dot products grow with
        # head_size, pushing softmax toward a near-one-hot regime and
        # killing gradients through it.
        scores = scores / math.sqrt(self.head_size)

        # Apply the causal mask: forbid attending to future positions by
        # setting their score to -inf before softmax (softmax(-inf) = 0).
        scores = scores.masked_fill(self.causal_mask[:T, :T] == 0, float("-inf"))

        # Softmax over "which key" -> a proper probability distribution
        # per query position. This IS the "attention weight matrix".
        weights = F.softmax(scores, dim=-1)   # (B, T, T)
        weights = self.dropout(weights)

        # Weighted sum of values: each position's output is a blend of
        # every ALLOWED position's value vector.
        out = weights @ v   # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        if return_weights:
            return out, weights
        return out
