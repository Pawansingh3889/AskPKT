"""
Multi-head attention, feed-forward network, residual connections, and
layer normalization -- combined into one Transformer block.

Maps to your notes -> Transformer Internals:
    Multi-head attention: head splitting, parallel heads, per-head
        projections, concatenation, output projection
    Feed-forward: two linear layers, up/down projection, hidden dimension,
        activation functions, position-wise application
    Residual connections: skip connections, identity path, gradient flow,
        residual stream, additive updates
    Layer normalisation: mean and variance, gain and bias parameters,
        pre-norm versus post-norm
    Block stacking: layer depth, repeated blocks
"""

import torch
import torch.nn as nn
from model.attention import SelfAttention


class MultiHeadAttention(nn.Module):
    """Run several SelfAttention heads in parallel, each in its own
    smaller subspace, then concatenate and mix back together.

    Why several heads instead of one big one: each head can specialise
    -- one might learn to track nearby words, another longer-range
    agreement, another something else entirely. One head averages
    everything into a single blend; several let those patterns coexist
    without interfering.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must divide evenly across heads"
        head_size = n_embd // n_head  # the "head splitting"

        # Per-head projections: n_head independent SelfAttention modules,
        # each with its own Wq/Wk/Wv, each working in its own
        # head_size-dimensional subspace.
        self.heads = nn.ModuleList([
            SelfAttention(n_embd, head_size, block_size, dropout) for _ in range(n_head)
        ])
        # Output projection: mixes the concatenated heads back together
        # into one n_embd-wide vector per position.
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parallel heads, concatenated along the feature dimension:
        # n_head * head_size == n_embd again.
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network: the same small MLP applied
    independently to every position -- no mixing across positions here,
    that was attention's job. Up-projects to a wider hidden dimension
    (conventionally 4x), applies a nonlinearity, projects back down.
    """

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        hidden = 4 * n_embd  # "usually four times the model dimension"
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden),   # up projection
            nn.GELU(),                   # activation function
            nn.Linear(hidden, n_embd),   # down projection
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer block: multi-head attention, then feed-forward,
    each wrapped in a residual connection and preceded by layer norm
    (pre-norm: normalize, then compute, then add back to the un-normed
    input -- the modern convention, more stable to train deep stacks of
    these than the original post-norm placement).
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual / skip connection: x + f(x), never x REPLACED by f(x).
        # This is the "identity path" / "residual stream" -- gradients
        # can flow straight through the addition even when f(x)'s own
        # gradient is tiny, which is what makes deep stacks of these
        # trainable at all instead of vanishing.
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
