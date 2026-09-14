"""
Self-attention: how each position decides what to look at.

Maps to your notes -> Transformer Internals -> Self-attention:
    queries, keys, values, projection matrices, scaled dot product,
    scaling by sqrt(d), softmax weights, weighted sum of values,
    causal mask, attention scores, attention weight matrix
Also -> KV cache:
    prefill vs decode -- see the kv_cache/use_cache parameters below

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

    def forward(self, x: torch.Tensor, kv_cache=None, use_cache: bool = False,
                return_weights: bool = False):
        """x: (batch, seq_len, n_embd) -> (batch, seq_len, head_size)

        kv_cache: None, or a (past_k, past_v) tuple of everything computed
            on EARLIER calls -- each (batch, past_len, head_size).
        use_cache: if True, also return the (possibly extended) cache so
            the caller can pass it into the next call.

        Three ways this gets called:
          1. Training / no caching at all: kv_cache=None, use_cache=False.
             Identical to the original implementation -- unchanged.
          2. "Prefill": kv_cache=None, use_cache=True. Runs a normal full
             forward over the whole prompt (T can be > 1) and returns the
             freshly computed (k, v) as the STARTING cache.
          3. "Decode step": kv_cache=(past_k, past_v), use_cache=True. x is
             just the ONE new token (T=1); its k, v get appended onto the
             cache instead of recomputing anything for earlier positions.
        """
        B, T, C = x.shape

        q = self.query(x)   # (B, T, head_size) -- "what am I looking for"
        k = self.key(x)     # (B, T, head_size) -- "what do I contain"
        v = self.value(x)   # (B, T, head_size) -- "what do I hand over"

        if kv_cache is not None:
            # Append this step's new k, v onto everyone computed before --
            # the whole point of the cache: we never recompute past K/V.
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        Tk = k.shape[1]  # total keys available: past cache + this call's new ones

        # Attention scores: how well does each query match each key?
        # (B, T, head_size) @ (B, head_size, Tk) -> (B, T, Tk)
        scores = q @ k.transpose(-2, -1)

        # Scale by sqrt(head_size): unscaled dot products grow with
        # head_size, pushing softmax toward a near-one-hot regime and
        # killing gradients through it.
        scores = scores / math.sqrt(self.head_size)

        if kv_cache is None:
            # No prior cache: either a normal full-sequence forward
            # (training) or the "prefill" step seeding the cache from a
            # prompt. Both need the real causal mask, T positions against
            # T keys, exactly as before.
            scores = scores.masked_fill(self.causal_mask[:T, :Tk] == 0, float("-inf"))
        # else: a decode step. The new query is always causally AFTER
        # everything already sitting in the cache, so it's allowed to see
        # all of it -- no mask needed, there's nothing in the future here.

        # Softmax over "which key" -> a proper probability distribution
        # per query position. This IS the "attention weight matrix".
        weights = F.softmax(scores, dim=-1)   # (B, T, Tk)
        weights = self.dropout(weights)

        # Weighted sum of values: each position's output is a blend of
        # every ALLOWED position's value vector.
        out = weights @ v   # (B, T, Tk) @ (B, Tk, head_size) -> (B, T, head_size)

        if use_cache:
            return out, (k, v)
        if return_weights:
            return out, weights
        return out
