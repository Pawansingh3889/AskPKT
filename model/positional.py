"""
Positional encoding: gives the model a sense of order.

Maps to your notes -> Positional Encoding -> Absolute methods:
    sinusoidal encoding, sine and cosine frequencies, fixed versus learned,
    additive injection, max position limit

Why it's needed (your notes -> "why it's needed"):
    Attention itself is permutation-invariant -- it looks at a *set* of
    token vectors and weighs them by similarity, with no notion of which
    came first. Without positional information, "dog bites man" and
    "man bites dog" would attend identically, because attention only
    sees which tokens are present, not their order.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    The original ("Attention Is All You Need") absolute method: a fixed
    table of shape (max_len, n_embd), one row per position, computed once
    from sine/cosine waves of different frequencies -- never trained.

    Even-numbered dimensions get sin(position / 10000^(2i/n_embd)),
    odd-numbered dimensions get the matching cos. Low dimensions oscillate
    fast (encode fine-grained position), high dimensions oscillate slowly
    (encode coarse position) -- like a clock with a second hand, minute
    hand, and hour hand all encoding "time" at different resolutions.
    """

    def __init__(self, max_len: int, n_embd: int):
        super().__init__()
        self.max_len = max_len  # the "max position limit"

        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd)
        )  # one frequency per pair of dimensions

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims: sine
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims: cosine

        # register_buffer, NOT nn.Parameter: this is the "fixed" branch of
        # fixed-versus-learned. It moves with the model (.to(device),
        # state_dict) but nn.init/optimizer never touch it -- there is no
        # gradient to compute because it's math, not a learned weight.
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        assert seq_len <= self.max_len, (
            f"sequence length {seq_len} exceeds max_len {self.max_len} "
            "this encoding was built for"
        )
        return self.pe[:seq_len]  # (seq_len, n_embd)

    def add_to(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Additive injection: position info is added elementwise to the
        token embeddings, not concatenated -- same shape in, same shape out."""
        seq_len = token_embeddings.shape[-2]
        return token_embeddings + self.forward(seq_len)
