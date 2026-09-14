"""
The full GPT model: embeddings -> positional encoding -> stack of
transformer blocks -> final layer norm -> output head -> logits.

Maps to your notes -> Hidden State visualisation -> Logits:
    raw scores, unembedding matrix, vocabulary projection
And -> Embeddings -> Embedding Matrix:
    weight tying, output projection, unembedding
"""

import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding
from model.positional import SinusoidalPositionalEncoding
from model.block import Block


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int,
                 block_size: int, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size

        self.token_emb = TokenEmbedding(vocab_size, n_embd)
        self.pos_enc = SinusoidalPositionalEncoding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm, before the output head

        # Output head / unembedding: hidden state (n_embd) -> vocab-sized
        # logits, one score per possible next token.
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: nn.Linear's weight has shape (vocab_size, n_embd)
        # -- IDENTICAL to token_emb.weight's shape -- so instead of
        # letting lm_head learn its own separate matrix, we point it at
        # the exact same Parameter object. One shared matrix now does
        # both jobs: reading a token in (embedding) and scoring every
        # possible token out (unembedding). This roughly halves the
        # parameter count of the two largest tables in a small model.
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor, kv_cache=None, use_cache: bool = False,
                start_pos: int = 0):
        """idx: (batch, seq_len) token ids -> (batch, seq_len, vocab_size) logits.

        kv_cache: None, or a list of one (k, v) cache per block (as
            returned by an earlier use_cache=True call).
        use_cache: if True, also return the updated per-block cache list.
        start_pos: the ABSOLUTE position of idx's first token in the full
            sequence -- 0 during training/prefill, but the cache's current
            length during incremental decoding (see positional.py)."""
        B, T = idx.shape
        assert start_pos + T <= self.block_size, (
            f"position {start_pos + T} exceeds block_size {self.block_size}"
        )
        x = self.pos_enc.add_to(self.token_emb(idx), start_pos=start_pos)

        if use_cache:
            new_caches = []
            for i, block in enumerate(self.blocks):
                block_cache = kv_cache[i] if kv_cache is not None else None
                x, new_cache = block(x, kv_cache=block_cache, use_cache=True)
                new_caches.append(new_cache)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            return logits, new_caches

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_emb.weight.numel()  # would double-subtract if not tied
        return n
