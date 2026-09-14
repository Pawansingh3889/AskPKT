"""
Token embeddings: the lookup table that turns each token id into a
trainable vector.

Maps to your notes -> Embeddings -> Token embeddings:
    lookup table, embedding dimension, token ID to vector, input embeddings,
    weight tying, initialisation, learned parameters, frozen vs trainable
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    The embedding "matrix" really is just that: one row per vocabulary
    entry, n_embd columns wide. Looking up a token's embedding is
    literally indexing into this matrix by its id -- no computation,
    just a table read. What makes it learnable is that the read is
    differentiable: gradients flow back into exactly the rows that
    were looked up, and nowhere else.
    """

    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        # The actual parameter table: shape (vocab_size, n_embd).
        # One learnable row per token id. This IS the "input embeddings".
        self.weight = nn.Parameter(torch.empty(vocab_size, n_embd))

        # Initialisation scale: small random values (std=0.02) is the
        # convention GPT-2 and most transformer implementations use --
        # large enough to break symmetry between rows, small enough that
        # variance doesn't blow up once this feeds into deeper layers.
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, seq_len) ints -> (batch, seq_len, n_embd) vectors.

        Fancy indexing IS the lookup table: self.weight[5] gives row 5,
        and indexing with a whole tensor of ids gives one row per id,
        batched automatically.
        """
        return self.weight[token_ids]

    def freeze(self):
        """Frozen vs trainable: stop gradients flowing into this table --
        e.g. if you'd loaded pretrained embeddings and didn't want
        training to disturb them."""
        self.weight.requires_grad_(False)

    def unfreeze(self):
        self.weight.requires_grad_(True)
