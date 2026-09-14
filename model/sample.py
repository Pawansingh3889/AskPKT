"""
Sampling: turning a probability distribution over the vocabulary into
one chosen next token.

Maps to your notes -> Hidden State visualisation -> Softmax / Sampling:
    temperature's effect on the curve, greedy, top-k, top-p, min-p

Written from scratch: the truncation logic (temperature, top-k, top-p)
is hand-implemented, since that logic is the actual concept to learn.
The one primitive we don't reimplement is the final random draw
(torch.multinomial) -- that's "roll a weighted die", not a transformer
concept.
"""

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature reshapes the distribution BEFORE softmax.
    temperature < 1 sharpens it (more confident, less random).
    temperature > 1 flattens it (more random, more surprising).
    temperature -> 0 is the limit where it becomes greedy (argmax)."""
    return logits / temperature


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Always pick the single highest-probability token. Deterministic:
    the same logits always give the same output, no randomness at all."""
    return torch.argmax(logits, dim=-1, keepdim=True)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k highest logits; set everything else to -inf so
    softmax assigns it exactly zero probability."""
    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k, dim=-1)
    threshold = values[..., -1, None]  # the k-th highest value, per row
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: sort tokens by probability, keep the smallest
    prefix whose cumulative probability is >= p, drop the long tail."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)

    # Drop a token only once the cumulative probability BEFORE it has
    # already reached p -- this keeps the token that CROSSES the
    # threshold, rather than excluding it.
    drop = (cumulative - probs) > p
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))

    # Scatter back into the original (unsorted) vocabulary order.
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


def sample(logits: torch.Tensor, temperature: float = 1.0,
           top_k: int = None, top_p: float = None) -> torch.Tensor:
    """The full pipeline for ONE position's logits: temperature -> optional
    top-k -> optional top-p -> softmax -> weighted random draw.
    logits: (..., vocab_size) -> (..., 1) chosen token id."""
    if temperature == 0:
        return greedy_sample(logits)

    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
