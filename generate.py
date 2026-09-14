"""
Text generation, two ways: recompute-everything vs KV cache.

Maps to your notes -> KV cache:
    prefill vs decode, autoregressive generation, sequence length scaling,
    time to first token, throughput versus latency

Run:
    python3 generate.py
"""

import time

import torch

from model.gpt import GPT
from model.sample import sample
from tokenizer.bpe import BPETokenizer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def generate_naive(model, tok, prompt: str, max_new_tokens: int,
                    temperature: float = 0.0, top_k=None, device=None):
    """The way every earlier milestone generated text: at EVERY step, feed
    the model the whole sequence so far and recompute attention over all
    of it from scratch, even though almost all of that work is identical
    to what the previous step just did."""
    ids = tok.encode(prompt)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.block_size:]], device=device)
        logits = model(x)
        next_id = sample(logits[0, -1], temperature=temperature, top_k=top_k).item()
        ids.append(next_id)
    return ids


@torch.no_grad()
def generate_cached(model, tok, prompt: str, max_new_tokens: int,
                     temperature: float = 0.0, top_k=None, device=None):
    """KV-cache generation: PREFILL the prompt once (one forward pass,
    builds the initial cache), then DECODE one token at a time -- each
    step only computes Q/K/V for the ONE new token and reuses every
    earlier position's K/V straight out of the cache."""
    ids = tok.encode(prompt)

    # --- prefill: one forward pass over the whole prompt, seeds the cache ---
    x = torch.tensor([ids], device=device)
    logits, kv_cache = model(x, use_cache=True, start_pos=0)
    pos = x.shape[1]  # absolute position the NEXT generated token will sit at

    next_id = sample(logits[0, -1], temperature=temperature, top_k=top_k).item()
    ids.append(next_id)

    # --- decode: one new token at a time, reusing the cache ---
    for _ in range(max_new_tokens - 1):
        x = torch.tensor([[ids[-1]]], device=device)  # ONLY the newest token
        logits, kv_cache = model(x, kv_cache=kv_cache, use_cache=True, start_pos=pos)
        pos += 1
        next_id = sample(logits[0, -1], temperature=temperature, top_k=top_k).item()
        ids.append(next_id)

    return ids


def main():
    device = get_device()
    print(f"device: {device}")

    tok = BPETokenizer()
    tok.load("tokenizer/vocab.json")

    ckpt = torch.load("checkpoints/gpt_shakespeare.pt", map_location=device)
    model = GPT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint from iteration {ckpt['iter']}, {model.num_params():,} params")

    prompt = "ROMEO:"
    max_new_tokens = 100

    # --- correctness first: greedy (temperature=0, deterministic) generation
    # from BOTH paths must produce the IDENTICAL token sequence. If caching
    # changes the output at all, it's wrong, no matter how fast it is. ---
    ids_naive = generate_naive(model, tok, prompt, max_new_tokens, temperature=0, device=device)
    ids_cached = generate_cached(model, tok, prompt, max_new_tokens, temperature=0, device=device)
    print()
    print(f"naive  output: {tok.decode(ids_naive)!r}")
    print(f"cached output: {tok.decode(ids_cached)!r}")
    print(f"IDENTICAL token sequences: {ids_naive == ids_cached}")

    # --- now benchmark speed, same settings, several repeats for stability ---
    def bench(fn, repeats=3):
        times = []
        for _ in range(repeats):
            t0 = time.time()
            fn(model, tok, prompt, max_new_tokens, temperature=0.8, top_k=40, device=device)
            times.append(time.time() - t0)
        return min(times)

    t_naive = bench(generate_naive)
    t_cached = bench(generate_cached)
    print()
    print(f"naive:  {t_naive:.3f}s for {max_new_tokens} tokens "
          f"({max_new_tokens / t_naive:.1f} tok/s)")
    print(f"cached: {t_cached:.3f}s for {max_new_tokens} tokens "
          f"({max_new_tokens / t_cached:.1f} tok/s)")
    print(f"speedup: {t_naive / t_cached:.2f}x")


if __name__ == "__main__":
    main()
