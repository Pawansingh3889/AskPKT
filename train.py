"""
Training loop: teach the GPT model to predict the next token.

Maps to your notes -> Training:
    pretraining, next-token prediction, cross-entropy loss,
    backpropagation, gradient descent, learning rate schedules

Run:
    python3 train.py                  # full run
    MAX_ITERS=20 python3 train.py     # quick smoke test
"""

import math
import os
import time

import torch
import torch.nn.functional as F

from model.gpt import GPT
from model.sample import sample
from tokenizer.bpe import BPETokenizer

# ---- hyperparameters, sized for an 8GB Apple M1 (see README) ----
# All overridable via environment variables, so a bigger run doesn't
# require editing this file -- e.g.:
#   N_EMBD=256 N_HEAD=8 N_LAYER=6 BLOCK_SIZE=192 MAX_ITERS=8000 \
#   CHECKPOINT_PATH=checkpoints/gpt_shakespeare_big.pt python3 train.py
N_EMBD = int(os.environ.get("N_EMBD", 128))
N_HEAD = int(os.environ.get("N_HEAD", 4))
N_LAYER = int(os.environ.get("N_LAYER", 4))
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 128))
DROPOUT = float(os.environ.get("DROPOUT", 0.1))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))

MAX_ITERS = int(os.environ.get("MAX_ITERS", 2000))
EVAL_INTERVAL = int(os.environ.get("EVAL_INTERVAL", 250))
EVAL_ITERS = 50

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 3e-4))
WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", 200))
MIN_LR = float(os.environ.get("MIN_LR", 3e-5))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/gpt_shakespeare.pt")
PROMPT = "ROMEO:"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr(it: int) -> float:
    """Learning rate schedule: linear warmup, then cosine decay to MIN_LR.
    Warmup avoids a destructive early update while the (still near-random)
    weights are at their most unstable; cosine decay lets learning slow
    down smoothly as the model converges, instead of a hard cliff."""
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    if it > MAX_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    """A batch of random (input, target) windows. target is input shifted
    by one position -- this IS "next-token prediction": at every
    position, the label is simply whatever token comes right after."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    """Average loss over several random batches, in eval mode (dropout
    off), for a more stable read than a single noisy training-batch loss."""
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def generate_sample(model, tok, device, max_new_tokens=60):
    model.eval()
    ids = tok.encode(PROMPT)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-BLOCK_SIZE:]], device=device)
        logits = model(x)
        next_id = sample(logits[0, -1], temperature=0.8, top_k=40).item()
        ids.append(next_id)
    model.train()
    return tok.decode(ids)


def main():
    device = get_device()
    print(f"device: {device}")

    tok = BPETokenizer()
    tok.load("tokenizer/vocab.json")

    with open("data/tinyshakespeare.txt") as f:
        text = f.read()
    print("encoding corpus with the trained BPE tokenizer...")
    ids = tok.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus: {len(data):,} tokens ({n:,} train / {len(data) - n:,} val)")

    model = GPT(vocab_size=tok.vocab_size, n_embd=N_EMBD, n_head=N_HEAD,
                n_layer=N_LAYER, block_size=BLOCK_SIZE, dropout=DROPOUT).to(device)
    print(f"model: {model.num_params():,} params on {device}")

    # cross-entropy loss + AdamW: the standard pretraining recipe.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # RESUME_FROM: continue an existing checkpoint instead of starting fresh.
    # Also lets a long run be split into several shorter PROCESSES instead of
    # one long-running one -- each fresh launch starts with clean memory,
    # which matters on an 8GB machine where a single very long MPS process
    # can accumulate memory pressure over thousands of iterations.
    start_iter = 0
    resume_from = os.environ.get("RESUME_FROM")
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"resumed model AND optimizer from {resume_from} at iter {ckpt['iter']}")
        else:
            print(f"resumed model from {resume_from} at iter {ckpt['iter']} "
                  f"(no saved optimizer state -- starting fresh momentum)")
        start_iter = ckpt["iter"]

    os.makedirs("checkpoints", exist_ok=True)
    t0 = time.time()

    for it in range(start_iter, MAX_ITERS + 1):
        lr = get_lr(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
            losses = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - t0
            print(f"iter {it:5d} | train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.0f}s")
            sample_text = generate_sample(model, tok, device)
            print(f"  sample: {sample_text!r}")

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),  # needed to resume training cleanly
                "iter": it,
                "config": dict(vocab_size=tok.vocab_size, n_embd=N_EMBD, n_head=N_HEAD,
                                n_layer=N_LAYER, block_size=BLOCK_SIZE, dropout=DROPOUT),
            }, CHECKPOINT_PATH)

        # --- the actual training step: forward, loss, backward, update ---
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()          # backpropagation
        optimizer.step()         # gradient descent update

    print(f"training done in {time.time() - t0:.0f}s, checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
