# AskPKT

A decoder-only transformer built from scratch in PyTorch, tokenizer through
fine-tuning and KV cache, no `transformers` library, no pretrained weights,
no `nn.MultiheadAttention`. Sized for an 8GB Apple M1. See README.md for the
full milestone list, hyperparameters, and the scaling-up writeup.

## Architecture (build order, each maps to one file)

```
tokenizer/bpe.py      → BPE tokenizer
model/embeddings.py   → token embeddings
model/positional.py   → positional encoding
model/attention.py    → self-attention (SelfAttention), causal mask
model/block.py         → multi-head attention + FFN + residual + LayerNorm
model/gpt.py            → output head
model/sample.py          → sampling
train.py                  → training loop; hyperparameters are env-var
                            overridable (N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE,
                            BATCH_SIZE, RESUME_FROM, CHECKPOINT_PATH, ...)
generate.py                → KV cache (generate_cached vs generate_naive)
finetune.py                  → fine-tuning
```

`explore.ipynb` and `train_interactive.ipynb` are for hands-on experimentation
against a real trained checkpoint, not part of the core pipeline.

## Dev commands

```bash
source .venv/bin/activate
pip install -r requirements.txt
python train.py                              # fresh training run
RESUME_FROM=checkpoints/gpt_shakespeare.pt python train.py   # resume, split a long run into several
python generate.py                            # sample from a trained checkpoint
```

## Memory constraint that actually bit once, worth knowing before repeating it

This machine has 8GB RAM and runs a browser/editor alongside training. A
long `batch_size=64` run got OOM-killed partway through. The fix that
worked: `RESUME_FROM` to split one long run into several shorter processes,
plus dropping to `batch_size=16`. If a run dies with no Python traceback,
suspect OOM before suspecting the code.

## What NOT to do

- **Don't add the `transformers` library or a pretrained checkpoint.** The
  entire point is that every mechanism is visible in hand-written code.
- **Don't hardcode hyperparameters where an env-var override already
  exists.** `train.py`'s whole design is that a bigger run is a different
  invocation, not a code edit.
- **Don't change the milestone checklist in README.md without updating it.**
