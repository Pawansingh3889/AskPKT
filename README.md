# AskPKT — TinyGPT from scratch

A decoder-only transformer language model, built term-by-term from a study-notes roadmap
(tokenization → embeddings → positional encoding → transformer internals → hidden
states/sampling → KV cache → training → fine-tuning). No `transformers`, no pretrained
weights, no `nn.MultiheadAttention` — every piece is hand-written so each concept in the
notes maps to real code.

Sized for an 8 GB Apple M1 (MPS backend, ~1–1.5M param model, trains in minutes).

## Setup
```
source .venv/bin/activate
pip install -r requirements.txt
```

## Milestones
- [x] 1. BPE tokenizer — `tokenizer/bpe.py`
- [x] 2. Embeddings — `model/embeddings.py`
- [x] 3. Positional encoding — `model/positional.py`
- [x] 4. Self-attention — `model/attention.py`
- [x] 5. Multi-head + FFN + residual + LayerNorm — `model/block.py`
- [x] 6. Output head + sampling — `model/gpt.py`, `model/sample.py`
- [x] 7. Training loop — `train.py`
- [x] 8. KV cache — `generate.py`
- [x] 9. Fine-tuning — `finetune.py`

## Hyperparameters
| param | value |
|---|---|
| vocab_size | ~800–1024 (learned) |
| n_embd | 128 |
| n_head | 4 |
| n_layer | 4 |
| block_size | 128 |
| batch_size | 32–64 |

## Data
`data/tinyshakespeare.txt` — Tiny Shakespeare, ~1.1MB, public domain.

## Scaling up
`checkpoints/gpt_shakespeare_big.pt` — a second, larger model
(`n_embd=192, n_head=6, n_layer=5, block_size=160`, 2,418,432 params,
2.6x the original) trained for 8000 iterations, reaching val loss
4.4889 (perplexity ~89) vs the original's 4.52 (~92) — a real, modest
improvement from more capacity + more training.

## Continued training, same size
Resumed the original 922,880-param model (`RESUME_FROM`, optimizer state
intact) via `train_interactive.ipynb` and kept training it past where it
was originally stopped: **iter 6600, val loss 4.4824** — better than the
original's documented 4.52, and within noise of the bigger model's 4.4889
(above), using the *original's* parameter count. More training on the
small model closed most of the capacity gap, at least this far; not a
claim that capacity stops mattering, just that this run hadn't converged
yet when the earlier number was written down. Checkpoint itself isn't
committed (`checkpoints/` is gitignored, binary weights don't belong in
git); the evidence is the training-loop output saved in
`train_interactive.ipynb`.

`explore.ipynb` was run against this updated checkpoint too: real
next-token probabilities after `"ROMEO:"` (`but` 7.5%, `I` 6.1%, `the`
4.8%, ...), temperature-sampled output at 0.3 vs. greedy, and nearest-
neighbour embeddings for subword tokens (`'lov'`'s nearest neighbours
are `'liv'`, `'do'`, `'lif'`, `'tru'`, a genuinely interesting cluster
for a byte-level tokenizer that's never seen a dictionary).

Getting there required working around real 8GB-RAM constraints: a
single long `batch_size=64` run got OOM-killed by the OS partway
through (this machine also runs a browser, editor, etc. sharing the
same memory). Fix was two-fold: `train.py` now supports
`RESUME_FROM=<checkpoint>` so a run can be split into several shorter
PROCESSES instead of one long-running one, and dropping to
`batch_size=16` reduced peak memory enough to finish reliably. All
hyperparameters (`N_EMBD`, `N_HEAD`, `N_LAYER`, `BLOCK_SIZE`,
`BATCH_SIZE`, `CHECKPOINT_PATH`, `RESUME_FROM`, etc.) are overridable
as environment variables -- see the comment at the top of `train.py`.

## Exploring the trained model
`explore.ipynb` — a Jupyter notebook (open in VS Code with the Python +
Jupyter extensions, select the `.venv` kernel) for hands-on experimentation
against the real trained checkpoint: your own prompts, temperature
comparisons, real next-token probabilities, and nearest-neighbour
embeddings now that they're actually trained.

## Training interactively
`train_interactive.ipynb` — train in controllable chunks instead of one
fixed `train.py` run: re-run one cell to train `ITERS_PER_CALL` more
iterations at a time, check the loss curve and a live sample anytime,
save a checkpoint whenever you want. Two modes: `MODE = "resume"` keeps
training the existing checkpoint (optimizer momentum included), or
`MODE = "fresh"` starts a new model with whatever architecture you set
in the config cell (`N_EMBD`, `N_HEAD`, `N_LAYER`, `BLOCK_SIZE`,
`DROPOUT`, `LEARNING_RATE`, `WEIGHT_DECAY`, `WARMUP_ITERS`,
`SCHEDULE_HORIZON`, `BATCH_SIZE`).
