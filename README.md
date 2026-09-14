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
- [ ] 9. Fine-tuning — `finetune.py`

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
