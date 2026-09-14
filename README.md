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
- [ ] 1. BPE tokenizer — `tokenizer/bpe.py`
- [ ] 2. Embeddings — `model/embeddings.py`
- [ ] 3. Positional encoding — `model/positional.py`
- [ ] 4. Self-attention — `model/attention.py`
- [ ] 5. Multi-head + FFN + residual + LayerNorm — `model/block.py`
- [ ] 6. Output head + sampling — `model/gpt.py`, `model/sample.py`
- [ ] 7. Training loop — `train.py`
- [ ] 8. KV cache — `generate.py`
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
