"""
Fine-tuning: continue training a pretrained checkpoint on a narrow
dataset, and see its behaviour shift toward it.

Maps to your notes -> Fine-tuning:
    supervised fine-tuning, dataset curation, overfitting
Also touches -> Training:
    catastrophic forgetting -- does specializing on ROMEO's lines make
    the model WORSE at other characters? We check, rather than assume.

Run:
    python3 finetune.py
"""

import re
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from model.gpt import GPT
from model.sample import sample
from tokenizer.bpe import BPETokenizer

BASE_CHECKPOINT = "checkpoints/gpt_shakespeare.pt"
FINETUNED_CHECKPOINT = "checkpoints/gpt_romeo_finetuned.pt"
CHARACTER_DATA_PATH = "data/character_romeo.txt"
CHARACTER_NAME = "ROMEO"

FT_ITERS = 400
FT_LEARNING_RATE = 5e-5   # noticeably lower than pretraining's 3e-4 -- fine-tuning
                          # nudges an already-capable model, it shouldn't need
                          # (and risks damaging it with) the original large steps
FT_BATCH_SIZE = 32
EVAL_INTERVAL = 100


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_character_lines(corpus_path: str, character: str) -> str:
    """Dataset curation: pull out just ONE character's dialogue from the
    full play text, using the "SPEAKER NAME:" header convention Tiny
    Shakespeare uses."""
    speaker_re = re.compile(r"^([A-Z][A-Za-z ]{1,30}):$")
    with open(corpus_path) as f:
        lines = f.read().split("\n")

    speeches = defaultdict(list)
    current = None
    for line in lines:
        m = speaker_re.match(line.strip())
        if m:
            current = m.group(1)
        elif current and line.strip():
            speeches[current].append(line)
        elif not line.strip():
            current = None

    return "\n".join(speeches[character])


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = GPT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


@torch.no_grad()
def generate(model, tok, prompt: str, device, max_new_tokens=80, temperature=0.8, top_k=40):
    model.eval()
    ids = tok.encode(prompt)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.block_size:]], device=device)
        logits = model(x)
        next_id = sample(logits[0, -1], temperature=temperature, top_k=top_k).item()
        ids.append(next_id)
    model.train()
    return tok.decode(ids)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def main():
    device = get_device()
    print(f"device: {device}")

    tok = BPETokenizer()
    tok.load("tokenizer/vocab.json")

    # --- dataset curation ---
    character_text = extract_character_lines("data/tinyshakespeare.txt", CHARACTER_NAME)
    with open(CHARACTER_DATA_PATH, "w") as f:
        f.write(character_text)
    print(f"{CHARACTER_NAME}'s dialogue: {len(character_text):,} chars, "
          f"saved to {CHARACTER_DATA_PATH}")

    ids = torch.tensor(tok.encode(character_text), dtype=torch.long)
    n = int(0.9 * len(ids))
    train_data, val_data = ids[:n], ids[n:]
    print(f"encoded: {len(ids):,} tokens ({n:,} train / {len(ids) - n:,} val) "
          f"-- compare to the 443,727-token FULL corpus this model pretrained on")

    # --- load the pretrained (base) model ---
    model, base_ckpt = load_model(BASE_CHECKPOINT, device)
    block_size = model.block_size
    print(f"loaded base checkpoint from iteration {base_ckpt['iter']}, "
          f"{model.num_params():,} params")

    # --- BEFORE: what the base model sounds like, pre-fine-tuning ---
    print("\n=== BEFORE fine-tuning ===")
    before_romeo = generate(model, tok, "ROMEO:", device)
    before_other = generate(model, tok, "KING RICHARD III:", device)
    print(f"ROMEO:            {before_romeo!r}")
    print(f"KING RICHARD III: {before_other!r}")

    # --- fine-tune: continue training, on ONLY Romeo's lines, at a low LR ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LEARNING_RATE, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for it in range(FT_ITERS + 1):
        if it % EVAL_INTERVAL == 0 or it == FT_ITERS:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(20):
                    x, y = get_batch(val_data, block_size, FT_BATCH_SIZE, device)
                    logits = model(x)
                    losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item())
            val_loss = sum(losses) / len(losses)
            model.train()
            print(f"  iter {it:4d} | val loss {val_loss:.4f} | {time.time()-t0:.0f}s")

        x, y = get_batch(train_data, block_size, FT_BATCH_SIZE, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": base_ckpt["iter"] + FT_ITERS,
        "config": base_ckpt["config"],
        "finetuned_on": CHARACTER_NAME,
    }, FINETUNED_CHECKPOINT)
    print(f"\nsaved fine-tuned checkpoint to {FINETUNED_CHECKPOINT}")

    # --- AFTER: same prompts, fine-tuned model ---
    print("\n=== AFTER fine-tuning on ROMEO's lines ===")
    after_romeo = generate(model, tok, "ROMEO:", device)
    after_other = generate(model, tok, "KING RICHARD III:", device)
    print(f"ROMEO:            {after_romeo!r}")
    print(f"KING RICHARD III: {after_other!r}")

    print("\n=== Side by side ===")
    print(f"ROMEO:            before: {before_romeo!r}")
    print(f"ROMEO:            after:  {after_romeo!r}")
    print(f"KING RICHARD III: before: {before_other!r}")
    print(f"KING RICHARD III: after:  {after_other!r}")


if __name__ == "__main__":
    main()
