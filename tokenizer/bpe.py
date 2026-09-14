"""
Byte Pair Encoding (BPE) tokenizer, built from scratch.

Maps to your notes -> Tokenization -> Algorithms:
    merge rules, greedy longest-match, pre-tokenization, vocabulary building

How it works:
    1. Start with the raw UTF-8 bytes of the text as the base vocabulary (0-255).
       This is "byte-level BPE" (what GPT-2 uses) -- operating on bytes rather
       than characters means we can tokenize ANY text, including emoji and
       non-English scripts, without ever hitting an unknown character.
    2. Count how often every adjacent PAIR of tokens occurs in the text.
    3. Merge the single MOST FREQUENT pair into one new token.
    4. Repeat until we reach the target vocab size.

This is the "build up" side of your notes (BPE), as opposed to unigram's
"carve away" side.
"""

import json
from collections import Counter


class BPETokenizer:
    def __init__(self):
        # id -> bytes. Starts as the 256 raw byte values (the base vocabulary).
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        # (id1, id2) -> new_id. The learned merge rules, IN THE ORDER LEARNED.
        # Order matters: early merges are more fundamental, later merges build on them.
        self.merges = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """Learn merge rules from `text` until vocab reaches `vocab_size`."""
        assert vocab_size >= 256, "vocab_size must cover the base 256 bytes"
        num_merges = vocab_size - 256

        # Pre-tokenization step: turn the raw text into a list of byte-ids.
        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            pair_counts = self._get_pair_counts(ids)
            if not pair_counts:
                break  # no more pairs left to merge (text fully collapsed)

            # The merge rule: pick the single most frequent adjacent pair.
            top_pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i

            ids = self._merge(ids, top_pair, new_id)
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            if verbose:
                print(f"merge {i + 1}/{num_merges}: {top_pair} -> {new_id} "
                      f"({self.vocab[new_id]!r}), {pair_counts[top_pair]} occurrences")

        return ids  # the training text, fully encoded with the final vocab

    @staticmethod
    def _get_pair_counts(ids):
        counts = Counter()
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
        return counts

    @staticmethod
    def _merge(ids, pair, new_id):
        """Greedy longest-match: scan left to right, replace every occurrence
        of `pair` with `new_id`, without overlapping matches."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def encode(self, text: str) -> list:
        """Text -> token ids, by repeatedly applying learned merges in the
        order they were learned (earliest merge = applied first)."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pair_counts = self._get_pair_counts(ids)
            # Of the pairs present in this text, apply whichever was learned
            # EARLIEST during training (lowest merge index = most fundamental).
            candidate = min(pair_counts, key=lambda p: self.merges.get(p, float("inf")))
            if candidate not in self.merges:
                break  # none of the remaining pairs have a merge rule
            ids = self._merge(ids, candidate, self.merges[candidate])
        return ids

    def decode(self, ids: list) -> str:
        """Token ids -> text. Round-trip fidelity: decode(encode(x)) == x."""
        byte_seq = b"".join(self.vocab[idx] for idx in ids)
        return byte_seq.decode("utf-8", errors="replace")

    def save(self, path: str):
        data = {f"{a},{b}": idx for (a, b), idx in self.merges.items()}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair_str, idx in data.items():  # dict preserves insertion order
            a, b = map(int, pair_str.split(","))
            self.merges[(a, b)] = idx
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
