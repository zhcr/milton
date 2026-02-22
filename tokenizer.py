"""BPE tokenizer trained on Paradise Lost.

Implements byte-pair encoding from scratch. The vocabulary is built
entirely from Milton's text. Spaces are encoded as part of tokens
(prefixed to words, like GPT-2) so decode is just concatenation.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

SPECIAL_TOKENS = {
    "<|pad|>": 0,
    "<|eos|>": 1,
    "<|user|>": 2,
    "<|milton|>": 3,
}


class Tokenizer:
    def __init__(self):
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self.special_tokens = dict(SPECIAL_TOKENS)
        self._word_cache: dict[str, list[int]] = {}

    def train(self, text: str, vocab_size: int = 4096):
        """Train BPE on the given text."""
        # Pre-tokenize: split into words with leading whitespace attached
        # e.g. "the fruit\nOf" -> ["the", " fruit", "\n", "Of"]
        chunks = re.findall(r"[^\S\n]*\S+|\n", text)
        word_freqs: dict[tuple[str, ...], int] = Counter()
        for chunk in chunks:
            word = tuple(chunk)
            word_freqs[word] += 1

        chars = set()
        for word in word_freqs:
            for ch in word:
                chars.add(ch)

        self.vocab = dict(SPECIAL_TOKENS)
        for ch in sorted(chars):
            self.vocab[ch] = len(self.vocab)

        num_merges = vocab_size - len(self.vocab)
        self.merges = []

        print(f"Base vocabulary: {len(self.vocab)} tokens ({len(chars)} unique chars)")
        print(f"Unique words: {len(word_freqs)}")
        print(f"Training {num_merges} merges...")

        # Build initial pair frequency index
        pair_freqs: Counter[tuple[str, str]] = Counter()
        pair_to_words: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)

        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += freq
                pair_to_words[pair].add(word)

        for merge_i in range(num_merges):
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.__getitem__)
            if pair_freqs[best_pair] < 2:
                break

            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)

            affected_words = list(pair_to_words.pop(best_pair, set()))
            del pair_freqs[best_pair]

            for word in affected_words:
                if word not in word_freqs:
                    continue
                freq = word_freqs[word]

                for i in range(len(word) - 1):
                    p = (word[i], word[i + 1])
                    if p != best_pair:
                        pair_freqs[p] -= freq
                        if p in pair_to_words:
                            pair_to_words[p].discard(word)

                new_word = _apply_merge(word, best_pair)
                del word_freqs[word]
                word_freqs[new_word] = word_freqs.get(new_word, 0) + freq

                for i in range(len(new_word) - 1):
                    p = (new_word[i], new_word[i + 1])
                    pair_freqs[p] += freq
                    pair_to_words[p].add(new_word)

            if (merge_i + 1) % 500 == 0:
                pair_freqs = Counter({p: f for p, f in pair_freqs.items() if f > 0})
                print(f"  {merge_i + 1}/{num_merges} merges")

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Final vocabulary: {len(self.vocab)} tokens")

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids."""
        parts = re.split(r"(<\|[a-z]+\|>)", text)
        token_ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                token_ids.append(self.special_tokens[part])
            else:
                token_ids.extend(self._encode_chunk(part))
        return token_ids

    def _encode_chunk(self, text: str) -> list[int]:
        chunks = re.findall(r"[^\S\n]*\S+|\n", text)
        token_ids = []
        for chunk in chunks:
            token_ids.extend(self._encode_word(chunk))
        return token_ids

    def _encode_word(self, chunk: str) -> list[int]:
        # Check cache first — Paradise Lost has ~17K unique words
        if chunk in self._word_cache:
            return self._word_cache[chunk]

        word = tuple(chunk)
        for merge_pair in self.merges:
            word = _apply_merge(word, merge_pair)

        ids = []
        for token in word:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                for ch in token:
                    if ch in self.vocab:
                        ids.append(self.vocab[ch])

        self._word_cache[chunk] = ids
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text. Just concatenate — spaces are in the tokens."""
        parts = []
        for idx in ids:
            if idx in self.inv_vocab:
                tok = self.inv_vocab[idx]
                if tok not in self.special_tokens:
                    parts.append(tok)
        return "".join(parts)

    def save(self, path: Path):
        data = {
            "merges": self.merges,
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls()
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab = data["vocab"]
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        tok.special_tokens = data["special_tokens"]
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.special_tokens["<|pad|>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<|eos|>"]

    @property
    def user_id(self) -> int:
        return self.special_tokens["<|user|>"]

    @property
    def milton_id(self) -> int:
        return self.special_tokens["<|milton|>"]


def _apply_merge(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(pair[0] + pair[1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def main():
    text_path = DATA_DIR / "paradise_lost.txt"
    if not text_path.exists():
        print("Run get_data.py first.")
        return

    text = text_path.read_text(encoding="utf-8")
    tok = Tokenizer()
    tok.train(text, vocab_size=4096)

    save_path = DATA_DIR / "tokenizer.json"
    tok.save(save_path)
    print(f"Saved tokenizer to {save_path}")

    sample = text[:200]
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"\nRound-trip test:")
    print(f"  Original:  {sample!r}")
    print(f"  Decoded:   {decoded!r}")
    print(f"  Tokens:    {len(encoded)}")
    print(f"  Match:     {decoded == sample}")

    all_tokens = tok.encode(text)
    print(f"\nParadise Lost: {len(all_tokens)} tokens ({len(text)} chars)")
    print(f"Compression ratio: {len(text) / len(all_tokens):.1f} chars/token")


if __name__ == "__main__":
    main()
