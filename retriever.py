"""TF-IDF passage retriever over Paradise Lost.

Splits the poem into passages, builds a term-frequency index,
and retrieves the most relevant passage for a given query.
Pure Python, no dependencies beyond stdlib.
"""

import math
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Common English words to ignore — keeps the index focused on Milton's vocabulary
STOP_WORDS = frozenset(
    "the and of to in a is that it for with as was on are be at by this "
    "have from or an but not what all were when we there can had has his "
    "her their which will each do how if they my than been its no would "
    "she other into more some him so could them only then did these very "
    "just about over such also may after most too should those now where "
    "who whom thy thee thou hath doth shall thy thine ye hast art wilt "
    "nor ere yet thus hence forth upon unto".split()
)

PASSAGE_LINES = 20
MAX_PASSAGE_LINES = 30


class Retriever:
    def __init__(self, text: str | None = None):
        if text is None:
            text = (DATA_DIR / "paradise_lost.txt").read_text(encoding="utf-8")
        self.passages = self._split_passages(text)
        self.tf_idf, self.vocab = self._build_index(self.passages)

    def _split_passages(self, text: str) -> list[str]:
        """Split into ~20-line passages, breaking at blank lines when possible."""
        lines = text.split("\n")
        passages = []
        current: list[str] = []

        for line in lines:
            current.append(line)
            at_break = line.strip() == ""
            if (len(current) >= PASSAGE_LINES and at_break) or len(current) >= MAX_PASSAGE_LINES:
                passages.append("\n".join(current).strip())
                current = []

        if current:
            passages.append("\n".join(current).strip())

        return [p for p in passages if len(p.strip()) > 30]

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"[a-z']+", text.lower())
        return [w for w in words if w not in STOP_WORDS and len(w) > 1]

    def _build_index(
        self, passages: list[str]
    ) -> tuple[list[dict[str, float]], dict[str, float]]:
        n = len(passages)
        tf: list[dict[str, float]] = []
        df: dict[str, int] = {}

        for passage in passages:
            words = self._tokenize(passage)
            total = len(words) if words else 1
            word_counts: dict[str, int] = {}
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

            tf_doc: dict[str, float] = {}
            for w, c in word_counts.items():
                tf_doc[w] = c / total
                df[w] = df.get(w, 0) + 1
            tf.append(tf_doc)

        idf: dict[str, float] = {}
        for w, count in df.items():
            idf[w] = math.log(n / (1 + count))

        tf_idf: list[dict[str, float]] = []
        for tf_doc in tf:
            scores = {}
            for w, t in tf_doc.items():
                scores[w] = t * idf.get(w, 0)
            tf_idf.append(scores)

        return tf_idf, idf

    def find(self, query: str, top_k: int = 1) -> list[str]:
        """Return the top_k most relevant passages for the query."""
        query_words = self._tokenize(query)
        if not query_words:
            return self.passages[:top_k]

        scores: list[tuple[float, int]] = []
        for i, doc_scores in enumerate(self.tf_idf):
            score = sum(doc_scores.get(w, 0) for w in query_words)
            scores.append((score, i))

        scores.sort(reverse=True)
        return [self.passages[idx] for _, idx in scores[:top_k]]

    def keywords(self, passage_idx: int, top_n: int = 5) -> list[str]:
        """Return the top_n most distinctive words for a passage."""
        if passage_idx >= len(self.tf_idf):
            return []
        scores = self.tf_idf[passage_idx]
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:top_n]]


if __name__ == "__main__":
    r = Retriever()
    print(f"{len(r.passages)} passages indexed")
    print()

    queries = ["darkness", "the fall of man", "love between Adam and Eve", "Satan", "heaven and light"]
    for q in queries:
        results = r.find(q, top_k=1)
        preview = results[0][:150].replace("\n", " / ")
        print(f"  '{q}' -> {preview}...")
        print()
