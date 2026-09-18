"""Hybrid retrieval: BM25 (exact words) + TF-IDF cosine (overlapping vocabulary).

BM25 is strong when the question repeats words from the document ("what is the
leave policy"). Cosine similarity over TF-IDF is more forgiving when the wording
differs. Scores from the two are normalised and blended, which beats either one
alone on a small corpus.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunk import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "to", "and", "is", "are", "was", "were", "be", "in", "on", "for",
        "it", "this", "that", "with", "as", "at", "by", "or", "from", "what", "which", "how",
        "do", "does", "did", "can", "i", "we", "you", "our", "my"}


def stem(word: str) -> str:
    """A deliberately small suffix stripper.

    Without it, a question about a "hotel" never matches a document that says
    "Hotels". A full Porter stemmer is overkill for this corpus and mangles
    words like "policies" into shapes that read badly in an extracted answer.
    """
    if word.endswith(("ss", "us", "is")):  # access, status, analysis
        return word
    for suffix in ("ies", "es", "ing", "ed", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str, *, drop_stopwords: bool = True, stemming: bool = True) -> list[str]:
    tokens = TOKEN.findall(text.lower())
    tokens = [t for t in tokens if not drop_stopwords or t not in STOP]
    return [stem(t) for t in tokens] if stemming else tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float
    cosine: float


class BM25:
    """Okapi BM25 (Robertson & Sparck Jones), written out."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = documents
        self.lengths = np.array([len(d) for d in documents], dtype=float)
        self.avg_length = float(self.lengths.mean()) if len(documents) else 0.0
        self.freqs = [Counter(d) for d in documents]
        df = Counter()
        for doc in documents:
            df.update(set(doc))
        n = len(documents)
        # the +0.5 smoothing keeps idf positive for terms in most documents
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([f[term] for f in self.freqs], dtype=float)
            denom = tf + self.k1 * (1 - self.b + self.b * self.lengths / (self.avg_length or 1))
            out += idf * (tf * (self.k1 + 1)) / np.where(denom == 0, 1, denom)
        return out


def _normalise(values: np.ndarray) -> np.ndarray:
    if not len(values) or values.max() <= 0:
        return np.zeros_like(values)
    return values / values.max()


class HybridIndex:
    def __init__(self, chunks: list[Chunk], *, alpha: float = 0.6):
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.chunks = chunks
        self.alpha = alpha  # weight on BM25; (1 - alpha) on cosine
        texts = [c.searchable for c in chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, question: str, k: int = 4, *, min_score: float = 0.05) -> list[Hit]:
        if not question.strip():
            return []
        bm = self.bm25.scores(tokenize(question))
        cos = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()
        blended = self.alpha * _normalise(bm) + (1 - self.alpha) * _normalise(cos)
        order = np.argsort(blended)[::-1][:k]
        return [Hit(self.chunks[i], float(blended[i]), float(bm[i]), float(cos[i]))
                for i in order if blended[i] >= min_score]

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> HybridIndex:
        with open(path, "rb") as fh:
            index = pickle.load(fh)  # noqa: S301  (only ever loads an index this tool wrote)
        if not isinstance(index, HybridIndex):
            raise TypeError(f"{path} does not contain a HybridIndex")
        return index
