"""Turning retrieved chunks into an answer.

The default generator is **extractive**: it picks the sentences that overlap the
question most, and every sentence keeps the citation of the chunk it came from.
Nothing is invented, so the answer can always be checked against the source.

`answer_question(..., generator=my_llm)` swaps in any callable
`(question, chunks) -> str` if you would rather have a language model write the
prose. The retrieval half of the pipeline does not change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .chunk import split_sentences
from .index import Hit, HybridIndex, tokenize

NO_ANSWER = "I could not find that in these documents."


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str]
    hits: list[Hit]
    confidence: float

    def format(self, show_sources: bool = True) -> str:
        out = self.text
        if show_sources and self.citations:
            out += "\n\nSources:\n" + "\n".join(f"  [{i + 1}] {c}" for i, c in enumerate(self.citations))
        return out


def _sentence_scores(question: str, sentences: list[str]) -> list[float]:
    q = set(tokenize(question))
    scores = []
    for s in sentences:
        words = tokenize(s)
        if not words:
            scores.append(0.0)
            continue
        overlap = sum(1 for w in set(words) if w in q)
        # favour sentences that cover the question without padding it out
        scores.append(overlap / (len(q) or 1) + 0.25 * overlap / len(set(words)))
    return scores


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3,
                      relative_cutoff: float = 0.55) -> str:
    """Pick the sentences that answer the question, best first.

    A sentence is scored on its own overlap with the question and on how well the
    chunk it came from scored, so a strong sentence in a weak chunk does not
    outrank the passage the retriever actually liked. Sentences well below the
    best one are dropped instead of padding the answer out to `max_sentences`.
    """
    scored: list[tuple[float, str]] = []
    for hit in hits:
        sentences = split_sentences(hit.chunk.text)
        for sentence, score in zip(sentences, _sentence_scores(question, sentences), strict=True):
            if score > 0:
                # heading words count too: "Hotels" lives under "Travel"
                heading_bonus = 0.15 if set(tokenize(hit.chunk.heading)) & set(tokenize(question)) else 0.0
                scored.append(((score + heading_bonus) * (0.5 + hit.score), sentence))
    if not scored:
        return hits[0].chunk.text[:400] if hits else NO_ANSWER
    scored.sort(key=lambda t: t[0], reverse=True)
    best = scored[0][0]
    chosen, seen = [], set()
    for score, sentence in scored:
        if score < best * relative_cutoff:
            break
        key = sentence.lower()[:60]
        if key in seen:
            continue
        seen.add(key)
        chosen.append(sentence)
        if len(chosen) >= max_sentences:
            break
    return " ".join(chosen)


def answer_question(question: str, index: HybridIndex, *, k: int = 4, max_sentences: int = 3,
                    generator: Callable[[str, list[Hit]], str] | None = None) -> Answer:
    hits = index.search(question, k=k)
    if not hits:
        return Answer(question, NO_ANSWER, [], [], 0.0)
    text = generator(question, hits) if generator else extractive_answer(question, hits, max_sentences)
    citations = list(dict.fromkeys(h.chunk.citation for h in hits))
    return Answer(question, text, citations, hits, hits[0].score)
