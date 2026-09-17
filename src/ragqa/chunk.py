"""Turning documents into retrievable chunks.

Chunking is where most RAG systems are won or lost. Two rules here:

1. **Split on structure first.** Markdown headings mark topic boundaries, so a
   chunk never straddles two sections, and every chunk remembers its heading
   path ("Leave policy > Annual leave"). That heading is prepended to the text,
   because a question usually repeats words from the heading.
2. **Overlap.** Long sections are split into windows that share a sentence, so a
   fact that sits on a window boundary is still retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    text: str
    source: str
    heading: str = ""
    index: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.index}" + (f" ({self.heading})" if self.heading else "")

    @property
    def searchable(self) -> str:
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(text.strip()) if s.strip()]


def chunk_document(text: str, source: str, *, max_words: int = 180, overlap_sentences: int = 1) -> list[Chunk]:
    """Split one document into chunks, keeping the markdown heading path."""
    sections: list[tuple[str, list[str]]] = []
    path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer and any(line.strip() for line in buffer):
            sections.append((" > ".join(path), buffer.copy()))
        buffer.clear()

    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
        else:
            buffer.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        sentences = split_sentences(body) or [body]
        window: list[str] = []
        words = 0
        for sentence in sentences:
            n = len(sentence.split())
            if window and words + n > max_words:
                chunks.append(Chunk(" ".join(window), source, heading, len(chunks)))
                window = window[-overlap_sentences:] if overlap_sentences else []
                words = sum(len(s.split()) for s in window)
            window.append(sentence)
            words += n
        if window:
            chunks.append(Chunk(" ".join(window), source, heading, len(chunks)))
    return chunks


def load_corpus(folder: str | Path, *, patterns: tuple[str, ...] = ("*.md", "*.txt"), **kwargs) -> list[Chunk]:
    """Read every document under `folder` and return all chunks."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"no such folder: {folder}")
    chunks: list[Chunk] = []
    files = sorted(p for pattern in patterns for p in folder.rglob(pattern))
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.extend(chunk_document(text, path.relative_to(folder).as_posix(), **kwargs))
    if not chunks:
        raise ValueError(f"no documents matching {patterns} under {folder}")
    return chunks
