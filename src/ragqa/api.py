"""A small HTTP API around the same pipeline.

    ragqa serve            # http://127.0.0.1:8000/docs
    curl "http://127.0.0.1:8000/ask?q=how+much+annual+leave+do+I+get"
"""

from __future__ import annotations

from pathlib import Path

from .answer import answer_question
from .chunk import load_corpus
from .index import HybridIndex


def create_app(index_path: Path, corpus_path: Path):
    from fastapi import FastAPI, HTTPException, Query

    index = HybridIndex.load(index_path) if Path(index_path).exists() else HybridIndex(load_corpus(corpus_path))
    app = FastAPI(title="ragqa", version="1.0.0",
                  description="Question answering over a local document collection.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(index.chunks),
                "documents": len({c.source for c in index.chunks})}

    @app.get("/ask")
    def ask(q: str = Query(min_length=2, description="your question"),
            k: int = Query(4, ge=1, le=20)) -> dict:
        if not q.strip():
            raise HTTPException(status_code=400, detail="empty question")
        answer = answer_question(q, index, k=k)
        return {
            "question": answer.question,
            "answer": answer.text,
            "citations": answer.citations,
            "confidence": round(answer.confidence, 3),
            "chunks": [{"citation": h.chunk.citation, "score": round(h.score, 3),
                        "text": h.chunk.text} for h in answer.hits],
        }

    @app.get("/sources")
    def sources() -> dict:
        return {"documents": sorted({c.source for c in index.chunks})}

    return app


def run(index_path: Path, corpus_path: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(index_path, corpus_path), host=host, port=port)
