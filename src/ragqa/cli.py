"""ragqa: index a folder of documents, then ask questions about them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .answer import answer_question
from .chunk import load_corpus
from .index import HybridIndex

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "corpus"
DEFAULT_INDEX = ROOT / "index" / "corpus.pkl"


def _index(path: Path, corpus: Path, alpha: float) -> HybridIndex:
    if path.exists():
        return HybridIndex.load(path)
    print(f"{path} not found, indexing {corpus} …", file=sys.stderr)
    index = HybridIndex(load_corpus(corpus), alpha=alpha)
    index.save(path)
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ragqa", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("index", help="build the search index from a folder")
    b.add_argument("corpus", nargs="?", type=Path, default=DEFAULT_CORPUS)
    b.add_argument("-o", "--output", type=Path, default=DEFAULT_INDEX)
    b.add_argument("--max-words", type=int, default=180, help="target chunk size")
    b.add_argument("--alpha", type=float, default=0.6, help="weight of BM25 against cosine")

    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question", nargs="+")
    a.add_argument("-i", "--index", type=Path, default=DEFAULT_INDEX)
    a.add_argument("-c", "--corpus", type=Path, default=DEFAULT_CORPUS)
    a.add_argument("-k", type=int, default=4, help="chunks to retrieve")
    a.add_argument("--alpha", type=float, default=0.6)
    a.add_argument("--json", action="store_true")
    a.add_argument("--show-chunks", action="store_true")

    c = sub.add_parser("chat", help="ask questions in a loop")
    c.add_argument("-i", "--index", type=Path, default=DEFAULT_INDEX)
    c.add_argument("-c", "--corpus", type=Path, default=DEFAULT_CORPUS)

    s = sub.add_parser("serve", help="run the HTTP API (needs the 'api' extra)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("-i", "--index", type=Path, default=DEFAULT_INDEX)
    s.add_argument("-c", "--corpus", type=Path, default=DEFAULT_CORPUS)

    args = ap.parse_args(argv)

    if args.cmd == "index":
        chunks = load_corpus(args.corpus, max_words=args.max_words)
        index = HybridIndex(chunks, alpha=args.alpha)
        index.save(args.output)
        docs = len({c.source for c in chunks})
        print(f"indexed {len(chunks)} chunks from {docs} documents -> {args.output}")
        return 0

    if args.cmd == "serve":
        from .api import run
        run(args.index, args.corpus, host=args.host, port=args.port)
        return 0

    index = _index(args.index, args.corpus, getattr(args, "alpha", 0.6))

    if args.cmd == "chat":
        print("Ask a question, or press Ctrl+D to quit.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question:
                print("\n" + answer_question(question, index).format())

    answer = answer_question(" ".join(args.question), index, k=args.k)
    if args.json:
        print(json.dumps({
            "question": answer.question, "answer": answer.text, "citations": answer.citations,
            "confidence": round(answer.confidence, 3),
            "chunks": [{"citation": h.chunk.citation, "score": round(h.score, 3),
                        "bm25": round(h.bm25, 3), "cosine": round(h.cosine, 3)} for h in answer.hits],
        }, indent=2))
        return 0
    print(answer.format())
    if args.show_chunks:
        print("\nRetrieved chunks:")
        for h in answer.hits:
            print(f"\n  [{h.chunk.citation}]  score {h.score:.3f} (bm25 {h.bm25:.2f}, cosine {h.cosine:.2f})")
            print("  " + h.chunk.text[:220].replace("\n", " ") + ("…" if len(h.chunk.text) > 220 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
