# RAG Document QA

[![CI](https://github.com/umer-78/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/umer-78/rag-document-qa/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![No API key](https://img.shields.io/badge/API%20key-not%20required-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

Ask questions about your own documents and get an answer **with citations**.
Retrieval-augmented generation without the generation bill: chunking that
respects document structure, hybrid BM25 + TF-IDF retrieval, and an extractive
answerer that can only quote what is in your files.

It runs offline, needs no API key, and a language model can be plugged in later
without touching the retrieval half.

```text
$ ragqa ask "how many annual leave days do I get" --show-chunks
Full-time employees get 24 days of paid annual leave a year, plus public
holidays. Sick days do not come out of annual leave. Leave accrues monthly at 2 days per completed month.

Sources:
  [1] employee-handbook.md#2 (Northwind Labs Employee Handbook > Annual leave)
  [2] employee-handbook.md#3 (Northwind Labs Employee Handbook > Sick leave)
  [3] employee-handbook.md#4 (Northwind Labs Employee Handbook > Parental leave)
  [4] engineering-onboarding.md#6 (Engineering Onboarding > Incidents)

Retrieved chunks:

  [employee-handbook.md#2 (Northwind Labs Employee Handbook > Annual leave)]  score 0.990 (bm25 8.66, cosine 0.35)
  Full-time employees get 24 days of paid annual leave a year, plus public holidays. Leave accrues monthly at 2 days per completed month. Up to 5 unused days carry into the next year and must be used before 31 March; anyth…

  [employee-handbook.md#3 (Northwind Labs Employee Handbook > Sick leave)]  score 0.966 (bm25 8.17, cosine 0.36)
  10 paid sick days a year, no medical certificate needed for the first 3 consecutive days. Sick days do not come out of annual leave.

  [employee-handbook.md#4 (Northwind Labs Employee Handbook > Parental leave)]  score 0.390 (bm25 4.48, cosine 0.07)
  Primary carers get 16 weeks at full pay, secondary carers 4 weeks at full pay. Both can be taken in up to three blocks within the first 18 months.

```

## How it works

```
documents ─▶ chunk (by heading, with overlap) ─▶ index (BM25 + TF-IDF)
                                                     │
                            question ────────────────┤
                                                     ▼
                                   top-k chunks ─▶ answer + citations
```

1. **Chunking** (`chunk.py`). Splits on markdown headings first, so a chunk never
   crosses a topic boundary, and each chunk carries its heading path
   ("Expenses and Travel > Travel"). Long sections become overlapping windows, so
   a fact on a boundary is still findable.
2. **Retrieval** (`index.py`). Okapi BM25 written out in full, blended with
   TF-IDF cosine similarity. BM25 is better when the question reuses the
   document's words; cosine is better when it does not. A small suffix stemmer
   makes "hotel" match "Hotels".
3. **Answering** (`answer.py`). Sentences are ranked by overlap with the question
   and by the score of the chunk they came from; weak ones are dropped rather
   than padding the answer. Every sentence comes from the corpus — a test asserts
   that nothing is invented.

## Quick start

```bash
git clone https://github.com/umer-78/rag-document-qa.git
cd rag-document-qa
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ragqa index                                   # index the sample corpus
ragqa ask "how much annual leave do I get"
ragqa ask "what is the hotel limit abroad" --show-chunks
ragqa ask "how long is the on-call rotation" --json
ragqa chat                                    # ask repeatedly
```

Point it at your own notes:

```bash
ragqa index ~/notes -o index/notes.pkl
ragqa ask -i index/notes.pkl "what did we decide about the database"
```

## HTTP API

```bash
pip install -e ".[api]"
ragqa serve            # http://127.0.0.1:8000/docs
curl "http://127.0.0.1:8000/ask?q=how%20much%20annual%20leave&k=3"
```

`/health` reports the index size, `/sources` lists the indexed documents, and
`/ask` returns the answer, the citations and the retrieved chunks.

## Plugging in a language model

`answer_question` takes any `generator(question, chunks) -> str`:

```python
from ragqa import HybridIndex, answer_question, load_corpus

index = HybridIndex(load_corpus("corpus"))

def with_llm(question, hits):
    context = "\n\n".join(f"[{h.chunk.citation}]\n{h.chunk.text}" for h in hits)
    return my_model(f"Answer only from the context.\n\n{context}\n\nQuestion: {question}")

print(answer_question("what is the notice period", index, generator=with_llm).format())
```

The retrieval, chunking and citation code stays the same. That is the part worth
owning: a weak retriever makes even a strong model guess.

## Sample corpus

`corpus/` is the handbook of **Northwind Labs, a company that does not exist**:
five documents covering leave, security, engineering onboarding, expenses and
support. Replace them with your own files and re-index.

## Tests

```bash
ruff check .
python -m pytest -q     # 23 tests
```

Including: chunks never cross a heading, windows overlap, BM25 matches the
formula computed by hand, retrieval finds the right document for five different
questions, unknown questions return "not found" rather than a guess, every
answered sentence exists in the corpus, and the HTTP API responds correctly.

## Limits

- Retrieval is lexical. A question phrased with entirely different vocabulary
  ("time off" against "annual leave") can miss; embeddings would help, at the
  cost of a model download.
- The extractive answerer quotes sentences, so answers read like the source.
  That is a deliberate trade for never inventing anything.
- Indexes are pickled, so only load ones you created.

## License

[MIT](LICENSE)
