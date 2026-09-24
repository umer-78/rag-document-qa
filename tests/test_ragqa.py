import math
from pathlib import Path

import pytest

from ragqa import HybridIndex, answer_question, chunk_document, load_corpus
from ragqa.answer import NO_ANSWER, extractive_answer
from ragqa.chunk import split_sentences
from ragqa.cli import main
from ragqa.index import BM25, stem, tokenize

CORPUS = Path(__file__).resolve().parents[1] / "corpus"

DOC = """# Handbook

## Leave

Employees get 24 days of paid annual leave. Leave accrues monthly.

## Notice

Notice is 30 days in the first year.
"""


# --------------------------------------------------------------------- chunking
def test_chunks_follow_headings():
    chunks = chunk_document(DOC, "handbook.md")
    assert [c.heading for c in chunks] == ["Handbook > Leave", "Handbook > Notice"]
    assert "24 days" in chunks[0].text
    assert "24 days" not in chunks[1].text, "a chunk must not cross a heading"


def test_citation_and_searchable_text():
    chunk = chunk_document(DOC, "handbook.md")[0]
    assert chunk.citation == "handbook.md#0 (Handbook > Leave)"
    assert chunk.searchable.startswith("Handbook > Leave")


def test_long_sections_are_split_with_overlap():
    body = " ".join(f"Sentence number {i} about policy." for i in range(60))
    chunks = chunk_document(f"# T\n\n## S\n\n{body}", "big.md", max_words=40, overlap_sentences=1)
    assert len(chunks) > 3
    # Pairs of neighbours, so the offset list is one shorter by construction.
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert split_sentences(a.text)[-1] == split_sentences(b.text)[0], "windows should overlap"


def test_sentence_splitting():
    assert split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]


def test_load_corpus_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        load_corpus(tmp_path / "empty")


# -------------------------------------------------------------------- retrieval
def test_stemmer_keeps_words_readable():
    assert stem("hotels") == "hotel"
    assert stem("policies") == "policy"
    assert stem("access") == "access"   # too short to strip
    assert tokenize("Hotels in Lahore") == ["hotel", "lahore"]


def test_bm25_matches_the_formula():
    docs = [["a", "b"], ["a", "a", "c"], ["d"]]
    bm = BM25(docs, k1=1.5, b=0.75)
    scores = bm.scores(["a"])
    assert scores[2] == 0, "a document without the term scores zero"
    assert scores[1] > scores[0], "more occurrences score higher"
    n, df = 3, 2
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    tf, length, avg = 1, 2, (2 + 3 + 1) / 3
    expected = idf * (tf * 2.5) / (tf + 1.5 * (1 - 0.75 + 0.75 * length / avg))
    assert scores[0] == pytest.approx(expected)


def test_unknown_words_score_zero():
    bm = BM25([["a"], ["b"]])
    assert bm.scores(["zzz"]).tolist() == [0.0, 0.0]


@pytest.fixture(scope="module")
def index():
    return HybridIndex(load_corpus(CORPUS))


def test_index_rejects_bad_input():
    with pytest.raises(ValueError):
        HybridIndex([])
    with pytest.raises(ValueError):
        HybridIndex(chunk_document(DOC, "d.md"), alpha=2)


@pytest.mark.parametrize("question,expected_source", [
    ("how many annual leave days do I get", "employee-handbook.md"),
    ("how quickly are critical vulnerabilities patched", "security-policy.md"),
    ("how long is the on-call rotation", "engineering-onboarding.md"),
    ("what is the daily allowance abroad", "expenses-and-travel.md"),
    ("when can support approve a refund", "support-runbook.md"),
])
def test_retrieval_finds_the_right_document(index, question, expected_source):
    hits = index.search(question, k=3)
    assert hits, "expected at least one hit"
    assert hits[0].chunk.source == expected_source


def test_search_handles_empty_and_unknown_questions(index):
    assert index.search("   ") == []
    assert index.search("xyzzy plugh frobnicate", k=3) == []


# ---------------------------------------------------------------------- answers
def test_answer_quotes_the_document_and_cites_it(index):
    answer = answer_question("how many days of annual leave do employees get", index)
    assert "24 days" in answer.text
    assert any("employee-handbook.md" in c for c in answer.citations)
    assert 0 < answer.confidence <= 1
    assert "Sources:" in answer.format()


def test_answer_is_extractive_every_sentence_comes_from_the_corpus(index):
    answer = answer_question("what happens when a secret is exposed", index)
    corpus_text = " ".join(c.text for c in index.chunks)
    for sentence in split_sentences(answer.text):
        assert sentence in corpus_text, f"invented sentence: {sentence!r}"


def test_no_answer_when_nothing_matches(index):
    answer = answer_question("what is the airspeed velocity of an unladen swallow", index)
    assert answer.text == NO_ANSWER
    assert answer.citations == []


def test_extractive_answer_without_hits():
    assert extractive_answer("q", []) == NO_ANSWER


def test_generator_can_be_swapped(index):
    def fake_llm(question, hits):
        return f"[llm] {len(hits)} chunks for {question}"

    answer = answer_question("annual leave", index, generator=fake_llm)
    assert answer.text.startswith("[llm]")
    assert answer.citations, "citations still come from retrieval"


# -------------------------------------------------------------------------- app
def test_index_round_trip(tmp_path, index):
    path = tmp_path / "i.pkl"
    index.save(path)
    loaded = HybridIndex.load(path)
    assert len(loaded.chunks) == len(index.chunks)
    assert loaded.search("annual leave")[0].chunk.citation == index.search("annual leave")[0].chunk.citation
    (tmp_path / "bad.pkl").write_bytes(b"\x80\x04K\x01.")
    with pytest.raises(TypeError):
        HybridIndex.load(tmp_path / "bad.pkl")


def test_cli_index_and_ask(tmp_path, capsys):
    idx = tmp_path / "i.pkl"
    assert main(["index", str(CORPUS), "-o", str(idx)]) == 0
    assert "indexed" in capsys.readouterr().out
    assert main(["ask", "-i", str(idx), "-c", str(CORPUS), "how", "much", "annual", "leave"]) == 0
    assert "24 days" in capsys.readouterr().out
    assert main(["ask", "-i", str(idx), "-c", str(CORPUS), "--json", "on-call", "rotation"]) == 0
    assert '"citations"' in capsys.readouterr().out


def test_http_api():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ragqa.api import create_app

    client = TestClient(create_app(Path("does-not-exist.pkl"), CORPUS))
    assert client.get("/health").json()["documents"] == 6
    home = client.get("/", follow_redirects=False)
    assert home.status_code == 307 and home.headers["location"] == "/docs"
    body = client.get("/ask", params={"q": "how many annual leave days"}).json()
    assert "24 days" in body["answer"]
    assert body["citations"]
    assert client.get("/ask", params={"q": "x"}).status_code == 422
    assert "support-runbook.md" in client.get("/sources").json()["documents"]
    del fastapi
