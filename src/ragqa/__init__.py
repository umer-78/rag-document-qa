"""Retrieval-augmented question answering over local documents."""

from .chunk import Chunk, chunk_document, load_corpus
from .index import HybridIndex
from .answer import Answer, answer_question

__all__ = ["Answer", "Chunk", "HybridIndex", "answer_question", "chunk_document", "load_corpus"]
__version__ = "1.0.0"
