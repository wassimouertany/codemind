"""Exception hierarchy. Never `raise Exception` and never `except Exception`."""

from __future__ import annotations


class CodeMindError(Exception):
    """Base class for every error this system raises deliberately."""


class IngestionError(CodeMindError):
    """Raised when a repository cannot be loaded, parsed or indexed."""


class RepositoryNotFoundError(IngestionError):
    """The path or URL does not point at a usable git repository."""


class UnsupportedLanguageError(IngestionError):
    """No grammar is registered for this file extension."""


class ChunkingError(IngestionError):
    """tree-sitter produced a tree the chunker could not walk."""


class RetrievalError(CodeMindError):
    """Hybrid search, reranking or the vector store failed."""


class LLMError(CodeMindError):
    """The inference backend was unreachable or returned an unusable response."""


class GroundingError(CodeMindError):
    """An answer cited evidence that does not exist in the retrieved context."""
