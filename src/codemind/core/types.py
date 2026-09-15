"""Domain types shared across ingestion, retrieval and the agent graph.

These are deliberately plain dataclasses rather than Pydantic models: they are
created in hot loops during ingestion (hundreds of thousands of instances) and
never cross an HTTP boundary. API-facing models live in `api/v1/schemas.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Language(StrEnum):
    """Programming languages the ingestion pipeline understands."""

    PYTHON = "python"
    JAVA = "java"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"


class Layer(StrEnum):
    """Architectural layer guessed from path and symbol naming conventions.

    Used as a retrieval filter ("only look at controllers") and as a hint to the
    dependency agent. A guess, never a guarantee — `UNKNOWN` is a valid answer
    and far better than a confident wrong label.
    """

    CONTROLLER = "controller"
    SERVICE = "service"
    REPOSITORY = "repository"
    MODEL = "model"
    CONFIG = "config"
    TEST = "test"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class SourceFile:
    """A single source file selected for indexing.

    `relative_path` is always POSIX-style and relative to the repository root,
    so chunk identities stay stable across machines and clones.
    """

    absolute_path: Path
    relative_path: str
    language: Language
    size_bytes: int
    content: str

    @property
    def line_count(self) -> int:
        return self.content.count("\n") + 1


@dataclass(slots=True)
class CodeChunk:
    """One indexable unit of code — normally a function, method or class.

    `context_header` is prepended to `body` before embedding. Without it a chunk
    like `return this.repo.findById(id);` is semantically meaningless and hybrid
    search cannot connect it to a question about `/users/:id`.
    """

    relative_path: str
    language: Language
    symbol_name: str
    kind: str
    start_line: int
    end_line: int
    body: str
    context_header: str = ""
    layer: Layer = Layer.UNKNOWN
    parent_symbol: str | None = None
    imports: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        """Deterministic ID: same code at the same path yields the same ID.

        Content is part of the hash so re-ingesting an unchanged repository is a
        no-op, while an edited function produces a new ID and supersedes the old.
        """
        digest = hashlib.blake2b(
            f"{self.relative_path}:{self.symbol_name}:{self.body}".encode(),
            digest_size=16,
        )
        return digest.hexdigest()

    @property
    def point_uuid(self) -> str:
        """`chunk_id` as a UUID string — Qdrant accepts only ints or UUIDs."""
        h = self.chunk_id
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    @property
    def embedding_text(self) -> str:
        """Exactly what goes to the embedder. Never embed `body` alone."""
        return f"{self.context_header}\n{self.body}" if self.context_header else self.body

    @property
    def citation(self) -> str:
        """`file:line` reference the grounding gate validates answers against."""
        return f"{self.relative_path}:{self.start_line}"
