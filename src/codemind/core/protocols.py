"""Structural interfaces that let lower layers depend on higher ones.

`ingestion.pipeline` orchestrates the whole ingest, which means it drives the
embedder, the sparse encoder and the vector store — all of which live in
`retrieval`, a *higher* layer. Importing them directly inverts the dependency
order the import-linter contract enforces, and the contract is right: ingestion
must stay usable without a vector database attached.

Protocols resolve that. The pipeline depends on the shape of a store, not on
`QdrantStore`; the concrete objects are injected by whoever composes the
system (a script, or the FastAPI lifespan). This is the same treatment
`LLMClient` gets in `llm/base.py`, for the same reason.

They are `runtime_checkable` so a test can assert the real implementations still
satisfy them — structural typing otherwise fails silently at the seam when a
signature drifts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from qdrant_client import models

from codemind.core.types import CodeChunk


@runtime_checkable
class DenseEncoder(Protocol):
    """Embeds chunks. Implemented by `retrieval.embedder.Embedder`."""

    async def embed_chunks(self, chunks: Sequence[CodeChunk]) -> list[list[float]]: ...


@runtime_checkable
class SparseEncoder(Protocol):
    """Encodes BM25 vectors. Implemented by `retrieval.sparse.SparseEncoder`."""

    async def encode_documents(self, texts: Sequence[str]) -> list[models.SparseVector]: ...


@runtime_checkable
class ChunkStore(Protocol):
    """Persists chunks. Implemented by `retrieval.qdrant_store.QdrantStore`.

    Only the write path appears here: the pipeline never searches, and a
    narrower protocol is a narrower thing to keep in sync.
    """

    @property
    def collection_name(self) -> str: ...

    async def ensure_collection(self, *, recreate: bool = False) -> bool: ...

    async def delete_by_file(self, relative_path: str) -> None: ...

    async def upsert_chunks(
        self,
        chunks: Sequence[CodeChunk],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[models.SparseVector],
    ) -> int: ...
