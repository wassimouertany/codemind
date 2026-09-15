"""Qdrant collection management and chunk persistence.

One collection holds both halves of hybrid retrieval as **named vectors** on
the same point:

    dense  : 768-d cosine, from `jina-embeddings-v2-base-code`
    sparse : BM25 over code-aware tokens

Named vectors rather than two collections is what makes the Query API's
server-side RRF fusion possible in a single round trip — see `hybrid.py`. Two
collections would mean two queries, client-side fusion, and no way to express
"the same chunk scored by both branches".

Memory, because 16 GB is the whole budget:

* `on_disk=True` keeps original float32 vectors in mmapped storage, not RSS.
* int8 scalar quantization keeps a 4x-smaller copy resident (`always_ram=True`)
  and rescores against the on-disk originals. For 100k chunks that is ~75 MB
  resident instead of ~300 MB.
* `on_disk_payload=True` — payloads carry full chunk bodies, which are the
  largest thing here and are only needed for points that actually come back.

**`Modifier.IDF` is not optional.** fastembed's `Qdrant/bm25` emits the
term-frequency half of BM25 only; the inverse-document-frequency half depends
on corpus statistics that live server-side. Without this modifier Qdrant scores
every term as equally rare, and a query containing a ubiquitous token like
`get` is dominated by it. The symptom is subtle — sparse recall that is poor
rather than zero — so it is asserted in the integration test.

Point IDs are UUIDs derived from `CodeChunk.chunk_id`. Qdrant accepts only
unsigned ints or UUIDs, and the chunk ID is a 16-byte blake2b digest, which is
exactly a UUID's width. Re-ingesting an unchanged repository therefore upserts
every point onto itself — the resume path in `scripts/ingest_repo.py` depends
on that being a no-op rather than a duplicate.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import RetrievalError
from codemind.core.types import CodeChunk, Language, Layer

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

INDEXED_PAYLOAD_FIELDS: dict[str, models.PayloadSchemaType] = {
    # Both are exact-match filters ("only this file", "only Java"), so KEYWORD
    # is right and TEXT would be wrong — a tokenizing index would match
    # `OrderService.java` for the term `order`.
    "file_path": models.PayloadSchemaType.KEYWORD,
    "language": models.PayloadSchemaType.KEYWORD,
}

_UUID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def point_id_for(chunk_id: str) -> str:
    """Map a chunk ID to a Qdrant point ID, deterministically.

    The chunk ID is 32 hex characters — a UUID's worth of bytes — so the common
    path is a direct reinterpretation with no information lost. The namespace
    fallback exists only so a future change to `digest_size` degrades into
    different-but-stable IDs rather than a crash mid-ingest.
    """
    try:
        return str(uuid.UUID(hex=chunk_id))
    except ValueError:
        return str(uuid.uuid5(_UUID_NAMESPACE, chunk_id))


class QdrantStore:
    """Owns the collection and the chunk<->point mapping.

    Holds an `AsyncQdrantClient`, so it is created in the FastAPI lifespan (or
    an ingest script's `main`) and closed on shutdown. Search lives in
    `hybrid.py`; this class is storage only.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._settings = resolved
        self._collection = resolved.qdrant_collection
        self._batch_size = resolved.qdrant_upsert_batch
        # An injected client is how tests get `:memory:` without a live server.
        self._client = client or AsyncQdrantClient(
            url=resolved.qdrant_url, timeout=int(resolved.qdrant_timeout)
        )
        self._owns_client = client is None

    @property
    def client(self) -> AsyncQdrantClient:
        """For `hybrid.py`, which issues the Query API call against this collection."""
        return self._client

    @property
    def collection_name(self) -> str:
        return self._collection

    async def close(self) -> None:
        """Only closes a client this store created. An injected one outlives it."""
        if self._owns_client:
            await self._client.close()

    # ------------------------------------------------------------------ #
    # collection lifecycle
    # ------------------------------------------------------------------ #

    async def ensure_collection(self, *, recreate: bool = False) -> bool:
        """Create the collection and its payload indexes if absent. Idempotent.

        Returns True if it created the collection. Safe to call on every boot
        and from `scripts/bootstrap_qdrant.py`; an existing collection is
        checked for a dimension mismatch rather than silently reused, because
        mixing 768-d and 1024-d vectors produces garbage rankings with no error.
        """
        if recreate and await self._client.collection_exists(self._collection):
            logger.warning("deleting collection %s", self._collection)
            await self._client.delete_collection(self._collection)

        if await self._client.collection_exists(self._collection):
            await self._verify_existing_collection()
            await self._ensure_payload_indexes()
            return False

        logger.info(
            "creating collection %s (dim=%d, quantization=%s, on_disk=%s)",
            self._collection,
            self._settings.embedding_dim,
            self._settings.qdrant_quantization,
            self._settings.qdrant_on_disk,
        )
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={DENSE_VECTOR: self._dense_params()},
            sparse_vectors_config={SPARSE_VECTOR: self._sparse_params()},
            # Bodies are the bulk of a payload and are only read for points
            # that come back. Keeping them off the heap is most of the win.
            on_disk_payload=self._settings.qdrant_on_disk,
        )
        await self._ensure_payload_indexes()
        return True

    def _dense_params(self) -> models.VectorParams:
        quantization = (
            models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    # Clips the top and bottom 0.5% before scaling, so a single
                    # outlier dimension cannot flatten the rest of the range.
                    quantile=0.99,
                    # The point of the exercise: the small quantized copy stays
                    # resident and the float32 originals stay on disk.
                    always_ram=True,
                )
            )
            if self._settings.qdrant_quantization
            else None
        )
        return models.VectorParams(
            size=self._settings.embedding_dim,
            distance=models.Distance.COSINE,
            on_disk=self._settings.qdrant_on_disk,
            quantization_config=quantization,
        )

    def _sparse_params(self) -> models.SparseVectorParams:
        return models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=self._settings.qdrant_on_disk),
            # See module docstring: fastembed supplies TF, Qdrant supplies IDF.
            modifier=models.Modifier.IDF,
        )

    async def _verify_existing_collection(self) -> None:
        """Fail loudly if the live collection does not match current settings."""
        info = await self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or DENSE_VECTOR not in vectors:
            raise RetrievalError(
                f"collection {self._collection!r} has no named vector {DENSE_VECTOR!r}; "
                "it predates the hybrid schema — recreate it"
            )
        actual = vectors[DENSE_VECTOR].size
        if actual != self._settings.embedding_dim:
            raise RetrievalError(
                f"collection {self._collection!r} stores {actual}-d vectors but "
                f"EMBEDDING_DIM is {self._settings.embedding_dim}; recreate the collection"
            )
        sparse = info.config.params.sparse_vectors
        if not sparse or SPARSE_VECTOR not in sparse:
            raise RetrievalError(
                f"collection {self._collection!r} has no named vector {SPARSE_VECTOR!r}; "
                "it predates the hybrid schema — recreate it"
            )

    async def _ensure_payload_indexes(self) -> None:
        """Create the filter indexes. Idempotent — Qdrant no-ops on a repeat."""
        for field, schema in INDEXED_PAYLOAD_FIELDS.items():
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=schema,
                wait=True,
            )

    # ------------------------------------------------------------------ #
    # writing
    # ------------------------------------------------------------------ #

    async def upsert_chunks(
        self,
        chunks: Sequence[CodeChunk],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[models.SparseVector],
    ) -> int:
        """Upsert chunks with both vectors. Returns the number of points written.

        The three sequences are zipped positionally — the encoders guarantee
        one vector per input, including empties, precisely so this alignment
        holds. A length mismatch means an encoder dropped a result, which would
        attach every subsequent vector to the wrong chunk, so it raises rather
        than truncating.
        """
        if not chunks:
            return 0
        if not len(chunks) == len(dense_vectors) == len(sparse_vectors):
            raise RetrievalError(
                f"vector/chunk misalignment: {len(chunks)} chunks, "
                f"{len(dense_vectors)} dense, {len(sparse_vectors)} sparse"
            )

        points = [
            models.PointStruct(
                id=point_id_for(chunk.chunk_id),
                vector={DENSE_VECTOR: list(dense), SPARSE_VECTOR: sparse},
                payload=payload_of(chunk),
            )
            for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True)
        ]

        for start in range(0, len(points), self._batch_size):
            batch = points[start : start + self._batch_size]
            await self._client.upsert(self._collection, points=batch, wait=True)
            logger.debug("upserted %d points (%d/%d)", len(batch), start + len(batch), len(points))
        return len(points)

    async def delete_by_file(self, relative_path: str) -> None:
        """Drop every chunk from one file — the unit re-ingestion replaces.

        Needed because a deleted or shrunken function leaves orphan points that
        keep matching queries and citing lines that no longer exist.
        """
        await self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path", match=models.MatchValue(value=relative_path)
                        )
                    ]
                )
            ),
            wait=True,
        )

    # ------------------------------------------------------------------ #
    # reading
    # ------------------------------------------------------------------ #

    async def count(self) -> int:
        result = await self._client.count(self._collection, exact=True)
        return result.count

    async def get_chunk(self, chunk_id: str) -> CodeChunk | None:
        """Fetch one chunk by its content-addressed ID."""
        points = await self._client.retrieve(
            self._collection, ids=[point_id_for(chunk_id)], with_payload=True
        )
        if not points:
            return None
        return chunk_from_payload(points[0].payload)


# --------------------------------------------------------------------------- #
# payload mapping
# --------------------------------------------------------------------------- #


def payload_of(chunk: CodeChunk) -> dict[str, Any]:
    """Serialise a chunk into a Qdrant payload.

    The whole chunk is stored, body included, so a result can be rendered and
    cited without touching the filesystem — the working tree may have moved on
    since ingestion, and the grounding gate is what reconciles the two.

    `file_path` rather than `relative_path`: it is the indexed filter key and
    the name used everywhere downstream, including the `file:line` citation.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "file_path": chunk.relative_path,
        "language": chunk.language.value,
        "symbol_name": chunk.symbol_name,
        "kind": chunk.kind,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "body": chunk.body,
        "context_header": chunk.context_header,
        "layer": chunk.layer.value,
        "parent_symbol": chunk.parent_symbol,
        "imports": chunk.imports,
    }


def chunk_from_payload(payload: dict[str, Any] | None) -> CodeChunk:
    """Rebuild a chunk from a Qdrant payload.

    Unknown `language` or `layer` values are tolerated — a collection written
    by an older build should degrade to `Layer.UNKNOWN`, not crash a live query.
    An unparseable language is fatal, since nothing downstream can render a
    chunk whose grammar is unknown.
    """
    if not payload:
        raise RetrievalError("Qdrant point has no payload; was it written without one?")
    try:
        language = Language(payload["language"])
    except (KeyError, ValueError) as exc:
        raise RetrievalError(f"payload has unusable language {payload.get('language')!r}") from exc

    try:
        layer = Layer(payload.get("layer", Layer.UNKNOWN.value))
    except ValueError:
        layer = Layer.UNKNOWN

    try:
        return CodeChunk(
            relative_path=payload["file_path"],
            language=language,
            symbol_name=payload["symbol_name"],
            kind=payload["kind"],
            start_line=int(payload["start_line"]),
            end_line=int(payload["end_line"]),
            body=payload["body"],
            context_header=payload.get("context_header", ""),
            layer=layer,
            parent_symbol=payload.get("parent_symbol"),
            imports=list(payload.get("imports") or []),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RetrievalError(f"malformed chunk payload: {exc}") from exc
