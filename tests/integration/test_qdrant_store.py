"""Tests for the Qdrant collection schema and chunk persistence.

Runs against `AsyncQdrantClient(":memory:")`, which round-trips collection
config faithfully — so the schema assertions (IDF modifier, int8 quantization,
on_disk) are real. Payload indexes are the exception: local mode accepts and
ignores them, so `test_payload_indexes_exist_on_a_live_server` checks those
against a real Qdrant and skips when none is reachable.

The end-to-end test ingests `tests/fixtures/sample_repo/` with the real
encoders, as CLAUDE.md requires for retrieval changes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient, models

from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import RetrievalError
from codemind.core.types import CodeChunk, Language, Layer
from codemind.ingestion.ast_chunker import AstChunker
from codemind.ingestion.repo_loader import RepositoryLoader
from codemind.retrieval.embedder import get_embedder
from codemind.retrieval.qdrant_store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    QdrantStore,
    chunk_from_payload,
    payload_of,
    point_id_for,
)
from codemind.retrieval.sparse import get_sparse_encoder

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


def settings_with(**overrides: object) -> Settings:
    return Settings(qdrant_collection="test_chunks", qdrant_upsert_batch=3, **overrides)  # type: ignore[arg-type]


@pytest.fixture
async def store() -> AsyncIterator[QdrantStore]:
    client = AsyncQdrantClient(":memory:")
    yield QdrantStore(settings_with(), client=client)
    await client.close()


def make_chunk(symbol: str = "placeOrder", *, path: str = "src/OrderService.java") -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language=Language.JAVA,
        symbol_name=symbol,
        kind="method",
        start_line=10,
        end_line=20,
        body=f"public void {symbol}() {{ repo.save(order); }}",
        context_header=f"{path} | com.demo.service | OrderService | void {symbol}()",
        layer=Layer.SERVICE,
        parent_symbol="OrderService",
        imports=["com.demo.repository.OrderRepository"],
    )


def fake_dense(n: int, dim: int = 768) -> list[list[float]]:
    return [[float(i + 1)] + [0.0] * (dim - 1) for i in range(n)]


def fake_sparse(n: int) -> list[models.SparseVector]:
    return [models.SparseVector(indices=[i + 1], values=[1.0]) for i in range(n)]


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


async def test_collection_has_named_dense_and_sparse_vectors(store: QdrantStore) -> None:
    await store.ensure_collection()
    params = (await store.client.get_collection(store.collection_name)).config.params

    assert isinstance(params.vectors, dict)
    dense = params.vectors[DENSE_VECTOR]
    assert dense.size == 768
    assert dense.distance is models.Distance.COSINE
    assert params.sparse_vectors is not None
    assert SPARSE_VECTOR in params.sparse_vectors


async def test_sparse_vector_uses_server_side_idf(store: QdrantStore) -> None:
    """fastembed's BM25 supplies TF only. Without IDF every term weighs the same."""
    await store.ensure_collection()
    params = (await store.client.get_collection(store.collection_name)).config.params
    assert params.sparse_vectors is not None
    assert params.sparse_vectors[SPARSE_VECTOR].modifier is models.Modifier.IDF


async def test_dense_vectors_are_int8_quantized_and_on_disk(store: QdrantStore) -> None:
    """The RAM budget: quantized copy resident, float32 originals mmapped."""
    await store.ensure_collection()
    params = (await store.client.get_collection(store.collection_name)).config.params
    assert isinstance(params.vectors, dict)
    dense = params.vectors[DENSE_VECTOR]

    assert dense.on_disk is True
    assert isinstance(dense.quantization_config, models.ScalarQuantization)
    assert dense.quantization_config.scalar.type is models.ScalarType.INT8
    assert dense.quantization_config.scalar.always_ram is True


async def test_quantization_and_on_disk_follow_settings() -> None:
    client = AsyncQdrantClient(":memory:")
    store = QdrantStore(
        settings_with(qdrant_quantization=False, qdrant_on_disk=False), client=client
    )
    await store.ensure_collection()
    params = (await client.get_collection(store.collection_name)).config.params
    assert isinstance(params.vectors, dict)
    assert params.vectors[DENSE_VECTOR].quantization_config is None
    assert not params.vectors[DENSE_VECTOR].on_disk
    await client.close()


# --------------------------------------------------------------------------- #
# idempotency and guards
# --------------------------------------------------------------------------- #


async def test_ensure_collection_is_idempotent_and_preserves_data(store: QdrantStore) -> None:
    assert await store.ensure_collection() is True
    await store.upsert_chunks([make_chunk()], fake_dense(1), fake_sparse(1))

    assert await store.ensure_collection() is False
    assert await store.count() == 1


async def test_recreate_drops_existing_points(store: QdrantStore) -> None:
    await store.ensure_collection()
    await store.upsert_chunks([make_chunk()], fake_dense(1), fake_sparse(1))

    assert await store.ensure_collection(recreate=True) is True
    assert await store.count() == 0


async def test_dimension_mismatch_with_existing_collection_fails_loudly() -> None:
    """Mixing 768-d and 1024-d vectors ranks garbage silently. Refuse instead."""
    client = AsyncQdrantClient(":memory:")
    await QdrantStore(settings_with(), client=client).ensure_collection()

    mismatched = QdrantStore(settings_with(embedding_dim=1024), client=client)
    with pytest.raises(RetrievalError, match="EMBEDDING_DIM"):
        await mismatched.ensure_collection()
    await client.close()


async def test_pre_hybrid_collection_is_rejected() -> None:
    """A single unnamed vector collection cannot serve the Query API prefetches."""
    client = AsyncQdrantClient(":memory:")
    await client.create_collection(
        "test_chunks",
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )
    with pytest.raises(RetrievalError, match="hybrid schema"):
        await QdrantStore(settings_with(), client=client).ensure_collection()
    await client.close()


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #


async def test_upsert_batches_across_the_configured_size(store: QdrantStore) -> None:
    """Batch size is 3 here; 7 chunks forces three calls, including a short tail."""
    await store.ensure_collection()
    chunks = [make_chunk(f"method{i}") for i in range(7)]
    assert await store.upsert_chunks(chunks, fake_dense(7), fake_sparse(7)) == 7
    assert await store.count() == 7


async def test_reupserting_unchanged_chunks_does_not_duplicate(store: QdrantStore) -> None:
    """Content-addressed IDs: a resumed ingest must be a no-op, not a doubling."""
    await store.ensure_collection()
    chunks = [make_chunk("a"), make_chunk("b")]
    await store.upsert_chunks(chunks, fake_dense(2), fake_sparse(2))
    await store.upsert_chunks(chunks, fake_dense(2), fake_sparse(2))
    assert await store.count() == 2


async def test_misaligned_vectors_raise_instead_of_truncating(store: QdrantStore) -> None:
    """One dropped vector would attach every later one to the wrong chunk."""
    await store.ensure_collection()
    with pytest.raises(RetrievalError, match="misalignment"):
        await store.upsert_chunks([make_chunk("a"), make_chunk("b")], fake_dense(1), fake_sparse(2))


async def test_empty_upsert_is_a_no_op(store: QdrantStore) -> None:
    await store.ensure_collection()
    assert await store.upsert_chunks([], [], []) == 0


async def test_empty_sparse_vector_is_accepted(store: QdrantStore) -> None:
    """An all-keyword chunk encodes to an empty sparse vector. It must still index."""
    await store.ensure_collection()
    empty = models.SparseVector(indices=[], values=[])
    assert await store.upsert_chunks([make_chunk()], fake_dense(1), [empty]) == 1


async def test_delete_by_file_removes_only_that_file(store: QdrantStore) -> None:
    await store.ensure_collection()
    chunks = [
        make_chunk("a", path="src/OrderService.java"),
        make_chunk("b", path="src/OrderService.java"),
        make_chunk("c", path="src/PaymentService.java"),
    ]
    await store.upsert_chunks(chunks, fake_dense(3), fake_sparse(3))

    await store.delete_by_file("src/OrderService.java")
    assert await store.count() == 1
    assert await store.get_chunk(chunks[2].chunk_id) is not None


# --------------------------------------------------------------------------- #
# payload mapping
# --------------------------------------------------------------------------- #


def test_payload_round_trips_every_field() -> None:
    original = make_chunk()
    restored = chunk_from_payload(payload_of(original))
    assert restored == original
    assert restored.chunk_id == original.chunk_id


def test_payload_uses_file_path_as_the_indexed_key() -> None:
    payload = payload_of(make_chunk())
    assert payload["file_path"] == "src/OrderService.java"
    assert payload["language"] == "java"
    assert "relative_path" not in payload


def test_unknown_layer_degrades_instead_of_crashing() -> None:
    """A collection written by an older build must not break live queries."""
    payload = payload_of(make_chunk()) | {"layer": "gateway"}
    assert chunk_from_payload(payload).layer is Layer.UNKNOWN


def test_unknown_language_is_fatal() -> None:
    with pytest.raises(RetrievalError, match="language"):
        chunk_from_payload(payload_of(make_chunk()) | {"language": "cobol"})


def test_missing_payload_is_fatal() -> None:
    with pytest.raises(RetrievalError):
        chunk_from_payload(None)


def test_point_id_is_a_deterministic_uuid() -> None:
    chunk_id = make_chunk().chunk_id
    assert point_id_for(chunk_id) == point_id_for(chunk_id)
    assert len(point_id_for(chunk_id)) == 36
    assert point_id_for("not-hex") == point_id_for("not-hex")


# --------------------------------------------------------------------------- #
# end to end on the fixture repo, real encoders
# --------------------------------------------------------------------------- #


@pytest.mark.slow
async def test_fixture_repo_indexes_and_sparse_finds_exact_symbol(store: QdrantStore) -> None:
    """Load -> chunk -> encode both ways -> upsert -> query each branch.

    The sparse assertion is the early warning for failure mode #1: if index-time
    and query-time tokenization diverge, `OrderRepository` stops coming back.
    """
    chunks = AstChunker().chunk_all(RepositoryLoader(FIXTURE).collect())
    assert chunks

    embedder, sparse = get_embedder(), get_sparse_encoder()
    dense_vectors = await embedder.embed_chunks(chunks)
    sparse_vectors = await sparse.encode_documents([c.embedding_text for c in chunks])

    await store.ensure_collection()
    assert await store.upsert_chunks(chunks, dense_vectors, sparse_vectors) == len(chunks)
    assert await store.count() == len(chunks)

    # An exact method name must come back at rank 1. `findByCustomerId` is
    # deliberately distinctive: `findById` exists in both OrderService and the
    # Python repo, so it cannot tell a working index from a lucky one.
    for method in ("findByCustomerId", "placeOrder"):
        response = await store.client.query_points(
            store.collection_name,
            query=await sparse.encode_query(method),
            using=SPARSE_VECTOR,
            limit=3,
            with_payload=True,
        )
        assert response.points, "sparse branch returned nothing — check tokenizer symmetry"
        assert chunk_from_payload(response.points[0].payload).symbol_name == method

    dense_response = await store.client.query_points(
        store.collection_name,
        query=await embedder.embed_query("where are orders saved to the database"),
        using=DENSE_VECTOR,
        limit=5,
    )
    assert dense_response.points


# --------------------------------------------------------------------------- #
# live server only
# --------------------------------------------------------------------------- #


async def test_payload_indexes_exist_on_a_live_server() -> None:
    """Local mode ignores payload indexes, so this is the only real check."""
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url, timeout=2)
    try:
        await client.get_collections()
    except Exception:  # noqa: BLE001 — any connection failure means "no server here"
        await client.close()
        pytest.skip(f"no Qdrant reachable at {settings.qdrant_url}")

    store = QdrantStore(Settings(qdrant_collection="codemind_test_indexes"), client=client)
    try:
        await store.ensure_collection(recreate=True)
        schema = (await client.get_collection(store.collection_name)).payload_schema
        assert schema["file_path"].data_type is models.PayloadSchemaType.KEYWORD
        assert schema["language"].data_type is models.PayloadSchemaType.KEYWORD
    finally:
        await client.delete_collection(store.collection_name)
        await client.close()
