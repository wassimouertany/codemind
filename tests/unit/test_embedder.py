"""Tests for dense embedding.

The two that matter beyond shape-checking:

* `test_embed_chunks_uses_the_context_header` — the invariant that keeps a
  one-line `return repo.find(id)` chunk retrievable at all.
* `test_dimension_mismatch_fails_at_construction` — proves the guard fires at
  load rather than surfacing as a Qdrant upsert error mid-ingest.
"""

from __future__ import annotations

import pytest

from codemind.core.config import Settings
from codemind.core.exceptions import RetrievalError
from codemind.core.types import CodeChunk, Language, Layer
from codemind.retrieval.embedder import Embedder, get_embedder

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """One model load for the whole module — construction is the expensive part."""
    return Embedder()


def make_chunk(body: str, *, header: str = "", symbol: str = "findById") -> CodeChunk:
    return CodeChunk(
        relative_path="src/main/java/com/demo/repository/OrderRepository.java",
        language=Language.JAVA,
        symbol_name=symbol,
        kind="method",
        start_line=10,
        end_line=12,
        body=body,
        context_header=header,
        layer=Layer.REPOSITORY,
    )


# --------------------------------------------------------------------------- #
# shape and type contract
# --------------------------------------------------------------------------- #


def test_returns_one_vector_per_input_at_configured_dimension(embedder: Embedder) -> None:
    vectors = embedder.embed_texts_sync(["class OrderService {}", "def handler(): pass"])
    assert len(vectors) == 2
    assert all(len(vector) == embedder.dimension for vector in vectors)


def test_empty_input_returns_empty_list(embedder: Embedder) -> None:
    assert embedder.embed_texts_sync([]) == []


def test_vectors_are_plain_floats(embedder: Embedder) -> None:
    """numpy floats would pass here but be rejected by Qdrant's strict models."""
    vector = embedder.embed_texts_sync(["placeOrder"])[0]
    assert all(type(value) is float for value in vector)


def test_embed_query_returns_one_vector(embedder: Embedder) -> None:
    assert len(embedder.embed_query_sync("why does POST /orders return 500")) == (
        embedder.dimension
    )


# --------------------------------------------------------------------------- #
# the context header invariant
# --------------------------------------------------------------------------- #


def test_embed_chunks_uses_the_context_header(embedder: Embedder) -> None:
    """A chunk must embed as header + body, never body alone.

    Two chunks with identical bodies but different headers have to land in
    different places, or every thin delegating method in the repo collapses
    onto one point and retrieval cannot tell them apart.
    """
    body = "return this.repo.findById(id);"
    orders = make_chunk(body, header="OrderRepository | com.demo.repository | findById")
    payments = make_chunk(body, header="PaymentRepository | com.demo.billing | findById")

    vectors = embedder.embed_chunks_sync([orders, payments])
    assert vectors[0] != vectors[1]


def test_chunk_without_a_header_still_embeds(embedder: Embedder) -> None:
    """`embedding_text` falls back to the body; ingestion must not crash on it."""
    vector = embedder.embed_chunks_sync([make_chunk("return 1;")])[0]
    assert len(vector) == embedder.dimension


def test_header_matches_embedding_text_path(embedder: Embedder) -> None:
    """`embed_chunks` must be exactly `embed_texts` over `embedding_text`."""
    chunk = make_chunk("return this.repo.findById(id);", header="OrderRepository | findById")
    assert embedder.embed_chunks_sync([chunk]) == embedder.embed_texts_sync([chunk.embedding_text])


# --------------------------------------------------------------------------- #
# semantic sanity
# --------------------------------------------------------------------------- #


def test_related_code_is_closer_than_unrelated_code(embedder: Embedder) -> None:
    """A weak but real check that the model is a code model and loaded correctly."""
    query, payment, parser = embedder.embed_texts_sync(
        [
            "how are payment timeouts handled",
            "class PaymentTimeoutException extends RuntimeException {}",
            "def parse_csv_row(line: str) -> list[str]: return line.split(',')",
        ]
    )

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm = sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5
        return dot / norm

    assert cosine(query, payment) > cosine(query, parser)


# --------------------------------------------------------------------------- #
# configuration guards
# --------------------------------------------------------------------------- #


def test_dimension_mismatch_fails_at_construction() -> None:
    """Fail at load, not at upsert an hour into an ingest."""
    settings = Settings(embedding_dim=1024)  # jina v2 base code is 768
    with pytest.raises(RetrievalError, match="EMBEDDING_DIM"):
        Embedder(settings)


def test_device_comes_from_settings() -> None:
    """Never a hardcoded `.cuda()`. The 4 GB card belongs to the LLM while serving."""
    assert Embedder(Settings(embedding_device="cpu")).device == "cpu"


# --------------------------------------------------------------------------- #
# async surface and lifetime
# --------------------------------------------------------------------------- #


async def test_async_matches_sync(embedder: Embedder) -> None:
    texts = ["OrderController placeOrder", "PaymentTimeoutException"]
    assert await embedder.embed_texts(texts) == embedder.embed_texts_sync(texts)
    assert await embedder.embed_query("place order") == embedder.embed_query_sync("place order")


def test_get_embedder_is_a_singleton() -> None:
    """Heavy models load once. A second load would double RSS for nothing."""
    assert get_embedder() is get_embedder()
