"""Hybrid retrieval against a live Qdrant with the sample repo ingested.

Live rather than `:memory:` because this is the Day 5 gate: the Query API's
prefetch + server-side RRF path is what production runs, and the payload
indexes that back filtered search do nothing in local mode.

The whole module skips when no Qdrant is reachable at QDRANT_URL. It writes
only to its own throwaway collection and never touches `codemind_chunks`.

`test_exact_method_name_is_rank_one` is the load-bearing sanity check from the
plan. If it fails, suspect tokenizer asymmetry before anything else.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, models

from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import RetrievalError
from codemind.core.types import Language
from codemind.ingestion.ast_chunker import AstChunker
from codemind.ingestion.repo_loader import RepositoryLoader
from codemind.retrieval.embedder import Embedder, get_embedder
from codemind.retrieval.hybrid import HybridRetriever
from codemind.retrieval.qdrant_store import QdrantStore
from codemind.retrieval.schemas import RetrievalSource
from codemind.retrieval.sparse import SparseEncoder, get_sparse_encoder

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"
COLLECTION = "codemind_test_hybrid"

pytestmark = pytest.mark.slow


def _qdrant_reachable(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def hybrid_settings(**overrides: object) -> Settings:
    return Settings(qdrant_collection=COLLECTION, **overrides)  # type: ignore[arg-type]


@pytest.fixture(scope="module", autouse=True)
def ingested_fixture() -> Iterator[int]:
    """Ingest the sample repo once per module, and drop the collection after.

    Uses `asyncio.run` with its own client so no async client outlives the
    event loop it was created on; each test opens a fresh one.
    """
    url = get_settings().qdrant_url
    if not _qdrant_reachable(url):
        pytest.skip(f"no Qdrant reachable at {url}")

    chunks = AstChunker().chunk_all(RepositoryLoader(FIXTURE).collect())
    embedder, sparse = get_embedder(), get_sparse_encoder()

    async def ingest() -> None:
        store = QdrantStore(hybrid_settings())
        try:
            await store.ensure_collection(recreate=True)
            await store.upsert_chunks(
                chunks,
                await embedder.embed_chunks(chunks),
                await sparse.encode_documents([c.embedding_text for c in chunks]),
            )
        finally:
            await store.close()

    async def drop() -> None:
        client = AsyncQdrantClient(url=url)
        try:
            await client.delete_collection(COLLECTION)
        finally:
            await client.close()

    asyncio.run(ingest())
    yield len(chunks)
    asyncio.run(drop())


def make_retriever(store: QdrantStore, settings: Settings) -> HybridRetriever:
    embedder: Embedder = get_embedder()
    sparse: SparseEncoder = get_sparse_encoder()
    return HybridRetriever(store, embedder, sparse, settings)


@pytest.fixture
async def store() -> AsyncIterator[QdrantStore]:
    store = QdrantStore(hybrid_settings())
    yield store
    await store.close()


@pytest.fixture
def retriever(store: QdrantStore) -> HybridRetriever:
    return make_retriever(store, hybrid_settings())


# --------------------------------------------------------------------------- #
# the Day 5 sanity check
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["findByCustomerId", "placeOrder", "processPayment"])
async def test_exact_method_name_is_rank_one(retriever: HybridRetriever, method: str) -> None:
    result = await retriever.search(method)
    assert result.chunks, "hybrid search returned nothing"
    top = result.chunks[0]
    assert top.chunk.symbol_name == method, (
        f"expected {method} at rank 1, got {[c.chunk.symbol_name for c in result.top(5)]}"
    )
    assert top.rank == 1


async def test_spaced_query_finds_camel_case_method(retriever: HybridRetriever) -> None:
    """ "find by customer id" shares no raw token with `findByCustomerId`.

    Only the code-aware tokenizer, applied identically at index and query time,
    connects them.
    """
    result = await retriever.search("find by customer id")

    # Top-2, not rank 1 — and that gap is a measured RRF property, not slack.
    # Sparse ranks findByCustomerId first by a wide margin (8.8 vs 3.4), but
    # dense ranks it 4th behind `find_by_id`. RRF fuses *ranks* and discards
    # that margin, so find_by_id (dense 1, sparse 3) edges out
    # findByCustomerId (dense 4, sparse 1): 0.750 vs 0.700. Reordering on
    # content is the cross-encoder's job (Day 6); tighten this to rank 1 once
    # the reranker sits behind fusion.
    top_two = [c.chunk.symbol_name for c in result.top(2)]
    assert "findByCustomerId" in top_two, f"got {top_two}"


async def test_dbsf_keeps_the_sparse_confidence_margin_rrf_discards(
    store: QdrantStore,
) -> None:
    """The case above, fixed by fusion choice rather than by reranking.

    DBSF normalises each branch's scores before summing, so sparse's 8.8-vs-3.4
    margin survives: findByCustomerId 1.729 vs find_by_id 1.375 on the fixture.
    """
    retriever = make_retriever(store, hybrid_settings(fusion_method="dbsf"))
    result = await retriever.search("find by customer id")
    assert result.chunks[0].chunk.symbol_name == "findByCustomerId", (
        f"got {[c.chunk.symbol_name for c in result.top(5)]}"
    )
    assert result.chunks[0].rank == 1


def test_fusion_defaults_to_rrf(retriever: HybridRetriever) -> None:
    """No behaviour change unless FUSION_METHOD is set."""
    assert retriever.fusion is models.Fusion.RRF


def test_fusion_method_setting_maps_to_qdrant_enum(store: QdrantStore) -> None:
    assert make_retriever(store, hybrid_settings(fusion_method="dbsf")).fusion is models.Fusion.DBSF
    assert make_retriever(store, hybrid_settings(fusion_method="rrf")).fusion is models.Fusion.RRF


async def test_reference_question_surfaces_the_order_call_path(
    retriever: HybridRetriever,
) -> None:
    """Not an answer yet — just that the right layer of the bug is in reach."""
    result = await retriever.search("why does POST /orders sometimes return 500")
    top_symbols = {c.chunk.symbol_name for c in result.top(3)}
    assert {"createOrder", "placeOrder"} <= top_symbols


# --------------------------------------------------------------------------- #
# result contract
# --------------------------------------------------------------------------- #


async def test_every_chunk_carries_its_fusion_score(retriever: HybridRetriever) -> None:
    result = await retriever.search("order payment")
    assert result.chunks
    for position, scored in enumerate(result.chunks, start=1):
        assert scored.rrf_score is not None
        assert scored.rrf_score == scored.score
        assert scored.rank == position
        assert scored.source is RetrievalSource.HYBRID
        assert scored.rerank_score is None


async def test_results_are_ordered_by_fusion_score(retriever: HybridRetriever) -> None:
    scores = [c.score for c in (await retriever.search("repository save")).chunks]
    assert scores == sorted(scores, reverse=True)


async def test_fused_candidates_matches_returned_chunks(retriever: HybridRetriever) -> None:
    result = await retriever.search("payment gateway timeout")
    assert result.fused_candidates == len(result.chunks)
    assert not result.reranked


async def test_per_branch_scores_are_not_invented(retriever: HybridRetriever) -> None:
    """A fused query does not report branch scores. Unset beats guessed."""
    for scored in (await retriever.search("placeOrder")).chunks:
        assert scored.dense_score is None
        assert scored.sparse_score is None


# --------------------------------------------------------------------------- #
# configuration: nothing hardcoded
# --------------------------------------------------------------------------- #


async def test_limit_is_the_sum_of_prefetch_settings(store: QdrantStore) -> None:
    retriever = make_retriever(store, hybrid_settings(prefetch_dense=2, prefetch_sparse=3))
    assert retriever.fused_limit == 5
    assert len(await retriever.search("order")) <= 5


async def test_default_limit_comes_from_settings(
    retriever: HybridRetriever, ingested_fixture: int
) -> None:
    settings = hybrid_settings()
    expected = settings.prefetch_dense + settings.prefetch_sparse
    assert retriever.fused_limit == expected
    assert len(await retriever.search("order")) <= min(expected, ingested_fixture)


# --------------------------------------------------------------------------- #
# degenerate input and filters
# --------------------------------------------------------------------------- #


async def test_stopword_only_query_falls_back_to_dense(retriever: HybridRetriever) -> None:
    """Empty sparse vector: the sparse branch contributes nothing, dense still answers."""
    result = await retriever.search("public static void")
    assert result.chunks


async def test_empty_query_is_rejected(retriever: HybridRetriever) -> None:
    with pytest.raises(RetrievalError, match="empty query"):
        await retriever.search("   ")


async def test_language_filter_applies_to_both_branches(retriever: HybridRetriever) -> None:
    """Uses the `language` payload index; filtering happens before fusion."""
    python_only = models.Filter(
        must=[models.FieldCondition(key="language", match=models.MatchValue(value="python"))]
    )
    result = await retriever.search("find user by id", query_filter=python_only)
    assert result.chunks
    assert all(c.chunk.language is Language.PYTHON for c in result.chunks)


async def test_missing_collection_raises_retrieval_error() -> None:
    settings = Settings(qdrant_collection="codemind_does_not_exist")
    store = QdrantStore(settings)
    try:
        with pytest.raises(RetrievalError, match="codemind_does_not_exist"):
            await make_retriever(store, settings).search("placeOrder")
    finally:
        await store.close()
