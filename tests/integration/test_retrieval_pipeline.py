"""`RetrievalPipeline` and `compose` on the fixture repo.

The expansion cases use the real `GraphExpander` over the fixture's real
symbol graph, so the neighbours appended are the planted bug path's, not a
stub's guess. Qdrant is `:memory:`: expansion needs payload filters only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient, models

from codemind.core.config import Settings
from codemind.core.types import CodeChunk
from codemind.ingestion.ast_chunker import AstChunker
from codemind.ingestion.repo_loader import RepositoryLoader
from codemind.ingestion.symbol_graph import SymbolGraph
from codemind.retrieval.expansion import GraphExpander
from codemind.retrieval.pipeline import RetrievalPipeline, compose
from codemind.retrieval.qdrant_store import QdrantStore
from codemind.retrieval.reranker import Reranker
from codemind.retrieval.schemas import (
    ChunkExpander,
    ChunkReranker,
    RetrievalResult,
    RetrievalSource,
    ScoredChunk,
)
from tests.agent_support import StubRetriever

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


@pytest.fixture(scope="module")
def ingredients() -> tuple[list[CodeChunk], SymbolGraph]:
    sources = RepositoryLoader(FIXTURE).collect()
    chunks = AstChunker().chunk_all(sources)
    return chunks, SymbolGraph.build(sources, chunks)


@pytest.fixture
async def store(ingredients: tuple[list[CodeChunk], SymbolGraph]) -> AsyncIterator[QdrantStore]:
    chunks, _ = ingredients
    client = AsyncQdrantClient(":memory:")
    store = QdrantStore(Settings(qdrant_collection="test_pipeline"), client=client)
    await store.ensure_collection()
    await store.upsert_chunks(
        chunks,
        [[1.0] + [0.0] * 767 for _ in chunks],
        [models.SparseVector(indices=[1], values=[1.0]) for _ in chunks],
    )
    yield store
    await client.close()


def fused(chunks: list[CodeChunk], *symbols: str) -> list[ScoredChunk]:
    by_name = {c.symbol_name: c for c in chunks if c.kind != "class"}
    return [
        ScoredChunk(chunk=by_name[s], score=1.0 / r, source=RetrievalSource.HYBRID, rank=r)
        for r, s in enumerate(symbols, start=1)
    ]


@dataclass
class ReversingReranker:
    """Reverses the candidates — an order no fusion would produce, so it is visible."""

    calls: list[int | None] = field(default_factory=list)

    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], *, top_k: int | None = None
    ) -> list[ScoredChunk]:
        self.calls.append(top_k)
        return list(reversed(chunks))[:top_k]


@dataclass
class RecordingExpander:
    max_extra: int = 2
    calls: int = 0

    async def expand(self, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        self.calls += 1
        return list(chunks)


def test_the_real_stages_satisfy_the_protocols() -> None:
    assert isinstance(Reranker.__new__(Reranker), ChunkReranker)
    assert isinstance(GraphExpander.__new__(GraphExpander), ChunkExpander)


async def test_flags_off_means_fused_order_cut_to_final_k_even_with_stages_passed(
    ingredients: tuple[list[CodeChunk], SymbolGraph],
) -> None:
    """The shipped default: RERANK_ENABLED=false, EXPANSION_DEPTH=0."""
    chunks, _ = ingredients
    candidates = fused(chunks, "placeOrder", "processPayment", "createOrder", "callGateway")
    reranker, expander = ReversingReranker(), RecordingExpander()
    pipeline = RetrievalPipeline(
        StubRetriever(candidates),
        reranker=reranker,
        expander=expander,
        settings=Settings(final_k=3, rerank_enabled=False, expansion_depth=0),
    )

    result = await pipeline.search("q")

    assert result.chunks == candidates[:3]
    assert (result.reranked, result.expanded) == (False, False)
    assert reranker.calls == []
    assert expander.calls == 0


async def test_rerank_on_takes_the_rerankers_order_at_final_k(
    ingredients: tuple[list[CodeChunk], SymbolGraph],
) -> None:
    chunks, _ = ingredients
    candidates = fused(chunks, "placeOrder", "processPayment", "createOrder", "callGateway")
    reranker = ReversingReranker()
    pipeline = RetrievalPipeline(
        StubRetriever(candidates),
        reranker=reranker,
        settings=Settings(final_k=3, rerank_enabled=True),
    )

    result: RetrievalResult = await pipeline.search("q")

    assert [s.chunk.symbol_name for s in result.chunks] == [
        "callGateway",
        "createOrder",
        "processPayment",
    ]
    assert reranker.calls == [3]
    assert (result.reranked, result.expanded) == (True, False)


async def test_enabled_without_a_stage_passed_is_a_no_op(
    ingredients: tuple[list[CodeChunk], SymbolGraph],
) -> None:
    chunks, _ = ingredients
    candidates = fused(chunks, "placeOrder", "processPayment")
    pipeline = RetrievalPipeline(
        StubRetriever(candidates), settings=Settings(rerank_enabled=True, expansion_depth=1)
    )
    result = await pipeline.search("q")
    assert result.chunks == candidates
    assert (result.reranked, result.expanded) == (False, False)


async def test_expansion_pulls_the_planted_bug_path_around_its_seed(
    ingredients: tuple[list[CodeChunk], SymbolGraph], store: QdrantStore
) -> None:
    """One seed, `placeOrder`; its caller and callee are the reference answer's other hops."""
    chunks, graph = ingredients
    expander = GraphExpander(store, graph, Settings(expansion_depth=1, expansion_max_extra=2))
    candidates = fused(chunks, "placeOrder", "findByCustomerId", "OrderController")

    out = await compose("q", candidates, final_k=3, expander=expander)

    assert out[0].chunk.symbol_name == "placeOrder"  # the seed keeps the top
    assert {s.chunk.symbol_name for s in out[1:]} == {"createOrder", "processPayment"}
    assert all(s.source is RetrievalSource.EXPANDED for s in out[1:])


async def test_a_seed_without_neighbours_is_backfilled_from_the_displaced_tail(
    ingredients: tuple[list[CodeChunk], SymbolGraph], store: QdrantStore
) -> None:
    """`findByCustomerId` has no callers or callees, so the tail must fill its budget."""
    chunks, graph = ingredients
    expander = GraphExpander(store, graph, Settings(expansion_depth=1, expansion_max_extra=2))
    candidates = fused(chunks, "findByCustomerId", "OrderController", "getOrder", "callGateway")

    out = await compose("q", candidates, final_k=4, expander=expander)

    assert [s.chunk.symbol_name for s in out] == [
        "findByCustomerId",
        "OrderController",
        "getOrder",
        "callGateway",
    ]
    assert all(s.source is RetrievalSource.HYBRID for s in out)


async def test_pipeline_runs_rerank_then_expansion_when_both_are_on(
    ingredients: tuple[list[CodeChunk], SymbolGraph], store: QdrantStore
) -> None:
    chunks, graph = ingredients
    settings = Settings(final_k=3, rerank_enabled=True, expansion_depth=1, expansion_max_extra=1)
    candidates = fused(chunks, "findById", "getOrder", "placeOrder")
    pipeline = RetrievalPipeline(
        StubRetriever(candidates),
        reranker=ReversingReranker(),
        expander=GraphExpander(store, graph, settings),
        settings=settings,
    )

    result = await pipeline.search("q")

    names = [s.chunk.symbol_name for s in result.chunks]
    assert names[:2] == ["placeOrder", "getOrder"]  # reranked seeds
    assert names[2] in {"createOrder", "processPayment", "findById"}  # a seed's neighbour
    assert result.chunks[2].source is RetrievalSource.EXPANDED
    assert (result.reranked, result.expanded) == (True, True)
