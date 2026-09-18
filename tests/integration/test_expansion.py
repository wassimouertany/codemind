"""Tests for call-graph expansion.

Built on the fixture's real bug path — `OrderController.createOrder` →
`OrderService.placeOrder` → `PaymentService.processPayment` — because that is
the chain Gate 2 has to reproduce.

Uses `:memory:` Qdrant: expansion needs payload *filters*, which local mode
supports, not payload *indexes*, which it ignores.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient, models

from codemind.core.config import Settings
from codemind.core.types import CodeChunk, Language, Layer
from codemind.ingestion.ast_chunker import AstChunker
from codemind.ingestion.repo_loader import RepositoryLoader
from codemind.ingestion.symbol_graph import SymbolGraph
from codemind.retrieval.expansion import GraphExpander
from codemind.retrieval.qdrant_store import QdrantStore
from codemind.retrieval.schemas import RetrievalSource, ScoredChunk

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


def expansion_settings(**overrides: object) -> Settings:
    return Settings(qdrant_collection="test_expansion", **overrides)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def ingredients() -> tuple[list[CodeChunk], SymbolGraph]:
    sources = RepositoryLoader(FIXTURE).collect()
    chunks = AstChunker().chunk_all(sources)
    return chunks, SymbolGraph.build(sources, chunks)


@pytest.fixture
async def store(ingredients: tuple[list[CodeChunk], SymbolGraph]) -> AsyncIterator[QdrantStore]:
    """A store holding every fixture chunk, with placeholder vectors.

    Expansion fetches by payload filter and never scores, so the vectors only
    have to exist and have the right width.
    """
    chunks, _ = ingredients
    client = AsyncQdrantClient(":memory:")
    store = QdrantStore(expansion_settings(), client=client)
    await store.ensure_collection()
    await store.upsert_chunks(
        chunks,
        [[1.0] + [0.0] * 767 for _ in chunks],
        [models.SparseVector(indices=[1], values=[1.0]) for _ in chunks],
    )
    yield store
    await client.close()


def hit(chunks: list[CodeChunk], symbol: str, *, rank: int = 1, score: float = 9.0) -> ScoredChunk:
    match = next(c for c in chunks if c.symbol_name == symbol and c.kind != "class")
    return ScoredChunk(
        chunk=match, score=score, source=RetrievalSource.HYBRID, rerank_score=score, rank=rank
    )


def names(results: list[ScoredChunk]) -> list[str]:
    return [scored.chunk.symbol_name for scored in results]


# --------------------------------------------------------------------------- #
# what expansion is for
# --------------------------------------------------------------------------- #


async def test_expansion_pulls_in_the_caller_and_the_callee(
    store: QdrantStore, ingredients: tuple[list[CodeChunk], SymbolGraph]
) -> None:
    """One hit on `placeOrder` should surface both neighbouring hops of the bug."""
    chunks, graph = ingredients
    results = await GraphExpander(store, graph, expansion_settings()).expand(
        [hit(chunks, "placeOrder")]
    )

    found = names(results)
    assert found[0] == "placeOrder", "the reranked hit must stay first"
    assert "createOrder" in found, f"caller missing from {found}"
    assert "processPayment" in found, f"callee missing from {found}"


async def test_expanded_chunks_are_marked_and_unscored(
    store: QdrantStore, ingredients: tuple[list[CodeChunk], SymbolGraph]
) -> None:
    """A graph guess must never masquerade as a cross-encoder judgement."""
    chunks, graph = ingredients
    results = await GraphExpander(store, graph, expansion_settings()).expand(
        [hit(chunks, "placeOrder")]
    )

    expanded = [scored for scored in results if scored.source is RetrievalSource.EXPANDED]
    assert expanded
    for scored in expanded:
        assert scored.score == 0.0
        assert scored.rerank_score is None


async def test_reranked_hits_are_never_displaced(
    store: QdrantStore, ingredients: tuple[list[CodeChunk], SymbolGraph]
) -> None:
    chunks, graph = ingredients
    hits = [hit(chunks, "placeOrder", rank=1), hit(chunks, "processPayment", rank=2, score=8.0)]
    results = await GraphExpander(store, graph, expansion_settings()).expand(hits)

    assert names(results)[:2] == ["placeOrder", "processPayment"]
    assert [scored.rank for scored in results] == list(range(1, len(results) + 1))


async def test_no_duplicates_of_chunks_already_retrieved(
    store: QdrantStore, ingredients: tuple[list[CodeChunk], SymbolGraph]
) -> None:
    """`processPayment` is both a hit and a callee of `placeOrder`."""
    chunks, graph = ingredients
    hits = [hit(chunks, "placeOrder", rank=1), hit(chunks, "processPayment", rank=2)]
    results = await GraphExpander(store, graph, expansion_settings()).expand(hits)

    ids = [scored.chunk.chunk_id for scored in results]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# settings drive everything
# --------------------------------------------------------------------------- #


async def test_depth_zero_disables_expansion(
    store: QdrantStore, ingredients: tuple[list[CodeChunk], SymbolGraph]
) -> None:
    """The ablation's off switch."""
    chunks, graph = ingredients
    hits = [hit(chunks, "placeOrder")]
    assert await GraphExpander(store, graph, expansion_settings(expansion_depth=0)).expand(
        hits
    ) == list(hits)


async def test_max_extra_caps_appended_chunks(
    store: QdrantStore, ingredients: tuple[list[CodeChunk], SymbolGraph]
) -> None:
    """One hub symbol must not flood the context window."""
    chunks, graph = ingredients
    results = await GraphExpander(store, graph, expansion_settings(expansion_max_extra=1)).expand(
        [hit(chunks, "placeOrder")]
    )
    assert len(results) == 2


async def test_depth_two_reaches_further_than_depth_one(
    store: QdrantStore, ingredients: tuple[list[CodeChunk], SymbolGraph]
) -> None:
    """From the controller, depth 2 should reach past the service into payments."""
    chunks, graph = ingredients
    hits = [hit(chunks, "createOrder")]
    shallow = await GraphExpander(store, graph, expansion_settings(expansion_depth=1)).expand(hits)
    deep = await GraphExpander(store, graph, expansion_settings(expansion_depth=2)).expand(hits)

    assert "processPayment" not in names(shallow)
    assert "processPayment" in names(deep)


# --------------------------------------------------------------------------- #
# degradation, never an error
# --------------------------------------------------------------------------- #


async def test_chunk_with_no_graph_node_is_a_no_op(
    store: QdrantStore, ingredients: tuple[list[CodeChunk], SymbolGraph]
) -> None:
    """A file edited since the graph was built, or a merged accessor chunk."""
    _, graph = ingredients
    orphan = ScoredChunk(
        chunk=CodeChunk(
            relative_path="src/NotIndexed.java",
            language=Language.JAVA,
            symbol_name="ghostMethod",
            kind="function",
            start_line=1,
            end_line=3,
            body="void ghostMethod() {}",
            layer=Layer.UNKNOWN,
        ),
        score=5.0,
        rank=1,
    )
    assert await GraphExpander(store, graph, expansion_settings()).expand([orphan]) == [orphan]


async def test_empty_input_returns_empty(
    store: QdrantStore, ingredients: tuple[list[CodeChunk], SymbolGraph]
) -> None:
    _, graph = ingredients
    assert await GraphExpander(store, graph, expansion_settings()).expand([]) == []


async def test_works_on_a_graph_restored_from_disk(
    store: QdrantStore, ingredients: tuple[list[CodeChunk], SymbolGraph], tmp_path: Path
) -> None:
    """`SymbolGraph.load` rebuilds nodes but not `symbols`; expansion must not care."""
    chunks, graph = ingredients
    path = tmp_path / "graph.json"
    graph.save(path)

    restored = SymbolGraph.load(path)
    results = await GraphExpander(store, restored, expansion_settings()).expand(
        [hit(chunks, "placeOrder")]
    )
    assert "processPayment" in names(results)
