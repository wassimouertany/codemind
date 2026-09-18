"""Tests for cross-encoder reranking.

Two carry weight:

* `test_rerank_recovers_the_chunk_rrf_ranked_second` — the concrete case this
  stage exists for, measured on the fixture in Day 5.
* `test_fusion_score_survives_reranking` — without it the Day 7 ablation cannot
  compare hybrid against hybrid+rerank on the same retrieved set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codemind.core.config import Settings
from codemind.core.types import CodeChunk, Language, Layer
from codemind.ingestion.ast_chunker import AstChunker
from codemind.ingestion.repo_loader import RepositoryLoader
from codemind.retrieval.reranker import Reranker, get_reranker
from codemind.retrieval.schemas import RetrievalSource, ScoredChunk

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    """One model load for the whole module — construction is the expensive part."""
    return Reranker()


@pytest.fixture(scope="module")
def fixture_chunks() -> list[CodeChunk]:
    return AstChunker().chunk_all(RepositoryLoader(FIXTURE).collect())


def fused(chunks: list[CodeChunk], names: list[str]) -> list[ScoredChunk]:
    """Build a candidate list in a given order, with descending fusion scores."""
    by_name = {c.symbol_name: c for c in chunks}
    return [
        ScoredChunk(
            chunk=by_name[name],
            score=1.0 - index * 0.05,
            source=RetrievalSource.HYBRID,
            rrf_score=1.0 - index * 0.05,
            rank=index + 1,
        )
        for index, name in enumerate(names)
    ]


def synthetic(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=CodeChunk(
                relative_path=f"src/File{i}.java",
                language=Language.JAVA,
                symbol_name=f"method{i}",
                kind="method",
                start_line=i,
                end_line=i + 3,
                body=f"public void method{i}() {{ return; }}",
                context_header=f"src/File{i}.java | Class{i} | void method{i}()",
                layer=Layer.SERVICE,
            ),
            score=1.0 - i * 0.001,
            rrf_score=1.0 - i * 0.001,
            rank=i + 1,
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# what reranking is for
# --------------------------------------------------------------------------- #


def test_rerank_recovers_the_chunk_rrf_ranked_second(
    reranker: Reranker, fixture_chunks: list[CodeChunk]
) -> None:
    """RRF put `find_by_id` first for this query; reading the code fixes it.

    Fusion sees only ranks, so dense's flat scores outvoted sparse's 8.815 vs
    3.434 margin. The cross-encoder reads the question against each chunk body.
    """
    candidates = fused(fixture_chunks, ["find_by_id", "findByCustomerId", "findById"])
    result = reranker.rerank_sync("find by customer id", candidates)
    assert result[0].chunk.symbol_name == "findByCustomerId", (
        f"got {[c.chunk.symbol_name for c in result]}"
    )


def test_rerank_orders_by_cross_encoder_score(
    reranker: Reranker, fixture_chunks: list[CodeChunk]
) -> None:
    result = reranker.rerank_sync(
        "where is a payment charged", fused(fixture_chunks, ["findById", "processPayment", "save"])
    )
    scores = [c.score for c in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0].chunk.symbol_name == "processPayment"


# --------------------------------------------------------------------------- #
# the ablation contract
# --------------------------------------------------------------------------- #


def test_fusion_score_survives_reranking(
    reranker: Reranker, fixture_chunks: list[CodeChunk]
) -> None:
    candidates = fused(fixture_chunks, ["placeOrder", "processPayment"])
    result = reranker.rerank_sync("place an order", candidates)

    for scored in result:
        assert scored.rrf_score is not None
        assert scored.rerank_score == scored.score
        assert scored.score != scored.rrf_score, "rerank score should differ from fusion score"


def test_inputs_are_not_mutated(reranker: Reranker, fixture_chunks: list[CodeChunk]) -> None:
    """An ablation run reranks a list it must still be able to score un-reranked."""
    candidates = fused(fixture_chunks, ["placeOrder", "processPayment"])
    before = [(c.score, c.rank, c.rerank_score) for c in candidates]
    reranker.rerank_sync("place an order", candidates)
    assert [(c.score, c.rank, c.rerank_score) for c in candidates] == before


def test_ranks_are_renumbered_from_one(reranker: Reranker, fixture_chunks: list[CodeChunk]) -> None:
    result = reranker.rerank_sync(
        "order", fused(fixture_chunks, ["placeOrder", "processPayment", "findById"])
    )
    assert [c.rank for c in result] == list(range(1, len(result) + 1))
    assert all(c.source is RetrievalSource.HYBRID for c in result), (
        "source is provenance, not stage"
    )


# --------------------------------------------------------------------------- #
# limits, all from Settings
# --------------------------------------------------------------------------- #


def test_returns_rerank_top_k_by_default(reranker: Reranker) -> None:
    assert len(reranker.rerank_sync("order", synthetic(12))) == reranker.top_k


def test_explicit_top_k_overrides_the_setting(reranker: Reranker) -> None:
    assert len(reranker.rerank_sync("order", synthetic(12), top_k=2)) == 2


def test_candidates_are_capped_at_the_prefetch_sum() -> None:
    """The cap is the latency ceiling: it must come from Settings, not a literal."""
    small = Reranker(Settings(prefetch_dense=2, prefetch_sparse=3, rerank_top_k=5))
    assert small.max_candidates == 5
    assert len(small.rerank_sync("order", synthetic(40))) == 5


def test_fewer_candidates_than_top_k_returns_what_exists(reranker: Reranker) -> None:
    assert len(reranker.rerank_sync("order", synthetic(2))) == 2


def test_empty_candidates_returns_empty(reranker: Reranker) -> None:
    assert reranker.rerank_sync("order", []) == []


def test_device_comes_from_settings(reranker: Reranker) -> None:
    """Never hardcoded. The 4 GB card belongs to the LLM while serving."""
    assert reranker.device == "cpu"


# --------------------------------------------------------------------------- #
# async surface and lifetime
# --------------------------------------------------------------------------- #


async def test_async_matches_sync(reranker: Reranker, fixture_chunks: list[CodeChunk]) -> None:
    candidates = fused(fixture_chunks, ["placeOrder", "processPayment", "findById"])
    assert [c.chunk.symbol_name for c in await reranker.rerank("place an order", candidates)] == [
        c.chunk.symbol_name for c in reranker.rerank_sync("place an order", candidates)
    ]


def test_get_reranker_is_a_singleton() -> None:
    """Heavy models load once. A second load would double RSS for nothing."""
    assert get_reranker() is get_reranker()
