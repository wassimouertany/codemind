"""Tests for the retrieval result types.

`test_reranking_preserves_the_fusion_score` is the load-bearing one. The Week 1
ablation table compares hybrid against hybrid+rerank on the same retrieved set,
which is only possible while both scores survive on the same object.
"""

from __future__ import annotations

from codemind.core.types import CodeChunk, Language, Layer
from codemind.retrieval.schemas import RetrievalResult, RetrievalSource, ScoredChunk


def make_chunk(
    path: str = "src/main/java/com/demo/service/OrderService.java",
    *,
    symbol: str = "placeOrder",
    language: Language = Language.JAVA,
    layer: Layer = Layer.SERVICE,
    start_line: int = 42,
) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language=language,
        symbol_name=symbol,
        kind="method",
        start_line=start_line,
        end_line=start_line + 8,
        body="public Order placeOrder(OrderRequest request) { ... }",
        layer=layer,
    )


def scored(chunk: CodeChunk | None = None, score: float = 0.5, **kwargs: object) -> ScoredChunk:
    return ScoredChunk(chunk=chunk or make_chunk(), score=score, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# ScoredChunk
# --------------------------------------------------------------------------- #


def test_citation_is_file_colon_line() -> None:
    """The exact form the grounding gate parses and validates."""
    assert scored(make_chunk(start_line=42)).citation == (
        "src/main/java/com/demo/service/OrderService.java:42"
    )


def test_reranking_preserves_the_fusion_score() -> None:
    """Rerank overwrites `score` but must not destroy what it replaced."""
    fused = scored(score=0.031, rrf_score=0.031, source=RetrievalSource.HYBRID)
    reranked = fused.with_rerank_score(8.4, rank=1)

    assert reranked.score == 8.4
    assert reranked.rerank_score == 8.4
    assert reranked.rrf_score == 0.031
    assert reranked.rank == 1
    assert reranked.source is RetrievalSource.HYBRID


def test_with_rerank_score_backfills_rrf_from_score() -> None:
    """A chunk that skipped explicit fusion still keeps its pre-rerank score."""
    assert scored(score=0.7).with_rerank_score(3.1, rank=2).rrf_score == 0.7


def test_with_rerank_score_does_not_mutate_the_original() -> None:
    """An ablation run reranks a set it must still be able to score un-reranked."""
    original = scored(score=0.031, rrf_score=0.031)
    original.with_rerank_score(8.4, rank=1)
    assert original.score == 0.031
    assert original.rerank_score is None


def test_branch_scores_survive_fusion() -> None:
    """Which branch found a chunk is the ablation study's whole question."""
    chunk = scored(score=0.03, dense_score=0.81, sparse_score=12.5, source=RetrievalSource.HYBRID)
    assert chunk.dense_score == 0.81
    assert chunk.sparse_score == 12.5


# --------------------------------------------------------------------------- #
# RetrievalResult
# --------------------------------------------------------------------------- #


def test_empty_result_is_falsy_and_has_no_citations() -> None:
    """A question the repo cannot answer retrieves nothing. That is a valid state."""
    result = RetrievalResult(query="what is the airspeed velocity of a swallow")
    assert len(result) == 0
    assert result.citations == []
    assert result.file_paths == []


def test_citations_are_in_rank_order() -> None:
    result = RetrievalResult(
        query="why does POST /orders return 500",
        chunks=[
            scored(make_chunk("a/Controller.java", start_line=10), score=9.0),
            scored(make_chunk("b/Service.java", start_line=20), score=7.0),
        ],
    )
    assert result.citations == ["a/Controller.java:10", "b/Service.java:20"]


def test_file_paths_deduplicates_but_keeps_rank_order() -> None:
    """Recall@k is scored per file, and three methods from one file are one hit."""
    result = RetrievalResult(
        query="q",
        chunks=[
            scored(make_chunk("Service.java", symbol="placeOrder")),
            scored(make_chunk("Controller.java", symbol="post")),
            scored(make_chunk("Service.java", symbol="validate")),
        ],
    )
    assert result.file_paths == ["Service.java", "Controller.java"]


def test_top_k_is_clamped_to_available_chunks() -> None:
    """Asking for 5 when 2 were retrieved returns 2, not an error."""
    result = RetrievalResult(query="q", chunks=[scored(), scored()])
    assert len(result.top(5)) == 2
    assert len(result.top(1)) == 1


def test_total_ms_sums_the_stages() -> None:
    """README latency numbers are built from these fields, not ad-hoc timing."""
    result = RetrievalResult(query="q", retrieval_ms=120.5, rerank_ms=340.25)
    assert result.total_ms == 460.75


def test_filter_by_layer_and_language() -> None:
    result = RetrievalResult(
        query="q",
        chunks=[
            scored(make_chunk("Controller.java", layer=Layer.CONTROLLER)),
            scored(make_chunk("Service.java", layer=Layer.SERVICE)),
            scored(make_chunk("repo.py", language=Language.PYTHON, layer=Layer.REPOSITORY)),
        ],
    )
    assert [c.chunk.relative_path for c in result.by_layer(Layer.SERVICE)] == ["Service.java"]
    assert [c.chunk.relative_path for c in result.by_language(Language.PYTHON)] == ["repo.py"]
