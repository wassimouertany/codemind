"""Types carried across the retrieval pipeline.

One shape flows through every stage — store, fusion, rerank, expansion — so a
node can be added or ablated without changing what the next stage reads:

    qdrant_store  -> list[ScoredChunk]   (RRF score)
    reranker      -> list[ScoredChunk]   (cross-encoder score, rrf_score kept)
    expansion     -> list[ScoredChunk]   (+ graph neighbours, source=EXPANDED)
    hybrid        -> RetrievalResult     (chunks + the stage timings/counts)

Scores from different stages are on different scales and are deliberately kept
in separate fields rather than overwritten. The ablation table in Week 1 needs
to compare "hybrid" against "hybrid+rerank" on the same retrieved set, and
that is impossible if reranking destroys the fusion score it replaced.

These are plain dataclasses, not Pydantic models: they are created per query in
the hundreds and never cross an HTTP boundary. API-facing models live in
`api/v1/schemas.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from codemind.core.types import CodeChunk, Language, Layer


class RetrievalSource(StrEnum):
    """How a chunk entered the result set.

    Recorded per chunk because the ablation study asks exactly this question:
    a chunk only sparse found is evidence the BM25 branch is earning its place.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    """Both prefetch branches returned it — the strongest signal fusion has."""
    EXPANDED = "expanded"
    """Pulled in by `expansion` as a caller, callee or parent class, not by search."""


@dataclass(slots=True)
class ScoredChunk:
    """A retrieved chunk with its scores and provenance.

    `chunk` is reconstructed from the Qdrant payload rather than re-read from
    disk, so a result stays valid even if the working tree has moved on. The
    grounding gate is what checks a citation against the real file.
    """

    chunk: CodeChunk
    score: float
    """Score from the most recent stage. What callers rank by."""
    source: RetrievalSource = RetrievalSource.HYBRID
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    """Fusion score, preserved when the reranker overwrites `score`."""
    rerank_score: float | None = None
    rank: int | None = None
    """1-based position after the final stage. Set by whoever ranks last."""

    @property
    def citation(self) -> str:
        """`file:line`, the form the grounding gate validates."""
        return self.chunk.citation

    def with_rerank_score(self, score: float, rank: int) -> ScoredChunk:
        """Return a copy carrying the cross-encoder score, keeping the fusion score.

        Non-mutating so an ablation run can rerank a retrieved set without
        destroying the pre-rerank ordering it is being compared against.
        """
        return ScoredChunk(
            chunk=self.chunk,
            score=score,
            source=self.source,
            dense_score=self.dense_score,
            sparse_score=self.sparse_score,
            rrf_score=self.rrf_score if self.rrf_score is not None else self.score,
            rerank_score=score,
            rank=rank,
        )


@dataclass(slots=True)
class RetrievalResult:
    """Everything one retrieval produced, plus what it cost.

    The counts and timings are not decoration: Langfuse spans and the Week 1
    ablation table are both built from them, and README latency numbers come
    from here rather than `time.time()` scattered through the code.
    """

    query: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    dense_candidates: int = 0
    sparse_candidates: int = 0
    fused_candidates: int = 0
    reranked: bool = False
    expanded: bool = False
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.rerank_ms

    @property
    def citations(self) -> list[str]:
        """`file:line` for every chunk, in rank order."""
        return [scored.citation for scored in self.chunks]

    @property
    def file_paths(self) -> list[str]:
        """Distinct files, in rank order. The unit the retrieval goldset scores against."""
        seen: dict[str, None] = {}
        for scored in self.chunks:
            seen.setdefault(scored.chunk.relative_path, None)
        return list(seen)

    def top(self, k: int) -> list[ScoredChunk]:
        """First `k` chunks. `k` comes from Settings — never hardcode it."""
        return self.chunks[:k]

    def by_layer(self, layer: Layer) -> list[ScoredChunk]:
        return [scored for scored in self.chunks if scored.chunk.layer is layer]

    def by_language(self, language: Language) -> list[ScoredChunk]:
        return [scored for scored in self.chunks if scored.chunk.language is language]
