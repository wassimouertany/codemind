"""The full retrieval contract — fuse, rerank, expand — composed in one place.

    HybridRetriever.search      -> fused candidates (PREFETCH_DENSE + PREFETCH_SPARSE)
    reranker (optional)         -> top FINAL_K by cross-encoder score
    expander (optional)         -> seeds + call-graph neighbours, backfilled to FINAL_K

`compose` is the single definition of how those stages combine. The Gate 1
ablation harness and the agent graph both call it, so the
`hybrid_rerank_expansion` row measured at Gate 1 is exactly what the graph
runs when both stages are on — which is the only way Gate 2's answer
completeness measurement of expansion means anything.

Both optional stages are no-ops unless they are passed in *and* their flag is
on (`RERANK_ENABLED`, `EXPANSION_DEPTH > 0`). Both ship off on Gate 1 evidence,
so the default pipeline is fused search cut to `FINAL_K`, unchanged.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from codemind.core.config import Settings, get_settings
from codemind.retrieval.schemas import (
    ChunkExpander,
    ChunkReranker,
    RetrievalResult,
    Retriever,
    ScoredChunk,
)


async def compose(
    query: str,
    fused: Sequence[ScoredChunk],
    *,
    final_k: int,
    reranker: ChunkReranker | None = None,
    expander: ChunkExpander | None = None,
) -> list[ScoredChunk]:
    """Rerank and expand `fused` into exactly-ranked, at most `final_k` chunks.

    With an expander, the top `final_k - max_extra` chunks are seeds and the
    neighbours compete for the tail; when a seed has fewer neighbours than the
    budget, the displaced tail backfills. The list is therefore `final_k` long
    whenever enough candidates exist, so expansion is compared against the
    6th-10th chunk it displaces rather than landing past the cutoff.
    """
    chunks = (
        await reranker.rerank(query, fused, top_k=final_k)
        if reranker is not None
        else list(fused[:final_k])
    )
    if expander is None:
        return chunks

    # `max_extra` must leave at least one seed to walk from.
    extra_budget = min(expander.max_extra, final_k - 1)
    seeds, remainder = chunks[: final_k - extra_budget], chunks[final_k - extra_budget :]
    return backfill(await expander.expand(seeds), remainder, final_k)


def backfill(
    chunks: Sequence[ScoredChunk], remainder: Sequence[ScoredChunk], final_k: int
) -> list[ScoredChunk]:
    """Top up `chunks` from `remainder` until the list is exactly `final_k` long.

    Without backfill a symbol with no callers would hand back a short list, and
    a configuration returning 6 chunks cannot be compared with one returning
    10 — the gap reads as a quality difference. Duplicates are skipped: a
    neighbour may already sit in the reranked tail.
    """
    out = list(chunks[:final_k])
    seen = {scored.chunk.chunk_id for scored in out}
    for scored in remainder:
        if len(out) >= final_k:
            break
        if scored.chunk.chunk_id in seen:
            continue
        seen.add(scored.chunk.chunk_id)
        out.append(scored)
    return out


class RetrievalPipeline:
    """A `Retriever` that runs the whole contract. What the agent graph searches with."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        reranker: ChunkReranker | None = None,
        expander: ChunkExpander | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._retriever = retriever
        self._reranker = reranker if resolved.rerank_enabled else None
        self._expander = expander if resolved.expansion_depth > 0 else None
        self._final_k = resolved.final_k

    @property
    def reranks(self) -> bool:
        return self._reranker is not None

    @property
    def expands(self) -> bool:
        return self._expander is not None

    async def search(self, query: str) -> RetrievalResult:
        result = await self._retriever.search(query)
        chunks = await compose(
            query,
            result.chunks,
            final_k=self._final_k,
            reranker=self._reranker,
            expander=self._expander,
        )
        return dataclasses.replace(
            result, chunks=chunks, reranked=self.reranks, expanded=self.expands
        )
