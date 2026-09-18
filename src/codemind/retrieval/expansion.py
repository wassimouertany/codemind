"""Call-graph expansion around reranked hits.

Retrieval answers "which code looks like the question". Diagnosis needs "and
what does that code touch". For the reference question the reranker returns
`OrderService.placeOrder` — the method that actually fails — but the answer is
only complete with `OrderController.createOrder` above it (how a request gets
there) and `PaymentService.processPayment` below it (what throws). Neither
shares enough vocabulary with the question to be retrieved on its own.

So expansion is a graph walk, not a search: take each reranked hit, find its
node in the symbol graph, pull callers and callees within `EXPANSION_DEPTH`
hops, and append those chunks.

**Appending is the whole contract.** Expansion never reorders, rescores or
displaces a reranked hit. The cross-encoder read those chunks against the
question and the graph did not; letting a neighbour outrank a hit would trade a
measured judgement for a structural guess. Expanded chunks carry
`source=EXPANDED` and `score=0.0` for exactly that reason — the score is not
comparable to a cross-encoder score, so it is left empty rather than invented.

Bounded by `EXPANSION_MAX_EXTRA`: one hub symbol (a logger, a base-class
method) can have dozens of callers, and an unbounded walk would fill the
context window with noise and blow the token budget the synthesiser enforces.

Degrades to a no-op, never an error, when the graph has no node for a chunk.
That is the normal case for merged accessor chunks (which the graph skips),
skeleton chunks, and any file edited since the graph was built.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from codemind.core.config import Settings, get_settings
from codemind.core.types import CodeChunk
from codemind.ingestion.symbol_graph import SymbolGraph
from codemind.retrieval.qdrant_store import QdrantStore
from codemind.retrieval.schemas import RetrievalSource, ScoredChunk

logger = logging.getLogger(__name__)

EXPANDED_SCORE = 0.0
"""Expanded chunks carry no comparable score. See the module docstring."""


class GraphExpander:
    """Appends call-graph neighbours to a reranked result set."""

    def __init__(
        self,
        store: QdrantStore,
        graph: SymbolGraph,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._store = store
        self._graph = graph
        self._depth = resolved.expansion_depth
        self._max_extra = resolved.expansion_max_extra

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def max_extra(self) -> int:
        return self._max_extra

    async def expand(self, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Return the reranked chunks unchanged, followed by their neighbours.

        `EXPANSION_DEPTH=0` disables expansion entirely — the switch the Day 7
        ablation flips to measure what this stage is worth.
        """
        results = list(chunks)
        if not results or self._depth == 0 or self._max_extra == 0:
            return results

        wanted = self._neighbour_symbols(results)
        if not wanted:
            return results

        extra = await self._store.fetch_by_symbols(wanted, limit=self._max_extra * 4)
        appended = self._append(results, extra)
        logger.debug(
            "expansion: %d hits -> %d neighbours requested, %d appended",
            len(results),
            len(wanted),
            len(appended) - len(results),
        )
        return appended

    def _neighbour_symbols(self, results: Sequence[ScoredChunk]) -> list[tuple[str, str]]:
        """Collect (file_path, symbol_name) for every neighbour worth fetching.

        Reads `path` and `simple_name` off the graph nodes rather than the
        `symbols` dict, because a graph restored with `SymbolGraph.load()` has
        node attributes but no rebuilt `symbols` mapping.
        """
        already = {(scored.chunk.relative_path, scored.chunk.symbol_name) for scored in results}
        wanted: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for scored in results:
            qualified = SymbolGraph._qualify(scored.chunk)
            for neighbour in self._graph.neighbours(qualified, depth=self._depth):
                attrs = self._graph.graph.nodes.get(neighbour, {})
                key = (attrs.get("path", ""), attrs.get("simple_name", ""))
                if not all(key) or key in already or key in seen:
                    continue
                seen.add(key)
                wanted.append(key)
        return wanted

    def _append(self, results: list[ScoredChunk], extra: Sequence[CodeChunk]) -> list[ScoredChunk]:
        """Append neighbours after the reranked hits, capped and deduplicated."""
        present = {scored.chunk.chunk_id for scored in results}
        out = list(results)
        rank = max((scored.rank or 0) for scored in results) + 1

        for chunk in extra:
            if len(out) - len(results) >= self._max_extra:
                break
            if chunk.chunk_id in present:
                continue
            present.add(chunk.chunk_id)
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=EXPANDED_SCORE,
                    source=RetrievalSource.EXPANDED,
                    rank=rank,
                )
            )
            rank += 1
        return out
