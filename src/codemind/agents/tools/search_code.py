"""search_code: hybrid search as a tool a node can call with its own query.

The retrieve node searches with the user's question; a node that needs a
second look ("where is PaymentTimeoutException defined?") calls this with a
query of its own. It runs the same `Retriever` the graph was built with —
normally the `RetrievalPipeline`, so fusion method, prefetch depths, rerank and
expansion all follow Settings exactly as the retrieve node's search does. The
number of hits defaults to `FINAL_K`.
"""

from __future__ import annotations

from codemind.agents.tools.schemas import CodeHit
from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import RetrievalError, ToolError
from codemind.retrieval.schemas import Retriever


class SearchCode:
    def __init__(self, retriever: Retriever, settings: Settings | None = None) -> None:
        self._retriever = retriever
        self._default_limit = (settings or get_settings()).final_k

    async def __call__(self, query: str, *, limit: int | None = None) -> list[CodeHit]:
        if not query.strip():
            raise ToolError("search_code needs a non-empty query")
        k = self._default_limit if limit is None else limit
        if k < 1:
            raise ToolError(f"search_code limit must be at least 1, got {k}")
        try:
            result = await self._retriever.search(query)
        except RetrievalError as exc:
            raise ToolError(f"search_code failed: {exc}") from exc
        return [
            CodeHit(
                rank=rank,
                file_path=scored.chunk.relative_path,
                start_line=scored.chunk.start_line,
                end_line=scored.chunk.end_line,
                symbol=_display(scored.chunk.parent_symbol, scored.chunk.symbol_name),
                kind=scored.chunk.kind,
                score=scored.score,
                source=scored.source.value,
            )
            for rank, scored in enumerate(result.chunks[:k], start=1)
        ]


def _display(parent: str | None, symbol: str) -> str:
    return f"{parent}.{symbol}" if parent and parent != symbol else symbol
