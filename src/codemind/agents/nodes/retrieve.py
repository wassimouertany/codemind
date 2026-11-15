"""Retrieve node: hybrid search as a graph step.

The node owns no retrieval logic. Fusion method and prefetch depths belong to
the injected `Retriever` — `HybridRetriever` reads them from Settings when the
lifespan builds it — and the node only cuts the ranking to `FINAL_K` from the
front. It never re-sorts: the retriever's order is the fused (or reranked)
order, and the Gate 1 numbers describe exactly that order.

A `RetrievalError` does not fail the graph. It is recorded, `retrieved` stays
empty, and the specialists answer "no evidence" instead of reasoning over
nothing.
"""

from __future__ import annotations

import logging

from codemind.agents.state import CodeMindState, Node, RetrieveUpdate
from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import RetrievalError
from codemind.retrieval.schemas import Retriever

logger = logging.getLogger(__name__)

NODE_NAME = "retrieve"


def make_retrieve(retriever: Retriever, settings: Settings | None = None) -> Node[RetrieveUpdate]:
    final_k = (settings or get_settings()).final_k

    async def retrieve(state: CodeMindState) -> RetrieveUpdate:
        try:
            result = await retriever.search(state["question"])
        except RetrievalError as exc:
            logger.warning("retrieval failed: %s", exc)
            return {
                "retrieved": [],
                "attempts": 1,
                "errors": [f"{NODE_NAME}: {exc}"],
                "messages": [{"node": NODE_NAME, "content": "retrieval failed; 0 chunks"}],
            }
        chunks = result.chunks[:final_k]
        return {
            "retrieved": chunks,
            "attempts": 1,
            "messages": [
                {
                    "node": NODE_NAME,
                    "content": f"{len(chunks)} of {len(result.chunks)} candidates kept",
                }
            ],
        }

    return retrieve
