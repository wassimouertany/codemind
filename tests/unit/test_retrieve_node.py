"""Retrieve node: cut to FINAL_K from the front, never reorder, never crash."""

from __future__ import annotations

from codemind.agents.nodes.retrieve import make_retrieve
from codemind.agents.state import initial_state
from codemind.core.config import Settings
from codemind.core.exceptions import RetrievalError
from codemind.retrieval.hybrid import HybridRetriever
from codemind.retrieval.schemas import Retriever
from tests.agent_support import StubRetriever, scored


def test_hybrid_retriever_satisfies_the_protocol() -> None:
    assert isinstance(HybridRetriever.__new__(HybridRetriever), Retriever)


async def test_searches_with_the_question() -> None:
    retriever = StubRetriever([scored("a")])
    await make_retrieve(retriever)(initial_state("Where is the owner controller?"))
    assert retriever.queries == ["Where is the owner controller?"]


async def test_keeps_the_first_final_k_in_the_retrievers_order() -> None:
    """Scores deliberately out of order: fused or reranked rank is the order, not score."""
    chunks = [scored(f"m{i}", score=s) for i, s in enumerate([0.1, 0.9, 0.3, 0.8, 0.2, 0.7])]
    update = await make_retrieve(StubRetriever(chunks), Settings(final_k=4))(initial_state("q"))

    assert update["retrieved"] == chunks[:4]
    assert update["attempts"] == 1
    assert update["messages"] == [{"node": "retrieve", "content": "4 of 6 candidates kept"}]
    assert "errors" not in update


async def test_fewer_candidates_than_final_k_keeps_them_all() -> None:
    chunks = [scored("a"), scored("b")]
    update = await make_retrieve(StubRetriever(chunks), Settings(final_k=10))(initial_state("q"))
    assert update["retrieved"] == chunks


async def test_a_retrieval_error_is_recorded_not_raised() -> None:
    retriever = StubRetriever(error=RetrievalError("hybrid query failed: connection refused"))
    update = await make_retrieve(retriever)(initial_state("q"))

    assert update["retrieved"] == []
    assert update["attempts"] == 1
    assert update["errors"] == ["retrieve: hybrid query failed: connection refused"]


async def test_returns_only_the_keys_it_owns() -> None:
    update = await make_retrieve(StubRetriever([scored("a")]))(initial_state("q"))
    assert set(update) <= {"retrieved", "attempts", "messages", "errors"}
