"""The compiled graph: named nodes, a real run, and the recursion limit."""

from __future__ import annotations

import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph

from codemind.agents.graph import build_graph, run_graph
from codemind.agents.state import CodeMindState, Route
from codemind.core.config import Settings
from codemind.core.exceptions import LLMError
from tests.agent_support import StubLLM, StubRetriever


def test_router_is_a_named_node() -> None:
    """Day 13 reads per-node spans by name; an anonymous supervisor has none."""
    graph = build_graph(StubLLM(lambda p, s: {"route": "bug"}), StubRetriever())
    assert set(graph.get_graph().nodes) == {"__start__", "router", "__end__"}


async def test_question_runs_start_to_end_through_the_router() -> None:
    graph = build_graph(StubLLM(lambda p, s: {"route": "dependency"}), StubRetriever())
    final = await run_graph(graph, "Which classes use PetTypeRepository?")

    assert final["question"] == "Which classes use PetTypeRepository?"
    assert final["route"] == [Route.DEPENDENCY]
    assert final["errors"] == []
    assert [m["node"] for m in final["messages"]] == ["router"]


async def test_an_unreachable_llm_degrades_the_run_instead_of_raising() -> None:
    graph = build_graph(StubLLM(lambda p, s: LLMError("connection refused")), StubRetriever())
    final = await run_graph(graph, "q")
    assert final["route"] == [Route.BUG]
    assert len(final["errors"]) == 1


async def test_run_graph_passes_the_configured_recursion_limit() -> None:
    """A node that routes to itself must stop at the limit, not spin forever."""

    async def again(state: CodeMindState) -> CodeMindState:
        return {"attempts": 1}

    builder = StateGraph(CodeMindState)
    builder.add_node("again", again)
    builder.add_edge(START, "again")
    builder.add_conditional_edges("again", lambda state: "again")
    looping = builder.compile()

    with pytest.raises(GraphRecursionError, match="Recursion limit of 3"):
        await run_graph(looping, "q", settings=Settings(graph_recursion_limit=3))


def test_default_recursion_limit_is_25() -> None:
    assert Settings().graph_recursion_limit == 25
