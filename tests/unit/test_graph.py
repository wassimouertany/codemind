"""The compiled graph: named nodes, routing to the right specialist, the recursion limit."""

from __future__ import annotations

import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph

from codemind.agents.graph import build_graph, run_graph
from codemind.agents.nodes.router import RouteDecision
from codemind.agents.nodes.specialist import SpecialistAnswer
from codemind.agents.state import CodeMindState, Route
from codemind.core.config import Settings
from codemind.core.exceptions import LLMError
from tests.agent_support import StubLLM, StubRetriever, scored

CHUNKS = [scored("placeOrder", lines=(10, 20)), scored("save", lines=(30, 32))]


def scripted(route: str) -> StubLLM:
    """Routes to `route`; any specialist answers with one citation of chunk [1]."""

    def respond(prompt: str, schema: type) -> object:
        if schema is RouteDecision:
            return {"route": route}
        assert schema is SpecialistAnswer
        return {"claims": [{"chunk": 1, "statement": "c"}]}

    return StubLLM(respond)


def test_every_node_is_named() -> None:
    """Day 13 reads per-node spans by name; an anonymous supervisor has none."""
    graph = build_graph(scripted("bug"), StubRetriever())
    assert set(graph.get_graph().nodes) == {
        "__start__",
        "router",
        "retrieve",
        "code_analyst",
        "dependency_agent",
        "locator",
        "__end__",
    }


@pytest.mark.parametrize(
    ("route", "specialist"),
    [
        ("bug", "code_analyst"),
        ("architecture", "code_analyst"),
        ("dependency", "dependency_agent"),
        ("locator", "locator"),
    ],
)
async def test_each_route_reaches_its_specialist_once(route: str, specialist: str) -> None:
    graph = build_graph(scripted(route), StubRetriever(CHUNKS))
    final = await run_graph(graph, "q")

    assert final["route"] == [Route(route)]
    assert [m["node"] for m in final["messages"]] == ["router", "retrieve", specialist]
    assert final["retrieved"] == CHUNKS
    assert final["evidence"][0]["start_line"] == 10
    assert final["attempts"] == 1
    assert final["errors"] == []


async def test_locator_is_a_short_circuit_with_no_generation() -> None:
    llm = scripted("locator")
    await run_graph(build_graph(llm, StubRetriever(CHUNKS)), "Where is placeOrder?")
    assert llm.schemas == [RouteDecision]


async def test_an_unreachable_llm_degrades_the_run_instead_of_raising() -> None:
    llm = StubLLM(lambda p, s: LLMError("connection refused"))
    final = await run_graph(build_graph(llm, StubRetriever(CHUNKS)), "q")

    assert final["route"] == [Route.BUG]
    assert [m["node"] for m in final["messages"]] == ["router", "retrieve", "code_analyst"]
    assert len(final["errors"]) == 2  # router fallback + analyst failure
    assert final["evidence"] == []


async def test_final_k_from_settings_reaches_the_specialist() -> None:
    chunks = [scored(f"m{i}", lines=(i, i)) for i in range(1, 8)]
    llm = scripted("bug")
    final = await run_graph(
        build_graph(llm, StubRetriever(chunks), settings=Settings(final_k=3)), "q"
    )
    assert len(final["retrieved"]) == 3
    assert "[3] " in llm.prompts[1]
    assert "[4] " not in llm.prompts[1]


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
