"""The compiled graph: named nodes, routing to the right specialist, the recursion limit."""

from __future__ import annotations

import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph

from codemind.agents.graph import build_graph, run_graph
from codemind.agents.nodes.router import RouteDecision
from codemind.agents.nodes.specialist import SpecialistAnswer
from codemind.agents.nodes.synthesize import ArchitectureAnswer, BugAnswer, DependencyAnswer
from codemind.agents.state import CodeMindState, Route
from codemind.core.config import Settings
from codemind.core.exceptions import LLMError
from tests.agent_support import StubLLM, StubRetriever, scored

CHUNKS = [scored("placeOrder", lines=(10, 20)), scored("save", lines=(30, 32))]


def scripted(route: str) -> StubLLM:
    """Routes to `route`; any specialist answers with one citation of chunk [1]."""

    def respond(prompt: str, schema: type) -> object:
        return {
            RouteDecision: {"route": route},
            SpecialistAnswer: {"claims": [{"chunk": 1, "statement": "c"}]},
            BugAnswer: {"root_cause": "rc", "recommended_fix": ["fix"], "suggested_test": "t"},
            ArchitectureAnswer: {"mechanism": "m"},
            DependencyAnswer: {"impacted": ["X"], "reasoning": "r"},
        }[schema]

    return StubLLM(respond)


def test_every_node_is_named() -> None:
    """Day 13 reads per-node spans by name; an anonymous supervisor has none."""
    graph = build_graph(scripted("bug"), StubRetriever())
    assert set(graph.get_graph().nodes) == {
        "__start__",
        "router",
        "retrieve",
        "bug_analyst",
        "architecture_analyst",
        "dependency_agent",
        "locator",
        "synthesize",
        "__end__",
    }


@pytest.mark.parametrize(
    ("route", "specialist"),
    [
        ("bug", "bug_analyst"),
        ("architecture", "architecture_analyst"),
        ("dependency", "dependency_agent"),
        ("locator", "locator"),
    ],
)
async def test_each_route_reaches_its_specialist_once(route: str, specialist: str) -> None:
    graph = build_graph(scripted(route), StubRetriever(CHUNKS))
    final = await run_graph(graph, "q")

    assert final["route"] == [Route(route)]
    tail = [] if route == "locator" else ["synthesize"]
    assert [m["node"] for m in final["messages"]] == ["router", "retrieve", specialist, *tail]
    assert len(final["answer"]) == (0 if route == "locator" else 1)
    assert final["retrieved"] == CHUNKS
    assert final["evidence"][0]["start_line"] == 10
    assert final["attempts"] == 1
    assert final["errors"] == []


async def test_the_answer_carries_the_specialists_evidence() -> None:
    final = await run_graph(build_graph(scripted("bug"), StubRetriever(CHUNKS)), "q")
    assert final["answer"][0]["evidence"] == final["evidence"]
    assert final["answer"][0]["fields"]["recommended_fix"] == ["fix"]  # type: ignore[index]


async def test_locator_is_a_short_circuit_with_no_generation() -> None:
    llm = scripted("locator")
    await run_graph(build_graph(llm, StubRetriever(CHUNKS)), "Where is placeOrder?")
    assert llm.schemas == [RouteDecision]


async def test_an_unreachable_llm_degrades_the_run_instead_of_raising() -> None:
    llm = StubLLM(lambda p, s: LLMError("connection refused"))
    final = await run_graph(build_graph(llm, StubRetriever(CHUNKS)), "q")

    assert final["route"] == [Route.BUG]
    assert [m["node"] for m in final["messages"]] == [
        "router",
        "retrieve",
        "bug_analyst",
        "synthesize",
    ]
    assert len(final["errors"]) == 3  # router fallback, analyst failure, degraded answer
    assert final["evidence"] == []
    assert final["answer"][0]["degraded"] == "no resolved claims to synthesize from"


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


class _CountingReranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query, chunks, *, top_k=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        return list(reversed(chunks))[:top_k]


async def test_rerank_passed_but_disabled_never_runs() -> None:
    """Default behaviour is unchanged: RERANK_ENABLED ships false."""
    reranker = _CountingReranker()
    graph = build_graph(
        scripted("bug"),
        StubRetriever(CHUNKS),
        reranker=reranker,
        settings=Settings(rerank_enabled=False),
    )
    final = await run_graph(graph, "q")
    assert reranker.calls == 0
    assert final["retrieved"] == CHUNKS


async def test_rerank_passed_and_enabled_reorders_what_the_specialist_sees() -> None:
    reranker = _CountingReranker()
    graph = build_graph(
        scripted("bug"),
        StubRetriever(CHUNKS),
        reranker=reranker,
        settings=Settings(rerank_enabled=True),
    )
    final = await run_graph(graph, "q")
    assert reranker.calls == 1
    assert final["retrieved"] == list(reversed(CHUNKS))
