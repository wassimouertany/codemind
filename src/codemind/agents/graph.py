"""The CodeMind agent graph, hand-built so every node is a named span.

`create_supervisor` would hide the routing inside one opaque agent; a
hand-built `StateGraph` gives each node its own name in the Langfuse trace,
which is what Day 13's per-node latency breakdown is read from.

    START -> router -> retrieve -> code_analyst      (bug, architecture) -> END
                                -> dependency_agent  (dependency)        -> END
                                -> locator           (locator)           -> END

Locator is a short circuit: it formats the top hits and never calls the model.
Synthesize and grounding attach after the specialists on Days 11-12.

Always run it through `run_graph`, which passes `recursion_limit`. Hitting the
limit almost always means a routing-prompt bug, not a limit that is too low.
"""

from __future__ import annotations

from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from codemind.agents.nodes.code_analyst import NODE_NAME as CODE_ANALYST
from codemind.agents.nodes.code_analyst import make_code_analyst
from codemind.agents.nodes.dependency_agent import NODE_NAME as DEPENDENCY_AGENT
from codemind.agents.nodes.dependency_agent import make_dependency_agent
from codemind.agents.nodes.locator import NODE_NAME as LOCATOR
from codemind.agents.nodes.locator import make_locator
from codemind.agents.nodes.retrieve import NODE_NAME as RETRIEVE
from codemind.agents.nodes.retrieve import make_retrieve
from codemind.agents.nodes.router import NODE_NAME as ROUTER
from codemind.agents.nodes.router import make_router
from codemind.agents.state import CodeMindState, Route, current_route, initial_state
from codemind.core.config import Settings, get_settings
from codemind.llm.base import LLMClient
from codemind.retrieval.schemas import Retriever

CodeMindGraph = CompiledStateGraph[CodeMindState, None, CodeMindState, CodeMindState]

SPECIALIST_FOR_ROUTE: dict[Route, str] = {
    Route.BUG: CODE_ANALYST,
    Route.ARCHITECTURE: CODE_ANALYST,
    Route.DEPENDENCY: DEPENDENCY_AGENT,
    Route.LOCATOR: LOCATOR,
}


def choose_specialist(state: CodeMindState) -> str:
    return SPECIALIST_FOR_ROUTE[current_route(state)]


def build_graph(
    llm: LLMClient, retriever: Retriever, *, settings: Settings | None = None
) -> CodeMindGraph:
    """Compile the graph. Every dependency is injected; nothing concrete is built here."""
    resolved = settings or get_settings()
    builder = StateGraph(CodeMindState)
    builder.add_node(ROUTER, make_router(llm))
    builder.add_node(RETRIEVE, make_retrieve(retriever, resolved))
    builder.add_node(CODE_ANALYST, make_code_analyst(llm))
    builder.add_node(DEPENDENCY_AGENT, make_dependency_agent(llm))
    builder.add_node(LOCATOR, make_locator(resolved))

    builder.add_edge(START, ROUTER)
    builder.add_edge(ROUTER, RETRIEVE)
    specialists = sorted(set(SPECIALIST_FOR_ROUTE.values()))
    builder.add_conditional_edges(RETRIEVE, choose_specialist, specialists)
    for specialist in specialists:
        builder.add_edge(specialist, END)
    return builder.compile()


async def run_graph(
    graph: CodeMindGraph, question: str, *, settings: Settings | None = None
) -> CodeMindState:
    """Run one question to completion with the configured recursion limit."""
    config = RunnableConfig(recursion_limit=(settings or get_settings()).graph_recursion_limit)
    return cast(CodeMindState, await graph.ainvoke(initial_state(question), config=config))
