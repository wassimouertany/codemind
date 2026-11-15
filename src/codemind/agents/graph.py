"""The CodeMind agent graph, hand-built so every node is a named span.

`create_supervisor` would hide the routing inside one opaque agent; a
hand-built `StateGraph` gives each node its own name in the Langfuse trace,
which is what Day 13's per-node latency breakdown is read from.

Today the graph is `START -> router -> END`. The retrieve, specialist,
synthesize and grounding nodes attach here as they land; `retriever` is part of
the signature now so callers do not change when they do.

Always run it through `run_graph`, which passes `recursion_limit`. Hitting the
limit almost always means a routing-prompt bug, not a limit that is too low.
"""

from __future__ import annotations

from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from codemind.agents.nodes.router import NODE_NAME as ROUTER
from codemind.agents.nodes.router import make_router
from codemind.agents.state import CodeMindState, initial_state
from codemind.core.config import Settings, get_settings
from codemind.llm.base import LLMClient
from codemind.retrieval.schemas import Retriever

CodeMindGraph = CompiledStateGraph[CodeMindState, None, CodeMindState, CodeMindState]


def build_graph(llm: LLMClient, retriever: Retriever) -> CodeMindGraph:
    """Compile the graph. Both dependencies are injected; nothing concrete is built here."""
    builder = StateGraph(CodeMindState)
    builder.add_node(ROUTER, make_router(llm))
    builder.add_edge(START, ROUTER)
    builder.add_edge(ROUTER, END)
    return builder.compile()


async def run_graph(
    graph: CodeMindGraph, question: str, *, settings: Settings | None = None
) -> CodeMindState:
    """Run one question to completion with the configured recursion limit."""
    config = RunnableConfig(recursion_limit=(settings or get_settings()).graph_recursion_limit)
    return cast(CodeMindState, await graph.ainvoke(initial_state(question), config=config))
