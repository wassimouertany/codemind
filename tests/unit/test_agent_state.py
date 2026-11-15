"""Reducers on `CodeMindState`: writes from different nodes accumulate.

The fan-out test is the one that matters. Two nodes in the same superstep
writing the same key raise `InvalidUpdateError` on a last-value channel, and
on a sequential path the second write silently erases the first.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from codemind.agents.state import (
    DEFAULT_ROUTE,
    CodeMindState,
    Route,
    current_route,
    initial_state,
)


def _writer(name: str) -> Any:
    async def node(state: CodeMindState) -> CodeMindState:
        return {
            "errors": [f"{name} failed"],
            "messages": [{"node": name, "content": name}],
            "evidence": [
                {"file_path": f"{name}.java", "start_line": 1, "end_line": 2, "claim": name}
            ],
            "attempts": 1,
        }

    return node


async def test_parallel_nodes_writing_the_same_keys_both_survive() -> None:
    builder = StateGraph(CodeMindState)
    builder.add_node("left", _writer("left"))
    builder.add_node("right", _writer("right"))
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge("left", END)
    builder.add_edge("right", END)

    final = await builder.compile().ainvoke(initial_state("q"))

    assert sorted(final["errors"]) == ["left failed", "right failed"]
    assert sorted(m["node"] for m in final["messages"]) == ["left", "right"]
    assert sorted(e["file_path"] for e in final["evidence"]) == ["left.java", "right.java"]
    assert final["attempts"] == 2


async def test_sequential_writes_append_instead_of_overwriting() -> None:
    builder = StateGraph(CodeMindState)
    builder.add_node("first", _writer("first"))
    builder.add_node("second", _writer("second"))
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)

    final = await builder.compile().ainvoke(initial_state("q"))

    assert final["errors"] == ["first failed", "second failed"]
    assert final["attempts"] == 2
    assert final["question"] == "q"


def test_current_route_is_the_latest_decision() -> None:
    state = initial_state("q")
    assert current_route(state) is DEFAULT_ROUTE
    state["route"] = [Route.ARCHITECTURE, Route.DEPENDENCY]
    assert current_route(state) is Route.DEPENDENCY
