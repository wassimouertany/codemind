"""find_definition: symbol name -> where it is defined, via the symbol graph.

Accepts the three spellings a node is likely to have in hand:

    placeOrder                                          simple name
    OrderService.placeOrder                             class-qualified
    src/.../OrderService.java::OrderService.placeOrder  graph node id

A simple name can match several definitions (overloads, same-named methods in
different classes); all are returned, sorted, and the caller decides. No match
is an empty list.
"""

from __future__ import annotations

from codemind.agents.tools.schemas import Definition
from codemind.core.exceptions import ToolError
from codemind.ingestion.symbol_graph import SymbolGraph


def resolve_symbol(graph: SymbolGraph, name: str) -> list[str]:
    """Graph node ids matching `name` in any of the accepted spellings."""
    name = name.strip()
    if not name:
        return []
    if "::" in name:
        return [name] if name in graph.graph else []
    simple = name.rsplit(".", 1)[-1]
    candidates = graph.find(simple)
    if "." in name:
        candidates = [q for q in candidates if q.split("::", 1)[-1] == name]
    return sorted(set(candidates))


def definition_of(graph: SymbolGraph, qualified_name: str) -> Definition:
    attrs = graph.graph.nodes[qualified_name]
    if "end_line" not in attrs:
        raise ToolError(
            f"symbol graph node {qualified_name!r} has no end_line; the graph predates "
            "line ranges — re-run ingestion to rebuild it"
        )
    return Definition(
        qualified_name=qualified_name,
        simple_name=attrs["simple_name"],
        kind=attrs["kind"],
        file_path=attrs["path"],
        start_line=attrs["line"],
        end_line=attrs["end_line"],
    )


class FindDefinition:
    def __init__(self, graph: SymbolGraph) -> None:
        self._graph = graph

    async def __call__(self, symbol: str) -> list[Definition]:
        return [definition_of(self._graph, q) for q in resolve_symbol(self._graph, symbol)]
