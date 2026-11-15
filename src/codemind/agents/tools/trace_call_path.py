"""trace_call_path: the call chain from an endpoint down to where it ends up.

Built on `SymbolGraph.path_between`, which follows `calls` edges only. Both
endpoints accept any spelling `find_definition` does. When a name is
ambiguous, every (source, target) pairing is tried and the shortest path wins,
ties broken by node id so the answer is deterministic.

This is a syntactic call graph: dynamic dispatch through an interface ends at
the interface method, and reflection or DI-by-name is invisible. A missing
path means the graph has no edge, not that the runtime has no call.
"""

from __future__ import annotations

from codemind.agents.tools.find_definition import definition_of, resolve_symbol
from codemind.agents.tools.schemas import CallPath
from codemind.ingestion.symbol_graph import SymbolGraph


class TraceCallPath:
    def __init__(self, graph: SymbolGraph) -> None:
        self._graph = graph

    async def __call__(self, source: str, target: str) -> CallPath:
        sources = resolve_symbol(self._graph, source)
        targets = resolve_symbol(self._graph, target)
        unresolved = tuple(
            name for name, found in ((source, sources), (target, targets)) if not found
        )
        if unresolved:
            return CallPath(source=source, target=target, hops=(), unresolved=unresolved)

        paths = [path for s in sources for t in targets if (path := self._graph.path_between(s, t))]
        if not paths:
            return CallPath(source=source, target=target, hops=())
        best = min(paths, key=lambda path: (len(path), path))
        return CallPath(
            source=source,
            target=target,
            hops=tuple(definition_of(self._graph, q) for q in best),
        )
