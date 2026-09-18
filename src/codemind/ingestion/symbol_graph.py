"""Symbol graph: who defines what, and who calls whom.

Retrieval finds chunks that *look* relevant. It cannot tell you that
`OrderController.createOrder` calls `OrderService.placeOrder` which calls
`PaymentService.processPayment` — and that chain is the actual answer to "why
does POST /orders return 500". No amount of embedding similarity produces it.

## The resolution problem

The AST gives call *sites*, not call *targets*. Seeing
`paymentService.processPayment(...)` tells us a method named `processPayment`
was invoked on something called `paymentService`. Mapping that to
`com.demo.service.PaymentService.processPayment` requires knowing the type of
`paymentService`, and we have no type checker.

So resolution is layered, most reliable first:

1. **Receiver type from a declaration.** Java gives us this directly:
   `private final PaymentService paymentService;` binds the name to a type. Same
   for constructor parameters and local variables.
2. **Unique simple name.** If exactly one symbol in the repository is called
   `processPayment`, the call is that one. Wrong only in the presence of an
   unrelated same-named method that is never the target.
3. **Import-scoped match.** Among several candidates, prefer one whose file is
   imported by the calling file.
4. **Give up.** Unresolved calls are counted, not guessed. A wrong edge is worse
   than a missing one: it produces a confident, false call path.

Every layer is recorded in `ResolutionStats`, so the resolution rate is a number
you can report rather than a claim you make.

## What this will miss, by construction

Dynamic dispatch through an interface resolves to the interface method, not the
implementation. Reflection, dependency-injection-by-name, lambdas passed as
callbacks and `getattr` are invisible. This is a syntactic approximation of a
call graph, not a sound one — which is fine for retrieval expansion, and would
not be fine for, say, dead-code elimination.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from codemind.core.types import CodeChunk, Language, SourceFile
from codemind.ingestion.language_registry import LanguageSpec, spec_for_language

if TYPE_CHECKING:  # pragma: no cover
    from tree_sitter import Node

logger = logging.getLogger(__name__)

CONTAINS = "contains"
CALLS = "calls"
IMPORTS = "imports"


@dataclass(slots=True, frozen=True)
class Symbol:
    """A named definition. `qualified_name` is the graph node id."""

    qualified_name: str
    simple_name: str
    kind: str
    relative_path: str
    start_line: int
    end_line: int
    language: str = ""
    parent: str | None = None

    @property
    def citation(self) -> str:
        return f"{self.relative_path}:{self.start_line}"


@dataclass(slots=True)
class ResolutionStats:
    """How each CALLS edge was resolved. Report this, don't claim accuracy."""

    call_sites: int = 0
    by_receiver_type: int = 0
    by_unique_name: int = 0
    by_import_scope: int = 0
    external: int = 0
    """Target name exists nowhere in the repository — a library or JDK call.
    These can never resolve and must not count against the resolution rate."""
    ambiguous_dropped: int = 0
    """Target name exists but several candidates matched and none won. These DO
    count against us: it is a resolution failure, not an out-of-scope call."""

    @property
    def resolved(self) -> int:
        return self.by_receiver_type + self.by_unique_name + self.by_import_scope

    @property
    def intra_repo_sites(self) -> int:
        """Call sites that could in principle resolve."""
        return self.resolved + self.ambiguous_dropped

    @property
    def resolution_rate(self) -> float:
        """Over resolvable sites only. The headline number to report.

        Dividing by all call sites would mostly measure how much of the JDK a
        project uses, which says nothing about this component's quality.
        """
        return self.resolved / self.intra_repo_sites if self.intra_repo_sites else 0.0

    @property
    def coverage_rate(self) -> float:
        """Over every call site, including library calls. Context, not quality."""
        return self.resolved / self.call_sites if self.call_sites else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "call_sites": self.call_sites,
            "intra_repo_sites": self.intra_repo_sites,
            "by_receiver_type": self.by_receiver_type,
            "by_unique_name": self.by_unique_name,
            "by_import_scope": self.by_import_scope,
            "external": self.external,
            "ambiguous_dropped": self.ambiguous_dropped,
            "resolution_rate": round(self.resolution_rate, 3),
            "coverage_rate": round(self.coverage_rate, 3),
        }


class SymbolGraph:
    """Directed graph of symbols. Nodes are qualified names; edges are typed."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph[str] = nx.DiGraph()
        self.symbols: dict[str, Symbol] = {}
        self.stats = ResolutionStats()
        self._by_simple_name: dict[str, list[str]] = defaultdict(list)
        self._parsers: dict[Language, Any] = {}

    # -- construction ------------------------------------------------------ #

    @classmethod
    def build(cls, sources: list[SourceFile], chunks: list[CodeChunk]) -> SymbolGraph:
        """Two passes: index every definition, then resolve every call site.

        Two passes are required — a call in the first file may target a method
        defined in the last one, so nothing can be resolved until the index is
        complete.
        """
        graph = cls()
        graph._index_definitions(chunks)
        graph._link_calls(sources, chunks)
        logger.info(
            "symbol graph: %d symbols, %d edges, %.1f%% of %d resolvable call sites",
            graph.graph.number_of_nodes(),
            graph.graph.number_of_edges(),
            graph.stats.resolution_rate * 100,
            graph.stats.intra_repo_sites,
        )
        return graph

    def _index_definitions(self, chunks: list[CodeChunk]) -> None:
        for chunk in chunks:
            if chunk.kind == "members":
                continue  # merged accessors have no single identity
            qualified = self._qualify(chunk)
            symbol = Symbol(
                qualified_name=qualified,
                simple_name=chunk.symbol_name,
                kind=chunk.kind,
                relative_path=chunk.relative_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                language=chunk.language.value,
                parent=chunk.parent_symbol,
            )
            self.symbols[qualified] = symbol
            self._by_simple_name[chunk.symbol_name].append(qualified)
            self.graph.add_node(
                qualified,
                kind=chunk.kind,
                path=chunk.relative_path,
                line=chunk.start_line,
                simple_name=chunk.symbol_name,
                language=chunk.language.value,
            )
            if chunk.parent_symbol:
                parent_q = f"{chunk.relative_path}::{chunk.parent_symbol}"
                if parent_q in self.symbols:
                    self.graph.add_edge(parent_q, qualified, type=CONTAINS)

    @staticmethod
    def _qualify(chunk: CodeChunk) -> str:
        """`path::Class.method` — unique, readable, and stable across machines."""
        if chunk.parent_symbol and chunk.kind != "class":
            return f"{chunk.relative_path}::{chunk.parent_symbol}.{chunk.symbol_name}"
        return f"{chunk.relative_path}::{chunk.symbol_name}"

    # -- call linking ------------------------------------------------------ #

    def _link_calls(self, sources: list[SourceFile], chunks: list[CodeChunk]) -> None:
        by_path = {s.relative_path: s for s in sources}
        chunks_by_path: dict[str, list[CodeChunk]] = defaultdict(list)
        for chunk in chunks:
            if chunk.kind != "class":
                chunks_by_path[chunk.relative_path].append(chunk)

        for path, file_chunks in chunks_by_path.items():
            source = by_path.get(path)
            if source is None:
                continue
            try:
                self._link_file(source, file_chunks)
            except Exception as exc:  # noqa: BLE001
                logger.warning("symbol graph: skipping %s (%s)", path, exc)

    def _link_file(self, source: SourceFile, file_chunks: list[CodeChunk]) -> None:
        spec = spec_for_language(source.language)
        raw = source.content.encode("utf-8")
        root = self._parser_for(source.language).parse(raw).root_node

        receivers = self._receiver_types(root, raw, spec)
        imported = self._imported_names(root, raw, spec)

        # line -> enclosing chunk, so a call site attributes to its caller
        line_owner: dict[int, CodeChunk] = {}
        for chunk in file_chunks:
            for line in range(chunk.start_line, chunk.end_line + 1):
                line_owner.setdefault(line, chunk)

        for call in self._find_calls(root, spec):
            line = call.start_point[0] + 1
            caller = line_owner.get(line)
            if caller is None:
                continue
            name, receiver = self._call_target(call, raw, source.language)
            if not name:
                continue
            self.stats.call_sites += 1
            target = self._resolve(name, receiver, receivers, imported, source.language)
            if target is None:
                continue
            caller_q = self._qualify(caller)
            if caller_q != target and caller_q in self.symbols:
                self.graph.add_edge(caller_q, target, type=CALLS, line=line)

    def _resolve(
        self,
        name: str,
        receiver: str | None,
        receivers: dict[str, str],
        imported: set[str],
        language: Language,
    ) -> str | None:
        """Apply the four layers in order. Never guess between equals."""
        # Same-language scoping first. A Java call can never target a Python
        # method, and without this a name only declared in one language (`save`,
        # inherited from JpaRepository in Java) resolves across the boundary and
        # fabricates a call path that looks entirely plausible.
        # dict.fromkeys: dedupe while keeping order. Java overloads
        # (`getPet(String)` and `getPet(int)`) index twice under one qualified
        # name; counting them as two candidates makes a single unambiguous
        # target look ambiguous and drops the edge.
        candidates = list(
            dict.fromkeys(
                q
                for q in self._by_simple_name.get(name, [])
                if self.symbols[q].language == language.value
            )
        )
        if not candidates:
            self.stats.external += 1
            return None

        # 1. receiver type — `paymentService` declared as `PaymentService`
        if receiver and (declared := receivers.get(_receiver_key(receiver))):
            typed = [q for q in candidates if f"::{declared}." in q]
            if len(typed) == 1:
                self.stats.by_receiver_type += 1
                return typed[0]

        # 2. unique simple name across the whole repository
        if len(candidates) == 1:
            self.stats.by_unique_name += 1
            return candidates[0]

        # 3. narrow by what the calling file imports
        scoped = [
            q
            for q in candidates
            if any(f"::{imp}." in q or q.endswith(f"::{imp}") for imp in imported)
        ]
        if len(scoped) == 1:
            self.stats.by_import_scope += 1
            return scoped[0]

        # 4. still ambiguous — a wrong edge is worse than a missing one
        self.stats.ambiguous_dropped += 1
        return None

    # -- AST helpers ------------------------------------------------------- #

    def _parser_for(self, language: Language) -> Any:
        if language not in self._parsers:
            from tree_sitter_language_pack import get_parser

            self._parsers[language] = get_parser(spec_for_language(language).grammar)
        return self._parsers[language]

    def _find_calls(self, root: Node, spec: LanguageSpec) -> list[Node]:
        found: list[Node] = []

        def visit(node: Node) -> None:
            if node.type in spec.call_nodes:
                found.append(node)
            for child in node.children:
                visit(child)

        visit(root)
        return found

    def _call_target(self, call: Node, raw: bytes, language: Language) -> tuple[str, str | None]:
        """Extract (method name, receiver variable) from a call site."""
        if language is Language.JAVA:
            name_node = call.child_by_field_name("name")
            obj_node = call.child_by_field_name("object")
            if name_node is None:  # object_creation_expression
                type_node = call.child_by_field_name("type")
                return (_text(type_node, raw) if type_node else "", None)
            return _text(name_node, raw), _text(obj_node, raw) if obj_node else None

        function = call.child_by_field_name("function")
        if function is None:
            return "", None
        if function.type in {"attribute", "member_expression"}:
            attr = function.child_by_field_name("attribute") or function.child_by_field_name(
                "property"
            )
            obj = function.child_by_field_name("object")
            return (
                _text(attr, raw) if attr else "",
                _text(obj, raw) if obj else None,
            )
        return _text(function, raw), None

    def _receiver_types(self, root: Node, raw: bytes, spec: LanguageSpec) -> dict[str, str]:
        """Map variable name -> declared type, where the language declares one."""
        mapping: dict[str, str] = {}
        if not spec.field_nodes:
            return mapping

        def visit(node: Node) -> None:
            if node.type in spec.field_nodes:
                type_node = node.child_by_field_name("type")
                if type_node is not None:
                    type_name = _text(type_node, raw).split("<")[0].strip()
                    for ident in _identifiers_in_declarator(node, raw):
                        mapping[ident] = type_name
            for child in node.children:
                visit(child)

        visit(root)
        return mapping

    def _imported_names(self, root: Node, raw: bytes, spec: LanguageSpec) -> set[str]:
        names: set[str] = set()
        for child in root.children:
            if child.type in spec.import_nodes:
                text = _text(child, raw).strip().rstrip(";")
                names.add(text.split(".")[-1].split(" ")[-1])
        return names

    # -- queries ----------------------------------------------------------- #

    def callers_of(self, qualified_name: str) -> list[str]:
        if qualified_name not in self.graph:
            return []
        return [
            u
            for u, _, d in self.graph.in_edges(qualified_name, data=True)
            if d.get("type") == CALLS
        ]

    def callees_of(self, qualified_name: str) -> list[str]:
        if qualified_name not in self.graph:
            return []
        return [
            v
            for _, v, d in self.graph.out_edges(qualified_name, data=True)
            if d.get("type") == CALLS
        ]

    def neighbours(self, qualified_name: str, depth: int = 1) -> list[str]:
        """Callers and callees within `depth` hops — what expansion.py needs."""
        if qualified_name not in self.graph:
            return []
        seen: set[str] = {qualified_name}
        frontier = {qualified_name}
        for _ in range(depth):
            nxt: set[str] = set()
            for node in frontier:
                nxt.update(self.callers_of(node))
                nxt.update(self.callees_of(node))
            nxt -= seen
            seen |= nxt
            frontier = nxt
        return sorted(seen - {qualified_name})

    def path_between(self, source: str, target: str) -> list[str]:
        """Shortest call path, e.g. Controller -> Service -> Repository.

        Empty when no path exists — which is a real answer, not a failure.
        """
        if source not in self.graph or target not in self.graph:
            return []
        calls_only = self.graph.edge_subgraph(
            [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == CALLS]
        )
        try:
            return list(nx.shortest_path(calls_only, source, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def find(self, simple_name: str) -> list[str]:
        return list(self._by_simple_name.get(simple_name, []))

    # -- persistence ------------------------------------------------------- #

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [{"id": n, **d} for n, d in self.graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.graph.edges(data=True)],
            "stats": self.stats.as_dict(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SymbolGraph:
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph = cls()
        for node in payload["nodes"]:
            node_id = node.pop("id")
            graph.graph.add_node(node_id, **node)
            graph._by_simple_name[node.get("simple_name", "")].append(node_id)
        for edge in payload["edges"]:
            graph.graph.add_edge(edge.pop("source"), edge.pop("target"), **edge)
        return graph


def _receiver_key(receiver: str) -> str:
    """Reduce a receiver expression to the variable name the field map holds.

    `this.vetRepository` -> `vetRepository`. Spring code writes `this.field`
    constantly, and without stripping it every field-typed call falls through
    to the weaker name-uniqueness layer. Multi-line or call-chain receivers
    (`p.getFileName()`, `LocalDate.now()`) have no single variable and return
    unchanged, so the lookup simply misses — which is correct.
    """
    text = receiver.strip()
    if "\n" in text or "(" in text:
        return text
    return text.rsplit(".", 1)[-1] if text.startswith("this.") else text


def _text(node: Node | None, raw: bytes) -> str:
    if node is None:
        return ""
    return raw[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _identifiers_in_declarator(node: Node, raw: bytes) -> list[str]:
    """Variable names bound by a declaration, ignoring the type identifier."""
    out: list[str] = []
    for child in node.children:
        if child.type in {"variable_declarator", "identifier"}:
            name = (
                child.child_by_field_name("name") if child.type == "variable_declarator" else child
            )
            if name is not None:
                out.append(_text(name, raw))
    return out
