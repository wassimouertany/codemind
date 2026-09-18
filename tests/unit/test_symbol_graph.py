"""Tests for the symbol graph.

The one that matters most is `test_no_cross_language_edges`. A wrong CALLS edge
produces a confident, fabricated call path — which is worse than no path at all,
because the answer looks right.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codemind.ingestion.ast_chunker import AstChunker
from codemind.ingestion.repo_loader import RepositoryLoader
from codemind.ingestion.symbol_graph import CALLS, CONTAINS, SymbolGraph

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"


@pytest.fixture(scope="module")
def graph() -> SymbolGraph:
    sources = RepositoryLoader(FIXTURE).collect()
    return SymbolGraph.build(sources, AstChunker().chunk_all(sources))


def node_for(graph: SymbolGraph, suffix: str) -> str:
    matches = [n for n in graph.graph if n.endswith(suffix)]
    assert matches, f"no node ending in {suffix!r}"
    return matches[0]


# --------------------------------------------------------------------------- #
# the call chain — the reason this module exists
# --------------------------------------------------------------------------- #


def test_full_call_path_from_controller_to_gateway(graph: SymbolGraph) -> None:
    start = node_for(graph, "OrderController.createOrder")
    end = node_for(graph, "PaymentService.callGateway")
    path = graph.path_between(start, end)

    assert len(path) == 4
    assert path[0].endswith("OrderController.createOrder")
    assert path[1].endswith("OrderService.placeOrder")
    assert path[2].endswith("PaymentService.processPayment")
    assert path[3].endswith("PaymentService.callGateway")


def test_place_order_calls_process_payment(graph: SymbolGraph) -> None:
    """The edge that explains the planted bug."""
    callees = graph.callees_of(node_for(graph, "OrderService.placeOrder"))
    assert any(c.endswith("PaymentService.processPayment") for c in callees)


def test_callers_of_place_order(graph: SymbolGraph) -> None:
    callers = graph.callers_of(node_for(graph, "OrderService.placeOrder"))
    assert any(c.endswith("OrderController.createOrder") for c in callers)


def test_no_path_where_none_exists(graph: SymbolGraph) -> None:
    """An empty path is a real answer, not a failure."""
    start = node_for(graph, "PaymentService.callGateway")
    end = node_for(graph, "OrderController.createOrder")
    assert graph.path_between(start, end) == []


def test_unknown_symbols_return_empty(graph: SymbolGraph) -> None:
    assert graph.callers_of("nope::Nope.nope") == []
    assert graph.callees_of("nope::Nope.nope") == []
    assert graph.neighbours("nope::Nope.nope") == []
    assert graph.path_between("a::b", "c::d") == []


# --------------------------------------------------------------------------- #
# correctness of resolution
# --------------------------------------------------------------------------- #


def test_no_cross_language_edges(graph: SymbolGraph) -> None:
    """A Java call can never target a Python method.

    Without language scoping, `orderRepository.save(order)` in Java resolved to
    UserRepository.save in Python — `save` is inherited from JpaRepository and
    so is declared nowhere in the Java source, making the Python one the unique
    match. The resulting call path was entirely plausible and entirely wrong.
    """
    for source, target, data in graph.graph.edges(data=True):
        if data.get("type") != CALLS:
            continue
        source_lang = graph.graph.nodes[source].get("language")
        target_lang = graph.graph.nodes[target].get("language")
        assert source_lang == target_lang, f"{source} -> {target} crosses languages"


def test_python_chain_is_intact(graph: SymbolGraph) -> None:
    callees = graph.callees_of(node_for(graph, "UserService.get_user_by_id"))
    assert any(c.endswith("UserRepository.find_by_id") for c in callees)


def test_receiver_type_resolution_is_used(graph: SymbolGraph) -> None:
    """Java field declarations bind a variable name to a type."""
    assert graph.stats.by_receiver_type > 0


def test_library_calls_are_external_not_failures(graph: SymbolGraph) -> None:
    """System.currentTimeMillis can never resolve; it must not count against us."""
    assert graph.stats.external > 0
    assert graph.stats.intra_repo_sites < graph.stats.call_sites


def test_resolution_is_perfect_on_the_fixture(graph: SymbolGraph) -> None:
    assert graph.stats.resolution_rate == 1.0
    assert graph.stats.ambiguous_dropped == 0


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


def test_contains_edges_link_class_to_methods(graph: SymbolGraph) -> None:
    cls = node_for(graph, "OrderService.java::OrderService")
    contained = [
        v for _, v, d in graph.graph.out_edges(cls, data=True) if d.get("type") == CONTAINS
    ]
    assert any(v.endswith("OrderService.placeOrder") for v in contained)


def test_neighbours_respects_depth(graph: SymbolGraph) -> None:
    start = node_for(graph, "OrderService.placeOrder")
    one = set(graph.neighbours(start, depth=1))
    two = set(graph.neighbours(start, depth=2))
    assert one <= two
    assert any(n.endswith("PaymentService.callGateway") for n in two)
    assert not any(n.endswith("PaymentService.callGateway") for n in one)


def test_find_by_simple_name(graph: SymbolGraph) -> None:
    assert any(q.endswith("OrderService.placeOrder") for q in graph.find("placeOrder"))
    assert graph.find("doesNotExist") == []


def test_merged_accessors_are_not_indexed(graph: SymbolGraph) -> None:
    """Merged chunks have no single identity, so they cannot be graph nodes."""
    assert not any("," in graph.graph.nodes[n].get("simple_name", "") for n in graph.graph)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #


def test_save_and_load_roundtrip(graph: SymbolGraph, tmp_path: Path) -> None:
    target = tmp_path / "graph.json"
    graph.save(target)
    restored = SymbolGraph.load(target)

    assert restored.graph.number_of_nodes() == graph.graph.number_of_nodes()
    assert restored.graph.number_of_edges() == graph.graph.number_of_edges()

    start = node_for(restored, "OrderController.createOrder")
    end = node_for(restored, "PaymentService.callGateway")
    assert len(restored.path_between(start, end)) == 4


# --------------------------------------------------------------------------- #
# resolution edge cases found on a real repository
# --------------------------------------------------------------------------- #


def test_overloads_do_not_look_ambiguous() -> None:
    """Java overloads share a qualified name; two entries is not two targets.

    `getPet(String)` and `getPet(int)` both index under
    `Owner.java::Owner.getPet`. Counting them as two candidates made a single
    unambiguous target look ambiguous and silently dropped 16 edges on Spring
    PetClinic. Bodies here are multi-statement on purpose — trivial one-liners
    merge into an accessor chunk, which the graph skips, so they would not
    exercise the path at all.
    """
    from codemind.core.types import Language, SourceFile

    code = """
class Owner {
    Pet getPet(String name) {
        Pet found = null;
        for (Pet p : pets) { if (p.name.equals(name)) { found = p; } }
        return found;
    }
    Pet getPet(int id) {
        Pet found = null;
        for (Pet p : pets) { if (p.id == id) { found = p; } }
        return found;
    }
    void show() {
        Pet a = getPet("x");
        System.out.println(a);
    }
}
"""
    source = SourceFile(Path("Owner.java"), "Owner.java", Language.JAVA, len(code), code)
    graph = SymbolGraph.build([source], AstChunker().chunk(source))

    assert len(graph.find("getPet")) == 2, "both overloads should index"
    assert len(set(graph.find("getPet"))) == 1, "under one qualified name"
    assert graph.stats.ambiguous_dropped == 0
    callees = graph.callees_of("Owner.java::Owner.show")
    assert any(c.endswith("Owner.getPet") for c in callees)


def test_this_prefixed_receiver_resolves_by_type() -> None:
    """`this.vetRepository.findAll()` must use the field's declared type.

    The receiver map is keyed on the bare field name, so without stripping
    `this.` every field-typed call fell through to the weaker name-uniqueness
    layer. Spring code writes `this.field` constantly.
    """
    from codemind.core.types import Language, SourceFile

    repo = "class VetRepository { void findAll() {} }"
    ctrl = """
class VetController {
    private final VetRepository vetRepository;
    void list() { this.vetRepository.findAll(); }
}
"""
    sources = [
        SourceFile(
            Path("VetRepository.java"), "VetRepository.java", Language.JAVA, len(repo), repo
        ),
        SourceFile(
            Path("VetController.java"), "VetController.java", Language.JAVA, len(ctrl), ctrl
        ),
    ]
    chunker = AstChunker()
    graph = SymbolGraph.build(sources, chunker.chunk_all(sources))

    assert graph.stats.by_receiver_type >= 1
    callees = graph.callees_of("VetController.java::VetController.list")
    assert any(c.endswith("VetRepository.findAll") for c in callees)


def test_call_chain_receivers_are_not_misresolved() -> None:
    """`p.getFileName().toString()` has no single variable — a miss is correct."""
    from codemind.ingestion.symbol_graph import _receiver_key

    assert _receiver_key("this.vetRepository") == "vetRepository"
    assert _receiver_key("owner") == "owner"
    assert _receiver_key("p.getFileName()") == "p.getFileName()"
