"""The four agent tools on the fixture repo.

The gate is `test_trace_reproduces_the_planted_bug_path`: the fixture's bug
must come out as the 4-hop chain the reference answer cites, with every hop's
line range resolving to that method in the real file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from codemind.agents.tools.find_definition import FindDefinition
from codemind.agents.tools.read_file import ReadFile
from codemind.agents.tools.schemas import CodeHit, Definition
from codemind.agents.tools.search_code import SearchCode
from codemind.agents.tools.trace_call_path import TraceCallPath
from codemind.core.config import Settings
from codemind.core.exceptions import RetrievalError, ToolError
from codemind.ingestion.ast_chunker import AstChunker
from codemind.ingestion.repo_loader import RepositoryLoader
from codemind.ingestion.symbol_graph import SymbolGraph
from tests.agent_support import StubRetriever, scored

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"
CONTROLLER = "src/main/java/com/demo/controller/OrderController.java"
PAYMENT = "src/main/java/com/demo/service/PaymentService.java"

PLANTED_PATH = [
    "OrderController.createOrder",
    "OrderService.placeOrder",
    "PaymentService.processPayment",
    "PaymentService.callGateway",
]


@pytest.fixture(scope="module")
def graph() -> SymbolGraph:
    sources = RepositoryLoader(FIXTURE).collect()
    return SymbolGraph.build(sources, AstChunker().chunk_all(sources))


@pytest.fixture(scope="module")
def reloaded(graph: SymbolGraph, tmp_path_factory: pytest.TempPathFactory) -> SymbolGraph:
    """What serving actually uses: the graph ingestion saved, loaded back."""
    path = tmp_path_factory.mktemp("graphs") / "fixture.json"
    graph.save(path)
    return SymbolGraph.load(path)


@pytest.fixture
def read_file() -> ReadFile:
    return ReadFile(FIXTURE)


# --------------------------------------------------------------------------- #
# trace_call_path — the gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("which", ["built", "reloaded"])
async def test_trace_reproduces_the_planted_bug_path(
    which: str, graph: SymbolGraph, reloaded: SymbolGraph, read_file: ReadFile
) -> None:
    trace = await TraceCallPath(graph if which == "built" else reloaded)(
        "OrderController.createOrder", "PaymentService.callGateway"
    )

    assert trace.found
    assert [hop.display_name for hop in trace.hops] == PLANTED_PATH
    for hop in trace.hops:
        text = "\n".join((await read_file(hop.file_path, hop.start_line, hop.end_line)).lines)
        assert f"{hop.simple_name}(" in text, (
            f"{hop.display_name} not at {hop.file_path}:{hop.start_line}"
        )


async def test_trace_accepts_simple_names(graph: SymbolGraph) -> None:
    trace = await TraceCallPath(graph)("createOrder", "callGateway")
    assert [hop.display_name for hop in trace.hops] == PLANTED_PATH


async def test_trace_follows_call_direction_only(graph: SymbolGraph) -> None:
    """The data layer does not call back up to the endpoint."""
    trace = await TraceCallPath(graph)("PaymentService.callGateway", "OrderController.createOrder")
    assert not trace.found
    assert trace.unresolved == ()


async def test_trace_names_an_unknown_endpoint_instead_of_saying_no_path(
    graph: SymbolGraph,
) -> None:
    trace = await TraceCallPath(graph)("createOrder", "chargeCard")
    assert not trace.found
    assert trace.unresolved == ("chargeCard",)


# --------------------------------------------------------------------------- #
# find_definition
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spelling",
    [
        "processPayment",
        "PaymentService.processPayment",
        f"{PAYMENT}::PaymentService.processPayment",
    ],
)
async def test_find_definition_accepts_every_spelling(graph: SymbolGraph, spelling: str) -> None:
    assert await FindDefinition(graph)(spelling) == [
        Definition(
            qualified_name=f"{PAYMENT}::PaymentService.processPayment",
            simple_name="processPayment",
            kind="function",
            file_path=PAYMENT,
            start_line=10,
            end_line=15,
        )
    ]


async def test_find_definition_survives_save_and_load(
    graph: SymbolGraph, reloaded: SymbolGraph
) -> None:
    assert await FindDefinition(reloaded)("placeOrder") == await FindDefinition(graph)("placeOrder")


async def test_ambiguous_simple_name_returns_every_definition(graph: SymbolGraph) -> None:
    """`findById` is defined in OrderService and inherited from the repository interface."""
    names = [d.display_name for d in await FindDefinition(graph)("findById")]
    assert names == sorted(names)
    assert "OrderService.findById" in names


async def test_wrong_class_qualifier_finds_nothing(graph: SymbolGraph) -> None:
    assert await FindDefinition(graph)("OrderService.processPayment") == []


async def test_unknown_symbol_is_an_empty_answer_not_an_error(graph: SymbolGraph) -> None:
    assert await FindDefinition(graph)("chargeCard") == []
    assert await FindDefinition(graph)("  ") == []


async def test_a_graph_without_line_ranges_says_to_reingest(graph: SymbolGraph) -> None:
    stale = SymbolGraph()
    stale.graph.add_node("A.java::A.run", kind="function", path="A.java", line=3, simple_name="run")
    stale._by_simple_name["run"].append("A.java::A.run")
    with pytest.raises(ToolError, match="re-run ingestion"):
        await FindDefinition(stale)("run")


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #


async def test_read_file_returns_exactly_the_requested_lines(read_file: ReadFile) -> None:
    slice_ = await read_file(CONTROLLER, 17, 20)
    assert slice_.lines == (
        "    @PostMapping",
        "    public Order createOrder(@RequestBody Order order) {",
        "        return orderService.placeOrder(order);",
        "    }",
    )
    assert (slice_.start_line, slice_.end_line, slice_.total_lines) == (17, 20, 26)


async def test_the_last_line_is_readable(read_file: ReadFile) -> None:
    assert (await read_file(CONTROLLER, 26, 26)).lines == ("}",)


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [(0, 3, "invalid line range"), (5, 4, "invalid line range"), (20, 27, "has 26 lines")],
)
async def test_out_of_bounds_ranges_are_refused_not_clamped(
    read_file: ReadFile, start: int, end: int, message: str
) -> None:
    with pytest.raises(ToolError, match=message):
        await read_file(CONTROLLER, start, end)


async def test_a_range_wider_than_the_limit_is_refused() -> None:
    with pytest.raises(ToolError, match="TOOL_READ_MAX_LINES"):
        await ReadFile(FIXTURE, Settings(tool_read_max_lines=3))(CONTROLLER, 1, 4)


@pytest.mark.parametrize(
    "path",
    ["../../../.env", "/etc/passwd", "src/../../conftest.py", "", "src/main/java/Nope.java"],
)
async def test_paths_outside_the_repository_or_missing_are_refused(
    read_file: ReadFile, path: str
) -> None:
    with pytest.raises(ToolError):
        await read_file(path, 1, 1)


async def test_a_symlink_pointing_out_of_the_repository_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("token\n")
    os.symlink(secret, repo / "link.txt")
    with pytest.raises(ToolError, match="escapes"):
        await ReadFile(repo)("link.txt", 1, 1)


async def test_undecodable_bytes_do_not_hide_the_file(read_file: ReadFile) -> None:
    """`blob.py` is deliberately non-UTF-8."""
    blob = next(FIXTURE.rglob("blob.py")).relative_to(FIXTURE).as_posix()
    assert (await read_file(blob, 1, 1)).total_lines >= 1


# --------------------------------------------------------------------------- #
# search_code
# --------------------------------------------------------------------------- #


async def test_search_code_returns_structured_hits_in_retriever_order() -> None:
    chunks = [
        scored(
            "placeOrder",
            path="src/OrderService.java",
            lines=(18, 23),
            parent="OrderService",
            score=0.2,
        ),
        scored("processPayment", path=PAYMENT, lines=(10, 15), parent="PaymentService", score=0.9),
    ]
    retriever = StubRetriever(chunks)
    hits = await SearchCode(retriever)("payment timeout")

    assert retriever.queries == ["payment timeout"]
    assert hits == [
        CodeHit(
            1, "src/OrderService.java", 18, 23, "OrderService.placeOrder", "method", 0.2, "hybrid"
        ),
        CodeHit(2, PAYMENT, 10, 15, "PaymentService.processPayment", "method", 0.9, "hybrid"),
    ]


async def test_search_code_defaults_to_final_k_and_honours_a_limit() -> None:
    retriever = StubRetriever([scored(f"m{i}", lines=(i, i)) for i in range(1, 9)])
    assert len(await SearchCode(retriever, Settings(final_k=5))("q")) == 5
    assert len(await SearchCode(retriever, Settings(final_k=5))("q", limit=2)) == 2


@pytest.mark.parametrize("query", ["", "   "])
async def test_search_code_refuses_an_empty_query(query: str) -> None:
    with pytest.raises(ToolError, match="non-empty"):
        await SearchCode(StubRetriever())(query)


async def test_search_code_wraps_retrieval_failures() -> None:
    retriever = StubRetriever(error=RetrievalError("qdrant unreachable"))
    with pytest.raises(ToolError, match="qdrant unreachable"):
        await SearchCode(retriever)("q")


async def test_graph_nodes_carry_their_layer_through_save_and_load(reloaded: SymbolGraph) -> None:
    """What a future trace_to_layer() walks to: the first REPOSITORY-layer node."""
    layers = {
        hop.display_name: reloaded.graph.nodes[hop.qualified_name]["layer"]
        for hop in (await TraceCallPath(reloaded)("createOrder", "callGateway")).hops
    }
    assert layers == {
        "OrderController.createOrder": "controller",
        "OrderService.placeOrder": "service",
        "PaymentService.processPayment": "service",
        "PaymentService.callGateway": "service",
    }
    repo = (await FindDefinition(reloaded)("findByCustomerId"))[0]
    assert reloaded.graph.nodes[repo.qualified_name]["layer"] == "repository"
