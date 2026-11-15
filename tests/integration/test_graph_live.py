"""The whole graph with the real router and specialists. Loads the 3B model onto the GPU.

Retrieval is stubbed with the fixture repo's real chunks, so this needs Ollama
but not Qdrant: it tests what the model does with context, not how the
context was found. The assertions are the properties that must hold for any
model — valid structured output, citations that resolve to retrieved lines,
no context overflow — plus one fact the fixture's bug makes unambiguous.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from codemind.agents.graph import build_graph, run_graph
from codemind.agents.state import Route
from codemind.core.config import get_settings
from codemind.ingestion.ast_chunker import AstChunker
from codemind.ingestion.repo_loader import RepositoryLoader
from codemind.llm.ollama_client import OllamaClient
from codemind.retrieval.schemas import ScoredChunk
from tests.agent_support import StubRetriever

pytestmark = pytest.mark.slow

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"

# The order a good retriever would return for the reference question.
ORDER = ["placeOrder", "processPayment", "createOrder", "callGateway", "findById", "OrderService"]


@pytest.fixture
def live_ollama_url() -> str:
    url = get_settings().ollama_base_url
    try:
        httpx.get(f"{url}/api/version", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"no Ollama reachable at {url}")
    return url


@pytest.fixture(scope="module")
def java_chunks() -> list[ScoredChunk]:
    chunks = [
        c
        for c in AstChunker().chunk_all(RepositoryLoader(FIXTURE).collect())
        if c.relative_path.endswith(".java") and c.kind != "class"
    ]
    rank = {name: i for i, name in enumerate(ORDER)}
    chunks.sort(key=lambda c: rank.get(c.symbol_name, len(ORDER)))
    return [ScoredChunk(chunk=c, score=1.0 / (i + 1)) for i, c in enumerate(chunks)]


async def ask(question: str, chunks: list[ScoredChunk], caplog: pytest.LogCaptureFixture) -> dict:  # type: ignore[type-arg]
    with caplog.at_level("WARNING", logger="codemind.llm.ollama_client"):
        async with OllamaClient() as llm:
            final = await run_graph(build_graph(llm, StubRetriever(chunks)), question)
    assert "truncat" not in caplog.text, caplog.text
    print(f"\n--- {question}")
    for answer in final["answer"]:
        for name, value in (answer["fields"] or {"degraded": answer["degraded"]}).items():
            print(f"  {name}: {value}")
    if not final["answer"]:
        print(final["messages"][-1]["content"])
    for e in final["evidence"]:
        print(f"    {e['file_path']}:{e['start_line']}-{e['end_line']}  {e['claim']}")
    retrieved = {
        (s.chunk.relative_path, s.chunk.start_line, s.chunk.end_line) for s in final["retrieved"]
    }
    cited = final["evidence"] + [e for a in final["answer"] for e in a["evidence"]]
    assert all((e["file_path"], e["start_line"], e["end_line"]) in retrieved for e in cited)
    return dict(final)


def prose(final: dict) -> str:  # type: ignore[type-arg]
    fields = final["answer"][-1]["fields"]
    return " ".join(v if isinstance(v, str) else " ".join(v) for v in fields.values())


async def test_reference_bug_question(
    live_ollama_url: str, java_chunks: list[ScoredChunk], caplog: pytest.LogCaptureFixture
) -> None:
    final = await ask("Why does POST /orders sometimes return 500?", java_chunks, caplog)
    assert final["route"] == [Route.BUG]
    assert final["errors"] == []
    answer = final["answer"][-1]
    assert answer["degraded"] is None
    fields = answer["fields"]
    assert "PaymentTimeoutException" in fields["root_cause"]
    assert fields["recommended_fix"] and all(step.strip() for step in fields["recommended_fix"])
    assert fields["suggested_test"].strip()
    cited = {e["file_path"].rsplit("/", 1)[-1] for e in answer["evidence"]}
    assert {"OrderService.java", "PaymentService.java"} <= cited  # where it is thrown and missed
    assert not re.search(r"\bline \d+|:\d+\b", prose(final))  # locations stay in evidence


async def test_dependency_question(
    live_ollama_url: str, java_chunks: list[ScoredChunk], caplog: pytest.LogCaptureFixture
) -> None:
    final = await ask("Which classes call PaymentService?", java_chunks, caplog)
    assert final["route"] == [Route.DEPENDENCY]
    assert final["errors"] == []
    answer = final["answer"][-1]
    assert answer["degraded"] is None
    assert "OrderService" in " ".join(answer["fields"]["impacted"])


async def test_locator_question(
    live_ollama_url: str, java_chunks: list[ScoredChunk], caplog: pytest.LogCaptureFixture
) -> None:
    final = await ask("Where is the method that places an order?", java_chunks, caplog)
    assert final["route"] == [Route.LOCATOR]
    assert [m["node"] for m in final["messages"]] == ["router", "retrieve", "locator"]
    assert "OrderService.java" in final["messages"][-1]["content"]
