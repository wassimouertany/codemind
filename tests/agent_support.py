"""Shared doubles and labelled data for agent tests. No model, no server."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from codemind.agents.nodes.router import make_router
from codemind.agents.state import Route, current_route, initial_state
from codemind.core.exceptions import LLMError, RetrievalError
from codemind.core.types import CodeChunk, Language, Layer
from codemind.llm.base import LLMClient, ModelT
from codemind.retrieval.schemas import RetrievalResult, ScoredChunk

ROUTER_QUESTIONS = Path(__file__).parent / "fixtures" / "router_questions.jsonl"


@dataclass(frozen=True, slots=True)
class RouterCase:
    source: str
    """The Gate 1 gold-set id the question and its label come from."""
    question: str
    route: Route


def load_router_cases() -> list[RouterCase]:
    rows = [json.loads(line) for line in ROUTER_QUESTIONS.read_text().splitlines() if line]
    return [RouterCase(r["source"], r["question"], Route(r["route"])) for r in rows]


@dataclass
class StubLLM:
    """An `LLMClient` that answers structured calls from `respond(prompt, schema)`."""

    respond: Callable[[str, type], object]
    prompts: list[str] = field(default_factory=list)
    schemas: list[type] = field(default_factory=list)

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        raise LLMError("StubLLM only answers structured calls")

    async def generate_structured(self, prompt: str, schema: type[ModelT]) -> ModelT:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        answer = self.respond(prompt, schema)
        if isinstance(answer, Exception):
            raise answer
        try:
            return schema.model_validate(answer)
        except ValidationError as exc:  # the protocol: every failure is an LLMError
            raise LLMError(f"does not match {schema.__name__}: {answer!r}") from exc


def scored(
    symbol: str,
    *,
    path: str = "src/main/java/demo/OwnerController.java",
    lines: tuple[int, int] = (10, 20),
    score: float = 0.5,
    parent: str | None = "OwnerController",
    imports: Sequence[str] = (),
    body: str | None = None,
) -> ScoredChunk:
    """A retrieved chunk with a real path and line range, for nodes to cite."""
    return ScoredChunk(
        chunk=CodeChunk(
            relative_path=path,
            language=Language.JAVA,
            symbol_name=symbol,
            kind="method",
            start_line=lines[0],
            end_line=lines[1],
            body=body or f"public void {symbol}() {{ }}",
            context_header=f"{path} | demo | {parent} | void {symbol}()",
            layer=Layer.CONTROLLER,
            parent_symbol=parent,
            imports=list(imports),
        ),
        score=score,
    )


@dataclass
class StubRetriever:
    """A `Retriever` returning fixed chunks, in exactly the order given."""

    chunks: list[ScoredChunk] = field(default_factory=list)
    error: RetrievalError | None = None
    queries: list[str] = field(default_factory=list)

    async def search(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return RetrievalResult(query=query, chunks=list(self.chunks))


async def route_accuracy(llm: LLMClient, cases: Sequence[RouterCase]) -> tuple[float, list[str]]:
    """Run the router node on every case; return accuracy and the misses, readable."""
    router = make_router(llm)
    misses: list[str] = []
    for case in cases:
        update = await router(initial_state(case.question))
        got = current_route({"route": update.get("route", [])})
        if got is not case.route:
            misses.append(f"[{case.source}] expected {case.route}, got {got}: {case.question}")
    return 1 - len(misses) / len(cases), misses
