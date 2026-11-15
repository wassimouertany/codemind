"""Shared doubles and labelled data for agent tests. No model, no server."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from codemind.agents.nodes.router import make_router
from codemind.agents.state import Route, current_route, initial_state
from codemind.core.exceptions import LLMError
from codemind.llm.base import LLMClient, ModelT
from codemind.retrieval.schemas import RetrievalResult

ROUTER_QUESTIONS = Path(__file__).parent / "fixtures" / "router_questions.jsonl"


@dataclass(frozen=True, slots=True)
class RouterCase:
    source: str
    """A Gate 1 gold-set id, or "handwritten" — the gold set has no test-category questions."""
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


class StubRetriever:
    async def search(self, query: str) -> RetrievalResult:
        return RetrievalResult(query=query)


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
