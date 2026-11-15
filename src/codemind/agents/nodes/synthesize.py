"""Synthesize: turn a specialist's resolved claims into the route's final answer.

One node; the prompt and the output schema are chosen by route. The schema is
the output contract, and it is enforced by the grammar rather than requested
in prose: on Day 9 the base model omitted every field it was allowed to omit
(no citations, no fix), so every field here is required, and the bug answer's
`recommended_fix` must have at least one step. A required slot guarantees a
value, not a good one — fix quality is measured, not assumed.

What the synthesizer may not do:

* mint citations. `evidence` is attached by this node from the specialist's
  claims, already resolved to `file:line`; the model never writes it. The
  prompt shows claims by number and code by path only, so the prose has no
  line numbers to copy.
* filter. Every claim is shown and every claim is carried through as
  evidence, padding included. Specialists emit, synthesis renders, and the
  grounding gate (Day 12) removes what is unsupported — one job per stage, so
  each can be tested alone.

Locator never reaches this node: its answer is the list of hits.
"""

from __future__ import annotations

import logging
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from codemind.agents.state import (
    CodeMindState,
    Evidence,
    FinalAnswer,
    Node,
    Route,
    SynthesizeUpdate,
    current_route,
)
from codemind.core.exceptions import LLMError
from codemind.llm.base import LLMClient
from codemind.llm.prompts import render_prompt
from codemind.retrieval.schemas import ScoredChunk

logger = logging.getLogger(__name__)

NODE_NAME = "synthesize"

# Every string is bounded so the grammar ends it; an unbounded field ran past
# num_predict on Day 9 and was truncated mid-JSON.
Step = Annotated[str, StringConstraints(min_length=1, max_length=300)]
Name = Annotated[str, StringConstraints(min_length=1, max_length=120)]


class BugAnswer(BaseModel):
    root_cause: str = Field(min_length=1, max_length=800)
    recommended_fix: list[Step] = Field(min_length=1, max_length=4)
    suggested_test: str = Field(min_length=1, max_length=500)


class ArchitectureAnswer(BaseModel):
    mechanism: str = Field(min_length=1, max_length=1500)


class DependencyAnswer(BaseModel):
    impacted: list[Name] = Field(min_length=1, max_length=10)
    reasoning: str = Field(min_length=1, max_length=1200)


CONTRACT: dict[Route, tuple[str, type[BaseModel]]] = {
    Route.BUG: ("synthesize_bug", BugAnswer),
    Route.ARCHITECTURE: ("synthesize_architecture", ArchitectureAnswer),
    Route.DEPENDENCY: ("synthesize_dependency", DependencyAnswer),
}


def make_synthesize(llm: LLMClient) -> Node[SynthesizeUpdate]:
    async def synthesize(state: CodeMindState) -> SynthesizeUpdate:
        route = current_route(state)
        claims = list(state.get("evidence") or [])
        if route not in CONTRACT:
            return _degraded(route, claims, f"no synthesis contract for route {route!s}")
        if not claims:
            return _degraded(route, claims, "no resolved claims to synthesize from")

        template, schema = CONTRACT[route]
        prompt = render_prompt(
            template,
            question=state["question"],
            claims=claims,
            sources=cited_sources(claims, state.get("retrieved") or []),
        )
        try:
            answer = await llm.generate_structured(prompt, schema)
        except LLMError as exc:
            logger.error("synthesis failed for a %s question: %s", route, exc)
            return _degraded(route, claims, f"synthesis failed: {exc}")

        final: FinalAnswer = {
            "route": route,
            "fields": answer.model_dump(),
            "evidence": claims,
            "degraded": None,
        }
        return {
            "answer": [final],
            "messages": [
                {"node": NODE_NAME, "content": f"{route} answer from {len(claims)} claims"}
            ],
        }

    return synthesize


def cited_sources(claims: list[Evidence], retrieved: list[ScoredChunk]) -> list[dict[str, object]]:
    """The retrieved chunks the claims rest on, once each, in first-cited order.

    Only cited chunks are shown: an uncited chunk in the prompt is an invitation
    to state something no claim supports. Each carries the numbers of the claims
    that cite it, so the model can tie code back to `[n]`.
    """
    by_location = {
        (s.chunk.relative_path, s.chunk.start_line, s.chunk.end_line): s.chunk for s in retrieved
    }
    sources: dict[tuple[str, int, int], list[int]] = {}
    for number, claim in enumerate(claims, start=1):
        key = (claim["file_path"], claim["start_line"], claim["end_line"])
        if key in by_location:
            sources.setdefault(key, []).append(number)
    return [
        {
            "chunk": by_location[key],
            "label": ("claim " if len(numbers) == 1 else "claims ")
            + ", ".join(f"[{n}]" for n in numbers),
        }
        for key, numbers in sources.items()
    ]


def _degraded(route: Route, claims: list[Evidence], reason: str) -> SynthesizeUpdate:
    final: FinalAnswer = {"route": route, "fields": None, "evidence": claims, "degraded": reason}
    return {
        "answer": [final],
        "errors": [f"{NODE_NAME}: {reason}"],
        "messages": [{"node": NODE_NAME, "content": f"{route} answer degraded: {reason}"}],
    }
