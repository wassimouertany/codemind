"""What the code analyst and the dependency agent share: reason over retrieved chunks.

The model sees the chunks numbered `[1]..[n]` and answers as a list of claims,
each one sentence tied to exactly one chunk number — never a path. The node
turns each number back into that chunk's real `file_path:start-end`, so an
answer cannot cite a file that was not retrieved: a fabricated path is
impossible by construction rather than caught later. A number outside `1..n`
is dropped and recorded in `errors`; the grounding gate (Day 12) then checks
that each claim is supported by the chunk it cites.

Why claims and not "analysis + citations": measured on the fixture with the
3B model, a free `analysis` string with an optional citation list produced no
citations at all, and forcing a non-empty list made it cite chunks 1..10 in
order. With the chunk number first in each claim, every claim on the
reference question described its cited chunk correctly. Putting the statement
first instead produced citations to chunks that did not exist.

With nothing retrieved the node does not call the model at all. A 3B model
asked to explain code it cannot see will explain code anyway.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from codemind.agents.state import (
    AnalysisUpdate,
    CodeMindState,
    Evidence,
    Node,
    current_route,
)
from codemind.core.exceptions import LLMError
from codemind.llm.base import LLMClient
from codemind.llm.prompts import render_prompt
from codemind.retrieval.schemas import ScoredChunk

logger = logging.getLogger(__name__)


MAX_CLAIMS = 6
"""The 3B model pads to whatever the cap allows, describing chunks the question
never asked about; 6 covers a controller -> service -> repository -> exception
trace plus a fix."""

MAX_STATEMENT_CHARS = 300
"""Enforced by the grammar, not just validated: an unbounded string ran past
`num_predict` on an architecture question and was truncated mid-JSON."""


class Claim(BaseModel):
    chunk: int
    """The bracketed number of the one chunk this claim rests on. First, so the
    model picks its evidence before it writes the sentence."""
    statement: str = Field(max_length=MAX_STATEMENT_CHARS)


class SpecialistAnswer(BaseModel):
    claims: list[Claim] = Field(min_length=1, max_length=MAX_CLAIMS)


def make_specialist(name: str, llm: LLMClient) -> Node[AnalysisUpdate]:
    """A specialist node named `name`, prompted by `configs/prompts/<name>.jinja2`."""

    async def specialist(state: CodeMindState) -> AnalysisUpdate:
        chunks = state.get("retrieved") or []
        if not chunks:
            return {
                "errors": [f"{name}: no retrieved context; analysis skipped"],
                "messages": [{"node": name, "content": "No code was retrieved for this question."}],
            }

        prompt = render_prompt(
            name,
            question=state["question"],
            route=current_route(state).value,
            chunks=[scored.chunk for scored in chunks],
        )
        try:
            answer = await llm.generate_structured(prompt, SpecialistAnswer)
        except LLMError as exc:
            logger.warning("%s produced no usable analysis: %s", name, exc)
            return {
                "errors": [f"{name}: {exc}"],
                "messages": [{"node": name, "content": "Analysis unavailable."}],
            }

        evidence, errors = resolve_claims(name, answer.claims, chunks)
        # Only claims that resolved reach the transcript: a dropped [13] must not
        # survive as text after its evidence was rejected.
        analysis = "\n".join(
            f"{claim.statement} [{claim.chunk}]"
            for claim in answer.claims
            if 1 <= claim.chunk <= len(chunks)
        )
        update: AnalysisUpdate = {
            "evidence": evidence,
            "messages": [{"node": name, "content": analysis}],
        }
        if errors:
            update["errors"] = errors
        return update

    return specialist


def resolve_claims(
    name: str, claims: list[Claim], chunks: list[ScoredChunk]
) -> tuple[list[Evidence], list[str]]:
    """Map each claim's chunk number to a real line range; reject numbers out of range."""
    evidence: list[Evidence] = []
    errors: list[str] = []
    for claim in claims:
        if not 1 <= claim.chunk <= len(chunks):
            errors.append(
                f"{name}: cited [{claim.chunk}] but only {len(chunks)} chunks were "
                f"retrieved; dropped claim {claim.statement!r}"
            )
            continue
        chunk = chunks[claim.chunk - 1].chunk
        evidence.append(
            {
                "file_path": chunk.relative_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "claim": claim.statement,
            }
        )
    return evidence, errors
