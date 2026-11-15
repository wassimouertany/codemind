"""Graph state: what every node reads, and the reducers that merge what they write.

Every key a node may write has an appending reducer. LangGraph's default for a
key without one is last-write-wins, and when two nodes in the same superstep
both write it, the graph raises rather than pick one. Neither is acceptable
here: the grounding retry re-enters retrieval, and the evidence from the first
pass must survive the second. So lists concatenate and `attempts` sums.

`question` is the one exception. It is the input, written once by the caller
and by no node, so it keeps the default channel; a node that tried to write it
would be a bug the default channel surfaces.

Nodes return a partial update containing only the keys they own — the
`*Update` TypedDicts below are those contracts, and mypy checks a node's return
against its own. Read the current route with `current_route`, never
`state["route"][0]`: routes append too, so the latest decision is the last.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Any, Protocol, TypedDict, TypeVar

from codemind.retrieval.schemas import ScoredChunk


class Route(StrEnum):
    """What kind of question this is. The router's only output.

    The four categories of the Gate 1 gold set (7/7/8/8 of 30). There is no
    `test` route: the gold set has no test questions, and with `INDEX_TESTS`
    off retrieval could never return a test file to answer one from. Test
    suggestions belong in the bug route's output instead.
    """

    BUG = "bug"
    ARCHITECTURE = "architecture"
    DEPENDENCY = "dependency"
    LOCATOR = "locator"


DEFAULT_ROUTE = Route.BUG
"""Where an unclassifiable question goes. Bug analysis traces a call path end
to end, so it is the specialist that degrades most gracefully on a question
meant for another."""


class Evidence(TypedDict):
    """A line range a claim rests on. The grounding gate resolves these against
    the retrieved chunk payloads before an answer may cite them."""

    file_path: str
    start_line: int
    end_line: int
    claim: str


class AgentMessage(TypedDict):
    """One node's contribution to the transcript, in execution order."""

    node: str
    content: str


class FinalAnswer(TypedDict):
    """The synthesized answer: the route's structured fields plus its evidence.

    `fields` is the route's output schema as a dict (`root_cause`,
    `recommended_fix`, ... for a bug), exactly as the grammar constrained it.
    `evidence` is carried through from the specialist's resolved claims, never
    produced by the synthesizer. `degraded` says why the answer is incomplete,
    or is None when it is not.
    """

    route: Route
    fields: dict[str, Any] | None
    evidence: list[Evidence]
    degraded: str | None


class CodeMindState(TypedDict, total=False):
    question: str
    route: Annotated[list[Route], operator.add]
    retrieved: Annotated[list[ScoredChunk], operator.add]
    evidence: Annotated[list[Evidence], operator.add]
    messages: Annotated[list[AgentMessage], operator.add]
    attempts: Annotated[int, operator.add]
    errors: Annotated[list[str], operator.add]
    answer: Annotated[list[FinalAnswer], operator.add]


class RouterUpdate(TypedDict, total=False):
    """The keys the router owns."""

    route: list[Route]
    messages: list[AgentMessage]
    errors: list[str]


class RetrieveUpdate(TypedDict, total=False):
    """The keys the retrieve node owns. Each pass counts as one attempt, so the
    grounding retry (Day 12) can tell a first pass from an expanded one."""

    retrieved: list[ScoredChunk]
    attempts: int
    messages: list[AgentMessage]
    errors: list[str]


class AnalysisUpdate(TypedDict, total=False):
    """The keys a specialist (code analyst, dependency agent, locator) owns."""

    evidence: list[Evidence]
    messages: list[AgentMessage]
    errors: list[str]


class SynthesizeUpdate(TypedDict, total=False):
    """The keys the synthesizer owns. Answers append, like routes: after the
    grounding retry the latest answer is the last one."""

    answer: list[FinalAnswer]
    messages: list[AgentMessage]
    errors: list[str]


UpdateT_co = TypeVar("UpdateT_co", covariant=True)


class Node(Protocol[UpdateT_co]):
    """A graph node returning the update type it owns.

    A protocol rather than `Callable`: LangGraph calls nodes as `node(state=...)`,
    and a bare `Callable[[CodeMindState], ...]` cannot promise a parameter name.
    """

    async def __call__(self, state: CodeMindState) -> UpdateT_co: ...


def initial_state(question: str) -> CodeMindState:
    """The input a graph run starts from. Every appending key starts empty."""
    return CodeMindState(
        question=question,
        route=[],
        retrieved=[],
        evidence=[],
        messages=[],
        attempts=0,
        errors=[],
        answer=[],
    )


def current_route(state: CodeMindState) -> Route:
    """The latest routing decision, or the default when none was made."""
    routes = state.get("route") or []
    return routes[-1] if routes else DEFAULT_ROUTE
