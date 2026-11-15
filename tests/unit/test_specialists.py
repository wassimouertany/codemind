"""Code analyst and dependency agent with a stubbed LLM.

What a stub can prove: the model sees numbered chunks and the question, the
route picks the task, each claim becomes evidence with its chunk's real path
and lines, and nothing is fabricated when the model cites a chunk that was not
retrieved or when there is nothing to cite at all. Whether the 3B model writes
good claims is `tests/integration/test_graph_live.py`.
"""

from __future__ import annotations

import pytest

from codemind.agents.nodes.code_analyst import make_code_analyst
from codemind.agents.nodes.dependency_agent import make_dependency_agent
from codemind.agents.nodes.specialist import MAX_CLAIMS, MAX_STATEMENT_CHARS, SpecialistAnswer
from codemind.agents.state import CodeMindState, Route, initial_state
from codemind.core.exceptions import LLMError
from codemind.retrieval.schemas import ScoredChunk
from tests.agent_support import StubLLM, scored

CHUNKS = [
    scored("processCreationForm", path="src/owner/OwnerController.java", lines=(45, 60)),
    scored(
        "save",
        path="src/owner/OwnerRepository.java",
        lines=(12, 14),
        parent="OwnerRepository",
        imports=["org.springframework.data.repository.Repository"],
    ),
]


def state_with(route: Route, chunks: list[ScoredChunk] = CHUNKS) -> CodeMindState:
    state = initial_state("Why does saving an owner fail?")
    state["route"] = [route]
    state["retrieved"] = list(chunks)
    return state


def answering(*claims: tuple[int, str]) -> StubLLM:
    return StubLLM(lambda p, s: {"claims": [{"chunk": n, "statement": t} for n, t in claims]})


async def test_prompt_numbers_chunks_with_real_locations_and_asks_the_question() -> None:
    llm = answering((1, "x"))
    await make_code_analyst(llm, Route.BUG)(state_with(Route.BUG))

    prompt = llm.prompts[0]
    assert llm.schemas == [SpecialistAnswer]
    assert "Question: Why does saving an owner fail?" in prompt
    assert "[1] src/owner/OwnerController.java:45-60" in prompt
    assert "[2] src/owner/OwnerRepository.java:12-14" in prompt
    assert prompt.index("[1] ") < prompt.index("[2] ")
    assert "public void save() { }" in prompt


async def test_bug_route_asks_for_a_root_cause_and_architecture_for_design() -> None:
    bug, arch = answering((1, "x")), answering((1, "x"))
    await make_code_analyst(bug, Route.BUG)(state_with(Route.BUG))
    await make_code_analyst(arch, Route.ARCHITECTURE)(state_with(Route.ARCHITECTURE))

    assert "root cause" in bug.prompts[0]
    assert "root cause" not in arch.prompts[0]
    assert "designed" in arch.prompts[0]


async def test_dependency_agent_sees_imports() -> None:
    llm = answering((1, "x"))
    await make_dependency_agent(llm)(state_with(Route.DEPENDENCY))
    assert "imports: org.springframework.data.repository.Repository" in llm.prompts[0]


async def test_each_claim_resolves_to_its_chunks_path_and_lines() -> None:
    llm = answering((1, "processCreationForm calls save"), (2, "save persists the owner"))
    update = await make_code_analyst(llm, Route.BUG)(state_with(Route.BUG))

    assert update["evidence"] == [
        {
            "file_path": "src/owner/OwnerController.java",
            "start_line": 45,
            "end_line": 60,
            "claim": "processCreationForm calls save",
        },
        {
            "file_path": "src/owner/OwnerRepository.java",
            "start_line": 12,
            "end_line": 14,
            "claim": "save persists the owner",
        },
    ]
    assert update["messages"] == [
        {
            "node": "bug_analyst",
            "content": "processCreationForm calls save [1]\nsave persists the owner [2]",
        }
    ]
    assert "errors" not in update


@pytest.mark.parametrize("bad", [0, 3, 13, -1])
async def test_a_claim_citing_an_unretrieved_chunk_is_dropped_everywhere(bad: int) -> None:
    """Measured: the 3B model cited [13] and [17] with 10 chunks in context."""
    llm = answering((1, "real"), (bad, "invented"))
    update = await make_code_analyst(llm, Route.BUG)(state_with(Route.BUG))

    assert [e["claim"] for e in update["evidence"]] == ["real"]
    assert "invented" not in update["messages"][0]["content"]
    assert len(update["errors"]) == 1
    assert f"cited [{bad}] but only 2 chunks" in update["errors"][0]


async def test_nothing_retrieved_means_no_llm_call_and_no_evidence() -> None:
    """A 3B model asked to explain code it cannot see will explain code anyway."""
    llm = answering((1, "invented"))
    update = await make_code_analyst(llm, Route.BUG)(state_with(Route.BUG, chunks=[]))

    assert llm.prompts == []
    assert "evidence" not in update
    assert update["errors"] == ["bug_analyst: no retrieved context; analysis skipped"]


async def test_an_llm_failure_is_recorded_not_raised() -> None:
    llm = StubLLM(lambda p, s: LLMError("does not match SpecialistAnswer"))
    update = await make_dependency_agent(llm)(state_with(Route.DEPENDENCY))

    assert "evidence" not in update
    assert update["errors"] == ["dependency_agent: does not match SpecialistAnswer"]
    assert update["messages"] == [{"node": "dependency_agent", "content": "Analysis unavailable."}]


def test_the_schema_bounds_claims_and_statement_length_for_the_grammar() -> None:
    """The backend compiles these into the grammar; an unbounded answer truncated mid-JSON."""
    schema = SpecialistAnswer.model_json_schema()
    claims = schema["properties"]["claims"]
    assert (claims["minItems"], claims["maxItems"]) == (1, MAX_CLAIMS)
    claim = schema["$defs"]["Claim"]
    assert list(claim["properties"]) == ["chunk", "statement"]  # evidence first
    assert claim["properties"]["statement"]["maxLength"] == MAX_STATEMENT_CHARS


async def test_returns_only_the_keys_it_owns() -> None:
    update = await make_code_analyst(answering((5, "bad")), Route.BUG)(state_with(Route.BUG))
    assert set(update) <= {"evidence", "messages", "errors"}


async def test_both_analyst_routes_share_one_prompt_but_get_their_own_node_name() -> None:
    """Spans are named after nodes; Day 13 must see which route each one answered."""
    bug, arch = answering((1, "x")), answering((1, "x"))
    bug_update = await make_code_analyst(bug, Route.BUG)(state_with(Route.BUG))
    arch_update = await make_code_analyst(arch, Route.ARCHITECTURE)(state_with(Route.ARCHITECTURE))

    assert bug_update["messages"][0]["node"] == "bug_analyst"
    assert arch_update["messages"][0]["node"] == "architecture_analyst"
    assert bug.prompts[0].startswith(arch.prompts[0][:60])  # same template


def test_specialist_prompts_forbid_line_numbers_in_claims() -> None:
    """The citation carries the range; Day 12 flags any bare line reference."""
    from codemind.llm.prompts import render_prompt

    for name in ("code_analyst", "dependency_agent"):
        prompt = render_prompt(name, question="q", route="bug", chunks=[])
        assert "Never write a line number in a claim" in prompt
