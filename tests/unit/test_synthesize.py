"""Synthesize with a stubbed LLM.

What a stub proves: the route picks the prompt and the schema, the schema
makes every field required, evidence is carried through from the claims
untouched (never written by the model, never filtered), only cited code is
shown, and every failure degrades the answer instead of raising. Whether the
3B model writes a good fix is `tests/integration/test_graph_live.py`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from codemind.agents.nodes.synthesize import (
    ArchitectureAnswer,
    BugAnswer,
    DependencyAnswer,
    cited_sources,
    make_synthesize,
)
from codemind.agents.state import CodeMindState, Evidence, Route, initial_state
from codemind.core.exceptions import LLMError
from codemind.llm.prompts import render_prompt
from tests.agent_support import StubLLM, scored

PLACE = scored(
    "placeOrder",
    path="src/OrderService.java",
    lines=(18, 23),
    parent="OrderService",
    body="paymentService.processPayment(order.getTotal());",
)
PAY = scored(
    "processPayment",
    path="src/PaymentService.java",
    lines=(10, 15),
    parent="PaymentService",
    body='throw new PaymentTimeoutException("gateway exceeded");',
)
UNCITED = scored(
    "findById", path="src/OrderService.java", lines=(25, 27), body="return repo.findById(id);"
)

CLAIMS: list[Evidence] = [
    {
        "file_path": "src/OrderService.java",
        "start_line": 18,
        "end_line": 23,
        "claim": "placeOrder does not catch the timeout",
    },
    {
        "file_path": "src/PaymentService.java",
        "start_line": 10,
        "end_line": 15,
        "claim": "processPayment throws PaymentTimeoutException",
    },
    {
        "file_path": "src/OrderService.java",
        "start_line": 18,
        "end_line": 23,
        "claim": "placeOrder calls processPayment",
    },
]

BUG = {
    "root_cause": "PaymentTimeoutException escapes placeOrder [1][2]",
    "recommended_fix": ["Catch PaymentTimeoutException in placeOrder and return 503"],
    "suggested_test": "Stub the gateway to exceed TIMEOUT_MS; assert 503",
}


def state_with(route: Route, claims: list[Evidence] = CLAIMS) -> CodeMindState:
    state = initial_state("Why does POST /orders sometimes return 500?")
    state["route"] = [route]
    state["retrieved"] = [PLACE, PAY, UNCITED]
    state["evidence"] = list(claims)
    return state


@pytest.mark.parametrize(
    ("route", "schema", "template_marker"),
    [
        (Route.BUG, BugAnswer, "recommended_fix"),
        (Route.ARCHITECTURE, ArchitectureAnswer, "mechanism"),
        (Route.DEPENDENCY, DependencyAnswer, "impacted"),
    ],
)
async def test_the_route_selects_prompt_and_schema(
    route: Route, schema: type, template_marker: str
) -> None:
    answers = {
        BugAnswer: BUG,
        ArchitectureAnswer: {"mechanism": "m"},
        DependencyAnswer: {"impacted": ["OrderService"], "reasoning": "r"},
    }
    llm = StubLLM(lambda p, s: answers[s])
    update = await make_synthesize(llm)(state_with(route))

    assert llm.schemas == [schema]
    assert template_marker in llm.prompts[0]
    assert update["answer"][0]["route"] is route
    assert update["answer"][0]["fields"] == answers[schema]
    assert update["answer"][0]["degraded"] is None
    assert "errors" not in update


async def test_evidence_is_every_claim_carried_through_unchanged() -> None:
    """No minting, no filtering: padding included, Day 12 decides what stays."""
    update = await make_synthesize(StubLLM(lambda p, s: BUG))(state_with(Route.BUG))
    assert update["answer"][0]["evidence"] == CLAIMS


async def test_the_prompt_numbers_claims_and_shows_only_cited_code_by_path() -> None:
    llm = StubLLM(lambda p, s: BUG)
    await make_synthesize(llm)(state_with(Route.BUG))
    prompt = llm.prompts[0]

    assert "[1] placeOrder does not catch the timeout" in prompt
    assert "[3] placeOrder calls processPayment" in prompt
    assert "Code for claims [1], [3] — src/OrderService.java" in prompt
    assert "Code for claim [2] — src/PaymentService.java" in prompt
    assert "return repo.findById(id);" not in prompt  # retrieved but uncited
    assert "18-23" not in prompt and ":18" not in prompt  # no line numbers to copy


def test_cited_sources_deduplicates_in_first_cited_order() -> None:
    sources = cited_sources(CLAIMS, [UNCITED, PAY, PLACE])
    assert [(s["chunk"].symbol_name, s["label"]) for s in sources] == [  # type: ignore[attr-defined]
        ("placeOrder", "claims [1], [3]"),
        ("processPayment", "claim [2]"),
    ]


def test_bug_schema_requires_every_field_and_at_least_one_fix_step() -> None:
    """Day 9: the base model silently omits anything optional."""
    schema = BugAnswer.model_json_schema()
    assert set(schema["required"]) == {"root_cause", "recommended_fix", "suggested_test"}
    assert schema["properties"]["recommended_fix"]["minItems"] == 1
    with pytest.raises(ValidationError):
        BugAnswer(root_cause="x", recommended_fix=[], suggested_test="t")
    with pytest.raises(ValidationError):
        BugAnswer(root_cause="x", recommended_fix=[""], suggested_test="t")


def test_every_contract_field_is_required_and_bounded_for_the_grammar() -> None:
    for schema in (BugAnswer, ArchitectureAnswer, DependencyAnswer):
        js = schema.model_json_schema()
        assert set(js["required"]) == set(js["properties"]), schema.__name__
        for name, prop in js["properties"].items():
            bounded = prop.get("maxLength") or (
                prop.get("maxItems") and prop["items"].get("maxLength")
            )
            assert bounded, f"{schema.__name__}.{name} is unbounded"


async def test_no_claims_means_no_llm_call_and_a_degraded_answer() -> None:
    llm = StubLLM(lambda p, s: BUG)
    update = await make_synthesize(llm)(state_with(Route.BUG, claims=[]))

    assert llm.prompts == []
    answer = update["answer"][0]
    assert answer["fields"] is None
    assert answer["degraded"] == "no resolved claims to synthesize from"
    assert update["errors"] == ["synthesize: no resolved claims to synthesize from"]


async def test_an_llm_failure_degrades_the_answer_and_keeps_the_evidence() -> None:
    llm = StubLLM(lambda p, s: LLMError("prompt of 9000 tokens ... does not fit LLM_NUM_CTX=8192"))
    update = await make_synthesize(llm)(state_with(Route.BUG))

    answer = update["answer"][0]
    assert answer["fields"] is None
    assert answer["evidence"] == CLAIMS
    assert "does not fit LLM_NUM_CTX" in answer["degraded"]
    assert len(update["errors"]) == 1


async def test_a_locator_question_has_no_contract_here() -> None:
    llm = StubLLM(lambda p, s: BUG)
    update = await make_synthesize(llm)(state_with(Route.LOCATOR))
    assert llm.prompts == []
    assert "no synthesis contract" in update["answer"][0]["degraded"]


def test_synthesis_prompts_forbid_line_numbers_and_new_citations() -> None:
    for name in ("synthesize_bug", "synthesize_architecture", "synthesize_dependency"):
        prompt = render_prompt(name, question="q", claims=CLAIMS, sources=[])
        assert "Never write a line number" in prompt
        assert "Do not invent claim numbers" in prompt


async def test_returns_only_the_keys_it_owns() -> None:
    update = await make_synthesize(StubLLM(lambda p, s: BUG))(state_with(Route.BUG))
    assert set(update) <= {"answer", "messages", "errors"}
