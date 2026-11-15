"""Router node with a stubbed LLM.

A stub cannot measure classification quality — it answers whatever it is
told to. What it can prove is everything around the model: the question
reaches the prompt, the call is structured with `RouteDecision`, the answer
lands in state as an enum, and every failure degrades to `bug` with a
recorded error instead of crashing the graph. Accuracy against the real model
is `tests/integration/test_router_live.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codemind.agents.nodes.router import RouteDecision, make_router
from codemind.agents.state import DEFAULT_ROUTE, Route, initial_state
from codemind.core.exceptions import LLMError
from codemind.llm.prompts import render_prompt
from tests.agent_support import StubLLM, load_router_cases, route_accuracy

CASES = load_router_cases()


def test_labelled_set_has_three_questions_per_route() -> None:
    assert len(CASES) == 12
    for route in Route:
        assert sum(case.route is route for case in CASES) == 3, route


def test_labelled_bug_architecture_dependency_questions_come_from_the_gold_set() -> None:
    assert all(c.source != "handwritten" for c in CASES if c.route is not Route.TEST)


async def test_router_carries_each_label_through_to_state() -> None:
    """The harness the live test uses, driven by an oracle. Must score 1.0."""
    labels = {case.question: case.route for case in CASES}

    def oracle(prompt: str, schema: type) -> object:
        question = next(q for q in labels if q in prompt)
        return {"route": labels[question].value}

    llm = StubLLM(oracle)
    accuracy, misses = await route_accuracy(llm, CASES)

    assert accuracy == 1.0, misses
    assert accuracy >= 0.85
    assert llm.schemas == [RouteDecision] * len(CASES)


async def test_the_question_is_rendered_into_the_prompt() -> None:
    llm = StubLLM(lambda prompt, schema: {"route": "dependency"})
    await make_router(llm)(initial_state("Which classes call OwnerRepository?"))
    assert llm.prompts[0].rstrip().endswith("Which classes call OwnerRepository?")


async def test_a_confident_route_records_no_error() -> None:
    update = await make_router(StubLLM(lambda p, s: {"route": "architecture"}))(initial_state("q"))
    assert update["route"] == [Route.ARCHITECTURE]
    assert "errors" not in update
    assert update["messages"] == [{"node": "router", "content": "route=architecture"}]


@pytest.mark.parametrize(
    "failure",
    [
        LLMError("returned output that does not match RouteDecision: 'Sure! bug'"),
        LLMError("Ollama unreachable at http://localhost:11434 after 3 attempts"),
    ],
    ids=["unparseable", "unreachable"],
)
async def test_failures_default_to_bug_and_are_recorded(failure: LLMError) -> None:
    update = await make_router(StubLLM(lambda p, s: failure))(initial_state("q"))

    assert update["route"] == [DEFAULT_ROUTE] == [Route.BUG]
    assert len(update["errors"]) == 1
    assert update["errors"][0].startswith("router: unclassifiable, defaulted to bug")
    assert str(failure) in update["errors"][0]


async def test_an_answer_outside_the_enum_falls_back_instead_of_adding_a_route() -> None:
    """A backend that ignores the grammar must not smuggle a fifth route in."""
    update = await make_router(StubLLM(lambda p, s: {"route": "locator"}))(initial_state("q"))
    assert update["route"] == [Route.BUG]
    assert "locator" in update["errors"][0]


def test_router_prompt_names_every_route() -> None:
    prompt = render_prompt("router", question="anything")
    for route in Route:
        assert f"- {route.value}:" in prompt


def test_prompt_examples_do_not_leak_labelled_questions() -> None:
    prompt = render_prompt("router", question="")
    assert not [c.question for c in CASES if c.question in prompt]


def test_a_missing_template_variable_is_an_error_not_an_empty_prompt() -> None:
    with pytest.raises(LLMError, match="router"):
        render_prompt("router")


def test_prompt_dev_set_is_disjoint_from_the_held_out_questions() -> None:
    """The router prompt is tuned on the dev set; overlap would fit the score."""
    dev = Path(__file__).resolve().parents[2] / "evaluation" / "datasets" / "router_dev.jsonl"
    dev_questions = {json.loads(line)["question"] for line in dev.read_text().splitlines() if line}
    assert dev_questions
    assert not dev_questions & {case.question for case in CASES}
