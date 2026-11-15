"""Locator: top hits as `file:line`, in retrieval order, with no model involved."""

from __future__ import annotations

from codemind.agents.nodes.locator import make_locator
from codemind.agents.state import CodeMindState, initial_state
from codemind.core.config import Settings
from codemind.retrieval.schemas import ScoredChunk
from tests.agent_support import scored


def state_with(chunks: list[ScoredChunk]) -> CodeMindState:
    state = initial_state("Where is the controller handling owner creation?")
    state["retrieved"] = chunks
    return state


async def test_lists_the_top_hits_with_file_and_lines_in_order() -> None:
    chunks = [
        scored("initCreationForm", path="src/owner/OwnerController.java", lines=(60, 64)),
        scored(
            "findById",
            path="src/owner/OwnerRepository.java",
            lines=(30, 31),
            parent="OwnerRepository",
        ),
        scored("Owner", path="src/owner/Owner.java", lines=(1, 90), parent=None),
    ]
    update = await make_locator(Settings(locator_top_k=5))(state_with(chunks))

    assert update["messages"] == [
        {
            "node": "locator",
            "content": "\n".join(
                [
                    "1. `src/owner/OwnerController.java:60-64` — OwnerController.initCreationForm (method)",
                    "2. `src/owner/OwnerRepository.java:30-31` — OwnerRepository.findById (method)",
                    "3. `src/owner/Owner.java:1-90` — Owner (method)",
                ]
            ),
        }
    ]
    assert [(e["file_path"], e["start_line"], e["end_line"]) for e in update["evidence"]] == [
        ("src/owner/OwnerController.java", 60, 64),
        ("src/owner/OwnerRepository.java", 30, 31),
        ("src/owner/Owner.java", 1, 90),
    ]


async def test_caps_at_locator_top_k() -> None:
    chunks = [scored(f"m{i}", lines=(i, i + 1)) for i in range(1, 9)]
    update = await make_locator(Settings(locator_top_k=3))(state_with(chunks))
    assert len(update["evidence"]) == 3
    assert update["messages"][0]["content"].count("\n") == 2


async def test_nothing_retrieved_is_reported_not_invented() -> None:
    update = await make_locator()(state_with([]))
    assert "evidence" not in update
    assert update["errors"] == ["locator: no retrieved context"]


async def test_returns_only_the_keys_it_owns() -> None:
    update = await make_locator()(state_with([scored("a")]))
    assert set(update) <= {"evidence", "messages", "errors"}
