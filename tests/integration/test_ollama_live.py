"""`OllamaClient` against a real Ollama server. Loads the 3B model onto the GPU.

Marked slow, and skips when nothing answers at `OLLAMA_BASE_URL`, so the
suite passes on a machine without Ollama and never heats the card by default.
"""

from __future__ import annotations

from enum import StrEnum

import httpx
import pytest
from pydantic import BaseModel

from codemind.core.config import get_settings
from codemind.llm.ollama_client import OllamaClient

pytestmark = pytest.mark.slow


class Answer(StrEnum):
    YES = "yes"
    NO = "no"


class Verdict(BaseModel):
    answer: Answer


@pytest.fixture
def live_ollama_url() -> str:
    url = get_settings().ollama_base_url
    try:
        httpx.get(f"{url}/api/version", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"no Ollama reachable at {url}")
    return url


async def test_generate_and_structured_round_trip(live_ollama_url: str) -> None:
    async with OllamaClient() as client:
        text = await client.generate("Reply with the single word: pong", max_tokens=8)
        verdict = await client.generate_structured(
            "Is Python a programming language? Answer as JSON.", Verdict
        )

    assert text.strip()
    assert verdict.answer is Answer.YES
