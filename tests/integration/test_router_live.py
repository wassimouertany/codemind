"""Router accuracy against the real model. Loads the 3B model onto the GPU.

The unit tests prove the plumbing with a stub; only a real model can say
whether the prompt separates the four routes. 12 labelled questions, 3 per
route; >= 85% means at most one miss.
"""

from __future__ import annotations

import httpx
import pytest

from codemind.core.config import get_settings
from codemind.llm.ollama_client import OllamaClient
from tests.agent_support import load_router_cases, route_accuracy

pytestmark = pytest.mark.slow

TARGET = 0.85


@pytest.fixture
def live_ollama_url() -> str:
    url = get_settings().ollama_base_url
    try:
        httpx.get(f"{url}/api/version", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"no Ollama reachable at {url}")
    return url


async def test_router_meets_the_accuracy_target(live_ollama_url: str) -> None:
    async with OllamaClient() as llm:
        accuracy, misses = await route_accuracy(llm, load_router_cases())
    print(f"router accuracy {accuracy:.3f}", *misses, sep="\n  ")
    assert accuracy >= TARGET, "\n".join(misses)
