"""`OllamaClient` against a mocked transport. No server, no model.

The retry tests are the ones that matter: retrying a 4xx turns "model not
found" into a 30-second wait for the same error, and not retrying a refused
connection turns an Ollama restart into a failed request.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum

import httpx
import pytest
from pydantic import BaseModel
from tenacity import wait_none

import codemind.llm.ollama_client as ollama_client
from codemind.core.config import Settings
from codemind.core.exceptions import LLMError
from codemind.llm.base import LLMClient
from codemind.llm.ollama_client import OllamaClient

Handler = Callable[[httpx.Request], httpx.Response]


class Colour(StrEnum):
    RED = "red"
    BLUE = "blue"


class Pick(BaseModel):
    colour: Colour


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_client, "RETRY_WAIT", wait_none())


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "ollama_base_url": "http://ollama.test",
        "llm_model": "test-model",
        "llm_max_tokens": 256,
        "llm_temperature": 0.3,
        "llm_num_ctx": 4096,
        "llm_max_retries": 3,
    }
    return Settings(**(base | overrides))  # type: ignore[arg-type]


def make_client(handler: Handler, **overrides: object) -> OllamaClient:
    s = settings(**overrides)
    transport = httpx.MockTransport(handler)
    return OllamaClient(
        s, client=httpx.AsyncClient(base_url=s.ollama_base_url, transport=transport)
    )


def ok(text: str, **extra: object) -> httpx.Response:
    return httpx.Response(200, json={"response": text, "done": True, **extra})


class Recorder:
    """A handler that records every request and replays scripted outcomes."""

    def __init__(self, *outcomes: httpx.Response | Exception) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def body(self, index: int = -1) -> dict[str, object]:
        result: dict[str, object] = json.loads(self.requests[index].content)
        return result


# --------------------------------------------------------------------------- #
# request shape
# --------------------------------------------------------------------------- #


def test_satisfies_the_protocol() -> None:
    assert isinstance(make_client(Recorder(ok(""))), LLMClient)


async def test_generate_posts_settings_to_api_generate() -> None:
    recorder = Recorder(ok("hello"))
    assert await make_client(recorder).generate("say hi") == "hello"

    request = recorder.requests[0]
    assert request.method == "POST"
    assert request.url == "http://ollama.test/api/generate"
    assert recorder.body() == {
        "model": "test-model",
        "prompt": "say hi",
        "stream": False,
        "options": {"num_ctx": 4096, "num_predict": 256, "temperature": 0.3},
    }


async def test_per_call_arguments_override_settings() -> None:
    recorder = Recorder(ok("x"))
    await make_client(recorder).generate("p", max_tokens=8, temperature=0.0, stop=["\n\n"])
    assert recorder.body()["options"] == {
        "num_ctx": 4096,
        "num_predict": 8,
        "temperature": 0.0,
        "stop": ["\n\n"],
    }


async def test_zero_temperature_is_not_mistaken_for_unset() -> None:
    recorder = Recorder(ok("x"))
    await make_client(recorder).generate("p", temperature=0.0)
    assert recorder.body()["options"]["temperature"] == 0.0  # type: ignore[index]


# --------------------------------------------------------------------------- #
# structured output
# --------------------------------------------------------------------------- #


async def test_structured_sends_the_json_schema_and_validates() -> None:
    recorder = Recorder(ok('{"colour": "blue"}'))
    pick = await make_client(recorder).generate_structured("pick one", Pick)

    assert pick == Pick(colour=Colour.BLUE)
    body = recorder.body()
    assert body["format"] == Pick.model_json_schema()
    assert body["options"]["temperature"] == 0.0  # type: ignore[index]


@pytest.mark.parametrize(
    "text",
    ['{"colour": "green"}', '{"colour": "bl', "Sure! The colour is blue.", "{}"],
    ids=["outside-enum", "truncated", "free-text", "missing-field"],
)
async def test_structured_output_that_fails_validation_is_an_llm_error(text: str) -> None:
    with pytest.raises(LLMError, match="does not match Pick"):
        await make_client(Recorder(ok(text))).generate_structured("pick one", Pick)


# --------------------------------------------------------------------------- #
# retries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "transient",
    [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")],
    ids=["connect-error", "read-timeout"],
)
async def test_transient_failures_are_retried_then_succeed(transient: Exception) -> None:
    recorder = Recorder(transient, transient, ok("recovered"))
    assert await make_client(recorder).generate("p") == "recovered"
    assert len(recorder.requests) == 3


async def test_retries_stop_at_the_configured_limit() -> None:
    recorder = Recorder(httpx.ConnectError("refused"))
    with pytest.raises(LLMError, match="after 2 attempts: ConnectError"):
        await make_client(recorder, llm_max_retries=2).generate("p")
    assert len(recorder.requests) == 2


@pytest.mark.parametrize("status", [400, 404, 422])
async def test_client_errors_are_never_retried(status: int) -> None:
    recorder = Recorder(httpx.Response(status, json={"error": "model 'test-model' not found"}))
    with pytest.raises(LLMError, match=f"returned {status}.*not found"):
        await make_client(recorder).generate("p")
    assert len(recorder.requests) == 1


async def test_server_errors_are_not_retried() -> None:
    """Ollama answers 500 for out-of-memory; asking again does not free VRAM."""
    recorder = Recorder(httpx.Response(500, text="CUDA error: out of memory"))
    with pytest.raises(LLMError, match=r"500.*out of memory"):
        await make_client(recorder).generate("p")
    assert len(recorder.requests) == 1


async def test_other_transport_errors_fail_fast_as_llm_error() -> None:
    recorder = Recorder(httpx.RemoteProtocolError("peer closed"))
    with pytest.raises(LLMError, match="RemoteProtocolError"):
        await make_client(recorder).generate("p")
    assert len(recorder.requests) == 1


# --------------------------------------------------------------------------- #
# malformed responses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>proxy</html>"),
        httpx.Response(200, json={"done": True}),
        httpx.Response(200, json=["not", "an", "object"]),
    ],
    ids=["not-json", "no-response-key", "not-an-object"],
)
async def test_malformed_success_bodies_are_llm_errors(response: httpx.Response) -> None:
    with pytest.raises(LLMError):
        await make_client(Recorder(response)).generate("p")


async def test_truncated_generation_is_returned_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="codemind.llm.ollama_client"):
        text = await make_client(Recorder(ok("partial", done_reason="length"))).generate("p")
    assert text == "partial"
    assert "truncated" in caplog.text


async def test_context_manager_closes_the_http_client() -> None:
    client = make_client(Recorder(ok("x")))
    async with client:
        pass
    assert client._client.is_closed
