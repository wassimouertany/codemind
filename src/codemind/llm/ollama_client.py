"""`LLMClient` over Ollama's native `/api/generate`, non-streaming.

Structured output uses Ollama's `format` field with the Pydantic model's JSON
schema. Ollama compiles that schema to a grammar, so an enum field cannot be
sampled outside its values; the response is still validated by Pydantic,
because a truncated generation (`num_predict` hit mid-object) is valid
grammar-so-far but not valid JSON.

Retries cover exactly two failures: the connection was refused
(`ConnectError`: Ollama restarting) and the response did not arrive in time
(`ReadTimeout`: a cold load on a hot GPU). A 4xx is a bug in the request — a
missing model, a malformed schema — and retrying it only multiplies the wait
before the same error. 5xx is not retried either: Ollama returns 500 for
out-of-memory, and asking again does not free VRAM.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import LLMError
from codemind.llm.base import ModelT

logger = logging.getLogger(__name__)

RETRYABLE: tuple[type[httpx.HTTPError], ...] = (httpx.ConnectError, httpx.ReadTimeout)

RETRY_WAIT = wait_exponential(multiplier=0.5, max=8)
"""0.5s, 1s, 2s ... capped at 8s. Long enough for Ollama to finish restarting."""

STRUCTURED_TEMPERATURE = 0.0
"""Classification wants the argmax label, not a sample from around it."""


class OllamaClient:
    """Async Ollama client. One instance per process, closed in the lifespan."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or httpx.AsyncClient(
            base_url=self._settings.ollama_base_url,
            timeout=self._settings.llm_timeout,
        )
        self.model = self._settings.llm_model

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        body = self._request(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
        return await self._post(body)

    async def generate_structured(self, prompt: str, schema: type[ModelT]) -> ModelT:
        body = self._request(prompt, temperature=STRUCTURED_TEMPERATURE)
        body["format"] = schema.model_json_schema()
        text = await self._post(body)
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(
                f"{self.model} returned output that does not match {schema.__name__}: "
                f"{text[:200]!r}"
            ) from exc

    def _request(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_ctx": self._settings.llm_num_ctx,
            "num_predict": self._settings.llm_max_tokens if max_tokens is None else max_tokens,
            "temperature": self._settings.llm_temperature if temperature is None else temperature,
        }
        if stop:
            options["stop"] = list(stop)
        return {"model": self.model, "prompt": prompt, "stream": False, "options": options}

    async def _post(self, body: dict[str, Any]) -> str:
        """POST once per attempt; wrap every failure in `LLMError`."""
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(RETRYABLE),
            stop=stop_after_attempt(self._settings.llm_max_retries),
            wait=RETRY_WAIT,
            before_sleep=lambda state: logger.warning(
                "ollama attempt %d failed (%s), retrying",
                state.attempt_number,
                type(state.outcome.exception()).__name__ if state.outcome else "unknown",
            ),
        )
        try:
            async for attempt in retrying:
                with attempt:
                    response = await self._client.post("/api/generate", json=body)
        except RetryError as exc:
            cause = exc.last_attempt.exception()
            raise LLMError(
                f"Ollama unreachable at {self._settings.ollama_base_url} after "
                f"{self._settings.llm_max_retries} attempts: {type(cause).__name__}"
            ) from cause
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {type(exc).__name__}: {exc}") from exc

        if response.is_error:
            raise LLMError(
                f"Ollama returned {response.status_code} for model {self.model}: "
                f"{_error_detail(response)}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError(f"Ollama returned a non-JSON body: {response.text[:200]!r}") from exc
        text = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            raise LLMError(f"Ollama response has no 'response' string: {str(payload)[:200]}")
        if payload.get("done_reason") == "length":
            logger.warning("ollama hit num_predict before finishing; output is truncated")
        return text

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


def _error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("error")
    except ValueError:
        detail = None
    return str(detail) if detail else response.text[:200]
