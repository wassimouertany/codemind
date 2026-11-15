"""The seam between agent nodes and whatever serves the model.

Nodes depend on `LLMClient`, never on `OllamaClient` or `VLLMClient`. The
concrete client is constructed once by whoever composes the system (the
FastAPI lifespan, a script) and injected into `build_graph`. That keeps the
Day 20 move to vLLM a one-line change at the composition root, and lets unit
tests drive every node with a stub instead of a 2 GB model on a 4 GB card.

This module imports nothing concrete on purpose. An import-linter contract
forbids `codemind.agents` from importing the client modules directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    """What a node may ask of a language model.

    Implementations raise `LLMError` for every failure — unreachable backend,
    rejected request, output that does not match the schema — so a node has
    exactly one exception type to degrade on.
    """

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        """Free-text completion. `None` means "use the configured default"."""
        ...

    async def generate_structured(self, prompt: str, schema: type[ModelT]) -> ModelT:
        """A completion constrained to `schema` and validated against it.

        The router depends on this: an enum field is enforced by the backend's
        grammar and re-checked by Pydantic, so no node ever parses free text.
        """
        ...
