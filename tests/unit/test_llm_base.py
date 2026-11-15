"""The `LLMClient` seam is structural; these keep it honest."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

import codemind.llm.base as base
from codemind.llm.base import LLMClient, ModelT


class _Minimal:
    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        return prompt

    async def generate_structured(self, prompt: str, schema: type[ModelT]) -> ModelT:
        return schema.model_validate({})


class _Empty(BaseModel):
    pass


def test_a_structural_implementation_satisfies_the_protocol() -> None:
    assert isinstance(_Minimal(), LLMClient)


def test_an_object_missing_structured_output_does_not() -> None:
    class FreeTextOnly:
        async def generate(self, prompt: str) -> str:
            return prompt

    assert not isinstance(FreeTextOnly(), LLMClient)


def test_base_imports_no_concrete_client() -> None:
    """vLLM swaps in below this module on Day 20; nothing concrete may leak up."""
    tree = ast.parse(Path(base.__file__).read_text())
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not {m for m in imported if m.startswith("codemind.")}
    assert "httpx" not in imported
