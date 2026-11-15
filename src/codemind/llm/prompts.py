"""Prompt templates from `configs/prompts/*.jinja2`. The only place prompts are rendered.

`StrictUndefined` makes a missing variable an error at render time. The
default renders it as an empty string, which produces a prompt that looks
fine and asks the model about nothing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

from codemind.core.config import get_settings
from codemind.core.exceptions import LLMError


@lru_cache(maxsize=4)
def prompt_environment(prompts_dir: Path | None = None) -> Environment:
    return Environment(
        loader=FileSystemLoader(prompts_dir or get_settings().prompts_dir),
        undefined=StrictUndefined,
        autoescape=False,  # prompts are not HTML; escaping would corrupt code
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(name: str, **context: Any) -> str:
    """Render `configs/prompts/<name>.jinja2` with `context`."""
    try:
        return prompt_environment().get_template(f"{name}.jinja2").render(**context)
    except TemplateError as exc:
        raise LLMError(f"cannot render prompt {name!r}: {exc}") from exc
