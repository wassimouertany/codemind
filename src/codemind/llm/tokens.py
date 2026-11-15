"""Exact prompt token counts, for budgeting against `LLM_NUM_CTX`.

Ollama has no tokenize endpoint and never errors on an overlong prompt — it
keeps roughly half the window and drops the rest, instructions first. A
characters-per-token estimate cannot guard against that: measured text runs
from 4.1 chars/token (rendered specialist prompts) down to 1.8 (dense
identifiers), so any fixed ratio is either unsafe or refuses prompts that fit.
The model's own tokenizer is exact: it matched Ollama's `prompt_eval_count` on
every prompt measured, plus a constant 29 tokens of chat template.

Only the tokenizer is loaded — a few MB, CPU, no model weights — once per
process, by whoever constructs the LLM client.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from tokenizers import Tokenizer

from codemind.core.exceptions import LLMError


class TokenCounter(Protocol):
    def __call__(self, text: str) -> int: ...


@lru_cache(maxsize=2)
def load_token_counter(name: str) -> TokenCounter:
    try:
        tokenizer = Tokenizer.from_pretrained(name)
    except OSError as exc:  # hub download / cache failures are OSError subclasses
        raise LLMError(f"cannot load tokenizer {name!r} (LLM_TOKENIZER): {exc}") from exc

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count
