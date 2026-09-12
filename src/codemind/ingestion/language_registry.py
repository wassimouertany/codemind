"""Maps file extensions to tree-sitter grammars and the node types worth chunking.

This registry is the single source of truth for "what counts as a chunkable unit"
in each language. The AST chunker (day 3) consumes it and contains no
language-specific branching of its own — adding a language means adding a
`LanguageSpec` here, nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from codemind.core.types import Language, Layer


@dataclass(slots=True, frozen=True)
class LanguageSpec:
    """Everything the chunker needs to know about one language."""

    language: Language
    grammar: str
    """Name passed to tree_sitter_language_pack.get_parser()."""

    extensions: frozenset[str]

    chunk_nodes: frozenset[str]
    """Node types that become their own chunk (functions, methods)."""

    container_nodes: frozenset[str]
    """Node types that hold chunks and supply the enclosing-name context
    (classes, interfaces). Chunked themselves only when they contain no
    chunkable children — e.g. a DTO or an interface of signatures."""

    import_nodes: frozenset[str]
    """Node types the symbol graph reads to build dependency edges."""

    name_field: str = "name"
    """Field name holding the identifier on chunk/container nodes."""


_SPECS: Final[tuple[LanguageSpec, ...]] = (
    LanguageSpec(
        language=Language.PYTHON,
        grammar="python",
        extensions=frozenset({".py", ".pyi"}),
        chunk_nodes=frozenset({"function_definition", "decorated_definition"}),
        container_nodes=frozenset({"class_definition"}),
        import_nodes=frozenset({"import_statement", "import_from_statement"}),
    ),
    LanguageSpec(
        language=Language.JAVA,
        grammar="java",
        extensions=frozenset({".java"}),
        chunk_nodes=frozenset({"method_declaration", "constructor_declaration"}),
        container_nodes=frozenset(
            {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}
        ),
        import_nodes=frozenset({"import_declaration"}),
    ),
    LanguageSpec(
        language=Language.TYPESCRIPT,
        grammar="typescript",
        extensions=frozenset({".ts", ".mts", ".cts"}),
        chunk_nodes=frozenset(
            {
                "function_declaration",
                "method_definition",
                "arrow_function",
                "function_signature",
            }
        ),
        container_nodes=frozenset(
            {"class_declaration", "interface_declaration", "type_alias_declaration"}
        ),
        import_nodes=frozenset({"import_statement"}),
    ),
    LanguageSpec(
        language=Language.JAVASCRIPT,
        grammar="javascript",
        extensions=frozenset({".js", ".mjs", ".cjs", ".jsx"}),
        chunk_nodes=frozenset({"function_declaration", "method_definition", "arrow_function"}),
        container_nodes=frozenset({"class_declaration"}),
        import_nodes=frozenset({"import_statement"}),
    ),
)

_BY_EXTENSION: Final[dict[str, LanguageSpec]] = {
    ext: spec for spec in _SPECS for ext in spec.extensions
}

_BY_LANGUAGE: Final[dict[Language, LanguageSpec]] = {spec.language: spec for spec in _SPECS}

SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(_BY_EXTENSION)


def spec_for_path(path: Path | str) -> LanguageSpec | None:
    """Return the spec for a path, or None if the extension is not indexable."""
    suffix = Path(path).suffix.lower()
    return _BY_EXTENSION.get(suffix)


def spec_for_language(language: Language) -> LanguageSpec:
    """Return the spec for a language. Raises KeyError if unregistered."""
    return _BY_LANGUAGE[language]


def is_supported(path: Path | str) -> bool:
    return Path(path).suffix.lower() in _BY_EXTENSION


# --------------------------------------------------------------------------- #
# Architectural layer inference
# --------------------------------------------------------------------------- #

_LAYER_PATTERNS: Final[tuple[tuple[re.Pattern[str], Layer], ...]] = (
    (re.compile(r"(^|[/_.])(test|tests|spec|__tests__)([/_.]|$)", re.I), Layer.TEST),
    (re.compile(r"(controller|resource|handler|router|endpoint|view)s?\b", re.I), Layer.CONTROLLER),
    (re.compile(r"(service|usecase|manager|facade)s?\b", re.I), Layer.SERVICE),
    (re.compile(r"(repository|repositories|dao|mapper|store)\b", re.I), Layer.REPOSITORY),
    (re.compile(r"(entity|entities|model|dto|schema|domain)s?\b", re.I), Layer.MODEL),
    (re.compile(r"(config|configuration|settings|properties)\b", re.I), Layer.CONFIG),
)


def infer_layer(relative_path: str, symbol_name: str = "") -> Layer:
    """Guess the architectural layer from path and symbol name.

    Tests are checked first: `UserServiceTest.java` is a test, not a service.
    Used as a retrieval filter and a hint — never as ground truth.
    """
    haystack = f"{relative_path} {symbol_name}"
    for pattern, layer in _LAYER_PATTERNS:
        if pattern.search(haystack):
            return layer
    return Layer.UNKNOWN
