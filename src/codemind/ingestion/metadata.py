"""Context headers and chunk enrichment.

Kept separate from the chunker because the header format is a *retrieval*
decision, not a parsing one. If Gate 1 shows recall is weak, this is the first
file to change — and changing it must not mean touching AST-walking code.
"""

from __future__ import annotations

from codemind.core.types import CodeChunk, Language
from codemind.ingestion.language_registry import infer_layer

MAX_SIGNATURE_CHARS = 200
MAX_IMPORTS_IN_HEADER = 8


def build_context_header(
    *,
    relative_path: str,
    package: str,
    parent_symbol: str | None,
    signature: str,
    language: Language,
    imports: list[str] | None = None,
) -> str:
    """Compose the line prepended to a chunk before embedding.

    A chunk body like `return this.repo.findById(id);` carries almost no
    retrievable signal on its own. The header supplies the path, the package,
    the enclosing class and the signature, which is what connects the chunk to a
    question phrased as "why does GET /users/{id} fail".

    Format:  path | package | Class | signature | imports: a, b, c
    """
    parts = [relative_path]
    if package:
        parts.append(package)
    if parent_symbol:
        parts.append(parent_symbol)
    if signature:
        parts.append(_truncate(signature, MAX_SIGNATURE_CHARS))

    header = " | ".join(parts)

    if imports:
        shown = [_short_import(i, language) for i in imports[:MAX_IMPORTS_IN_HEADER]]
        shown = [s for s in shown if s]
        if shown:
            header = f"{header} | imports: {', '.join(shown)}"

    return header


def enrich(chunk: CodeChunk) -> CodeChunk:
    """Fill in derived fields that need no AST access."""
    if chunk.layer.value == "unknown":
        chunk.layer = infer_layer(chunk.relative_path, chunk.symbol_name)
    return chunk


def module_path_for(relative_path: str, language: Language) -> str:
    """Best-effort module/package name when the source declares none.

    Java declares its package explicitly, so the chunker passes that through.
    Python and TypeScript do not, so we derive it from the file path.
    """
    if language is Language.JAVA:
        return ""

    path = relative_path.rsplit(".", 1)[0]
    for prefix in ("src/", "app/", "lib/", "source/"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    if path.endswith("/__init__"):
        path = path[: -len("/__init__")]
    return path.replace("/", ".")


def _short_import(statement: str, language: Language) -> str:
    """Reduce an import statement to the symbol that matters.

    `import com.demo.repository.OrderRepository;` -> `OrderRepository`
    Full paths waste header budget and add tokens BM25 will match on noisily.
    """
    text = statement.strip().rstrip(";").replace("\n", " ")

    if language is Language.JAVA:
        text = text.removeprefix("import ").removeprefix("static ").strip()
        if text.endswith(".*"):
            return text
        return text.rsplit(".", 1)[-1]

    if language is Language.PYTHON:
        if text.startswith("from "):
            rest = text[5:]
            module, _, names = rest.partition(" import ")
            return names.split(",")[0].strip() or module.strip()
        return text.removeprefix("import ").split(",")[0].strip().rsplit(".", 1)[-1]

    # TS/JS: `import { A, B } from "x"` -> A
    if "{" in text and "}" in text:
        inner = text[text.index("{") + 1 : text.index("}")]
        return inner.split(",")[0].strip()
    if " from " in text:
        return text.split(" from ")[0].removeprefix("import ").strip()
    return ""


def _truncate(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"
