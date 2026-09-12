"""AST-aware code chunking with tree-sitter.

Why this file exists: a 512-token sliding window over Java cuts through the
middle of methods. Half a method retrieves badly (the signature and the body
end up in different chunks) and reads badly (the LLM sees code that does not
compile). Chunking on the syntax tree means every chunk is a complete unit with
a name, a line range and an enclosing scope — which is also what makes
`file:line` citations possible.

Emission strategy per file:
  * one *skeleton* chunk per class/interface — the declaration, fields and
    method signatures, with bodies elided. Answers "what is this class".
  * one chunk per method/function. Answers "what does this code do".
  * consecutive tiny members (getters, setters) merge into one chunk rather
    than producing dozens of near-identical 3-line embeddings.
  * oversized functions split on statement boundaries, never mid-expression.

All language specifics come from `language_registry`. This module contains no
per-language branching beyond what the registry declares.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codemind.core.exceptions import ChunkingError
from codemind.core.types import CodeChunk, Language, SourceFile
from codemind.ingestion.language_registry import LanguageSpec, spec_for_language
from codemind.ingestion.metadata import build_context_header, enrich, module_path_for

if TYPE_CHECKING:  # pragma: no cover
    from tree_sitter import Node

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHUNK_CHARS = 4_000
DEFAULT_MIN_CHUNK_CHARS = 140
DEFAULT_MAX_SKELETON_CHARS = 2_000


@dataclass(slots=True)
class ChunkStats:
    files_chunked: int = 0
    chunks_emitted: int = 0
    skeletons: int = 0
    members: int = 0
    merged_small: int = 0
    split_large: int = 0
    parse_errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "files_chunked": self.files_chunked,
            "chunks_emitted": self.chunks_emitted,
            "skeletons": self.skeletons,
            "members": self.members,
            "merged_small": self.merged_small,
            "split_large": self.split_large,
            "parse_errors": self.parse_errors,
        }


class AstChunker:
    """Turns a `SourceFile` into a list of `CodeChunk`s."""

    def __init__(
        self,
        *,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
        max_skeleton_chars: int = DEFAULT_MAX_SKELETON_CHARS,
    ) -> None:
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.max_skeleton_chars = max_skeleton_chars
        self.stats = ChunkStats()
        self._parsers: dict[Language, Any] = {}

    # -- public ------------------------------------------------------------ #

    def chunk(self, source: SourceFile) -> list[CodeChunk]:
        spec = spec_for_language(source.language)
        raw = source.content.encode("utf-8")

        try:
            tree = self._parser_for(source.language).parse(raw)
        except Exception as exc:
            self.stats.parse_errors += 1
            raise ChunkingError(f"{source.relative_path}: parse failed: {exc}") from exc

        root = tree.root_node
        package = self._package_name(root, raw, spec, source)
        imports = self._imports(root, raw, spec)

        chunks: list[CodeChunk] = []
        for container in self._find(root, spec.container_nodes, spec):
            chunks.extend(self._chunk_container(container, raw, spec, source, package, imports))

        # functions not inside any container (module-level Python, exported TS)
        for node in self._find_top_level_members(root, spec):
            member = self._make_member_chunk(node, raw, spec, source, package, None, imports)
            if member is not None:
                chunks.append(member)

        self.stats.files_chunked += 1
        self.stats.chunks_emitted += len(chunks)
        return [enrich(c) for c in chunks]

    def chunk_all(self, sources: list[SourceFile]) -> list[CodeChunk]:
        out: list[CodeChunk] = []
        for source in sources:
            try:
                out.extend(self.chunk(source))
            except ChunkingError as exc:
                logger.warning("skipping %s: %s", source.relative_path, exc)
        return out

    # -- containers -------------------------------------------------------- #

    def _chunk_container(
        self,
        container: Node,
        raw: bytes,
        spec: LanguageSpec,
        source: SourceFile,
        package: str,
        imports: list[str],
    ) -> list[CodeChunk]:
        name = self._node_name(container, raw, spec) or "<anonymous>"
        members = list(self._find(container, spec.chunk_nodes, spec, skip_root=True))

        chunks: list[CodeChunk] = [
            self._make_skeleton_chunk(container, members, raw, spec, source, package, name, imports)
        ]

        chunks.extend(self._make_member_chunks(members, raw, spec, source, package, name, imports))
        return chunks

    def _make_skeleton_chunk(
        self,
        container: Node,
        members: list[Node],
        raw: bytes,
        spec: LanguageSpec,
        source: SourceFile,
        package: str,
        name: str,
        imports: list[str],
    ) -> CodeChunk:
        """Class declaration + fields + method signatures, bodies elided.

        This is what answers "what does OrderRepository extend" — information
        that lives in the declaration line and would otherwise be split across
        method chunks that never mention it.
        """
        body = _elide_bodies(container, members, raw, self.max_skeleton_chars)
        signature = _declaration_line(container, raw, spec)
        self.stats.skeletons += 1
        return CodeChunk(
            relative_path=source.relative_path,
            language=source.language,
            symbol_name=name,
            kind="class",
            start_line=container.start_point[0] + 1,
            end_line=container.end_point[0] + 1,
            body=body,
            context_header=build_context_header(
                relative_path=source.relative_path,
                package=package,
                parent_symbol=None,
                signature=signature,
                language=source.language,
                imports=imports,
            ),
            imports=list(imports),
        )

    # -- members ----------------------------------------------------------- #

    def _make_member_chunks(
        self,
        members: list[Node],
        raw: bytes,
        spec: LanguageSpec,
        source: SourceFile,
        package: str,
        parent: str | None,
        imports: list[str],
    ) -> list[CodeChunk]:
        """Emit member chunks, merging runs of tiny ones.

        A DTO with eight getters should not produce eight embeddings that are
        almost identical — they compete with each other in the ranking and
        crowd out the method that actually answers the question.
        """
        chunks: list[CodeChunk] = []
        pending: list[Node] = []

        def flush() -> None:
            if not pending:
                return
            if len(pending) == 1:
                chunk = self._make_member_chunk(
                    pending[0], raw, spec, source, package, parent, imports
                )
                if chunk is not None:
                    chunks.append(chunk)
            else:
                chunks.append(
                    self._make_merged_chunk(pending, raw, spec, source, package, parent, imports)
                )
                self.stats.merged_small += len(pending)
            pending.clear()

        for node in members:
            text = _text(node, raw)
            if len(text) < self.min_chunk_chars and _is_trivial(node, raw):
                pending.append(node)
                continue
            flush()
            chunk = self._make_member_chunk(node, raw, spec, source, package, parent, imports)
            if chunk is None:
                continue
            if len(chunk.body) > self.max_chunk_chars:
                chunks.extend(self._split_oversized(chunk))
            else:
                chunks.append(chunk)
        flush()
        return chunks

    def _make_member_chunk(
        self,
        node: Node,
        raw: bytes,
        spec: LanguageSpec,
        source: SourceFile,
        package: str,
        parent: str | None,
        imports: list[str],
    ) -> CodeChunk | None:
        name = self._node_name(node, raw, spec)
        if name is None:
            return None
        body = _text(node, raw)
        self.stats.members += 1
        return CodeChunk(
            relative_path=source.relative_path,
            language=source.language,
            symbol_name=name,
            kind="function",
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            body=body,
            context_header=build_context_header(
                relative_path=source.relative_path,
                package=package,
                parent_symbol=parent,
                signature=_signature_of(node, raw),
                language=source.language,
                imports=imports,
            ),
            parent_symbol=parent,
            imports=list(imports),
        )

    def _make_merged_chunk(
        self,
        nodes: list[Node],
        raw: bytes,
        spec: LanguageSpec,
        source: SourceFile,
        package: str,
        parent: str | None,
        imports: list[str],
    ) -> CodeChunk:
        names = [self._node_name(n, raw, spec) or "?" for n in nodes]
        body = "\n\n".join(_text(n, raw) for n in nodes)
        label = f"{parent or 'module'} accessors"
        return CodeChunk(
            relative_path=source.relative_path,
            language=source.language,
            symbol_name=", ".join(names),
            kind="members",
            start_line=nodes[0].start_point[0] + 1,
            end_line=nodes[-1].end_point[0] + 1,
            body=body,
            context_header=build_context_header(
                relative_path=source.relative_path,
                package=package,
                parent_symbol=parent,
                signature=label,
                language=source.language,
                imports=imports,
            ),
            parent_symbol=parent,
            imports=list(imports),
        )

    def _split_oversized(self, chunk: CodeChunk) -> list[CodeChunk]:
        """Split a very long function on blank lines, never mid-expression."""
        lines = chunk.body.splitlines()
        parts: list[list[str]] = [[]]
        size = 0
        for line in lines:
            if size + len(line) > self.max_chunk_chars and not line.strip().startswith((")", "}")):
                parts.append([])
                size = 0
            parts[-1].append(line)
            size += len(line) + 1

        out: list[CodeChunk] = []
        offset = 0
        for index, part in enumerate(p for p in parts if p):
            out.append(
                CodeChunk(
                    relative_path=chunk.relative_path,
                    language=chunk.language,
                    symbol_name=f"{chunk.symbol_name} (part {index + 1})",
                    kind="function_part",
                    start_line=chunk.start_line + offset,
                    end_line=chunk.start_line + offset + len(part) - 1,
                    body="\n".join(part),
                    context_header=chunk.context_header,
                    parent_symbol=chunk.parent_symbol,
                    imports=list(chunk.imports),
                )
            )
            offset += len(part)
        self.stats.split_large += 1
        return out

    # -- tree helpers ------------------------------------------------------ #

    def _parser_for(self, language: Language) -> Any:
        if language not in self._parsers:
            from tree_sitter_language_pack import get_parser

            self._parsers[language] = get_parser(spec_for_language(language).grammar)
        return self._parsers[language]

    def _find(
        self,
        root: Node,
        wanted: frozenset[str],
        spec: LanguageSpec,
        *,
        skip_root: bool = False,
    ) -> list[Node]:
        """Outermost matching nodes below `root`, in source order.

        Outermost only: a helper function nested inside a method belongs to that
        method's chunk. Emitting it separately duplicates the code and yields a
        chunk with no usable enclosing context.

        When searching for members (`skip_root=True`), a nested container ends
        the search on that branch — its members are collected by its own call.
        """
        found: list[Node] = []
        searching_members = wanted is spec.chunk_nodes

        def visit(node: Node, depth: int) -> None:
            if depth > 0:
                if node.type in wanted:
                    found.append(node)
                    return
                if searching_members and node.type in spec.container_nodes:
                    return
            for child in node.children:
                visit(child, depth + 1)

        visit(root, 0)
        found.sort(key=lambda n: n.start_byte)
        return found

    def _find_top_level_members(self, root: Node, spec: LanguageSpec) -> list[Node]:
        """Chunkable nodes that are not inside any container."""
        found: list[Node] = []

        def visit(node: Node, depth: int) -> None:
            if depth > 0 and node.type in spec.container_nodes:
                return
            if depth > 0 and node.type in spec.chunk_nodes:
                found.append(node)
                return
            for child in node.children:
                visit(child, depth + 1)

        visit(root, 0)
        return found

    def _node_name(self, node: Node, raw: bytes, spec: LanguageSpec) -> str | None:
        """Identifier for a node, unwrapping Python's decorated_definition."""
        target = node
        if node.type == "decorated_definition":
            inner = next(
                (c for c in node.children if c.type in {"function_definition", "class_definition"}),
                None,
            )
            if inner is None:
                return None
            target = inner

        name_node = target.child_by_field_name(spec.name_field)
        if name_node is not None:
            return _text(name_node, raw)

        for child in target.children:
            if child.type in {"identifier", "type_identifier", "property_identifier"}:
                return _text(child, raw)
        return None

    def _package_name(self, root: Node, raw: bytes, spec: LanguageSpec, source: SourceFile) -> str:
        if source.language is Language.JAVA:
            for child in root.children:
                if child.type == "package_declaration":
                    return _text(child, raw).removeprefix("package").strip().rstrip(";")
        return module_path_for(source.relative_path, source.language)

    def _imports(self, root: Node, raw: bytes, spec: LanguageSpec) -> list[str]:
        out: list[str] = []

        def visit(node: Node, depth: int) -> None:
            if depth > 2:
                return
            if node.type in spec.import_nodes:
                out.append(_text(node, raw))
                return
            for child in node.children:
                visit(child, depth + 1)

        visit(root, 0)
        return out


# --------------------------------------------------------------------------- #
# free functions
# --------------------------------------------------------------------------- #


ACCESSOR_PREFIXES = ("get", "set", "is", "has", "to")
DUNDER_TRIVIAL = {"__init__", "__repr__", "__str__", "__eq__", "__hash__"}


def _is_trivial(node: Node, raw: bytes) -> bool:
    """True when a member is a getter/setter-shaped one-liner safe to merge.

    Annotated members are never trivial. `@PostMapping createOrder()` is short
    but it is an HTTP endpoint — exactly what questions target — so it must keep
    its own chunk and its own embedding. Merging it into an "accessors" blob
    would make it unfindable.
    """
    if _has_annotation(node, raw):
        return False

    body = node.child_by_field_name("body")
    if body is None:
        return False

    statements = [
        child
        for child in body.children
        if child.is_named and child.type not in {"comment", "block_comment", "line_comment"}
    ]
    if len(statements) > 1:
        return False

    name_node = node.child_by_field_name("name")
    name = _text(name_node, raw) if name_node is not None else ""
    if name in DUNDER_TRIVIAL:
        return True
    return name.startswith(ACCESSOR_PREFIXES)


def _has_annotation(node: Node, raw: bytes) -> bool:
    """Java annotations, Python decorators, TS decorators."""
    if node.type == "decorated_definition":
        return True
    for child in node.children:
        if child.type in {"modifiers", "decorator"} and "@" in _text(child, raw):
            return True
        if child.type in {"annotation", "marker_annotation"}:
            return True
    return False


def _declaration_line(node: Node, raw: bytes, spec: LanguageSpec) -> str:
    """The line that actually declares the type, skipping annotations above it.

    `@RestController\npublic class OrderController {` should yield the second
    line — the first tells you nothing about what the class is.
    """
    name_node = node.child_by_field_name(spec.name_field)
    text = _text(node, raw)
    if name_node is None:
        return _first_line(text)
    target_row = name_node.start_point[0] - node.start_point[0]
    lines = text.splitlines()
    if 0 <= target_row < len(lines):
        return lines[target_row].strip().rstrip("{").strip()
    return _first_line(text)


def _text(node: Node, raw: bytes) -> str:
    return raw[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _first_line(text: str) -> str:
    return text.splitlines()[0].strip() if text else ""


def _signature_of(node: Node, raw: bytes) -> str:
    """Everything before the body — the part a reader needs to know what it takes."""
    body = node.child_by_field_name("body")
    if body is not None and body.start_byte > node.start_byte:
        return " ".join(raw[node.start_byte : body.start_byte].decode("utf-8", "replace").split())
    return _first_line(_text(node, raw))


def _elide_bodies(container: Node, members: list[Node], raw: bytes, limit: int) -> str:
    """Container source with member bodies replaced by `{ ... }`."""
    spans = sorted(
        (m.child_by_field_name("body") or m for m in members),
        key=lambda n: n.start_byte,
    )
    pieces: list[str] = []
    cursor = container.start_byte
    for span in spans:
        if span.start_byte < cursor:
            continue
        pieces.append(raw[cursor : span.start_byte].decode("utf-8", "replace"))
        pieces.append("{ ... }")
        cursor = span.end_byte
    pieces.append(raw[cursor : container.end_byte].decode("utf-8", "replace"))

    text = "".join(pieces)
    text = "\n".join(line for line in text.splitlines() if line.strip())
    return text if len(text) <= limit else text[: limit - 1] + "…"
