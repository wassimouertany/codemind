"""Tests for AST-aware chunking.

The load-bearing test is `test_no_chunk_splits_a_function_body`. Everything else
in the retrieval pipeline assumes chunks are syntactically complete units; if
that breaks, recall degrades in ways that are very hard to diagnose later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codemind.core.types import CodeChunk, Language
from codemind.ingestion.ast_chunker import AstChunker
from codemind.ingestion.repo_loader import RepositoryLoader

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"


@pytest.fixture(scope="module")
def chunks() -> list[CodeChunk]:
    loader = RepositoryLoader(FIXTURE)
    chunker = AstChunker()
    return chunker.chunk_all(loader.collect())


def by_name(chunks: list[CodeChunk], name: str) -> CodeChunk:
    matches = [c for c in chunks if c.symbol_name == name]
    assert matches, f"no chunk named {name!r}; have {[c.symbol_name for c in chunks]}"
    return matches[0]


# --------------------------------------------------------------------------- #
# the invariant
# --------------------------------------------------------------------------- #


def test_no_chunk_splits_a_function_body(chunks: list[CodeChunk]) -> None:
    """Braces and parens must balance in every non-skeleton chunk.

    A chunk cut mid-function has unbalanced delimiters. Skeleton chunks are
    exempt because their bodies are deliberately elided to `{ ... }`.
    """
    for chunk in chunks:
        if chunk.kind in {"class", "function_part"}:
            continue
        body = chunk.body
        assert body.count("{") == body.count("}"), f"{chunk.symbol_name}: unbalanced braces"
        assert body.count("(") == body.count(")"), f"{chunk.symbol_name}: unbalanced parens"


def test_every_chunk_has_valid_line_range(chunks: list[CodeChunk]) -> None:
    """Citations are `file:line`. A wrong range is a wrong citation."""
    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line
        source = (FIXTURE / chunk.relative_path).read_text(encoding="utf-8")
        assert chunk.end_line <= len(source.splitlines()) + 1


def test_chunk_ids_are_unique_and_stable(chunks: list[CodeChunk]) -> None:
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"

    again = AstChunker().chunk_all(RepositoryLoader(FIXTURE).collect())
    assert [c.chunk_id for c in again] == ids, "ids must be deterministic across runs"


# --------------------------------------------------------------------------- #
# the bug we planted
# --------------------------------------------------------------------------- #


def test_place_order_is_its_own_chunk(chunks: list[CodeChunk]) -> None:
    """The method containing the bug must be independently retrievable."""
    chunk = by_name(chunks, "placeOrder")
    assert "paymentService.processPayment" in chunk.body
    assert "orderRepository.save" in chunk.body
    assert chunk.parent_symbol == "OrderService"


def test_the_call_chain_is_fully_chunked(chunks: list[CodeChunk]) -> None:
    """Controller -> Service -> Service -> Repository, each separately findable."""
    for name in ("createOrder", "placeOrder", "processPayment", "findByCustomerId"):
        by_name(chunks, name)


def test_annotated_endpoints_are_never_merged(chunks: list[CodeChunk]) -> None:
    """`@PostMapping createOrder` is short but is an endpoint, not an accessor."""
    chunk = by_name(chunks, "createOrder")
    assert chunk.kind == "function"
    assert "PostMapping" in chunk.context_header


def test_trivial_accessors_are_merged(chunks: list[CodeChunk]) -> None:
    """Three getters on a DTO should be one chunk, not three near-duplicates."""
    merged = [c for c in chunks if c.kind == "members" and "Order.java" in c.relative_path]
    assert merged, "Order.java getters were not merged"
    assert "getId" in merged[0].symbol_name
    assert "setTotal" in merged[0].symbol_name


# --------------------------------------------------------------------------- #
# context headers
# --------------------------------------------------------------------------- #


def test_every_chunk_has_a_context_header(chunks: list[CodeChunk]) -> None:
    for chunk in chunks:
        assert chunk.context_header, f"{chunk.symbol_name} has no header"
        assert chunk.relative_path in chunk.context_header


def test_embedding_text_includes_header(chunks: list[CodeChunk]) -> None:
    chunk = by_name(chunks, "placeOrder")
    assert chunk.embedding_text.startswith(chunk.context_header)
    assert chunk.body in chunk.embedding_text


def test_java_package_is_extracted(chunks: list[CodeChunk]) -> None:
    chunk = by_name(chunks, "placeOrder")
    assert "com.demo.service" in chunk.context_header


def test_python_module_is_derived(chunks: list[CodeChunk]) -> None:
    chunk = by_name(chunks, "get_user_by_id")
    assert "user_service" in chunk.context_header


def test_class_skeleton_uses_declaration_not_annotation(chunks: list[CodeChunk]) -> None:
    """`@RestController` tells you nothing; `public class OrderController` does."""
    skeleton = next(c for c in chunks if c.kind == "class" and c.symbol_name == "OrderController")
    assert "class OrderController" in skeleton.context_header


def test_skeleton_elides_method_bodies(chunks: list[CodeChunk]) -> None:
    skeleton = next(c for c in chunks if c.kind == "class" and c.symbol_name == "OrderService")
    assert "{ ... }" in skeleton.body
    assert "paymentService.processPayment" not in skeleton.body


def test_interface_skeleton_keeps_extends_clause(chunks: list[CodeChunk]) -> None:
    """`extends JpaRepository<Order, Long>` only exists on the declaration."""
    skeleton = next(c for c in chunks if c.kind == "class" and c.symbol_name == "OrderRepository")
    assert "JpaRepository" in skeleton.body or "JpaRepository" in skeleton.context_header


def test_imports_are_attached(chunks: list[CodeChunk]) -> None:
    chunk = by_name(chunks, "placeOrder")
    assert any("OrderRepository" in i for i in chunk.imports)


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


def test_parent_symbol_is_set_for_methods(chunks: list[CodeChunk]) -> None:
    assert by_name(chunks, "processPayment").parent_symbol == "PaymentService"
    assert by_name(chunks, "get_user_by_id").parent_symbol == "UserService"


def test_layers_are_inferred(chunks: list[CodeChunk]) -> None:
    assert by_name(chunks, "createOrder").layer.value == "controller"
    assert by_name(chunks, "placeOrder").layer.value == "service"
    assert by_name(chunks, "findByCustomerId").layer.value == "repository"


def test_both_languages_are_chunked(chunks: list[CodeChunk]) -> None:
    languages = {c.language for c in chunks}
    assert Language.JAVA in languages
    assert Language.PYTHON in languages


def test_nested_helpers_are_not_double_emitted(chunks: list[CodeChunk]) -> None:
    """A private helper is one chunk, not one plus a copy inside its caller."""
    names = [c.symbol_name for c in chunks]
    assert names.count("callGateway") == 1


def test_stats_add_up() -> None:
    chunker = AstChunker()
    produced = chunker.chunk_all(RepositoryLoader(FIXTURE).collect())
    assert chunker.stats.files_chunked == 7
    assert chunker.stats.chunks_emitted == len(produced)
    assert chunker.stats.parse_errors == 0
    assert chunker.stats.skeletons > 0
    assert chunker.stats.members > 0


def test_oversized_function_is_split() -> None:
    """A very long method splits into parts that together cover the original."""
    from codemind.core.types import SourceFile

    lines = "\n".join(f"        int v{i} = {i};" for i in range(400))
    code = f"class Big {{\n    void huge() {{\n{lines}\n    }}\n}}\n"
    source = SourceFile(
        absolute_path=Path("Big.java"),
        relative_path="Big.java",
        language=Language.JAVA,
        size_bytes=len(code),
        content=code,
    )
    chunker = AstChunker(max_chunk_chars=1500)
    produced = chunker.chunk(source)
    parts = [c for c in produced if c.kind == "function_part"]
    assert len(parts) > 1
    assert chunker.stats.split_large == 1
    assert all(len(p.body) <= 2000 for p in parts)
