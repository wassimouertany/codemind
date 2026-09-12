"""Tests for the language registry and architectural layer inference."""

from __future__ import annotations

import pytest

from codemind.core.types import Language, Layer
from codemind.ingestion.language_registry import (
    SUPPORTED_EXTENSIONS,
    infer_layer,
    is_supported,
    spec_for_language,
    spec_for_path,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/main/java/com/demo/OrderService.java", Language.JAVA),
        ("app/user_service.py", Language.PYTHON),
        ("src/api/client.ts", Language.TYPESCRIPT),
        ("src/legacy/util.js", Language.JAVASCRIPT),
        ("components/Button.jsx", Language.JAVASCRIPT),
        ("types/models.pyi", Language.PYTHON),
    ],
)
def test_extension_mapping(path: str, expected: Language) -> None:
    spec = spec_for_path(path)
    assert spec is not None
    assert spec.language is expected


@pytest.mark.parametrize("path", ["README.md", "pom.xml", "image.png", "Makefile", "a.rs"])
def test_unsupported_extensions(path: str) -> None:
    assert spec_for_path(path) is None
    assert not is_supported(path)


def test_case_insensitive() -> None:
    assert spec_for_path("Order.JAVA") is not None


def test_every_language_has_a_spec() -> None:
    for language in Language:
        spec = spec_for_language(language)
        assert spec.grammar
        assert spec.chunk_nodes, f"{language} has no chunkable node types"
        assert spec.extensions


def test_extensions_are_unique_across_languages() -> None:
    """A duplicate extension would silently route files to the wrong grammar."""
    seen: set[str] = set()
    for language in Language:
        spec = spec_for_language(language)
        assert not (seen & spec.extensions), f"{language}: duplicate extension"
        seen |= spec.extensions
    assert seen == set(SUPPORTED_EXTENSIONS)


# --------------------------------------------------------------------------- #
# layer inference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/main/java/com/demo/controller/OrderController.java", Layer.CONTROLLER),
        ("src/main/java/com/demo/service/OrderService.java", Layer.SERVICE),
        ("src/main/java/com/demo/repository/OrderRepository.java", Layer.REPOSITORY),
        ("src/main/java/com/demo/model/Order.java", Layer.MODEL),
        ("src/config/AppConfig.java", Layer.CONFIG),
        ("app/routers/users.py", Layer.CONTROLLER),
        ("app/dao/user_dao.py", Layer.REPOSITORY),
        ("app/random_helper.py", Layer.UNKNOWN),
    ],
)
def test_infer_layer(path: str, expected: Layer) -> None:
    assert infer_layer(path) is expected


@pytest.mark.parametrize(
    "path",
    [
        "src/test/java/com/demo/service/OrderServiceTest.java",
        "tests/unit/test_user_service.py",
        "src/__tests__/api.spec.ts",
    ],
)
def test_tests_win_over_other_patterns(path: str) -> None:
    """OrderServiceTest.java is a test, not a service. Order of checks matters."""
    assert infer_layer(path) is Layer.TEST


def test_symbol_name_contributes() -> None:
    assert infer_layer("app/misc.py", "UserRepository") is Layer.REPOSITORY
