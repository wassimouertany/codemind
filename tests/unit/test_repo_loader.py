"""Tests for the repository loader.

The fixture deliberately contains noise the loader must reject: node_modules,
a minified bundle, a gitignored build directory, and a binary file with a .py
extension. Each rejection has its own assertion so a regression names itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codemind.core.exceptions import RepositoryNotFoundError
from codemind.core.types import Language
from codemind.ingestion.repo_loader import RepositoryLoader

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"


@pytest.fixture(scope="module")
def loaded() -> tuple[RepositoryLoader, list[str]]:
    loader = RepositoryLoader(FIXTURE)
    files = loader.collect()
    return loader, [f.relative_path for f in files]


def test_fixture_exists() -> None:
    assert FIXTURE.is_dir(), "run scripts/make_fixture.sh first"


def test_finds_the_java_call_chain(loaded: tuple[RepositoryLoader, list[str]]) -> None:
    _, paths = loaded
    for expected in (
        "src/main/java/com/demo/controller/OrderController.java",
        "src/main/java/com/demo/service/OrderService.java",
        "src/main/java/com/demo/service/PaymentService.java",
        "src/main/java/com/demo/repository/OrderRepository.java",
        "src/main/java/com/demo/model/Order.java",
    ):
        assert expected in paths


def test_finds_the_python_files(loaded: tuple[RepositoryLoader, list[str]]) -> None:
    _, paths = loaded
    assert "app/user_service.py" in paths
    assert "app/user_repository.py" in paths


def test_excludes_node_modules(loaded: tuple[RepositoryLoader, list[str]]) -> None:
    _, paths = loaded
    assert not any("node_modules" in p for p in paths)


def test_excludes_minified_bundles(loaded: tuple[RepositoryLoader, list[str]]) -> None:
    _, paths = loaded
    assert "app/vendor.min.js" not in paths


def test_respects_gitignore(loaded: tuple[RepositoryLoader, list[str]]) -> None:
    """target/ is in .gitignore AND in EXCLUDED_DIRECTORIES — belt and braces."""
    _, paths = loaded
    assert not any(p.startswith("target/") for p in paths)


def test_rejects_binary_with_source_extension(loaded: tuple[RepositoryLoader, list[str]]) -> None:
    """blob.py has a .py extension but is binary; it must be rejected."""
    loader, paths = loaded
    assert "app/blob.py" not in paths
    assert loader.stats.skipped_binary >= 1


def test_ignores_non_source_files(loaded: tuple[RepositoryLoader, list[str]]) -> None:
    _, paths = loaded
    assert "README.md" not in paths


def test_paths_are_relative_and_posix(loaded: tuple[RepositoryLoader, list[str]]) -> None:
    """Chunk IDs hash the path, so it must be stable across machines."""
    _, paths = loaded
    for p in paths:
        assert not p.startswith("/")
        assert "\\" not in p


def test_language_detection(loaded: tuple[RepositoryLoader, list[str]]) -> None:
    loader, _ = loaded
    assert loader.stats.per_language[Language.JAVA] == 5
    assert loader.stats.per_language[Language.PYTHON] == 2


def test_stats_are_consistent(loaded: tuple[RepositoryLoader, list[str]]) -> None:
    loader, paths = loaded
    assert loader.stats.files_selected == len(paths)
    assert loader.stats.files_seen > loader.stats.files_selected
    assert loader.stats.bytes_selected > 0


def test_content_is_decoded(loaded: tuple[RepositoryLoader, list[str]]) -> None:
    loader = RepositoryLoader(FIXTURE)
    service = next(f for f in loader.iter_files() if f.relative_path.endswith("OrderService.java"))
    assert "PaymentTimeoutException" in service.content
    assert service.line_count > 10


def test_exclude_tests_flag() -> None:
    with_tests = len(RepositoryLoader(FIXTURE, include_tests=True).collect())
    without = len(RepositoryLoader(FIXTURE, include_tests=False).collect())
    assert without <= with_tests


def test_deterministic_order() -> None:
    """Two walks must agree, or batched embedding is not reproducible."""
    first = [f.relative_path for f in RepositoryLoader(FIXTURE).iter_files()]
    second = [f.relative_path for f in RepositoryLoader(FIXTURE).iter_files()]
    assert first == second


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(RepositoryNotFoundError):
        RepositoryLoader(tmp_path / "does-not-exist")


def test_size_cap(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("x = 1\n" * 50_000)
    loader = RepositoryLoader(tmp_path, max_file_bytes=100)
    assert loader.collect() == []
    assert loader.stats.skipped_too_large == 1
