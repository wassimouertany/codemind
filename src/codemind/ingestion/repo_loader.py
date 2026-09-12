"""Clone a repository and select the source files worth indexing.

The selection logic matters more than it looks: a typical Java repo is 60%
generated code, vendored dependencies and build output. Indexing that noise
costs embedding time and, worse, pollutes retrieval with near-duplicate matches
that push the real answer below the rerank cutoff.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pathspec

from codemind.core.exceptions import RepositoryNotFoundError
from codemind.core.types import Language, SourceFile
from codemind.ingestion.language_registry import spec_for_path

logger = logging.getLogger(__name__)

DEFAULT_MAX_FILE_BYTES = 1_000_000
DEFAULT_MAX_LINE_LENGTH = 2_000
"""A single line beyond this almost always means minified or generated code."""

EXCLUDED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        ".idea",
        ".vscode",
        ".gradle",
        ".mvn",
        "node_modules",
        "bower_components",
        "vendor",
        "third_party",
        "venv",
        ".venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "site-packages",
        "build",
        "dist",
        "out",
        "target",
        "bin",
        "obj",
        "coverage",
        "htmlcov",
        ".next",
        ".nuxt",
        ".svelte-kit",
        "migrations",
        "generated",
        "gen",
    }
)

EXCLUDED_FILE_PATTERNS: tuple[str, ...] = (
    "*.min.js",
    "*.min.css",
    "*.bundle.js",
    "*.map",
    "*_pb2.py",
    "*_pb2_grpc.py",
    "*.pb.go",
    "*.generated.*",
    "*.g.dart",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
)


class PathMatcher(Protocol):
    """The only thing we need from a compiled ignore spec.

    Depending on this shape instead of `pathspec.PathSpec` keeps the code
    type-clean across pathspec versions: 1.0 declares PathSpec non-generic,
    1.1 declares it generic, and a hardcoded annotation breaks on one of them.
    """

    def match_file(self, file: str, /) -> bool: ...


def _compile_spec(lines: list[str] | tuple[str, ...]) -> PathMatcher:
    """Compile a gitignore-style spec across pathspec 0.x and 1.x.

    pathspec 1.0 renamed the "gitwildmatch" style to "gitignore" and deprecated
    the old name. Try the new one first so we stay quiet on modern versions.
    """
    try:
        return pathspec.PathSpec.from_lines("gitignore", list(lines))
    except (KeyError, ValueError, LookupError):
        return pathspec.PathSpec.from_lines("gitwildmatch", list(lines))


@dataclass(slots=True)
class LoadStats:
    """Counters for one walk. Surfaced in logs and the ingestion API response."""

    files_seen: int = 0
    files_selected: int = 0
    skipped_unsupported: int = 0
    skipped_excluded_dir: int = 0
    skipped_gitignored: int = 0
    skipped_too_large: int = 0
    skipped_generated: int = 0
    skipped_unreadable: int = 0
    skipped_binary: int = 0
    bytes_selected: int = 0
    per_language: dict[Language, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int | dict[str, int]]:
        return {
            "files_seen": self.files_seen,
            "files_selected": self.files_selected,
            "skipped_unsupported": self.skipped_unsupported,
            "skipped_excluded_dir": self.skipped_excluded_dir,
            "skipped_gitignored": self.skipped_gitignored,
            "skipped_too_large": self.skipped_too_large,
            "skipped_generated": self.skipped_generated,
            "skipped_unreadable": self.skipped_unreadable,
            "skipped_binary": self.skipped_binary,
            "bytes_selected": self.bytes_selected,
            "per_language": {lang.value: n for lang, n in self.per_language.items()},
        }


def clone_repository(url: str, destination: Path, *, depth: int = 1) -> Path:
    """Shallow-clone `url` into `destination`. Reuses an existing clone.

    Depth 1 by default: we index the current state of the code, not its history.
    A full clone of a large repository can be 10x the size for no benefit here.
    """
    from git import GitCommandError, Repo  # imported lazily: ~200ms

    destination = destination.expanduser().resolve()

    if (destination / ".git").exists():
        logger.info("reusing existing clone at %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("cloning %s into %s (depth=%d)", url, destination, depth)
    try:
        Repo.clone_from(url, destination, depth=depth)
    except GitCommandError as exc:  # pragma: no cover - network dependent
        raise RepositoryNotFoundError(f"could not clone {url}: {exc}") from exc
    return destination


class RepositoryLoader:
    """Walks a repository root and yields the files worth indexing."""

    def __init__(
        self,
        root: Path,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        respect_gitignore: bool = True,
        include_tests: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise RepositoryNotFoundError(f"not a directory: {self.root}")

        self.max_file_bytes = max_file_bytes
        self.include_tests = include_tests
        self.stats = LoadStats()

        self._gitignore = self._load_gitignore() if respect_gitignore else None
        self._excluded_files = _compile_spec(EXCLUDED_FILE_PATTERNS)

    # -- public ------------------------------------------------------------ #

    def iter_files(self) -> Iterator[SourceFile]:
        """Yield every indexable source file, in deterministic sorted order.

        Sorted so that two runs over the same commit produce the same order,
        which keeps batched embedding reproducible and makes diffs readable.
        """
        for path in sorted(self._walk()):
            source = self._read(path)
            if source is not None:
                yield source

    def collect(self) -> list[SourceFile]:
        """Materialize all files. Prefer `iter_files` for large repositories."""
        return list(self.iter_files())

    # -- internals --------------------------------------------------------- #

    def _load_gitignore(self) -> PathMatcher | None:
        gitignore = self.root / ".gitignore"
        if not gitignore.is_file():
            return None
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return None
        return _compile_spec(lines)

    def _walk(self) -> Iterator[Path]:
        """Depth-first walk that prunes excluded directories without descending."""
        stack = [self.root]
        while stack:
            current = stack.pop()
            try:
                entries = list(current.iterdir())
            except OSError:
                self.stats.skipped_unreadable += 1
                continue

            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if entry.name in EXCLUDED_DIRECTORIES or entry.name.startswith("."):
                        self.stats.skipped_excluded_dir += 1
                        continue
                    stack.append(entry)
                elif entry.is_file():
                    self.stats.files_seen += 1
                    if self._should_index(entry):
                        yield entry

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _should_index(self, path: Path) -> bool:
        relative = self._relative(path)

        if spec_for_path(path) is None:
            self.stats.skipped_unsupported += 1
            return False

        if self._excluded_files.match_file(relative):
            self.stats.skipped_generated += 1
            return False

        if self._gitignore is not None and self._gitignore.match_file(relative):
            self.stats.skipped_gitignored += 1
            return False

        if not self.include_tests and _looks_like_test(relative):
            return False

        try:
            if path.stat().st_size > self.max_file_bytes:
                self.stats.skipped_too_large += 1
                return False
        except OSError:
            self.stats.skipped_unreadable += 1
            return False

        return True

    def _read(self, path: Path) -> SourceFile | None:
        """Read and decode a file, rejecting binary and minified content."""
        try:
            raw = path.read_bytes()
        except OSError:
            self.stats.skipped_unreadable += 1
            return None

        # Binary detection. A NUL byte is the classic signal, but short random
        # blobs often contain none, so strict UTF-8 decoding is the real test:
        # source code is effectively always valid UTF-8, and anything that is
        # not decodable is not code we can chunk.
        if b"\x00" in raw[:8192]:
            self.stats.skipped_binary += 1
            return None

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            self.stats.skipped_binary += 1
            return None

        if _has_overlong_line(content, DEFAULT_MAX_LINE_LENGTH):
            self.stats.skipped_generated += 1
            return None

        spec = spec_for_path(path)
        if spec is None:  # pragma: no cover - already filtered in _should_index
            return None

        self.stats.files_selected += 1
        self.stats.bytes_selected += len(raw)
        self.stats.per_language[spec.language] = self.stats.per_language.get(spec.language, 0) + 1

        return SourceFile(
            absolute_path=path,
            relative_path=self._relative(path),
            language=spec.language,
            size_bytes=len(raw),
            content=content,
        )


def _has_overlong_line(content: str, limit: int) -> bool:
    return any(len(line) > limit for line in content.splitlines())


def _looks_like_test(relative_path: str) -> bool:
    lowered = relative_path.lower()
    return (
        "/test" in lowered
        or lowered.startswith("test")
        or lowered.endswith(("_test.py", "test.java", ".spec.ts", ".test.ts", ".test.js"))
    )
