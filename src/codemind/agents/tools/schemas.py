"""What the agent tools return. Structured data, never prose.

A node formats these for a prompt or turns them into `Evidence`; the tool
never decides how its result is worded. Every location is a repository-relative
POSIX path with an inclusive 1-based line range — the same convention as
`CodeChunk`, so a tool result and a retrieved chunk can be compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Definition:
    """Where a symbol is defined. `qualified_name` is the symbol graph node id."""

    qualified_name: str
    simple_name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int

    @property
    def display_name(self) -> str:
        """`OrderService.placeOrder` — the qualified name without its path."""
        return self.qualified_name.split("::", 1)[-1]


@dataclass(frozen=True, slots=True)
class CodeHit:
    """One search result, in the order the retrieval pipeline ranked it."""

    rank: int
    file_path: str
    start_line: int
    end_line: int
    symbol: str
    kind: str
    score: float
    source: str


@dataclass(frozen=True, slots=True)
class FileSlice:
    """Lines `start_line..end_line` of a file, and how long the file is."""

    file_path: str
    start_line: int
    end_line: int
    total_lines: int
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CallPath:
    """The shortest call chain from `source` to `target`, as definitions.

    `hops` is empty when no path exists, which is an answer, not a failure.
    `unresolved` names any endpoint that matched no symbol at all, so "no
    path" and "no such symbol" stay distinguishable.
    """

    source: str
    target: str
    hops: tuple[Definition, ...]
    unresolved: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return bool(self.hops)
