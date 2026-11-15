"""read_file: return an exact line range of a repository file, bounds-checked.

Every check refuses rather than repairs. A path outside the repository (`..`,
an absolute path, a symlink pointing out) is a `ToolError` — this tool reads
files a model asked for, so it must not become a way to read `.env`. A range
past the end of the file, inverted, or wider than `TOOL_READ_MAX_LINES` is a
`ToolError` too: clamping it would return different lines than were asked for,
and a citation built on them would point at code nobody read.

Lines are numbered 1-based and inclusive, like `CodeChunk`, so a chunk's
`start_line..end_line` can be passed straight through. Files are decoded as
UTF-8 with replacement: one undecodable byte must not hide a whole file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from codemind.agents.tools.schemas import FileSlice
from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import ToolError


class ReadFile:
    def __init__(self, repo_root: Path, settings: Settings | None = None) -> None:
        self._root = repo_root.resolve()
        self._max_lines = (settings or get_settings()).tool_read_max_lines

    async def __call__(self, path: str, start_line: int, end_line: int) -> FileSlice:
        return await asyncio.to_thread(self._read, path, start_line, end_line)

    def _read(self, path: str, start_line: int, end_line: int) -> FileSlice:
        target = self._inside_root(path)
        if start_line < 1 or end_line < start_line:
            raise ToolError(f"invalid line range {start_line}-{end_line} for {path}")
        if end_line - start_line + 1 > self._max_lines:
            raise ToolError(
                f"{path}:{start_line}-{end_line} is {end_line - start_line + 1} lines; "
                f"the limit is {self._max_lines} (TOOL_READ_MAX_LINES)"
            )
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            raise ToolError(f"cannot read {path}: {exc}") from exc
        if end_line > len(lines):
            raise ToolError(
                f"{path} has {len(lines)} lines; {start_line}-{end_line} is past the end"
            )
        return FileSlice(
            file_path=path,
            start_line=start_line,
            end_line=end_line,
            total_lines=len(lines),
            lines=tuple(lines[start_line - 1 : end_line]),
        )

    def _inside_root(self, path: str) -> Path:
        if not path or Path(path).is_absolute():
            raise ToolError(f"path must be relative to the repository root: {path!r}")
        target = (self._root / path).resolve()
        if not target.is_relative_to(self._root):
            raise ToolError(f"path escapes the repository root: {path!r}")
        if not target.is_file():
            raise ToolError(f"no such file in the repository: {path!r}")
        return target
