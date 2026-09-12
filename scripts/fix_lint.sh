#!/usr/bin/env bash
# Fixes the five `make lint` failures after Day 3.
#   1. RUF100  unnecessary `# noqa: BLE001` (the handler re-raises, so BLE never fires)
#   2. SIM102  nested if in _has_annotation
#   3. UP037   quoted type annotations (x2)
#   4. E902    ruff trying to lint the deliberately-binary test fixture
# Also enables BLE (blind-except) linting, which we should have had from day one.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> pyproject.toml: exclude fixtures, enable BLE"
python3 - <<'PY'
import pathlib
p = pathlib.Path("pyproject.toml")
s = p.read_text()

if "exclude = [\"tests/fixtures\"]" not in s:
    s = s.replace(
        'src = ["src", "tests"]',
        'src = ["src", "tests"]\n# blob.py is intentionally non-UTF-8; ruff must not try to parse fixtures\nexclude = ["tests/fixtures"]',
    )
if '"BLE"' not in s:
    s = s.replace(
        'select = ["E", "F", "I", "N", "UP", "B", "C4", "SIM", "TID", "RUF", "ASYNC"]',
        'select = ["E", "F", "I", "N", "UP", "B", "BLE", "C4", "SIM", "TID", "RUF", "ASYNC"]',
    )
p.write_text(s)
print("   ok")
PY

echo "==> ast_chunker.py: SIM102 + unused parameter + quoted annotation"
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/codemind/ingestion/ast_chunker.py")
s = p.read_text()

s = s.replace(
    '''        if child.type in {"modifiers", "decorator"}:
            if "@" in _text(child, raw):
                return True
        if child.type == "annotation" or child.type == "marker_annotation":
            return True''',
    '''        if child.type in {"modifiers", "decorator"} and "@" in _text(child, raw):
            return True
        if child.type in {"annotation", "marker_annotation"}:
            return True''',
)
s = s.replace(
    'def _declaration_line(node: Node, raw: bytes, spec: LanguageSpec, chunker: "AstChunker") -> str:',
    "def _declaration_line(node: Node, raw: bytes, spec: LanguageSpec) -> str:",
)
s = s.replace(
    "signature = _declaration_line(container, raw, spec, self)",
    "signature = _declaration_line(container, raw, spec)",
)
p.write_text(s)
print("   ok")
PY

echo "==> repo_loader.py: quoted annotation"
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/codemind/ingestion/repo_loader.py")
s = p.read_text()
s = s.replace(
    'def _compile_spec(lines: "list[str] | tuple[str, ...]") -> pathspec.PathSpec:',
    "def _compile_spec(lines: list[str] | tuple[str, ...]) -> pathspec.PathSpec:",
)
p.write_text(s)
print("   ok")
PY

echo "==> status.py: hide build caches and fixtures"
python3 - <<'PY'
import pathlib
p = pathlib.Path("scripts/status.py")
if p.exists():
    s = p.read_text()
    if ".import_linter_cache" not in s:
        s = s.replace(
            'SKIP_DIRS = {\n',
            'SKIP_DIRS = {\n    ".import_linter_cache", "fixtures",\n',
        )
        p.write_text(s)
print("   ok")
PY

echo "==> gitignore build caches"
grep -q '.import_linter_cache' .gitignore 2>/dev/null || printf '.import_linter_cache/\n.repos/\n' >> .gitignore

echo "==> removing the buggy v1 verification script"
rm -f scripts/verify_stack.py

echo "==> ruff --fix + format"
uv run ruff check --fix src tests || true
uv run ruff format src tests

echo ""
echo "==> verifying"
uv run ruff check src tests
uv run mypy
uv run lint-imports
uv run pytest -q
echo ""
echo "Day 3 closed."
