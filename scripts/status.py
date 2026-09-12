#!/usr/bin/env python
"""Show the project tree with per-file status and build progress.

    uv run python scripts/status.py           # tree + progress
    uv run python scripts/status.py --todo    # only files still to write
    uv run python scripts/status.py --next    # what to work on next

A file counts as WRITTEN when it no longer contains the bootstrap placeholder
`\"\"\"TODO.\"\"\"`. The DAY map below is the schedule from the 4-week plan.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = '"""TODO."""'

# file -> day it gets written
DAY: dict[str, str] = {
    "src/codemind/main.py": "D1",
    "src/codemind/core/config.py": "D1",
    "src/codemind/core/types.py": "D2",
    "src/codemind/core/exceptions.py": "D2",
    "src/codemind/ingestion/repo_loader.py": "D2",
    "src/codemind/ingestion/language_registry.py": "D2",
    "src/codemind/ingestion/ast_chunker.py": "D3",
    "src/codemind/ingestion/metadata.py": "D3",
    "src/codemind/retrieval/tokenizer.py": "D4",
    "src/codemind/retrieval/sparse.py": "D4",
    "src/codemind/retrieval/embedder.py": "D4",
    "src/codemind/retrieval/qdrant_store.py": "D4",
    "scripts/bootstrap_qdrant.py": "D4",
    "src/codemind/retrieval/hybrid.py": "D5",
    "src/codemind/retrieval/schemas.py": "D5",
    "src/codemind/retrieval/reranker.py": "D6",
    "src/codemind/retrieval/expansion.py": "D6",
    "src/codemind/ingestion/symbol_graph.py": "D6",
    "src/codemind/ingestion/pipeline.py": "D6",
    "scripts/ingest_repo.py": "D6",
    "evaluation/retrieval_eval.py": "D7",
    "src/codemind/agents/state.py": "D8",
    "src/codemind/agents/graph.py": "D8",
    "src/codemind/agents/nodes/router.py": "D8",
    "src/codemind/llm/base.py": "D8",
    "src/codemind/llm/ollama_client.py": "D8",
    "src/codemind/agents/nodes/retrieve.py": "D9",
    "src/codemind/agents/nodes/code_analyst.py": "D9",
    "src/codemind/agents/nodes/dependency_agent.py": "D9",
    "src/codemind/agents/nodes/test_agent.py": "D9",
    "src/codemind/agents/tools/search_code.py": "D10",
    "src/codemind/agents/tools/read_file.py": "D10",
    "src/codemind/agents/tools/find_definition.py": "D10",
    "src/codemind/agents/tools/trace_call_path.py": "D10",
    "src/codemind/agents/nodes/synthesize.py": "D11",
    "src/codemind/agents/nodes/grounding.py": "D12",
    "src/codemind/core/observability.py": "D13",
    "src/codemind/core/logging.py": "D13",
    "src/codemind/storage/models.py": "D13",
    "src/codemind/storage/session.py": "D13",
    "src/codemind/storage/checkpointer.py": "D13",
    "src/codemind/api/deps.py": "D14",
    "src/codemind/api/router.py": "D14",
    "src/codemind/api/v1/health.py": "D14",
    "src/codemind/api/v1/query.py": "D14",
    "src/codemind/api/v1/repositories.py": "D14",
    "src/codemind/api/v1/traces.py": "D14",
    "src/codemind/api/v1/schemas.py": "D14",
    "training/dataset/build_seed.py": "D15",
    "training/dataset/synthesize.py": "D15",
    "training/dataset/validate.py": "D15",
    "training/dataset/to_chatml.py": "D16",
    "training/train_qlora.py": "D17",
    "training/merge_and_export.py": "D19",
    "src/codemind/llm/vllm_client.py": "D20",
    "evaluation/model_benchmark.py": "D21",
    "evaluation/ragas_eval.py": "D22",
    "src/codemind/llm/judge.py": "D22",
}

SKIP_DIRS = {
    ".import_linter_cache", "fixtures",
    ".git", ".venv", "__pycache__", "data", ".repos", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "node_modules", "htmlcov", ".idea",
}

G, C, Y, D, B, R = "\033[92m", "\033[96m", "\033[93m", "\033[90m", "\033[1m", "\033[0m"


def is_stub(path: Path) -> bool:
    try:
        return PLACEHOLDER in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def status(rel: str, path: Path) -> tuple[str, str]:
    """Return (marker, state) for one file."""
    if path.suffix == ".py" and is_stub(path):
        return f"{D}[{DAY.get(rel, '  ?'):>4}]{R}", "stub"
    if rel in DAY or path.suffix in {".py", ".sh", ".toml", ".json", ".yml", ".md"}:
        return f"{G}[ ok ]{R}", "done"
    return "      ", "other"


def walk(directory: Path, prefix: str, out: list[str], counts: dict[str, int]) -> None:
    entries = sorted(
        (e for e in directory.iterdir() if e.name not in SKIP_DIRS),
        key=lambda e: (not e.is_dir(), e.name),
    )
    for i, entry in enumerate(entries):
        last = i == len(entries) - 1
        conn = "└── " if last else "├── "
        if entry.is_dir():
            out.append(f"       {prefix}{conn}{B}{entry.name}/{R}")
            walk(entry, prefix + ("    " if last else "│   "), out, counts)
        else:
            if entry.name == "__init__.py":
                continue
            rel = entry.relative_to(ROOT).as_posix()
            marker, state = status(rel, entry)
            counts[state] = counts.get(state, 0) + 1
            out.append(f"{marker} {prefix}{conn}{entry.name}")


def main() -> int:
    args = set(sys.argv[1:])
    written = {r for r in DAY if not is_stub(ROOT / r) and (ROOT / r).exists()}
    pending = {r for r in DAY if r not in written}

    if "--todo" in args:
        print(f"\n  {len(pending)} files still to write:\n")
        for rel in sorted(pending, key=lambda r: (int(DAY[r][1:]), r)):
            print(f"    {D}{DAY[rel]:>4}{R}  {rel}")
        print()
        return 0

    if "--next" in args:
        if not pending:
            print("\n  Nothing pending. Ship it.\n")
            return 0
        nxt = min(int(DAY[r][1:]) for r in pending)
        group = sorted(r for r in pending if int(DAY[r][1:]) == nxt)
        print(f"\n  Next up — Day {nxt}:\n")
        for rel in group:
            print(f"    {rel}")
        print()
        return 0

    out: list[str] = []
    counts: dict[str, int] = {}
    walk(ROOT, "", out, counts)
    print()
    print("\n".join(out))

    total = len(DAY)
    done = len(written)
    pct = done / total
    bar = "█" * int(pct * 40) + "░" * (40 - int(pct * 40))
    print()
    print(f"  {B}Progress{R}  [{G}{bar}{R}]  {done}/{total} files  ({pct:.0%})")

    by_day: dict[int, list[int]] = {}
    for rel, day in DAY.items():
        n = int(day[1:])
        d, t = by_day.setdefault(n, [0, 0])
        by_day[n] = [d + (rel in written), t + 1]
    line = "  ".join(
        f"{(G if d == t else Y if d else D)}D{n}:{d}/{t}{R}"
        for n, (d, t) in sorted(by_day.items())
    )
    print(f"  {line}")
    print(f"\n  {D}--todo  list pending files    --next  what to build now{R}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
