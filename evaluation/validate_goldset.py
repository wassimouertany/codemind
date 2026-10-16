"""Validate the retrieval gold set before it is used to measure anything.

    uv run python -m evaluation.validate_goldset
    uv run python -m evaluation.validate_goldset --collection codemind_petclinic

A gold file that is not in the index is a **permanent zero**: no amount of
prefetch, fusion or reranking can retrieve a chunk that was never written. Left
unchecked it looks exactly like a retrieval failure, so Recall@10 ends up
measuring a typo. This script makes that impossible to miss before Gate 1
numbers are recorded.

Structural checks (unique ids, known category, 1-4 gold files, non-empty
question) are cheap and catch hand-editing slips. The index check is the one
that matters, and when a path is missing the three closest indexed paths are
printed with their similarity: spotting one wrong segment in a 60-character
Java path by eye is miserable.

Exits non-zero on any failure, so `make` targets and CI can depend on it.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codemind.core.config import Settings, get_settings
from codemind.retrieval.qdrant_store import QdrantStore

DEFAULT_GOLDSET = Path("evaluation/datasets/retrieval_goldset.jsonl")
CATEGORIES = frozenset({"bug", "architecture", "dependency", "locator"})
MAX_GOLD_FILES = 4
CLOSE_MATCHES = 3
SCROLL_PAGE = 256


@dataclass(slots=True)
class GoldQuery:
    """One validated gold-set row."""

    query_id: str
    category: str
    question: str
    gold_files: list[str]
    notes: str = ""

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> GoldQuery:
        return cls(
            query_id=str(raw.get("id", "")),
            category=str(raw.get("category", "")),
            question=str(raw.get("question", "")),
            gold_files=list(raw.get("gold_files") or []),
            notes=str(raw.get("notes", "")),
        )


@dataclass(slots=True)
class ValidationReport:
    queries: list[GoldQuery] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    indexed_files: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def load_goldset(path: Path) -> list[GoldQuery]:
    """Parse the JSONL, failing loudly on the line that is malformed."""
    if not path.is_file():
        raise SystemExit(f"error: gold set not found at {path}")

    queries: list[GoldQuery] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: {path}:{number} is not valid JSON: {exc}") from exc
        queries.append(GoldQuery.from_raw(raw))
    return queries


def check_structure(queries: list[GoldQuery]) -> list[str]:
    """Everything checkable without touching Qdrant."""
    errors: list[str] = []

    seen: Counter[str] = Counter(query.query_id for query in queries)
    for query_id, count in seen.items():
        if not query_id:
            errors.append("a row has no id")
        elif count > 1:
            errors.append(f"id {query_id!r} appears {count} times — ids must be unique")

    for query in queries:
        label = query.query_id or "<no id>"
        if query.category not in CATEGORIES:
            errors.append(f"{label}: category {query.category!r} not in {sorted(CATEGORIES)}")
        if not query.question.strip():
            errors.append(f"{label}: question is empty")
        count = len(query.gold_files)
        if not 1 <= count <= MAX_GOLD_FILES:
            errors.append(f"{label}: {count} gold files, expected 1-{MAX_GOLD_FILES}")
        if len(set(query.gold_files)) != count:
            errors.append(f"{label}: duplicate paths in gold_files")
        for gold in query.gold_files:
            if not gold.strip():
                errors.append(f"{label}: empty path in gold_files")
    return errors


async def indexed_paths(store: QdrantStore) -> set[str]:
    """Every distinct `file_path` in the collection.

    Scrolls with `with_payload=["file_path"]` so bodies stay on disk — the
    payloads are the largest thing in the collection and none of them are
    needed here.
    """
    paths: set[str] = set()
    offset: Any = None
    while True:
        points, offset = await store.client.scroll(
            store.collection_name,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=["file_path"],
            with_vectors=False,
        )
        for point in points:
            if point.payload and (path := point.payload.get("file_path")):
                paths.add(str(path))
        if offset is None:
            break
    return paths


def check_against_index(queries: list[GoldQuery], indexed: set[str]) -> list[str]:
    """The check that actually protects the numbers."""
    if not indexed:
        return ["collection is empty — ingest the repository before validating"]

    pool = sorted(indexed)
    errors: list[str] = []
    for query in queries:
        for gold in query.gold_files:
            if gold in indexed:
                continue
            errors.append(f"{query.query_id}: gold path is not indexed: {gold}")
            for suggestion in difflib.get_close_matches(gold, pool, n=CLOSE_MATCHES, cutoff=0.0):
                ratio = difflib.SequenceMatcher(None, gold, suggestion).ratio()
                errors.append(f"    did you mean ({ratio:.0%})  {suggestion}")
    return errors


async def validate(goldset: Path, settings: Settings) -> ValidationReport:
    queries = load_goldset(goldset)
    report = ValidationReport(queries=queries)
    report.errors.extend(check_structure(queries))

    store = QdrantStore(settings)
    try:
        indexed = await indexed_paths(store)
    finally:
        await store.close()

    report.indexed_files = len(indexed)
    report.errors.extend(check_against_index(queries, indexed))
    return report


def print_report(report: ValidationReport, settings: Settings, goldset: Path) -> None:
    gold_paths = {gold for query in report.queries for gold in query.gold_files}
    print(f"\n  gold set     {goldset}")
    print(f"  collection   {settings.qdrant_collection} ({report.indexed_files} files indexed)")
    print(f"  questions    {len(report.queries)}")
    print(f"  gold paths   {len(gold_paths)} distinct")

    print("\n  per category")
    counts = Counter(query.category for query in report.queries)
    for category in sorted(CATEGORIES):
        print(f"    {category:<13}{counts.get(category, 0)}")
    unknown = sorted(set(counts) - CATEGORIES)
    for category in unknown:
        print(f"    {category!r:<13}{counts[category]}  <- not a valid category")

    if report.ok:
        print("\n  OK — every gold path is indexed\n")
        return

    print(f"\n  {len(report.errors)} problem(s):")
    for error in report.errors:
        prefix = "  " if error.startswith("    ") else "    - "
        print(f"{prefix}{error}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--collection", help="override QDRANT_COLLECTION")
    args = parser.parse_args()

    settings = Settings(qdrant_collection=args.collection) if args.collection else get_settings()
    report = asyncio.run(validate(args.goldset, settings))
    print_report(report, settings, args.goldset)
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
