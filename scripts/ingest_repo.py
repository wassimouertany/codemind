"""Ingest a repository into Qdrant: clone, chunk, build the symbol graph, index.

    uv run python scripts/ingest_repo.py --repo https://github.com/spring-projects/spring-petclinic
    uv run python scripts/ingest_repo.py --repo ./local/checkout --collection scratch
    uv run python scripts/ingest_repo.py --repo ./repo --no-resume      # skip per-file deletes
    uv run python scripts/ingest_repo.py --repo ./repo --recreate       # drop the collection first

`make ingest REPO=...` runs this with `EMBEDDING_DEVICE=cuda`, which is safe
only because the LLM is not loaded during ingestion. Everything else comes from
Settings.

Resume is on by default: each file's existing points are deleted before its new
ones are written, so a re-run over changed code leaves no chunks citing code
that moved. `--no-resume` is faster on a first ingest into an empty collection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import CodeMindError
from codemind.ingestion.pipeline import IngestionPipeline, IngestionReport
from codemind.retrieval.embedder import Embedder
from codemind.retrieval.qdrant_store import QdrantStore
from codemind.retrieval.sparse import SparseEncoder


class ProgressBar:
    """Minimal stderr progress bar. No dependency, and silent when piped.

    Ingestion is long enough that a silent terminal looks like a hang, but not
    worth pulling `rich` or `tqdm` into the runtime dependency set for.
    """

    WIDTH = 28

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._started = time.perf_counter()

    def __call__(self, stage: str, done: int, total: int) -> None:
        if not self._enabled:
            return
        filled = int(self.WIDTH * done / total) if total else self.WIDTH
        bar = "#" * filled + "." * (self.WIDTH - filled)
        elapsed = time.perf_counter() - self._started
        sys.stderr.write(f"\r  {stage:<6} [{bar}] {done}/{total}  {elapsed:5.1f}s")
        sys.stderr.flush()
        if done >= total:
            sys.stderr.write("\n")


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.collection:
        settings = Settings(qdrant_collection=args.collection)

    store = QdrantStore(settings)
    pipeline = IngestionPipeline(store, Embedder(settings), SparseEncoder(settings), settings)
    progress = ProgressBar(enabled=sys.stderr.isatty() and not args.json)

    try:
        report = await pipeline.ingest(
            args.repo,
            resume=not args.no_resume,
            recreate=args.recreate,
            progress=progress,
        )
    except (CodeMindError, UnexpectedResponse, ResponseHandlingException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        await store.close()

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        _print_report(report)
    return 0


def _print_report(report: IngestionReport) -> None:
    """Human-readable summary — the Day 6 numbers worth recording."""
    data = report.as_dict()
    load, chunking, resolution = data["load"], data["chunking"], data["resolution"]

    print(f"\n  repository   {data['repository']}")
    print(f"  collection   {data['collection']}")
    print(f"  files        {data['files']} selected of {load['files_seen']} seen")
    print(
        f"  chunks       {data['chunks']} "
        f"({chunking['skeletons']} skeletons, {chunking['members']} members, "
        f"{chunking['merged_small']} merged, {chunking['split_large']} split)"
    )
    print(
        f"  points       {data['points_upserted']} upserted, {data['files_deleted']} files cleared"
    )
    print(f"  symbols      {data['symbols']} nodes, {data['edges']} edges")
    print(
        f"  resolution   {resolution['resolution_rate']:.1%} of "
        f"{resolution['intra_repo_sites']} resolvable call sites "
        f"({resolution['external']} external, {resolution['ambiguous_dropped']} ambiguous)"
    )
    print(f"  graph        {data['graph_path']}")
    print(f"  wall time    {data['wall_seconds']}s")
    if chunking["parse_errors"]:
        print(f"  WARNING      {chunking['parse_errors']} files failed to parse")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="clone URL or local path")
    parser.add_argument("--collection", help="override QDRANT_COLLECTION")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="skip per-file deletes (faster into an empty collection, leaves orphans otherwise)",
    )
    parser.add_argument(
        "--recreate", action="store_true", help="drop the collection first (destroys all points)"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=get_settings().log_level,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
