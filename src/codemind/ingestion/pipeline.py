"""End-to-end ingestion: a repository URL or path in, an indexed collection out.

    clone -> load -> chunk -> symbol graph -> embed (dense + sparse) -> upsert
                                                 |
                                                 +-> graph saved beside the collection

Ordering is forced by dependencies, not preference. The symbol graph needs every
chunk before it can resolve a call in the first file to a method defined in the
last one, so it is built after chunking and before any upsert — a failure there
should stop the run before it writes half an index.

**Resume is per file, not per chunk.** `chunk_id` hashes path + symbol + body
with no line number, so an unchanged file produces byte-identical IDs and
re-upserting is a no-op. A *changed* file is the problem: renaming a method
leaves the old chunk in the collection forever, still matching queries and still
citing a line that no longer means anything. `delete_by_file` before each file's
upsert is what keeps the index honest, and it is why the unit of work here is a
file rather than a batch of chunks.

The store and encoders arrive as `core.protocols` Protocols, not concrete
classes: they live in `retrieval`, a higher layer, and ingestion must not import
upward. See `core/protocols.py`.

CPU-bound stages (tree-sitter parsing, graph construction, ONNX inference) all
run in worker threads. Ingestion is a script, so nothing is competing for the
event loop — but the same code runs behind `POST /repos`, where it would
otherwise block every other request for minutes.

Memory: sources are materialised as a list because the symbol graph needs a
second pass over them. For a repository large enough that this hurts, the fix is
to re-read files in the graph pass rather than to stream chunks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codemind.core.config import Settings, get_settings
from codemind.core.protocols import ChunkStore, DenseEncoder, SparseEncoder
from codemind.core.types import CodeChunk
from codemind.ingestion.ast_chunker import AstChunker, ChunkStats
from codemind.ingestion.repo_loader import LoadStats, RepositoryLoader, clone_repository
from codemind.ingestion.symbol_graph import ResolutionStats, SymbolGraph

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]
"""(stage, done, total) — drives the CLI progress bar without importing it here."""


@dataclass(slots=True)
class IngestionReport:
    """What one ingest did. Every Day 6 number to record lives here."""

    repository: str
    collection: str
    files: int = 0
    chunks: int = 0
    points_upserted: int = 0
    files_deleted: int = 0
    """Files cleared before re-upsert. Non-zero only on a resumed run."""
    graph_path: Path | None = None
    symbols: int = 0
    edges: int = 0
    wall_seconds: float = 0.0
    load: LoadStats = field(default_factory=LoadStats)
    chunking: ChunkStats = field(default_factory=ChunkStats)
    resolution: ResolutionStats = field(default_factory=ResolutionStats)

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "collection": self.collection,
            "files": self.files,
            "chunks": self.chunks,
            "points_upserted": self.points_upserted,
            "files_deleted": self.files_deleted,
            "graph_path": str(self.graph_path) if self.graph_path else None,
            "symbols": self.symbols,
            "edges": self.edges,
            "wall_seconds": round(self.wall_seconds, 1),
            "load": self.load.as_dict(),
            "chunking": self.chunking.as_dict(),
            "resolution": self.resolution.as_dict(),
        }


class IngestionPipeline:
    """Runs the full ingest. Models are injected: they load once, elsewhere."""

    def __init__(
        self,
        store: ChunkStore,
        embedder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._settings = resolved
        self._store = store
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._batch_chunks = resolved.qdrant_upsert_batch

    @property
    def graph_path(self) -> Path:
        """One graph per collection, so a rebuilt index never loads a stale graph."""
        return self._settings.graph_dir / f"{self._store.collection_name}.json"

    async def ingest(
        self,
        source: str | Path,
        *,
        resume: bool = True,
        recreate: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IngestionReport:
        """Ingest a local path or a clone URL.

        `resume=True` deletes each file's existing points before writing the new
        ones, so a re-run over changed code leaves no orphans. `recreate=True`
        drops the whole collection first and makes that per-file work redundant.
        """
        started = time.perf_counter()
        root = await asyncio.to_thread(self._resolve_source, source)
        report = IngestionReport(repository=str(source), collection=self._store.collection_name)

        loader = RepositoryLoader(root)
        sources = await asyncio.to_thread(loader.collect)
        report.load = loader.stats
        report.files = len(sources)
        _report(progress, "load", len(sources), len(sources))
        if not sources:
            logger.warning("no indexable files under %s", root)
            report.wall_seconds = time.perf_counter() - started
            return report

        chunker = AstChunker()
        chunks = await asyncio.to_thread(chunker.chunk_all, sources)
        report.chunking = chunker.stats
        report.chunks = len(chunks)
        _report(progress, "chunk", len(chunks), len(chunks))

        graph = await asyncio.to_thread(SymbolGraph.build, sources, chunks)
        report.resolution = graph.stats
        report.symbols = graph.graph.number_of_nodes()
        report.edges = graph.graph.number_of_edges()
        _report(progress, "graph", report.symbols, report.symbols)

        await self._store.ensure_collection(recreate=recreate)
        # Deleting per file is pointless work when the collection was just
        # dropped, and it doubles the request count on a fresh ingest.
        await self._index(chunks, report, clear_files=resume and not recreate, progress=progress)

        await asyncio.to_thread(graph.save, self.graph_path)
        report.graph_path = self.graph_path

        report.wall_seconds = time.perf_counter() - started
        logger.info(
            "ingested %s: %d files, %d chunks, %d points in %.1fs (resolution %.1f%%)",
            root,
            report.files,
            report.chunks,
            report.points_upserted,
            report.wall_seconds,
            report.resolution.resolution_rate * 100,
        )
        return report

    def _resolve_source(self, source: str | Path) -> Path:
        """Clone if it looks like a URL, otherwise treat it as a local path."""
        text = str(source)
        if text.startswith(("http://", "https://", "git@")):
            name = text.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]
            return clone_repository(text, self._settings.repos_dir / name)
        return Path(text).expanduser().resolve()

    async def _index(
        self,
        chunks: Sequence[CodeChunk],
        report: IngestionReport,
        *,
        clear_files: bool,
        progress: ProgressCallback | None,
    ) -> None:
        """Encode and upsert, batching whole files so deletes stay aligned."""
        by_file: dict[str, list[CodeChunk]] = {}
        for chunk in chunks:
            by_file.setdefault(chunk.relative_path, []).append(chunk)

        total = len(by_file)
        batch: list[CodeChunk] = []
        paths: list[str] = []

        for done, (path, file_chunks) in enumerate(by_file.items(), start=1):
            batch.extend(file_chunks)
            paths.append(path)
            if len(batch) >= self._batch_chunks:
                await self._flush(batch, paths, report, clear_files=clear_files)
                batch, paths = [], []
                _report(progress, "index", done, total)

        if batch:
            await self._flush(batch, paths, report, clear_files=clear_files)
        _report(progress, "index", total, total)

    async def _flush(
        self,
        batch: list[CodeChunk],
        paths: list[str],
        report: IngestionReport,
        *,
        clear_files: bool,
    ) -> None:
        """Clear superseded points for these files, then write the new ones.

        Deleting before encoding, rather than after, keeps the window where a
        file has no points as short as the upsert itself.
        """
        if clear_files:
            for path in paths:
                await self._store.delete_by_file(path)
            report.files_deleted += len(paths)

        dense, sparse = await asyncio.gather(
            self._embedder.embed_chunks(batch),
            self._sparse.encode_documents([chunk.embedding_text for chunk in batch]),
        )
        report.points_upserted += await self._store.upsert_chunks(batch, dense, sparse)


def _report(callback: ProgressCallback | None, stage: str, done: int, total: int) -> None:
    if callback is not None:
        callback(stage, done, total)
