"""End-to-end ingestion over the fixture repo.

`test_resume_replaces_a_changed_file_without_orphans` is the one that matters.
Orphaned points from a renamed method keep matching queries and keep citing
lines that have moved, which is exactly the failure the grounding gate cannot
catch — the citation resolves, it just describes code that no longer exists.
"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from codemind.core.config import Settings
from codemind.ingestion.pipeline import IngestionPipeline
from codemind.ingestion.symbol_graph import SymbolGraph
from codemind.retrieval.embedder import get_embedder
from codemind.retrieval.qdrant_store import QdrantStore
from codemind.retrieval.sparse import get_sparse_encoder

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning"),
]


@pytest.fixture
async def pipeline(tmp_path: Path) -> AsyncIterator[tuple[IngestionPipeline, QdrantStore]]:
    client = AsyncQdrantClient(":memory:")
    settings = Settings(qdrant_collection="test_pipeline", graph_dir=tmp_path / "graphs")
    store = QdrantStore(settings, client=client)
    yield IngestionPipeline(store, get_embedder(), get_sparse_encoder(), settings), store
    await client.close()


# --------------------------------------------------------------------------- #
# the full run
# --------------------------------------------------------------------------- #


async def test_ingest_indexes_every_chunk_and_persists_the_graph(
    pipeline: tuple[IngestionPipeline, QdrantStore],
) -> None:
    pipe, store = pipeline
    report = await pipe.ingest(FIXTURE)

    assert report.files == 7
    assert report.chunks == 23
    assert report.points_upserted == report.chunks
    assert await store.count() == report.chunks

    assert report.graph_path is not None
    assert report.graph_path.exists()
    restored = SymbolGraph.load(report.graph_path)
    assert restored.graph.number_of_nodes() == report.symbols
    assert restored.graph.number_of_edges() == report.edges


async def test_report_carries_all_three_stat_blocks(
    pipeline: tuple[IngestionPipeline, QdrantStore],
) -> None:
    """Day 6 asks for these numbers to be recorded, so they ride with the result."""
    report = (await pipeline[0].ingest(FIXTURE)).as_dict()

    assert report["load"]["files_selected"] == 7
    assert report["chunking"]["chunks_emitted"] == 23
    assert report["chunking"]["parse_errors"] == 0
    assert report["resolution"]["call_sites"] > 0
    assert 0.0 <= report["resolution"]["resolution_rate"] <= 1.0
    assert report["wall_seconds"] > 0


async def test_progress_callback_reports_every_stage(
    pipeline: tuple[IngestionPipeline, QdrantStore],
) -> None:
    seen: list[tuple[str, int, int]] = []
    await pipeline[0].ingest(FIXTURE, progress=lambda *args: seen.append(args))

    assert {stage for stage, _, _ in seen} == {"load", "chunk", "graph", "index"}
    assert seen[-1][1] == seen[-1][2], "the last report must show completion"


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #


async def test_reingesting_unchanged_code_is_idempotent(
    pipeline: tuple[IngestionPipeline, QdrantStore],
) -> None:
    """Content-addressed IDs: the same code must re-upsert onto itself."""
    pipe, store = pipeline
    first = await pipe.ingest(FIXTURE)
    second = await pipe.ingest(FIXTURE)

    assert await store.count() == first.chunks
    assert second.chunks == first.chunks
    assert second.files_deleted > 0, "resume should have cleared files before re-upsert"


async def test_resume_replaces_a_changed_file_without_orphans(
    pipeline: tuple[IngestionPipeline, QdrantStore], tmp_path: Path
) -> None:
    """Rename a method and the old chunk must be gone, not merely outranked."""
    pipe, store = pipeline
    work = tmp_path / "repo"
    shutil.copytree(FIXTURE, work)

    before = await pipe.ingest(work)
    assert any(c.symbol_name == "placeOrder" for c in await _all_chunks(store))

    target = work / "src/main/java/com/demo/service/OrderService.java"
    target.write_text(target.read_text().replace("placeOrder", "submitOrder"), encoding="utf-8")

    after = await pipe.ingest(work)
    symbols = {c.symbol_name for c in await _all_chunks(store)}

    assert "submitOrder" in symbols
    assert "placeOrder" not in symbols, "orphaned chunk survived re-ingestion"
    assert after.chunks == before.chunks


async def test_recreate_skips_per_file_deletes(
    pipeline: tuple[IngestionPipeline, QdrantStore],
) -> None:
    """Dropping the collection makes per-file clearing redundant work."""
    pipe, store = pipeline
    await pipe.ingest(FIXTURE)
    report = await pipe.ingest(FIXTURE, recreate=True)

    assert report.files_deleted == 0
    assert await store.count() == report.chunks


async def test_resume_disabled_leaves_orphans(
    pipeline: tuple[IngestionPipeline, QdrantStore], tmp_path: Path
) -> None:
    """The contrast case, so the resume test above is proving something."""
    pipe, store = pipeline
    work = tmp_path / "repo2"
    shutil.copytree(FIXTURE, work)
    await pipe.ingest(work, resume=False)

    target = work / "src/main/java/com/demo/service/OrderService.java"
    target.write_text(target.read_text().replace("placeOrder", "submitOrder"), encoding="utf-8")
    await pipe.ingest(work, resume=False)

    symbols = {c.symbol_name for c in await _all_chunks(store)}
    assert {"placeOrder", "submitOrder"} <= symbols


# --------------------------------------------------------------------------- #
# degenerate input
# --------------------------------------------------------------------------- #


async def test_empty_repository_reports_nothing_and_does_not_raise(
    pipeline: tuple[IngestionPipeline, QdrantStore], tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    report = await pipeline[0].ingest(empty)

    assert report.files == 0
    assert report.chunks == 0
    assert report.graph_path is None


async def _all_chunks(store: QdrantStore) -> list:
    points, _ = await store.client.scroll(store.collection_name, limit=500, with_payload=True)
    from codemind.retrieval.qdrant_store import chunk_from_payload

    return [chunk_from_payload(p.payload) for p in points]


# --------------------------------------------------------------------------- #
# the layering seam
# --------------------------------------------------------------------------- #


def test_concrete_implementations_satisfy_the_protocols() -> None:
    """Ingestion depends on shapes, not classes. Drift here fails silently."""
    from codemind.core.protocols import ChunkStore, DenseEncoder, SparseEncoder

    assert isinstance(QdrantStore(Settings()), ChunkStore)
    assert isinstance(get_embedder(), DenseEncoder)
    assert isinstance(get_sparse_encoder(), SparseEncoder)
