"""Hybrid retrieval: dense + BM25 fused server-side with RRF, in one round trip.

    query
      -> embedder        -> dense vector  -> Prefetch(using="dense",  limit=PREFETCH_DENSE)
      -> sparse encoder  -> sparse vector -> Prefetch(using="sparse", limit=PREFETCH_SPARSE)
                                          -> FusionQuery(FUSION_METHOD), limit = dense + sparse
                                          -> RetrievalResult

Reciprocal Rank Fusion scores by *rank*, not raw score, which is the only sane
way to combine these branches: cosine similarity lives in [-1, 1] and BM25 is
unbounded, so any weighted sum of the raw scores would need retuning per repo.

The sparse branch goes through `SparseEncoder.encode_query`, which applies
`tokenize_query()` — the same tokenizer the index side used. That symmetry is
the single most important property of the whole pipeline; see `tokenizer.py`.
The dense branch sees the raw query, because splitting identifiers would only
strip signal the embedding model can use.

What a fused query cannot tell you: Qdrant returns only the fused score, not
which branch contributed a point or where it ranked there. So `dense_score`,
`sparse_score` and the per-branch candidate counts stay unset rather than being
guessed. The Day 7 ablation measures branches by running them separately.

Latency is not measured here. Per CLAUDE.md, stage timings come from Langfuse
spans (Day 13), not `time.perf_counter()` scattered through retrieval code.
"""

from __future__ import annotations

import asyncio
import logging

from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import RetrievalError
from codemind.retrieval.embedder import Embedder
from codemind.retrieval.qdrant_store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    QdrantStore,
    chunk_from_payload,
)
from codemind.retrieval.schemas import RetrievalResult, RetrievalSource, ScoredChunk
from codemind.retrieval.sparse import SparseEncoder

logger = logging.getLogger(__name__)

_FUSION_BY_SETTING: dict[str, models.Fusion] = {
    "rrf": models.Fusion.RRF,
    "dbsf": models.Fusion.DBSF,
}


class HybridRetriever:
    """Runs one fused dense+sparse query against the chunk collection.

    Holds no models of its own — the embedder, sparse encoder and store are
    created once in the FastAPI lifespan and injected, so this is cheap to
    construct and safe to share across requests.
    """

    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        sparse_encoder: SparseEncoder,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._store = store
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._prefetch_dense = resolved.prefetch_dense
        self._prefetch_sparse = resolved.prefetch_sparse
        self._fusion = _FUSION_BY_SETTING[resolved.fusion_method]

    @property
    def fusion(self) -> models.Fusion:
        """Fusion applied to the prefetches, from `FUSION_METHOD`."""
        return self._fusion

    @property
    def fused_limit(self) -> int:
        """Candidates returned after fusion: every prefetch slot, deduplicated server-side."""
        return self._prefetch_dense + self._prefetch_sparse

    async def search(
        self,
        query: str,
        *,
        query_filter: models.Filter | None = None,
    ) -> RetrievalResult:
        """Retrieve fused candidates for `query`, ranked by RRF score.

        `query_filter` is applied to both prefetch branches, so a filtered
        search ranks within the filtered set rather than fusing two unfiltered
        lists and discarding afterwards. It is backed by the `file_path` and
        `language` payload indexes.

        Returns up to `PREFETCH_DENSE + PREFETCH_SPARSE` chunks; fewer when the
        branches overlap or the collection is small. An empty result is valid.
        """
        if not query.strip():
            raise RetrievalError("cannot retrieve for an empty query")

        # Both encoders run in worker threads, so these genuinely overlap.
        dense_vector, sparse_vector = await asyncio.gather(
            self._embedder.embed_query(query),
            self._sparse.encode_query(query),
        )

        # An all-stopword query yields an empty sparse vector. Qdrant accepts
        # it and the prefetch contributes nothing, so fusion degrades to
        # dense-only rather than failing — verified against a live server.
        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR,
                limit=self._prefetch_dense,
                filter=query_filter,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR,
                limit=self._prefetch_sparse,
                filter=query_filter,
            ),
        ]

        try:
            response = await self._store.client.query_points(
                collection_name=self._store.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=self._fusion),
                limit=self.fused_limit,
                with_payload=True,
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise RetrievalError(
                f"hybrid query against {self._store.collection_name!r} failed: {exc}"
            ) from exc

        chunks = [
            ScoredChunk(
                chunk=chunk_from_payload(point.payload),
                score=point.score,
                source=RetrievalSource.HYBRID,
                rrf_score=point.score,
                rank=rank,
            )
            for rank, point in enumerate(response.points, start=1)
        ]
        logger.debug("hybrid query %r -> %d fused candidates", query, len(chunks))

        return RetrievalResult(
            query=query,
            chunks=chunks,
            fused_candidates=len(chunks),
        )
