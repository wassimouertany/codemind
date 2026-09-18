"""Cross-encoder reranking of fused candidates.

Fusion ranks without ever reading a chunk against the question. Dense scores
query and chunk independently and BM25 counts terms, so both can only compare
two summaries. A cross-encoder puts the query and the chunk in *one* forward
pass with full attention between them, which is why it can fix mistakes neither
branch can see.

The fixture has a clean example. For "find by customer id", sparse is 2.6x more
confident in `findByCustomerId` (8.815 vs 3.434) but dense ranks it 4th, and RRF
fuses *ranks*, so `find_by_id` wins on position alone. Reading the question
against the chunk body is what recovers it.

Cost is the trade: this is the slowest stage in retrieval, and it grows linearly
with candidates. Three things keep it bounded:

* candidates are capped at `PREFETCH_DENSE + PREFETCH_SPARSE` — the most fusion
  can return, so the cap binds only if a caller passes a longer list;
* pairs are scored in batches of `RERANKER_BATCH_SIZE`, truncated to
  `RERANKER_MAX_LENGTH` tokens;
* the model is loaded once and stays warm. Loading it per request would cost
  more than the query.

It runs on **CPU** (`RERANKER_DEVICE`): during serving the 4 GB card is fully
committed to the LLM.

The fusion score is never destroyed — `with_rerank_score` copies it into
`rrf_score` while `score` becomes the cross-encoder's. Day 7's ablation compares
hybrid against hybrid+rerank on the same retrieved set, which is impossible if
reranking overwrites what it replaced.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import RetrievalError
from codemind.retrieval.schemas import ScoredChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder over (query, chunk) pairs. Construct once, reuse.

    Construction downloads (first run) and loads ~278M parameters, so it belongs
    in the FastAPI lifespan or at the top of a script — never at import time and
    never per request.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        self._model_name = resolved.reranker_model
        self._device = resolved.reranker_device
        self._batch_size = resolved.reranker_batch_size
        self._max_length = resolved.reranker_max_length
        self._top_k = resolved.rerank_top_k
        self._max_candidates = resolved.prefetch_dense + resolved.prefetch_sparse

        # Imported here, not at module scope: `import sentence_transformers`
        # pulls in torch and costs seconds even when nothing reranks.
        from sentence_transformers import CrossEncoder

        logger.info("loading reranker %s on %s", self._model_name, self._device)
        self._model: Any = CrossEncoder(
            self._model_name,
            device=self._device,
            max_length=self._max_length,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_candidates(self) -> int:
        """Most pairs ever scored for one query — the latency ceiling."""
        return self._max_candidates

    @property
    def top_k(self) -> int:
        return self._top_k

    def rerank_sync(
        self,
        query: str,
        chunks: Sequence[ScoredChunk],
        *,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Score, sort and truncate. Blocking — use the async form on the loop.

        Returns new `ScoredChunk`s; the inputs are left untouched so a caller
        can still score the pre-rerank ordering.
        """
        if not chunks:
            return []
        limit = top_k if top_k is not None else self._top_k

        considered = list(chunks[: self._max_candidates])
        if len(chunks) > self._max_candidates:
            logger.warning(
                "reranking only %d of %d candidates (cap = prefetch_dense + prefetch_sparse)",
                self._max_candidates,
                len(chunks),
            )

        # embedding_text, not body: the context header carries the path, class
        # and signature, which is exactly what distinguishes two identical
        # one-line delegating methods from each other.
        pairs = [(query, scored.chunk.embedding_text) for scored in considered]
        try:
            raw = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except (RuntimeError, ValueError) as exc:  # torch / tokenizer failures
            raise RetrievalError(f"reranking failed for {len(pairs)} pairs") from exc

        if len(raw) != len(considered):
            raise RetrievalError(
                f"reranker returned {len(raw)} scores for {len(considered)} candidates"
            )

        ordered = sorted(
            zip(considered, (float(score) for score in raw), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [
            scored.with_rerank_score(score, rank)
            for rank, (scored, score) in enumerate(ordered[:limit], start=1)
        ]

    async def rerank(
        self,
        query: str,
        chunks: Sequence[ScoredChunk],
        *,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Async wrapper. Runs in a worker thread — inference must not block the loop."""
        return await asyncio.to_thread(self.rerank_sync, query, chunks, top_k=top_k)


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Process-wide reranker for scripts and tests.

    The API passes its lifespan-owned instance explicitly instead of calling
    this, so the model's lifetime stays tied to the app's.
    """
    return Reranker()
