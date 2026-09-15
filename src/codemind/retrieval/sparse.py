"""BM25 sparse vectors over code-aware tokens.

This is the lexical half of hybrid retrieval. Dense embeddings are good at
"what does this mean" and bad at "find the identifier spelled exactly this
way" — a question naming `OrderRepository` or `PaymentTimeoutException` should
land on that symbol at rank 1, and only an inverted index does that reliably.

Two things make this module load-bearing:

**Every string goes through `codemind.retrieval.tokenizer` first.** Handed raw
source, BM25 sees `getUserById` as one opaque token that no human query will
ever match. See `tokenizer` for why symmetry between the two call sites is the
whole ballgame.

**Documents and queries are encoded differently, on purpose.** BM25 is not a
symmetric similarity. Term weights (TF saturation times IDF) belong on the document
side; the query side contributes each term once, with weight 1.0. fastembed
splits this across `embed()` and `query_embed()`, so `encode_documents` and
`encode_query` are genuinely different operations rather than an alias pair:

    doc   "order service find by customer id find find"
          -> values [1.661, 1.661, 1.985, 1.661, 1.661]   # `find` weighted up
    query "order service find by customer id find find"
          -> values [1.0, 1.0, 1.0, 1.0, 1.0]

Using `embed()` for a query would double-count term frequency and quietly skew
ranking toward queries that repeat a word.

Stemming is *not* done here. `Qdrant/bm25` applies Snowball internally — probed
directly, `order` and `orders` produce identical index sets — so adding
`py-rust-stemmers` on top would stem twice and desynchronise this module from
the tokenizer's documented "splits but does not stem" contract.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client import models

from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import RetrievalError
from codemind.retrieval.tokenizer import tokenize_query, tokenized_text

logger = logging.getLogger(__name__)

EMPTY_SPARSE_VECTOR = models.SparseVector(indices=[], values=[])
"""What a string of pure stopwords tokenizes to. Legal, and matches nothing."""


class SparseEncoder:
    """BM25 encoder over tokenized code. Construct once, reuse for the process.

    Constructing this downloads (first run) and loads the model, so it belongs
    in the FastAPI lifespan or at the top of an ingest script — never at import
    time and never per request.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        self._model_name = resolved.sparse_model
        self._batch_size = resolved.embedding_batch_size
        logger.info("loading sparse model %s", self._model_name)
        # BM25 is a lexical index — no neural inference, so it has no business
        # touching the GPU. fastembed's `cuda` default is AUTO, which would grab
        # a provider if one is visible; the 4 GB budget is reserved for the LLM.
        self._model = SparseTextEmbedding(self._model_name, cuda=False)

    @property
    def model_name(self) -> str:
        return self._model_name

    # ------------------------------------------------------------------ #
    # documents
    # ------------------------------------------------------------------ #

    def encode_documents_sync(self, texts: Sequence[str]) -> list[models.SparseVector]:
        """Encode chunk texts. Blocking — call from a thread, or use the async form.

        Returned in input order, one vector per input, including empties. The
        caller zips these against chunks positionally, so dropping a result
        would silently misalign every vector after it.
        """
        if not texts:
            return []
        prepared = [tokenized_text(text) for text in texts]
        try:
            raw = list(self._model.embed(prepared, batch_size=self._batch_size))
        except (RuntimeError, ValueError) as exc:  # onnxruntime / tokenizer failures
            raise RetrievalError(f"sparse encoding failed for {len(texts)} documents") from exc

        if len(raw) != len(texts):
            raise RetrievalError(
                f"sparse encoder returned {len(raw)} vectors for {len(texts)} documents"
            )

        vectors = [_to_sparse_vector(item) for item in raw]
        empty = sum(1 for vector in vectors if not vector.indices)
        if empty:
            # Not fatal: a chunk can legitimately be all keywords and braces.
            # It stays dense-retrievable. A *large* count means the tokenizer
            # stopword list is eating real vocabulary.
            logger.warning("%d/%d chunks produced an empty sparse vector", empty, len(texts))
        return vectors

    async def encode_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        """Async wrapper. Runs in a worker thread — this is CPU-bound work."""
        return await asyncio.to_thread(self.encode_documents_sync, texts)

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #

    def encode_query_sync(self, text: str) -> models.SparseVector:
        """Encode one query. Uses `query_embed`, not `embed` — see module docstring."""
        prepared = tokenize_query(text)
        if not prepared:
            logger.warning("query %r tokenized to nothing; sparse branch will not match", text)
            return EMPTY_SPARSE_VECTOR
        try:
            raw = next(iter(self._model.query_embed(prepared)), None)
        except (RuntimeError, ValueError) as exc:
            raise RetrievalError(f"sparse encoding failed for query {text!r}") from exc
        if raw is None:
            return EMPTY_SPARSE_VECTOR
        return _to_sparse_vector(raw)

    async def encode_query(self, text: str) -> models.SparseVector:
        """Async wrapper. Runs in a worker thread — this is CPU-bound work."""
        return await asyncio.to_thread(self.encode_query_sync, text)


def _to_sparse_vector(embedding: object) -> models.SparseVector:
    """Convert a fastembed `SparseEmbedding` to Qdrant's wire type.

    fastembed hands back numpy arrays; Qdrant's Pydantic models are declared
    `List[int]` / `List[float]` in strict mode and reject `np.int32`, so the
    conversion has to go through Python scalars rather than `.tolist()` on a
    view.
    """
    indices = getattr(embedding, "indices", None)
    values = getattr(embedding, "values", None)
    if indices is None or values is None:
        raise RetrievalError(f"unexpected sparse embedding type {type(embedding)!r}")
    return models.SparseVector(
        indices=[int(index) for index in indices],
        values=[float(value) for value in values],
    )


@lru_cache(maxsize=1)
def get_sparse_encoder() -> SparseEncoder:
    """Process-wide encoder for scripts and tests.

    The API passes its lifespan-owned instance explicitly instead of calling
    this, so the model's lifetime stays tied to the app's.
    """
    return SparseEncoder()
