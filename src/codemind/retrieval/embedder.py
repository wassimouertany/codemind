"""Dense embeddings for code chunks.

The dense half of hybrid retrieval. Where BM25 matches spellings, this matches
meaning — "why does checkout fail" finding `PaymentTimeoutException` without
sharing a single token with it.

Two invariants this module exists to enforce:

**Never embed `chunk.body` alone.** A chunk whose entire body is
`return this.repo.findById(id);` is semantically empty; the same three lines
appear in every repository class ever written. `CodeChunk.embedding_text`
prepends the context header (`path | module | class | signature`) so the vector
encodes *which* lookup this is. `embed_chunks` is the supported entry point
precisely so no call site has to remember this.

**The device comes from `Settings`, never from a hardcoded `.cuda()`.** During
serving the 4 GB card is fully committed to the LLM and this runs on CPU.
During `make ingest` the LLM is not loaded, so `EMBEDDING_DEVICE=cuda` makes a
full repository pass minutes rather than tens of minutes. fastembed's own
default is `Device.AUTO`, which would silently take the GPU whenever one is
visible — exactly the failure the env var exists to prevent — so the flag is
always passed explicitly.

Output dimension is checked against `settings.embedding_dim` at load time.
Qdrant rejects a mismatched vector at upsert with an error that names neither
the model nor the setting, which is a miserable thing to debug an hour into an
ingest; failing at construction costs one forward pass.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray

from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import RetrievalError
from codemind.core.types import CodeChunk

logger = logging.getLogger(__name__)

DenseVector = list[float]


class Embedder:
    """Dense encoder over code. Construct once, reuse for the process.

    Construction loads the model (~640 MB on disk for the jina code model), so
    it belongs in the FastAPI lifespan or at the top of an ingest script.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        self._model_name = resolved.embedding_model
        self._expected_dim = resolved.embedding_dim
        self._batch_size = resolved.embedding_batch_size
        self._device = resolved.embedding_device

        logger.info(
            "loading embedding model %s on %s (batch=%d)",
            self._model_name,
            self._device,
            self._batch_size,
        )
        self._model = TextEmbedding(self._model_name, cuda=self._device == "cuda")
        self._verify_dimension()

    def _verify_dimension(self) -> None:
        """One forward pass to prove the model matches the configured dimension.

        Cheap insurance: swapping `EMBEDDING_MODEL` without updating
        `EMBEDDING_DIM` otherwise surfaces as a Qdrant upsert rejection much
        later, or — worse, if the collection is recreated — as a silently
        useless index.
        """
        probe = next(iter(self._model.embed(["def probe(): pass"])), None)
        if probe is None:
            raise RetrievalError(f"embedding model {self._model_name} returned no vector")
        actual = int(probe.shape[-1])
        if actual != self._expected_dim:
            raise RetrievalError(
                f"embedding model {self._model_name} produces {actual}-d vectors but "
                f"EMBEDDING_DIM is {self._expected_dim}; fix the setting or the model, "
                "and recreate the Qdrant collection"
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._expected_dim

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------ #
    # text
    # ------------------------------------------------------------------ #

    def embed_texts_sync(self, texts: Sequence[str]) -> list[DenseVector]:
        """Embed raw strings. Blocking — call from a thread, or use the async form.

        Returned in input order, one vector per input. Prefer `embed_chunks`
        for chunks: it is the only path that guarantees the context header is
        included.
        """
        if not texts:
            return []
        try:
            raw = list(self._model.embed(texts, batch_size=self._batch_size))
        except (RuntimeError, ValueError) as exc:  # onnxruntime / tokenizer failures
            raise RetrievalError(f"dense encoding failed for {len(texts)} texts") from exc

        if len(raw) != len(texts):
            raise RetrievalError(f"embedder returned {len(raw)} vectors for {len(texts)} texts")
        return [_to_dense_vector(vector, self._expected_dim) for vector in raw]

    async def embed_texts(self, texts: Sequence[str]) -> list[DenseVector]:
        """Async wrapper. Runs in a worker thread — ONNX inference is CPU-bound."""
        return await asyncio.to_thread(self.embed_texts_sync, texts)

    # ------------------------------------------------------------------ #
    # chunks
    # ------------------------------------------------------------------ #

    def embed_chunks_sync(self, chunks: Sequence[CodeChunk]) -> list[DenseVector]:
        """Embed chunks via `embedding_text`, so the context header is always included."""
        return self.embed_texts_sync([chunk.embedding_text for chunk in chunks])

    async def embed_chunks(self, chunks: Sequence[CodeChunk]) -> list[DenseVector]:
        """Async wrapper. Runs in a worker thread — ONNX inference is CPU-bound."""
        return await asyncio.to_thread(self.embed_chunks_sync, chunks)

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #

    def embed_query_sync(self, text: str) -> DenseVector:
        """Embed one query.

        No tokenizer here, and no instruction prefix. The jina v2 code model is
        trained symmetrically on natural-language/code pairs, so a question is
        encoded exactly as a chunk is. Splitting identifiers would only strip
        signal the dense model can use.
        """
        vectors = self.embed_texts_sync([text])
        if not vectors:
            raise RetrievalError(f"dense encoding returned nothing for query {text!r}")
        return vectors[0]

    async def embed_query(self, text: str) -> DenseVector:
        """Async wrapper. Runs in a worker thread — ONNX inference is CPU-bound."""
        return await asyncio.to_thread(self.embed_query_sync, text)


def _to_dense_vector(vector: NDArray[np.number[Any]], expected_dim: int) -> DenseVector:
    """Convert one numpy row to the plain floats Qdrant's strict models require.

    Typed over `np.number` rather than `np.floating` because fastembed declares
    int8/int32 return types for its quantized models. Those are not in use here
    — the dimension check below is what actually constrains the output — but
    narrowing the annotation would reject the declared union at the call site.
    """
    if vector.shape[-1] != expected_dim:
        raise RetrievalError(
            f"embedder produced a {vector.shape[-1]}-d vector, expected {expected_dim}"
        )
    return [float(value) for value in vector]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide embedder for scripts and tests.

    The API passes its lifespan-owned instance explicitly instead of calling
    this, so the model's lifetime stays tied to the app's.
    """
    return Embedder()
