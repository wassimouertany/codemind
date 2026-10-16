"""Typed application settings. The only place that reads the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    # app
    codemind_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "codemind_chunks"
    qdrant_quantization: bool = True
    qdrant_on_disk: bool = True
    qdrant_timeout: float = Field(60.0, gt=0)
    qdrant_upsert_batch: int = Field(128, ge=1, le=2048)
    """Points per upsert call. 128 keeps a request well under the 32 MB body cap
    while still amortising round-trips over a ~50k-chunk ingest."""

    # storage
    index_tests: bool = False
    """Whether test sources are indexed. Off by default.

    Measured on petclinic: test files took 29-46% of every top-10, and the share
    tracked recall monotonically (29% -> 0.839, 46% -> 0.628). No gold answer
    lives in `src/test/`, yet tests share production vocabulary almost exactly,
    so they can only ever displace a correct chunk. Turn this on to index a
    repository *about* its tests, not to answer questions about its behaviour."""
    repos_dir: Path = Path("./.repos")
    """Where repositories are cloned. Keep it on the Linux filesystem: ingestion
    from /mnt/c crawls under WSL2."""
    graph_dir: Path = Path("./data/graphs")
    """Symbol graphs live beside the collection they describe, one JSON per
    collection, so a rebuilt collection never loads a stale graph."""
    database_url: str = "sqlite+aiosqlite:///./data/codemind.db"
    checkpoint_db: Path = Path("./data/checkpoints.db")

    # models
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    embedding_dim: int = 768
    embedding_device: Literal["cpu", "cuda"] = "cpu"
    embedding_batch_size: int = Field(32, ge=1, le=512)
    """Small on purpose. The GPU holds the LLM during serving, and on CPU a
    larger batch buys little while raising peak RSS against a 16 GB budget."""
    sparse_model: str = "Qdrant/bm25"
    rerank_enabled: bool = False
    """Whether the cross-encoder reranks fused candidates before synthesis.

    Off by default: at Gate 1 reranking matches DBSF on Recall@10, loses 0.086
    MRR, and costs 14s/query on CPU. The stage stays wired and tested; the
    ablation harness selects configurations explicitly and ignores this flag,
    so `hybrid_rerank` is still measured on every run.
    """
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: Literal["cpu", "cuda"] = "cpu"
    reranker_batch_size: int = Field(16, ge=1, le=256)
    """Cross-encoder pairs per forward pass. Smaller than the embedder's batch:
    pairs are query+chunk, so each row is roughly twice as long."""
    reranker_max_length: int = Field(512, ge=128, le=2048)
    """Tokens per pair. bge-reranker-base is trained at 512; longer inputs cost
    quadratic attention for text the model never learned to use."""

    # llm
    llm_provider: Literal["ollama", "vllm"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5-coder:3b-instruct-q4_K_M"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192

    # retrieval
    prefetch_dense: int = Field(20, ge=1, le=200)
    prefetch_sparse: int = Field(20, ge=1, le=200)
    fusion_method: Literal["rrf", "dbsf"] = "rrf"
    """How the dense and sparse prefetches are combined server-side.

    `rrf` fuses ranks only; `dbsf` normalises each branch's scores (mean +/- 3
    sigma) and sums them, so a branch's confidence margin survives fusion. On
    the fixture, "find by customer id" ranks findByCustomerId 2nd under RRF
    (sparse is 8.8 vs 3.4 sure, but rank discards that) and 1st under DBSF."""
    expansion_depth: int = Field(0, ge=0, le=3)
    """Call-graph hops pulled in around each reranked hit. 0 disables expansion.

    Off by default: at Gate 1 expansion costs 0.039 Recall@10 at fixed budget;
    re-evaluated at Gate 2. The stage stays wired and tested — a graph
    neighbour may well be worth more for answer completeness than for recall,
    and that is Gate 2's question, not this one. Past 2 the neighbourhood grows
    faster than the context window.
    """
    expansion_max_extra: int = Field(5, ge=0, le=100)
    """Hard cap on appended neighbours, so one hub symbol cannot flood context.

    Must stay below the final context budget: expansion walks *from* the
    reranked hits, so a budget that leaves no reranked seeds leaves nothing to
    expand. At a top-10 budget, 5 means "5 reranked hits plus up to 5 of their
    graph neighbours"."""
    rerank_top_k: int = Field(5, ge=1, le=50)

    # observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # evaluation only
    judge_model: str = ""
    judge_api_key: str = ""

    @property
    def prompts_dir(self) -> Path:
        return PROJECT_ROOT / "configs" / "prompts"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
