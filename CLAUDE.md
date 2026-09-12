# CLAUDE.md — CodeMind AI

## What this is

CodeMind ingests a real GitHub repository, indexes it with AST-aware chunking,
retrieves with hybrid search (dense + BM25 + cross-encoder rerank), and answers
technical questions through a LangGraph multi-agent workflow backed by a QLoRA
fine-tuned Qwen2.5-Coder-3B. Every answer cites `file:line` evidence.

Reference question the system must answer well:
> "Why does `POST /orders` sometimes return 500?"
→ trace Controller → Service → Repository → exception, cite files, propose a fix.

## Hardware reality — read this before suggesting anything

Dev machine: **RTX with 4 GB VRAM, 16 GB system RAM, Windows 11 + WSL2.**

This is the binding constraint on every model decision:

- The LLM is **3B, Q4_K_M, served by Ollama**. Not 7B. Not vLLM locally.
- The embedder and reranker run on **CPU** during serving. The GPU holds the LLM
  and nothing else.
- During `make ingest` the LLM is not loaded, so the embedder may use CUDA.
  This is why `EMBEDDING_DEVICE` is an env var and never a hardcoded `.cuda()`.
- **All QLoRA training happens on Kaggle** (30h/week free, 2× T4 16 GB).
  Local training is only ever a smoke test with Qwen2.5-Coder-0.5B, seq 512,
  50 samples, to prove the script runs before burning Kaggle hours.
- Never propose Postgres, ClickHouse, Elasticsearch, or self-hosted Langfuse.
  RAM budget does not allow it.

If a suggestion would need more than 4 GB VRAM or 2 GB of additional RAM,
say so instead of writing the code.

## Stack (do not substitute without asking)

- Python 3.12, `uv` only — never `pip install` directly
- FastAPI + Pydantic v2 + pydantic-settings
- Qdrant, one collection, named vectors `dense` (768d cosine) + `sparse` (BM25),
  int8 scalar quantization, vectors on disk
- tree-sitter via `tree-sitter-language-pack`
- Dense: `jinaai/jina-embeddings-v2-base-code` (161M)
- Sparse: `fastembed` `SparseTextEmbedding("Qdrant/bm25")`
- Rerank: `BAAI/bge-reranker-base` (278M). `bge-reranker-v2-m3` only for
  evaluation runs on rented GPU.
- LangGraph — hand-built `StateGraph`, **not** `create_supervisor` (named nodes
  must appear in the Langfuse trace)
- SQLite via `langgraph-checkpoint-sqlite` + async SQLAlchemy
- Langfuse Cloud
- Training: Unsloth + TRL, `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`

## Non-negotiable invariants

1. **Every chunk carries a context header.** Before embedding, prepend
   `<file_path> | <package/module> | <enclosing class> | <signature>` to the
   chunk body. Retrieval quality collapses without this.
2. **Code-aware sparse tokenization.** `retrieval/tokenizer.py` splits camelCase,
   PascalCase, snake_case and dotted paths into separate terms *before* BM25.
   `getUserById` → `get user by id`. Applied identically at index time and query
   time — any divergence silently kills sparse recall.
3. **Grounding gate is mandatory.** `agents/nodes/grounding.py` verifies every
   factual claim maps to a retrieved chunk and that cited paths actually exist.
   Unsupported → one retry with expanded context → then degrade the answer.
   Never fabricate a citation.
4. **No LLM calls outside `src/codemind/llm/`.** Nodes depend on the `LLMClient`
   Protocol.
5. **`llm/judge.py` is evaluation-only.** Enforced by an import-linter contract
   in `pyproject.toml`. Do not weaken the contract to make a test pass.
6. **Everything is traced.** Retrieval, reranking, each node and each LLM call
   are separate Langfuse spans in one trace. README latency numbers come from
   Langfuse, not `time.time()` sprinkled in the code.
7. **Async all the way down.** `AsyncQdrantClient`, async SQLAlchemy, async
   FastAPI. CPU-bound work (tree-sitter parsing, cross-encoder inference) goes
   to a thread pool — it must not block the event loop.
8. **Heavy models load once, in the FastAPI lifespan.** Never at import time,
   never per request.

## Conventions

- `src/` layout, absolute imports (`from codemind.retrieval.hybrid import ...`)
- `mypy --strict` on `src/codemind/`; `ruff` lint + format, line length 100
- Config only via `codemind.core.config.get_settings()` — no bare `os.getenv`
  in business logic
- Prompts in `configs/prompts/*.jinja2`, never inline f-strings
- Custom exceptions from `core/exceptions.py`; no bare `except Exception`
- Retrieval and ingestion changes require a test using
  `tests/fixtures/sample_repo/`

## Retrieval contract

```
query
  → tokenizer (code-aware) ──→ sparse vector ─┐
  → embedder ─────────────────→ dense vector ─┤
                                              ├─→ Qdrant Query API
                                              │   prefetch: dense top-20
                                              │   prefetch: sparse top-20
                                              │   fusion:   RRF → top-40
                                              ↓
                                      bge-reranker-base → top-5
                                              ↓
                                 expansion (callers/callees) → context
```

Tunables live in `.env` / `configs/retrieval.yaml`. Never hardcode `top_k`.

## Build order — each gate must pass before the next phase

1. **Ingestion** — repo_loader → language_registry → ast_chunker → symbol_graph
   *Gate:* chunk boundaries align with function/class boundaries on
   `tests/fixtures/sample_repo/`, verified by test.
2. **Retrieval** — tokenizer → embedder/sparse → qdrant_store → hybrid → reranker
   *Gate:* a Recall@10 number exists, measured on
   `evaluation/datasets/retrieval_goldset.jsonl` (30 hand-labelled questions).
   **No agent code is written before this number exists.**
3. **Agents** — state → graph → router → retrieve → specialists → synthesize →
   grounding
   *Gate:* the reference question produces a correct call-path answer with
   citations that resolve to real lines.
4. **API + tracing** — FastAPI, SSE, Langfuse spans
5. **Fine-tuning** — dataset → QLoRA on Kaggle → GGUF export → Ollama
   *Gate:* QLoRA beats base on a 100-example held-out set, identical contexts.
6. **Evaluation + packaging** — RAGAS, ablation table, Docker, README

## Known failure modes

- **Sparse recall is zero** → tokenizer applied at query time but not index time,
  or vice versa. Check both call sites.
- **Reranker is the latency bottleneck** → batch it, cap candidates at 40, run in
  a thread pool, keep it warm in the lifespan.
- **Router misclassifies** → structured output with an enum, not free-text parsing.
- **Graph loops forever** → `recursion_limit=25` on `invoke()`. A hit almost
  always means a routing-prompt bug, not a limit that is too low.
- **Training loss below 0.2** → overfitting. Fewer epochs or lower LoRA rank.
- **Answers cite files that don't exist** → the grounding gate isn't validating
  paths against retrieved chunk payloads.
- **Ingestion crawls** → repo is on `/mnt/c/`. Move it to the Linux filesystem.

## Out of scope

Frontend, auth, multi-tenancy, write access to repos, automatic PR creation.
If a task drifts here, stop and ask.

## Installed versions — verify APIs against these, not from memory

These are MAJOR versions ahead of most online tutorials. Before using any
API from langgraph, langfuse or ragas, check the installed source or docs.
Do not trust remembered API shapes.

- langgraph == 1.2.11
- langchain-core == 1.6.2
- langfuse == 4.15.2
- qdrant-client == 1.19.0
- fastembed == 0.8.0
- sentence-transformers == 6.0.1
- transformers == 5.17.0
- torch == 2.14.0
- ragas == 0.3.1
- pydantic == 2.13.5
- fastapi == 0.141.1
