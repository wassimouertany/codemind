# CodeMind AI

**An autonomous software-engineering agent that answers questions about a real codebase — with citations.**

> *"Why does `POST /orders` sometimes return 500?"*
>
> CodeMind traces the call path from controller to service to repository, finds
> the unhandled exception, and answers with `file:line` evidence for every claim.

![tests](https://img.shields.io/badge/tests-280%20passing-brightgreen)
![mypy](https://img.shields.io/badge/mypy-strict-blue)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

> **Status: week 1 of 4 complete.** Ingestion, hybrid retrieval and the Gate 1
> evaluation harness are built and measured. The agent layer, fine-tuning and
> RAGAS evaluation are in progress — see [Roadmap](#roadmap).

---

## The problem

A developer joining a SaaS codebase spends days working out which files own a
behaviour, what a change will break, and why an endpoint intermittently fails.
A general-purpose LLM can discuss the concepts but has never seen *this*
repository.

CodeMind turns a repository into a knowledge base that specialised agents can
query — and grounds every statement in code that actually exists.

## Architecture
                developer question
                        │
                   FastAPI
                        │
                LangGraph router
                        │
    ┌───────────────────┼───────────────────┐

Code Analyst Dependency Agent Test Agent
└───────────────────┼───────────────────┘
│
Hybrid retrieval
┌────────┴────────┐
BM25 sparse dense vectors
└────────┬────────┘
RRF / DBSF fusion
│
cross-encoder reranking (off by default)
│
call-graph expansion (off by default)
│
Qwen2.5-Coder-3B (QLoRA fine-tuned)
│
grounding verification
│
answer + file:line evidence


---

## Results

Measured on [Spring PetClinic](https://github.com/spring-projects/spring-petclinic)
against **30 hand-labelled questions** across four categories
(bug / architecture / dependency / locator).

### Gate 1 — retrieval ablation

| Configuration | Recall@5 | Recall@10 | MRR | Latency |
|---|---|---|---|---|
| dense only | 0.767 | 0.878 | 0.928 | ~1 s |
| BM25 only | 0.717 | 0.817 | 0.782 | ~0.2 s |
| hybrid (RRF) | 0.806 | 0.883 | 0.928 | ~0.9 s |
| **hybrid (DBSF)** | **0.822** | **0.894** | **0.944** | ~1 s |
| hybrid + rerank | 0.722 | 0.894 | 0.858 | ~14 s |
| hybrid + rerank + expansion | 0.722 | 0.856 | 0.850 | ~14 s |

All deltas carry a paired bootstrap 95% CI (10k resamples, same 30 questions).

### Three conclusions, two of them negative

**Fusion beats its own sparse branch, significantly.** `sparse_only → hybrid_rrf`
is +0.089 Recall@5 `[+0.022, +0.178]` and +0.067 Recall@10 `[+0.017, +0.128]`.

**Hybrid vs dense-only is *not* distinguishable at n=30.** DBSF leads on every
metric, but `dense_only → hybrid_dbsf` is +0.056 Recall@5 `[-0.017, +0.133]` —
the interval spans zero. DBSF ships because it leads on all five measures and
costs nothing extra, not because the gap is proven.

**The cross-encoder does not ship — and the regression is significant.**
`hybrid_dbsf → hybrid_rerank` is **−0.100 Recall@5, 95% CI [-0.172, -0.039]**:
the interval excludes zero. It matches DBSF on Recall@10, loses 0.086 MRR, and
costs ~14 s per query on CPU against ~1 s. It reorders without improving *which*
gold files are retrieved, and actively pushes them down. Retained behind
`RERANK_ENABLED=false` and
re-evaluated at Gate 2, where ranking quality inside the context window may
matter more than it does for Recall@10.

**Expansion costs 0.038 Recall@10** at a fixed 10-chunk budget — a graph
neighbour is worth less than the 6th–10th reranked chunk here. But Recall@10
asks "did the gold file appear in the context", and expansion exists to supply
the *caller and callee* of a failing method. That is a Gate 2 question.
`EXPANSION_DEPTH=0` until it is answered.

### Other measured facts

| | |
|---|---|
| Symbol-graph call resolution | **90.7%** of resolvable call sites |
| Ingestion | 25 files → 99 chunks, zero parse errors |
| Chunk size | median 273 chars, p95 1338 |
| Local inference | 58.6 tok/s (Qwen2.5-Coder-3B Q4_K_M, RTX 3050) |

Both ablation runs are committed: `evaluation/reports/gate1_retrieval.md` and
`gate1_retrieval_before_fixes.md`. The second documents its own invalidity — see
below.

---

## Gate 1 failed on the first run

Dense-only scored **0.839** Recall@10. Hybrid + rerank scored **0.744**. The
configuration with more machinery was worse.

Per-query inspection showed 29–46% of retrieved chunks were **test files**. Every
gold label was in `src/main/`, so a test could never be a correct answer — but
test files share production vocabulary almost exactly, and test method names
(`shouldReturnBadRequestWhenLastNameIsEmpty`) read like natural-language
questions, which the cross-encoder preferred.

The correlation was monotonic:

| Configuration | % of top-10 slots on tests | Recall@10 |
|---|---|---|
| dense only | 29% | 0.839 |
| hybrid (RRF) | 38% | 0.806 |
| hybrid (DBSF) | 40% | 0.811 |
| hybrid + rerank | 45% | 0.744 |
| BM25 only | 46% | 0.628 |

Excluding tests from the index moved BM25-only by **+0.150** and dense-only by
**+0.017** — the ordering and magnitude the diagnosis predicted.

**Five gold labels were also wrong**, found by reading the source rather than by
guessing. `findPetTypes()` is declared in `PetTypeRepository.java:37`;
`OwnerRepository` does not have it. In one case the reranker had ranked
`PetTypeRepository` first and scored 0.000 because the label said otherwise —
the retriever was right and the benchmark was wrong.

Correcting a benchmark because the system disagrees with it is how a benchmark
gets tuned to flatter the system, so every correction traces to a line number in
the source.

---

## Design decisions

**AST-aware chunking, not sliding windows.** A 512-token window cuts through
method bodies, so a signature and its implementation land in different chunks.
CodeMind chunks on the tree-sitter syntax tree: every chunk is a complete
function, method or class skeleton with an exact line range — which is the only
reason `file:line` citations are possible at all.

**Every chunk carries a context header.** `return this.repo.findById(id);` has
almost no retrievable signal alone. Each chunk is prefixed with
`path | package | Class | signature | imports` before embedding.

**Code-aware BM25 tokenization, symmetric by assertion.** BM25 stems English
words; handed `getUserById` it produces one opaque token. CodeMind splits
identifiers — `getUserById` → `get user by id`, `HTTPResponseCode` →
`http response code` — with the *same* function at index time and query time.
Asymmetry there collapses sparse recall silently, so symmetry is asserted by
test rather than assumed.

The measured result that justifies hybrid search: for *"why does POST /orders
return 500"*, identifier splitting alone yields **0** matching terms against the
method that fails, splitting plus stemming yields **1**, and nothing lexical
connects "500" to `PaymentTimeoutException` — that link is purely semantic.

**Four-layer call resolution that refuses to guess.** The AST gives call *sites*,
not targets. Resolution tries receiver type (the only sound layer), then unique
simple name, then import scope — then gives up and counts the failure. A wrong
edge produces a confident, fabricated call path, which is worse than no path.

**Model sized to the deployment target.** The fine-tune is
Qwen2.5-Coder-**3B**, not 7B, because the target device has 4 GB of VRAM. A 7B
Q4 model spills into system RAM and serves at a few tokens per second. Training
runs on a free Kaggle T4; serving is llama.cpp via Ollama.

---

## Stack

| Layer | Choice |
|---|---|
| API | FastAPI, Pydantic v2, Python 3.12 (`uv`) |
| Vector store | Qdrant — named vectors (`dense` + `sparse`), int8 quantization, on-disk |
| Chunking | tree-sitter — Java, Python, TypeScript, JavaScript |
| Dense embeddings | `jinaai/jina-embeddings-v2-base-code` (768d, ONNX via FastEmbed) |
| Sparse | BM25 with IDF modifier, over code-aware tokenization |
| Reranking | `BAAI/bge-reranker-base` cross-encoder (off by default — see Results) |
| Call graph | tree-sitter + networkx |
| Orchestration | LangGraph — hand-built `StateGraph` |
| Inference | Qwen2.5-Coder-3B Q4_K_M via Ollama |
| Fine-tuning | Unsloth + TRL, QLoRA 4-bit |
| Observability | Langfuse |
| Evaluation | RAGAS + custom IR metrics |

FastEmbed serves the dense model rather than sentence-transformers: the Jina
code model's `trust_remote_code` file calls `find_pruneable_heads_and_indices`,
removed in transformers 5.x. ONNX removes transformers from the retrieval path
entirely and runs faster on CPU.

---

## Quickstart

```bash
git clone https://github.com/wassimouertany/codemind.git
cd codemind

make install          # uv sync, seeds .env
make up               # Qdrant via docker compose
make dev              # http://localhost:8000/health

make ingest REPO=https://github.com/spring-projects/spring-petclinic
```

Requires Python 3.12, Docker, and Ollama with
`qwen2.5-coder:3b-instruct-q4_K_M` pulled.

### Evaluation

```bash
make validate-goldset   # every gold path must exist in the index
make eval-retrieval     # full ablation -> evaluation/reports/
```

The validator runs before the ablation by design: a gold path that is not
indexed is a permanent zero no configuration can fix.

---

## Development

```bash
make test             # pytest
make lint             # ruff + mypy --strict + import-linter
make status           # progress and what's next
```

The architecture is enforced mechanically, not by convention. Two
[import-linter](https://import-linter.readthedocs.io) contracts in
`pyproject.toml` keep the layering intact and keep the evaluation judge model
out of production code paths — if `judge.py` leaked into a request path, the
system would be grading itself with the model it is measured against.

`mypy --strict` runs over the whole package. `OPEN_ISSUES.md` tracks deferred
work, and every entry must state what blocks it.

---

## Roadmap

- [x] Repository loading with generated / vendored / binary filtering
- [x] AST-aware chunking with context headers
- [x] Symbol graph with four-layer call resolution
- [x] Hybrid retrieval (BM25 + dense, RRF and DBSF fusion)
- [x] Cross-encoder reranking and call-graph expansion
- [x] Retrieval evaluation, ablation study, significance testing
- [ ] LangGraph multi-agent workflow
- [ ] Grounding verification
- [ ] Langfuse tracing
- [ ] QLoRA fine-tuning and base-vs-tuned benchmark
- [ ] RAGAS evaluation
- [ ] Containerization and CI

---

## Limitations

- Four languages. Adding one means adding a `LanguageSpec` — cheap, not free.
- A file that parses but contains no chunkable node yields zero chunks. Correct
  for Java `package-info.java`; **wrong for a Python module of top-level
  statements**. Tracked in `OPEN_ISSUES.md`; the fix is deferred because it
  changes chunking and would invalidate the corpus Gate 1 was measured on.
- The symbol graph resolves calls syntactically, without type resolution —
  dynamic dispatch, reflection and DI-by-name are invisible. A syntactic
  approximation is fine for retrieval expansion and would not be fine for, say,
  dead-code elimination.
- Layer inference (controller / service / repository) is a naming heuristic and
  will mislabel unconventional codebases.
- At 99 chunks, a top-10 covers ~10% of the index, so absolute Recall@10 is
  inflated relative to a production-scale corpus. Compare configurations by
  delta; Recall@5 and MRR stay discriminative where Recall@10 saturates.
- Single repository, single user. No auth, no multi-tenancy.
- Read-only: CodeMind diagnoses and proposes, it does not modify code.

---

## Author

**Wassim Ouertani** — software engineering student at ISAMM, Tunisia.
[LinkedIn](https://linkedin.com/in/wassim-ouertani) ·
[GitHub](https://github.com/wassimouertany)

## License

MIT
