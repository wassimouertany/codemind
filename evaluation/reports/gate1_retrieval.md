# Gate 1 — retrieval baseline

Gold set: **30 questions** hand-labelled against `codemind_chunks`.

Every configuration draws from the same pool of 40 candidates (`PREFETCH_DENSE=20` + `PREFETCH_SPARSE=20`) and returns 10 chunks, so the rows differ by technique and not by depth.

## Reading these numbers

Corpus: **25 files / 99 chunks** after excluding tests. A top-10 is **~10% of the whole index**, so absolute Recall@10 is inflated relative to a production-scale corpus and does not transfer.

- Compare configurations by **delta**, not by absolute value.
- Prefer **Recall@5** and **MRR**: both stay discriminative at this size, while Recall@10 saturates.
- `Misses` (questions retrieving no gold file at all) is the figure that would survive a larger index.

Reranked rows were scored on **cpu** (`RERANKER_DEVICE`), batch 16, truncated to 512 tokens. Device and batch size change throughput, not cross-encoder scores — per-query latency is hardware-bound and is deliberately not reported here.

## Ablation

| Configuration | Recall@5 | Recall@10 | MRR | nDCG@10 | Precision@10 | Misses |
|---|---|---|---|---|---|---|
| dense_only | 0.767 | 0.878 | 0.928 | 0.839 | 0.467 | 0/30 |
| sparse_only | 0.717 | 0.817 | 0.782 | 0.724 | 0.383 | 1/30 |
| hybrid_rrf | 0.806 | 0.883 | 0.928 | 0.852 | 0.470 | 0/30 |
| hybrid_dbsf | 0.822 | 0.894 | 0.944 | 0.868 | 0.470 | 0/30 |
| hybrid_rerank | 0.722 | 0.894 | 0.858 | 0.818 | 0.460 | 0/30 |
| hybrid_rerank_expansion | 0.722 | 0.856 | 0.850 | 0.799 | 0.417 | 1/30 |

Misses = questions where no gold file was retrieved at all.

### Configurations

- `dense_only` — dense vector only, no fusion
- `sparse_only` — sparse vector only, no fusion
- `hybrid_rrf` — dense + sparse, RRF fusion
- `hybrid_dbsf` — dense + sparse, DBSF fusion
- `hybrid_rerank` — dense + sparse, RRF fusion, cross-encoder rerank
- `hybrid_rerank_expansion` — dense + sparse, RRF fusion, cross-encoder rerank, call-graph expansion (answers: is a graph neighbour worth more than the 6th-10th reranked chunk?)

## Significance

Paired bootstrap, 10,000 resamples of the 30 questions. Paired because every configuration answers the same questions. A CI spanning zero means the two are **not distinguishable** on this gold set — at n=30 that is the common case, and saying so is more useful than ranking noise.

### Recall@5

| Comparison | Delta and 95% CI |
|---|---|
| dense_only → sparse_only | -0.050 [95% CI -0.161, +0.061] — not distinguishable |
| sparse_only → hybrid_rrf | +0.089 [95% CI +0.022, +0.178] — improvement |
| hybrid_rrf → hybrid_dbsf | +0.017 [95% CI -0.033, +0.067] — not distinguishable |
| hybrid_dbsf → hybrid_rerank | -0.100 [95% CI -0.172, -0.039] — regression |
| hybrid_rerank → hybrid_rerank_expansion | +0.000 [95% CI +0.000, +0.000] — not distinguishable |
| dense_only → hybrid_rrf | +0.039 [95% CI -0.033, +0.111] — not distinguishable |
| dense_only → hybrid_dbsf | +0.056 [95% CI -0.017, +0.133] — not distinguishable |
| dense_only → hybrid_rerank | -0.044 [95% CI -0.128, +0.044] — not distinguishable |
| dense_only → hybrid_rerank_expansion | -0.044 [95% CI -0.128, +0.044] — not distinguishable |

### Recall@10

| Comparison | Delta and 95% CI |
|---|---|
| dense_only → sparse_only | -0.061 [95% CI -0.144, +0.017] — not distinguishable |
| sparse_only → hybrid_rrf | +0.067 [95% CI +0.017, +0.128] — improvement |
| hybrid_rrf → hybrid_dbsf | +0.011 [95% CI +0.000, +0.033] — not distinguishable |
| hybrid_dbsf → hybrid_rerank | -0.000 [95% CI -0.072, +0.072] — not distinguishable |
| hybrid_rerank → hybrid_rerank_expansion | -0.039 [95% CI -0.100, +0.011] — not distinguishable |
| dense_only → hybrid_rrf | +0.006 [95% CI -0.044, +0.061] — not distinguishable |
| dense_only → hybrid_dbsf | +0.017 [95% CI -0.022, +0.067] — not distinguishable |
| dense_only → hybrid_rerank | +0.017 [95% CI -0.056, +0.089] — not distinguishable |
| dense_only → hybrid_rerank_expansion | -0.022 [95% CI -0.111, +0.067] — not distinguishable |

## Recall@10 by category

| Configuration | architecture (n=7) | bug (n=7) | dependency (n=8) | locator (n=8) |
|---|---|---|---|---|
| dense_only | 1.000 | 0.667 | 0.833 | 1.000 |
| sparse_only | 1.000 | 0.643 | 0.625 | 1.000 |
| hybrid_rrf | 1.000 | 0.786 | 0.750 | 1.000 |
| hybrid_dbsf | 1.000 | 0.786 | 0.792 | 1.000 |
| hybrid_rerank | 1.000 | 0.786 | 0.792 | 1.000 |
| hybrid_rerank_expansion | 0.929 | 0.643 | 0.833 | 1.000 |

## Per-query regressions

**dense_only → sparse_only** (mean Recall@10 -0.061)

| Query | Before | After |
|---|---|---|
| dep-05 | 1.000 | 0.333 |
| bug-02 | 1.000 | 0.500 |
| bug-06 | 0.500 | 0.000 |
| dep-06 | 0.667 | 0.333 |
| dep-04 | 1.000 | 0.667 |
| dep-08 | 1.000 | 0.667 |

**sparse_only → hybrid_rrf** (mean Recall@10 +0.067)

No query regressed.

**hybrid_rrf → hybrid_dbsf** (mean Recall@10 +0.011)

No query regressed.

**hybrid_dbsf → hybrid_rerank** (mean Recall@10 +0.000)

| Query | Before | After |
|---|---|---|
| bug-02 | 1.000 | 0.500 |
| bug-03 | 1.000 | 0.500 |
| dep-04 | 1.000 | 0.667 |

**hybrid_rerank → hybrid_rerank_expansion** (mean Recall@10 -0.039)

| Query | Before | After |
|---|---|---|
| bug-04 | 1.000 | 0.500 |
| bug-06 | 0.500 | 0.000 |
| arch-06 | 1.000 | 0.500 |

---

Per-query detail: `gate1_per_query.json`.
