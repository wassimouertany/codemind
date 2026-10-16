# Gate 1 — retrieval baseline (BEFORE corpus and label fixes)

> Kept for comparison. This run is **invalid as a measurement** and is retained
> only because the before/after delta is the interesting result.
>
> Three faults, all fixed afterwards:
>
> 1. **Tests were indexed** — 45 files / 204 chunks, 20 of them test files.
>    Test chunks took 29-46% of every top-10 and the share tracked recall
>    monotonically (29% -> 0.839, 46% -> 0.628). No gold answer lives in
>    `src/test/`, so a test chunk can only displace a correct one.
> 2. **Unequal depth** — `hybrid_rerank_expansion` returned 5 chunks on
>    arch-01, arch-04 and loc-06 where every other configuration returned 10.
> 3. **Wrong gold labels** — four rows cited `OwnerRepository.java` for pet-type
>    lookups; `findPetTypes()` is declared in `PetTypeRepository.java:37`.
>
> Compare against `gate1_retrieval.md`.


Gold set: **30 questions** hand-labelled against `codemind_chunks`.

Every configuration draws from the same pool of 40 candidates (`PREFETCH_DENSE=20` + `PREFETCH_SPARSE=20`) and returns 10 chunks, so the rows differ by technique and not by depth.

## Ablation

| Configuration | Recall@5 | Recall@10 | MRR | nDCG@10 | Precision@10 | Misses |
|---|---|---|---|---|---|---|
| dense_only | 0.733 | 0.839 | 0.856 | 0.767 | 0.417 | 0/30 |
| sparse_only | 0.500 | 0.628 | 0.564 | 0.490 | 0.267 | 3/30 |
| hybrid_rrf | 0.706 | 0.806 | 0.832 | 0.730 | 0.387 | 0/30 |
| hybrid_dbsf | 0.700 | 0.811 | 0.823 | 0.731 | 0.383 | 0/30 |
| hybrid_rerank | 0.611 | 0.744 | 0.711 | 0.639 | 0.313 | 2/30 |
| hybrid_rerank_expansion | 0.611 | 0.672 | 0.694 | 0.599 | 0.326 | 3/30 |

Misses = questions where no gold file was retrieved at all.

### Configurations

- `dense_only` — dense vector only, no fusion
- `sparse_only` — sparse vector only, no fusion
- `hybrid_rrf` — dense + sparse, RRF fusion
- `hybrid_dbsf` — dense + sparse, DBSF fusion
- `hybrid_rerank` — dense + sparse, RRF fusion, cross-encoder rerank
- `hybrid_rerank_expansion` — dense + sparse, RRF fusion, cross-encoder rerank, call-graph expansion

## Recall@10 by category

| Configuration | architecture (n=7) | bug (n=7) | dependency (n=8) | locator (n=8) |
|--- |
|--- |
|--- |
|--- |
|---|
| dense_only | 0.929 | 0.667 | 0.750 | 1.000 |
| sparse_only | 0.714 | 0.548 | 0.500 | 0.750 |
| hybrid_rrf | 0.857 | 0.738 | 0.625 | 1.000 |
| hybrid_dbsf | 0.857 | 0.667 | 0.708 | 1.000 |
| hybrid_rerank | 1.000 | 0.524 | 0.584 | 0.875 |
| hybrid_rerank_expansion | 0.786 | 0.429 | 0.583 | 0.875 |

## Per-query regressions

**dense_only → sparse_only** (mean Recall@10 -0.211)

| Query | Before | After |
|---|---|---|
| loc-05 | 1.000 | 0.000 |
| loc-07 | 1.000 | 0.000 |
| dep-04 | 1.000 | 0.333 |
| dep-05 | 1.000 | 0.333 |
| bug-02 | 1.000 | 0.500 |
| bug-06 | 0.500 | 0.000 |
| arch-02 | 1.000 | 0.500 |
| arch-04 | 1.000 | 0.500 |
| arch-06 | 1.000 | 0.500 |
| bug-07 | 0.667 | 0.333 |
| dep-06 | 0.667 | 0.333 |
| dep-02 | 1.000 | 0.667 |

**sparse_only → hybrid_rrf** (mean Recall@10 +0.178)

No query regressed.

**hybrid_rrf → hybrid_dbsf** (mean Recall@10 +0.006)

| Query | Before | After |
|---|---|---|
| bug-03 | 1.000 | 0.500 |

**hybrid_dbsf → hybrid_rerank** (mean Recall@10 -0.067)

| Query | Before | After |
|---|---|---|
| loc-03 | 1.000 | 0.000 |
| bug-02 | 1.000 | 0.500 |
| bug-06 | 0.500 | 0.000 |
| dep-02 | 1.000 | 0.667 |
| dep-04 | 1.000 | 0.667 |
| dep-05 | 1.000 | 0.667 |
| dep-08 | 1.000 | 0.667 |

**hybrid_rerank → hybrid_rerank_expansion** (mean Recall@10 -0.072)

| Query | Before | After |
|---|---|---|
| bug-07 | 0.667 | 0.000 |
| arch-01 | 1.000 | 0.500 |
| arch-04 | 1.000 | 0.500 |
| arch-06 | 1.000 | 0.500 |
| dep-05 | 0.667 | 0.333 |
| dep-07 | 0.667 | 0.333 |

---

Per-query detail: `gate1_per_query.json`.
