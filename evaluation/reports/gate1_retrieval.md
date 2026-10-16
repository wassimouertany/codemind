# Gate 1 — retrieval baseline

Gold set: **3 questions** hand-labelled against `codemind_chunks`.

Every configuration draws from the same pool of 40 candidates (`PREFETCH_DENSE=20` + `PREFETCH_SPARSE=20`) and returns 10 chunks, so the rows differ by technique and not by depth.

## Reading these numbers

Corpus: **25 files / 99 chunks** after excluding tests. A top-10 is **~10% of the whole index**, so absolute Recall@10 is inflated relative to a production-scale corpus and does not transfer.

- Compare configurations by **delta**, not by absolute value.
- Prefer **Recall@5** and **MRR**: both stay discriminative at this size, while Recall@10 saturates.
- `Misses` (questions retrieving no gold file at all) is the figure that would survive a larger index.

## Ablation

| Configuration | Recall@5 | Recall@10 | MRR | nDCG@10 | Precision@10 | Misses |
|---|---|---|---|---|---|---|
| dense_only | 0.667 | 0.667 | 1.000 | 0.742 | 0.533 | 0/3 |

Misses = questions where no gold file was retrieved at all.

### Configurations

- `dense_only` — dense vector only, no fusion

## Significance

_needs at least two configurations._

## Recall@10 by category

| Configuration | bug (n=3) |
|---|---|
| dense_only | 0.667 |

## Per-query regressions
_needs at least two configurations._

---

Per-query detail: `gate1_per_query.json`.
