"""Information-retrieval metrics for Gate 1.

Pure functions over ranked lists. No Qdrant, no models, no I/O — which is why
every one of them can be unit-tested against worked examples rather than
trusted.

## Grading at file level, not chunk level

Gold labels are file paths. Labelling "which chunk answers this" by hand is
slow, brittle, and changes whenever the chunker changes; labelling "which file
answers this" is fast and stable. A retrieved chunk counts as relevant when its
file is in the gold set.

The consequence to keep in mind: several chunks from one gold file collapse to
one hit. That is intended — retrieving five chunks of `OrderService.java` is
not five times better than retrieving one, and scoring it that way would reward
a retriever that returns near-duplicates.

## Why these four

* **Recall@k** — did the answer make it into the context window at all? If the
  gold file is absent at k, no amount of reranking or prompting can recover it.
  This is the headline number for Gate 1.
* **MRR** — how high was the *first* correct file? Sensitive to the top of the
  list, which is where a small context budget actually looks.
* **nDCG@k** — rewards getting all gold files high, discounted by position.
  Distinguishes two configurations that share the same Recall@k.
* **Precision@k** — reported as context, not as a target. With 3 gold files and
  k=10, perfect retrieval still scores 0.3.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class QueryResult:
    """One question's outcome under one configuration.

    `retrieved_files` is ordered by rank and may repeat a file — several chunks
    from the same file is the normal case. Dedup happens inside the metrics, so
    callers pass exactly what the retriever returned.
    """

    query_id: str
    retrieved_files: Sequence[str]
    gold_files: Sequence[str]

    @property
    def gold(self) -> set[str]:
        return set(self.gold_files)


@dataclass(slots=True)
class MetricSummary:
    """Averaged metrics over a whole gold set."""

    configuration: str
    queries: int
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    precision_at_10: float = 0.0
    zero_recall: int = 0
    """Questions where no gold file was retrieved at all. The failure count
    that matters — these are unanswerable downstream, whatever the LLM does."""

    per_query: dict[str, float] = field(default_factory=dict)
    """Recall@10 per query id, for finding which questions a change fixed."""

    def as_row(self) -> dict[str, float | int | str]:
        return {
            "configuration": self.configuration,
            "recall@5": round(self.recall_at_5, 3),
            "recall@10": round(self.recall_at_10, 3),
            "mrr": round(self.mrr, 3),
            "ndcg@10": round(self.ndcg_at_10, 3),
            "precision@10": round(self.precision_at_10, 3),
            "zero_recall": self.zero_recall,
        }


def dedupe(files: Iterable[str]) -> list[str]:
    """First occurrence of each file, order preserved.

    Rank positions are taken from this deduplicated list, so a gold file whose
    first chunk is at position 2 gets rank 2 even if positions 3 and 4 are more
    chunks of the same file.
    """
    return list(dict.fromkeys(files))


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of gold files appearing in the top k retrieved chunks.

    `k` counts *chunks*, not distinct files — it is the context budget, and the
    budget is spent on chunks. Deduping before slicing would silently let a
    retriever inspect more than k results.
    """
    if not gold:
        return 0.0
    found = set(dedupe(retrieved[:k])) & gold
    return len(found) / len(gold)


def reciprocal_rank(retrieved: Sequence[str], gold: set[str]) -> float:
    """1 / rank of the first gold file, or 0 if none was retrieved."""
    for rank, path in enumerate(dedupe(retrieved), start=1):
        if path in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Normalised discounted cumulative gain, binary relevance.

    Each distinct gold file contributes 1/log2(rank+1) at its first appearance.
    The ideal ranking puts every gold file in the leading positions, so the
    normaliser is that sum over min(len(gold), k) positions.
    """
    if not gold:
        return 0.0

    ordered = dedupe(retrieved[:k])
    gain = sum(
        1.0 / math.log2(rank + 1) for rank, path in enumerate(ordered, start=1) if path in gold
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return gain / ideal if ideal else 0.0


def precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of the top k *chunks* whose file is relevant.

    Chunk-level on purpose: this measures how much of the context budget was
    spent usefully, and the budget is chunks. Bounded above by
    len(gold)/k when the retriever returns one chunk per file.
    """
    window = retrieved[:k]
    if not window:
        return 0.0
    return sum(1 for path in window if path in gold) / len(window)


def summarise(configuration: str, results: Sequence[QueryResult]) -> MetricSummary:
    """Average every metric over a gold set.

    Macro-averaged — each question counts once regardless of how many gold
    files it has. Micro-averaging would let a single question with eight gold
    files dominate one with two.
    """
    summary = MetricSummary(configuration=configuration, queries=len(results))
    if not results:
        return summary

    recalls_5: list[float] = []
    recalls_10: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    precisions: list[float] = []

    for result in results:
        gold = result.gold
        r10 = recall_at_k(result.retrieved_files, gold, 10)
        recalls_5.append(recall_at_k(result.retrieved_files, gold, 5))
        recalls_10.append(r10)
        rrs.append(reciprocal_rank(result.retrieved_files, gold))
        ndcgs.append(ndcg_at_k(result.retrieved_files, gold, 10))
        precisions.append(precision_at_k(result.retrieved_files, gold, 10))
        summary.per_query[result.query_id] = round(r10, 3)
        if r10 == 0.0:
            summary.zero_recall += 1

    n = len(results)
    summary.recall_at_5 = sum(recalls_5) / n
    summary.recall_at_10 = sum(recalls_10) / n
    summary.mrr = sum(rrs) / n
    summary.ndcg_at_10 = sum(ndcgs) / n
    summary.precision_at_10 = sum(precisions) / n
    return summary


def to_markdown(summaries: Sequence[MetricSummary]) -> str:
    """Render the ablation table that goes straight into the README."""
    if not summaries:
        return "_no results_\n"

    header = (
        "| Configuration | Recall@5 | Recall@10 | MRR | nDCG@10 | Precision@10 | Misses |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = "".join(
        f"| {s.configuration} | {s.recall_at_5:.3f} | {s.recall_at_10:.3f} | "
        f"{s.mrr:.3f} | {s.ndcg_at_10:.3f} | {s.precision_at_10:.3f} | "
        f"{s.zero_recall}/{s.queries} |\n"
        for s in summaries
    )
    return header + rows


def regressions(
    baseline: MetricSummary, candidate: MetricSummary
) -> list[tuple[str, float, float]]:
    """Queries the candidate made worse — the list an average hides.

    A configuration can lift mean Recall@10 while breaking questions the
    baseline answered. Those are the interesting cases, and the ablation
    write-up is much stronger for naming them.
    """
    out: list[tuple[str, float, float]] = []
    for query_id, before in baseline.per_query.items():
        after = candidate.per_query.get(query_id, 0.0)
        if after < before:
            out.append((query_id, before, after))
    return sorted(out, key=lambda row: row[1] - row[2], reverse=True)
