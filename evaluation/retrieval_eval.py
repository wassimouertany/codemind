"""Gate 1: measure retrieval over the hand-labelled gold set.

    uv run python -m evaluation.retrieval_eval --ablation        # all configs + reports
    uv run python -m evaluation.retrieval_eval --config hybrid_rrf
    uv run python -m evaluation.retrieval_eval --limit 5         # quick smoke run

Six configurations, each ablating one stage of the pipeline:

    dense_only              jina embeddings alone
    sparse_only             BM25 over code-aware tokens alone
    hybrid_rrf              both branches, rank fusion
    hybrid_dbsf             both branches, score fusion
    hybrid_rerank           + bge-reranker-base cross-encoder
    hybrid_rerank_expansion + call-graph neighbours of the top hits

## Comparability is the whole design

Every configuration draws from the **same candidate pool** (`PREFETCH_DENSE +
PREFETCH_SPARSE`) and returns the **same number of chunks** (`--final-k`,
default 10, matching Recall@10). Change either per configuration and the table
measures depth rather than technique.

The expansion row is built to answer one question: **is a graph neighbour worth
more than the 6th-10th reranked chunk?** It reranks to
`final_k - EXPANSION_MAX_EXTRA`, appends up to `EXPANSION_MAX_EXTRA` neighbours
of those seeds, and backfills from the displaced reranked tail when fewer
neighbours exist — so the list is always exactly `final_k`, and expansion really
does compete against reranked chunks instead of landing past the cutoff.

Models load once and are shared across every configuration: the cross-encoder
alone takes longer to load than a whole sparse run takes to execute.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

from codemind.core.config import Settings, get_settings
from codemind.core.exceptions import CodeMindError
from codemind.ingestion.symbol_graph import SymbolGraph
from codemind.retrieval.embedder import Embedder
from codemind.retrieval.expansion import GraphExpander
from codemind.retrieval.hybrid import HybridRetriever
from codemind.retrieval.qdrant_store import DENSE_VECTOR, SPARSE_VECTOR, QdrantStore
from codemind.retrieval.reranker import Reranker
from codemind.retrieval.schemas import ScoredChunk
from codemind.retrieval.sparse import SparseEncoder
from evaluation import metrics
from evaluation.significance import paired_bootstrap, verdict
from evaluation.validate_goldset import (
    DEFAULT_GOLDSET,
    GoldQuery,
    indexed_paths,
    load_goldset,
)

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("evaluation/reports")
ABLATION_REPORT = REPORTS_DIR / "gate1_retrieval.md"
PER_QUERY_REPORT = REPORTS_DIR / "gate1_per_query.json"
DEFAULT_FINAL_K = 10
"""Chunks each configuration returns. 10 because Recall@10 is the gate metric."""


@dataclass(slots=True, frozen=True)
class Configuration:
    """One row of the ablation table."""

    name: str
    branch: str | None = None
    """Single-branch query, or None for fused."""
    fusion: str = "rrf"
    rerank: bool = False
    expand: bool = False
    needs_graph: bool = False

    @property
    def description(self) -> str:
        if self.branch:
            return f"{self.branch} vector only, no fusion"
        parts = [f"dense + sparse, {self.fusion.upper()} fusion"]
        if self.rerank:
            parts.append("cross-encoder rerank")
        if self.expand:
            parts.append(
                "call-graph expansion (answers: is a graph neighbour worth more "
                "than the 6th-10th reranked chunk?)"
            )
        return ", ".join(parts)


CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration("dense_only", branch=DENSE_VECTOR),
    Configuration("sparse_only", branch=SPARSE_VECTOR),
    Configuration("hybrid_rrf"),
    Configuration("hybrid_dbsf", fusion="dbsf"),
    Configuration("hybrid_rerank", rerank=True),
    Configuration("hybrid_rerank_expansion", rerank=True, expand=True, needs_graph=True),
)
CONFIGURATIONS_BY_NAME = {config.name: config for config in CONFIGURATIONS}


class Harness:
    """Owns the models and runs any configuration against the gold set.

    Constructed once. Each configuration gets its own `HybridRetriever` (fusion
    is fixed at construction) but shares the underlying encoders and store.
    """

    def __init__(self, settings: Settings, *, final_k: int, needs_rerank: bool) -> None:
        self._settings = settings
        self._final_k = final_k
        self._store = QdrantStore(settings)
        self._embedder = Embedder(settings)
        self._sparse = SparseEncoder(settings)
        # The cross-encoder is ~278M parameters and the single slowest thing to
        # load, so a dense-only run must not pay for it.
        self._reranker = Reranker(settings) if needs_rerank else None
        self._graph: SymbolGraph | None = None
        self._retrievers: dict[str, HybridRetriever] = {}
        self._indexed_chunks: int | None = None

    @property
    def graph_path(self) -> Path:
        return self._settings.graph_dir / f"{self._store.collection_name}.json"

    def load_graph(self) -> SymbolGraph | None:
        """Load the persisted symbol graph, or None when it is absent."""
        if self._graph is None and self.graph_path.is_file():
            self._graph = SymbolGraph.load(self.graph_path)
            logger.info(
                "loaded symbol graph: %d nodes, %d edges",
                self._graph.graph.number_of_nodes(),
                self._graph.graph.number_of_edges(),
            )
        return self._graph

    def _retriever(self, fusion: str) -> HybridRetriever:
        if fusion not in self._retrievers:
            tuned = self._settings.model_copy(update={"fusion_method": fusion})
            self._retrievers[fusion] = HybridRetriever(
                self._store, self._embedder, self._sparse, tuned
            )
        return self._retrievers[fusion]

    async def close(self) -> None:
        await self._store.close()

    async def retrieve(self, config: Configuration, question: str) -> list[ScoredChunk]:
        """Run one question through one configuration, returning final-k chunks."""
        retriever = self._retriever(config.fusion)

        if config.branch:
            # Depth matches the fused pool, not a branch prefetch size.
            result = await retriever.search_branch(
                question, branch=config.branch, limit=retriever.fused_limit
            )
        else:
            result = await retriever.search(question)

        chunks = list(result.chunks)
        if not config.rerank:
            return chunks[: self._final_k]

        if self._reranker is None:  # pragma: no cover - guarded by needs_rerank
            raise CodeMindError("reranker was not loaded for a reranking configuration")

        reranked = await self._reranker.rerank(question, chunks, top_k=self._final_k)
        if not config.expand:
            return reranked

        graph = self.load_graph()
        if graph is None:
            raise CodeMindError(
                f"no symbol graph at {self.graph_path} — run scripts/ingest_repo.py first"
            )
        # Seeds keep the top of the list; neighbours compete for the tail.
        # `expansion_max_extra` must leave at least one seed to walk from.
        extra_budget = min(self._settings.expansion_max_extra, self._final_k - 1)
        seeds, remainder = (
            reranked[: self._final_k - extra_budget],
            reranked[self._final_k - extra_budget :],
        )
        # The shipped default is EXPANSION_DEPTH=0 (off, on Gate 1 evidence).
        # The ablation row exists to *measure* the stage, so it forces at least
        # one hop — otherwise this row silently becomes a copy of
        # hybrid_rerank and the table stops saying anything about expansion.
        expander = GraphExpander(
            self._store,
            graph,
            self._settings.model_copy(
                update={
                    "expansion_max_extra": extra_budget,
                    "expansion_depth": max(1, self._settings.expansion_depth),
                }
            ),
        )
        expanded = await expander.expand(seeds)
        return backfill(expanded, remainder, self._final_k)

    async def corpus_stats(self) -> tuple[int, int]:
        """(distinct files, chunks) in the collection — the report's denominator."""
        if self._indexed_chunks is None:
            self._indexed_chunks = await self._store.count()
        # Reuses the validator's paginating scroll: a one-shot `limit=N` would
        # silently undercount files on a repository-scale index.
        return len(await indexed_paths(self._store)), self._indexed_chunks

    async def _check_depth(self, config: Configuration, query_id: str, got: int) -> None:
        """Every configuration must return exactly `final_k` chunks.

        Unequal depth silently invalidates the whole table: a configuration
        returning 5 chunks cannot win Recall@10 against one returning 10, and
        the difference reads as a quality gap. This caught expansion handing
        back 5 chunks on arch-01, arch-04 and loc-06 while every other row had
        10. Skipped only when the index itself holds fewer chunks than `k`.
        """
        if self._indexed_chunks is None:
            self._indexed_chunks = await self._store.count()
        if self._indexed_chunks < self._final_k:
            return
        if got != self._final_k:
            raise CodeMindError(
                f"{config.name} returned {got} chunks for {query_id}, expected "
                f"{self._final_k} — configurations must be compared at equal depth"
            )

    async def run(
        self, config: Configuration, queries: Sequence[GoldQuery]
    ) -> tuple[metrics.MetricSummary, list[dict[str, object]]]:
        """Evaluate one configuration over the gold set."""
        started = time.perf_counter()
        results: list[metrics.QueryResult] = []
        details: list[dict[str, object]] = []

        for query in queries:
            chunks = await self.retrieve(config, query.question)
            paths = [scored.chunk.relative_path for scored in chunks]
            await self._check_depth(config, query.query_id, len(paths))
            results.append(
                metrics.QueryResult(
                    query_id=query.query_id,
                    retrieved_files=paths,
                    gold_files=query.gold_files,
                )
            )
            details.append(
                {
                    "configuration": config.name,
                    "query_id": query.query_id,
                    "category": query.category,
                    "question": query.question,
                    "gold_files": list(query.gold_files),
                    "retrieved_files": metrics.dedupe(paths)[: self._final_k],
                    "retrieved_chunks": paths,
                    "recall_at_10": round(metrics.recall_at_k(paths, set(query.gold_files), 10), 3),
                }
            )

        summary = metrics.summarise(config.name, results)
        logger.info(
            "%-24s recall@10=%.3f mrr=%.3f (%.1fs)",
            config.name,
            summary.recall_at_10,
            summary.mrr,
            time.perf_counter() - started,
        )
        return summary, details


SIGNIFICANCE_METRICS: tuple[tuple[str, int], ...] = (("Recall@5", 5), ("Recall@10", 10))


def significance_pairs(names: Sequence[str]) -> list[tuple[str, str]]:
    """Consecutive steps, plus every row against the dense-only baseline.

    Consecutive pairs answer "did this stage help"; dense-only pairs answer
    "is any of this better than the simplest thing that works", which is the
    question Gate 1 is actually asking.
    """
    pairs = list(pairwise(names))
    baseline = names[0] if names else ""
    pairs += [(baseline, name) for name in names[1:] if (baseline, name) not in pairs]
    return pairs


def significance_section(details: Sequence[dict[str, object]]) -> str:
    """Paired bootstrap over per-query recall, rendered per metric.

    Recomputed from the cached retrieved/gold lists rather than from
    `MetricSummary.per_query`, which only carries Recall@10 — and Recall@5 is
    the metric that still discriminates at this corpus size.
    """
    samples: dict[str, dict[int, dict[str, float]]] = {}
    for row in details:
        name = str(row["configuration"])
        # Detail rows are JSON-shaped (`dict[str, object]`) so they can be read
        # back from the cached report as readily as from a live run.
        chunks = [str(path) for path in cast("Sequence[object]", row["retrieved_chunks"])]
        gold = {str(path) for path in cast("Sequence[object]", row["gold_files"])}
        for _, k in SIGNIFICANCE_METRICS:
            samples.setdefault(name, {}).setdefault(k, {})[str(row["query_id"])] = (
                metrics.recall_at_k(chunks, gold, k)
            )

    ordered = [c.name for c in CONFIGURATIONS if c.name in samples]
    if len(ordered) < 2:
        return "\n## Significance\n\n_needs at least two configurations._\n"

    out = (
        "\n## Significance\n\n"
        "Paired bootstrap, 10,000 resamples of the 30 questions. Paired because every "
        "configuration answers the same questions. A CI spanning zero means the two are "
        "**not distinguishable** on this gold set — at n=30 that is the common case, and "
        "saying so is more useful than ranking noise.\n"
    )
    for label, k in SIGNIFICANCE_METRICS:
        out += f"\n### {label}\n\n| Comparison | Delta and 95% CI |\n|---|---|\n"
        for baseline, candidate in significance_pairs(ordered):
            ids = sorted(set(samples[baseline][k]) & set(samples[candidate][k]))
            before = [samples[baseline][k][i] for i in ids]
            after = [samples[candidate][k][i] for i in ids]
            out += f"| {baseline} → {candidate} | {verdict(*paired_bootstrap(before, after))} |\n"
    return out


def assert_questions_match(
    details: Sequence[dict[str, object]], queries: Sequence[GoldQuery]
) -> None:
    """Refuse to score cached retrievals against a question that has changed.

    Correcting a label often means rewording the question, and a cached
    `retrieved_chunks` list was produced for the *old* wording. Scoring it
    against new gold is silent nonsense: the number looks plausible and
    measures a retrieval nobody asked for. This trap recurs on every label
    fix, so it raises rather than warns.
    """
    current = {query.query_id: query.question for query in queries}
    stale = sorted(
        {
            str(row["query_id"])
            for row in details
            if str(row["query_id"]) in current
            and str(row.get("question", "")) != current[str(row["query_id"])]
        }
    )
    if stale:
        raise CodeMindError(
            "cached retrievals were produced for different question wording: "
            f"{', '.join(stale)} — re-retrieve them with "
            f"--only {','.join(stale)} instead of rescoring from cache"
        )


def load_cached_details(path: Path = PER_QUERY_REPORT) -> list[dict[str, object]]:
    """Per-query rows from a previous run, or empty when there is no cache."""
    if not path.is_file():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return list(loaded) if isinstance(loaded, list) else []


def merge_details(
    cached: Sequence[dict[str, object]], fresh: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    """Splice fresh rows over cached ones, keyed by (configuration, query_id).

    Lets a two-question re-retrieval replace exactly its own entries instead of
    forcing a full re-ablation for every corrected label.
    """
    merged = {(str(r["configuration"]), str(r["query_id"])): r for r in cached}
    merged.update({(str(r["configuration"]), str(r["query_id"])): r for r in fresh})
    return list(merged.values())


def summaries_from_details(
    details: Sequence[dict[str, object]], queries: Sequence[GoldQuery]
) -> list[metrics.MetricSummary]:
    """Rebuild every summary from per-query rows, in configuration order.

    Metrics recompute exactly from the stored chunk lists, so a spliced cache
    yields the same table a full run would.
    """
    gold = {query.query_id: list(query.gold_files) for query in queries}
    grouped: dict[str, list[metrics.QueryResult]] = {}
    for row in details:
        query_id = str(row["query_id"])
        if query_id not in gold:
            continue  # a question removed from the gold set since that run
        grouped.setdefault(str(row["configuration"]), []).append(
            metrics.QueryResult(
                query_id=query_id,
                retrieved_files=[
                    str(path) for path in cast("Sequence[object]", row["retrieved_chunks"])
                ],
                gold_files=gold[query_id],
            )
        )
    return [
        metrics.summarise(config.name, grouped[config.name])
        for config in CONFIGURATIONS
        if config.name in grouped
    ]


def backfill(
    chunks: Sequence[ScoredChunk], remainder: Sequence[ScoredChunk], final_k: int
) -> list[ScoredChunk]:
    """Top up `chunks` from `remainder` until the list is exactly `final_k` long.

    Expansion displaces the tail of the reranked list, but only as far as it
    actually has neighbours to offer. Without backfill a symbol with no callers
    would hand back a short list, and a configuration returning 6 chunks cannot
    be compared with one returning 10 — the gap reads as a quality difference.
    Duplicates are skipped: a neighbour may already sit in the reranked tail.
    """
    out = list(chunks[:final_k])
    seen = {scored.chunk.chunk_id for scored in out}
    for scored in remainder:
        if len(out) >= final_k:
            break
        if scored.chunk.chunk_id in seen:
            continue
        seen.add(scored.chunk.chunk_id)
        out.append(scored)
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def category_table(
    summaries: Sequence[metrics.MetricSummary],
    queries: Sequence[GoldQuery],
) -> str:
    """Recall@10 per category — where a configuration wins is as useful as whether.

    Sparse is expected to carry `locator` questions ("where is X implemented")
    and dense to carry `architecture` ones; a table that only reports the mean
    hides that trade entirely.
    """
    categories = sorted({query.category for query in queries})
    by_id = {query.query_id: query.category for query in queries}
    counts = Counter(by_id.values())

    # Parenthesised deliberately: without it Python concatenates the adjacent
    # literals first and `*` repeats the newline too, producing one separator
    # row per category instead of one row of separators.
    header = (
        "| Configuration | "
        + " | ".join(f"{c} (n={counts[c]})" for c in categories)
        + " |\n"
        + ("|---" * (len(categories) + 1))
        + "|\n"
    )
    rows = ""
    for summary in summaries:
        cells = []
        for category in categories:
            scores = [
                recall
                for query_id, recall in summary.per_query.items()
                if by_id.get(query_id) == category
            ]
            cells.append(f"{sum(scores) / len(scores):.3f}" if scores else "—")
        rows += f"| {summary.configuration} | " + " | ".join(cells) + " |\n"
    return header + rows


def regression_section(summaries: Sequence[metrics.MetricSummary]) -> str:
    """Per-query regressions between consecutive configurations."""
    if len(summaries) < 2:
        return "_needs at least two configurations._\n"

    out = ""
    for baseline, candidate in pairwise(summaries):
        rows = metrics.regressions(baseline, candidate)
        delta = candidate.recall_at_10 - baseline.recall_at_10
        out += f"\n**{baseline.configuration} → {candidate.configuration}** "
        out += f"(mean Recall@10 {delta:+.3f})\n\n"
        if not rows:
            out += "No query regressed.\n"
            continue
        out += "| Query | Before | After |\n|---|---|---|\n"
        out += "".join(f"| {qid} | {before:.3f} | {after:.3f} |\n" for qid, before, after in rows)
    return out


def scale_caveat(files: int, chunks: int, final_k: int) -> str:
    """State the corpus size next to the numbers it inflates.

    At 99 chunks a top-10 is ~10% of the whole index, so Recall@10 is generous
    in a way it would not be against a production-scale repository: a random
    ranker would score far above zero. The deltas between configurations stay
    meaningful; the absolute values do not transfer.
    """
    coverage = final_k / chunks * 100 if chunks else 0.0
    return (
        f"\n## Reading these numbers\n\n"
        f"Corpus: **{files} files / {chunks} chunks** after excluding tests. A top-{final_k} "
        f"is **~{coverage:.0f}% of the whole index**, so absolute Recall@{final_k} is inflated "
        f"relative to a production-scale corpus and does not transfer.\n\n"
        f"- Compare configurations by **delta**, not by absolute value.\n"
        f"- Prefer **Recall@5** and **MRR**: both stay discriminative at this size, "
        f"while Recall@{final_k} saturates.\n"
        f"- `Misses` (questions retrieving no gold file at all) is the figure that "
        f"would survive a larger index.\n"
    )


def write_ablation_report(
    summaries: Sequence[metrics.MetricSummary],
    queries: Sequence[GoldQuery],
    settings: Settings,
    final_k: int,
    corpus: tuple[int, int] = (0, 0),
    details: Sequence[dict[str, object]] = (),
) -> Path:
    assert_questions_match(details, queries)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Gate 1 — retrieval baseline\n",
        f"\nGold set: **{len(queries)} questions** hand-labelled against "
        f"`{settings.qdrant_collection}`.\n",
        f"\nEvery configuration draws from the same pool of "
        f"{settings.prefetch_dense + settings.prefetch_sparse} candidates "
        f"(`PREFETCH_DENSE={settings.prefetch_dense}` + "
        f"`PREFETCH_SPARSE={settings.prefetch_sparse}`) and returns "
        f"{final_k} chunks, so the rows differ by technique and not by depth.\n",
        scale_caveat(corpus[0], corpus[1], final_k),
        "\n## Ablation\n\n",
        metrics.to_markdown(summaries),
        "\nMisses = questions where no gold file was retrieved at all.\n",
        "\n### Configurations\n\n",
        "".join(
            f"- `{c.name}` — {c.description}\n"
            for c in CONFIGURATIONS
            if c.name in {s.configuration for s in summaries}
        ),
        significance_section(details),
        "\n## Recall@10 by category\n\n",
        category_table(summaries, queries),
        "\n## Per-query regressions\n",
        regression_section(summaries),
        f"\n---\n\nPer-query detail: `{PER_QUERY_REPORT.name}`.\n",
    ]
    ABLATION_REPORT.write_text("".join(lines), encoding="utf-8")
    return ABLATION_REPORT


def write_per_query_report(details: Sequence[dict[str, object]]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PER_QUERY_REPORT.write_text(json.dumps(list(details), indent=2), encoding="utf-8")
    return PER_QUERY_REPORT


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


async def run_evaluation(args: argparse.Namespace) -> int:
    settings = Settings(qdrant_collection=args.collection) if args.collection else get_settings()

    queries = load_goldset(args.goldset)
    if args.only:
        wanted = [q.strip() for q in args.only.split(",") if q.strip()]
        known = {query.query_id for query in queries}
        if missing := [q for q in wanted if q not in known]:
            print(f"error: unknown query id(s): {', '.join(missing)}", file=sys.stderr)
            return 1
        queries = [query for query in queries if query.query_id in wanted]
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("error: gold set is empty", file=sys.stderr)
        return 1

    selected = [CONFIGURATIONS_BY_NAME[args.config]] if args.config else list(CONFIGURATIONS)
    harness = Harness(
        settings,
        final_k=args.final_k,
        needs_rerank=any(config.rerank for config in selected),
    )

    summaries: list[metrics.MetricSummary] = []
    details: list[dict[str, object]] = []
    # Captured inside the try, because `finally` closes the Qdrant client and
    # the report is written after that.
    corpus = (0, 0)
    try:
        for config in selected:
            if config.needs_graph and harness.load_graph() is None:
                print(
                    f"skipping {config.name}: no symbol graph at {harness.graph_path}",
                    file=sys.stderr,
                )
                continue
            summary, rows = await harness.run(config, queries)
            summaries.append(summary)
            details.extend(rows)
        corpus = await harness.corpus_stats()
    except CodeMindError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        await harness.close()

    print()
    print(metrics.to_markdown(summaries))

    if args.ablation:
        report = write_ablation_report(summaries, queries, settings, args.final_k, corpus, details)
        per_query = write_per_query_report(details)
        print(f"  wrote {report}")
        print(f"  wrote {per_query}\n")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--collection", help="override QDRANT_COLLECTION")
    parser.add_argument(
        "--config", choices=sorted(CONFIGURATIONS_BY_NAME), help="run a single configuration"
    )
    parser.add_argument("--ablation", action="store_true", help="write the Gate 1 reports")
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    parser.add_argument(
        "--only",
        help="re-retrieve just these query ids (comma-separated); with --ablation the "
        "fresh rows are spliced over the cached ones instead of replacing the file",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=DEFAULT_FINAL_K,
        help=f"chunks returned per configuration (default {DEFAULT_FINAL_K})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.exit(asyncio.run(run_evaluation(args)))


if __name__ == "__main__":
    main()
