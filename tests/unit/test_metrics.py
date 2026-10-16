"""Tests for the retrieval metrics.

Every expected value here is computed by hand in the test body or stated in the
docstring. A metrics module that is merely self-consistent is worthless — these
numbers go in the README and get defended in interviews.
"""

from __future__ import annotations

import math

from evaluation.metrics import (
    MetricSummary,
    QueryResult,
    dedupe,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    regressions,
    summarise,
    to_markdown,
)

GOLD = {"A.java", "B.java"}
RANKED = ["C.java", "A.java", "A.java", "D.java", "B.java"]


# --------------------------------------------------------------------------- #
# dedupe
# --------------------------------------------------------------------------- #


def test_dedupe_keeps_first_occurrence_in_order() -> None:
    assert dedupe(RANKED) == ["C.java", "A.java", "D.java", "B.java"]


def test_dedupe_empty() -> None:
    assert dedupe([]) == []


# --------------------------------------------------------------------------- #
# recall
# --------------------------------------------------------------------------- #


def test_recall_counts_chunks_not_files() -> None:
    """k is the context budget, and the budget is spent on chunks.

    Top 3 chunks are C, A, A — only one gold file present, so 1/2.
    """
    assert recall_at_k(RANKED, GOLD, 3) == 0.5


def test_recall_at_5_finds_everything() -> None:
    assert recall_at_k(RANKED, GOLD, 5) == 1.0


def test_recall_is_zero_when_nothing_matches() -> None:
    assert recall_at_k(["X.java", "Y.java"], GOLD, 10) == 0.0


def test_recall_with_empty_gold_is_zero_not_a_crash() -> None:
    assert recall_at_k(RANKED, set(), 10) == 0.0


def test_duplicate_chunks_do_not_inflate_recall() -> None:
    """Five chunks of one gold file is one hit, not five."""
    assert recall_at_k(["A.java"] * 5, GOLD, 10) == 0.5


# --------------------------------------------------------------------------- #
# MRR
# --------------------------------------------------------------------------- #


def test_mrr_uses_first_relevant_position() -> None:
    """A is at deduped position 2, so 1/2."""
    assert reciprocal_rank(RANKED, GOLD) == 0.5


def test_mrr_is_one_when_first_result_is_relevant() -> None:
    assert reciprocal_rank(["A.java", "Z.java"], GOLD) == 1.0


def test_mrr_is_zero_when_nothing_relevant() -> None:
    assert reciprocal_rank(["X.java"], GOLD) == 0.0


# --------------------------------------------------------------------------- #
# nDCG
# --------------------------------------------------------------------------- #


def test_ndcg_matches_hand_computation() -> None:
    """Deduped: C, A, D, B. A at rank 2, B at rank 4.

    gain  = 1/log2(3) + 1/log2(5)
    ideal = 1/log2(2) + 1/log2(3)
    """
    gain = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    expected = gain / ideal
    assert math.isclose(expected, 0.6509, abs_tol=1e-4), "hand-check the constant"
    assert math.isclose(ndcg_at_k(RANKED, GOLD, 10), expected)


def test_ndcg_is_one_for_perfect_ranking() -> None:
    assert ndcg_at_k(["A.java", "B.java"], GOLD, 10) == 1.0


def test_ndcg_rewards_higher_placement() -> None:
    """Same recall, better order, higher score — the point of nDCG."""
    high = ndcg_at_k(["A.java", "B.java", "Z.java"], GOLD, 10)
    low = ndcg_at_k(["Z.java", "Y.java", "A.java", "B.java"], GOLD, 10)
    assert high > low


def test_ndcg_zero_when_nothing_relevant() -> None:
    assert ndcg_at_k(["X.java"], GOLD, 10) == 0.0


# --------------------------------------------------------------------------- #
# precision
# --------------------------------------------------------------------------- #


def test_precision_counts_relevant_chunks() -> None:
    """3 of the 5 returned chunks belong to gold files."""
    assert precision_at_k(RANKED, GOLD, 10) == 0.6


def test_precision_on_empty_result() -> None:
    assert precision_at_k([], GOLD, 10) == 0.0


# --------------------------------------------------------------------------- #
# summarise
# --------------------------------------------------------------------------- #


def make_results() -> list[QueryResult]:
    return [
        QueryResult("q1", ["A.java", "B.java"], ["A.java", "B.java"]),  # perfect
        QueryResult("q2", ["X.java", "Y.java", "C.java"], ["C.java"]),  # found at 3
        QueryResult("q3", ["X.java", "Y.java"], ["Z.java"]),  # missed
    ]


def test_summarise_macro_averages() -> None:
    summary = summarise("hybrid", make_results())
    assert summary.queries == 3
    # 1.0 + 1.0 + 0.0 over 3
    assert math.isclose(summary.recall_at_10, 2 / 3)
    # 1/1 + 1/3 + 0 over 3
    assert math.isclose(summary.mrr, (1.0 + 1 / 3) / 3)


def test_summarise_counts_total_misses() -> None:
    """The number that matters: questions no prompt can rescue."""
    assert summarise("hybrid", make_results()).zero_recall == 1


def test_summarise_records_per_query_recall() -> None:
    summary = summarise("hybrid", make_results())
    assert summary.per_query["q1"] == 1.0
    assert summary.per_query["q3"] == 0.0


def test_summarise_on_empty_set() -> None:
    summary = summarise("hybrid", [])
    assert summary.queries == 0
    assert summary.recall_at_10 == 0.0


def test_one_question_with_many_gold_files_does_not_dominate() -> None:
    """Macro-averaging: each question counts once, however many gold files."""
    big = QueryResult("big", ["A.java"], [f"F{i}.java" for i in range(8)])
    small = QueryResult("small", ["G.java"], ["G.java"])
    summary = summarise("x", [big, small])
    assert math.isclose(summary.recall_at_10, 0.5)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def test_markdown_table_has_a_row_per_configuration() -> None:
    table = to_markdown(
        [summarise("dense only", make_results()), summarise("hybrid", make_results())]
    )
    assert "| dense only |" in table
    assert "| hybrid |" in table
    assert table.count("\n") == 4  # header, separator, two rows


def test_markdown_on_empty_input() -> None:
    assert "no results" in to_markdown([])


def test_regressions_finds_queries_a_change_broke() -> None:
    """A mean can rise while individual questions break. Those are the story."""
    baseline = MetricSummary("base", 2, per_query={"q1": 1.0, "q2": 0.5})
    candidate = MetricSummary("new", 2, per_query={"q1": 0.0, "q2": 1.0})

    broken = regressions(baseline, candidate)
    assert broken == [("q1", 1.0, 0.0)]


def test_regressions_empty_when_nothing_got_worse() -> None:
    baseline = MetricSummary("base", 1, per_query={"q1": 0.5})
    candidate = MetricSummary("new", 1, per_query={"q1": 1.0})
    assert regressions(baseline, candidate) == []
