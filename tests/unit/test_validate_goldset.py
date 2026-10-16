"""Tests for gold-set validation.

The structural checks are pure functions over parsed rows, so they are unit
tested here; the index check is exercised against a real collection by
`make validate-goldset`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation.validate_goldset import (
    CATEGORIES,
    GoldQuery,
    check_against_index,
    check_structure,
    load_goldset,
)

from codemind.core.exceptions import CodeMindError

REAL_GOLDSET = Path("evaluation/datasets/retrieval_goldset.jsonl")


def query(**overrides: object) -> GoldQuery:
    base = {
        "id": "bug-01",
        "category": "bug",
        "question": "Why does saving an owner fail?",
        "gold_files": ["src/main/java/org/example/OwnerController.java"],
    }
    return GoldQuery.from_raw(base | overrides)


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


def test_a_valid_row_has_no_errors() -> None:
    assert check_structure([query()]) == []


def test_duplicate_ids_are_rejected() -> None:
    errors = check_structure([query(), query()])
    assert any("unique" in error for error in errors)


def test_unknown_category_is_rejected() -> None:
    errors = check_structure([query(category="perf")])
    assert any("category" in error for error in errors)


def test_every_declared_category_is_accepted() -> None:
    rows = [query(id=f"q{i}", category=c) for i, c in enumerate(sorted(CATEGORIES))]
    assert check_structure(rows) == []


def test_empty_question_is_rejected() -> None:
    assert any("question is empty" in e for e in check_structure([query(question="   ")]))


def test_gold_files_must_not_be_empty() -> None:
    assert any("expected 1-4" in e for e in check_structure([query(gold_files=[])]))


def test_more_than_four_gold_files_is_rejected() -> None:
    rows = [query(gold_files=[f"src/F{i}.java" for i in range(5)])]
    assert any("expected 1-4" in e for e in check_structure(rows))


def test_duplicate_gold_paths_are_rejected() -> None:
    rows = [query(gold_files=["src/A.java", "src/A.java"])]
    assert any("duplicate paths" in e for e in check_structure(rows))


def test_missing_id_is_reported() -> None:
    assert any("no id" in e for e in check_structure([query(id="")]))


# --------------------------------------------------------------------------- #
# index comparison
# --------------------------------------------------------------------------- #


def test_indexed_gold_paths_pass() -> None:
    path = "src/main/java/org/example/OwnerController.java"
    assert check_against_index([query(gold_files=[path])], {path, "src/Other.java"}) == []


def test_missing_path_is_reported_with_close_matches() -> None:
    """A single wrong segment in a long Java path is the realistic typo."""
    typo = "src/main/java/org/example/OwnerControler.java"
    indexed = {
        "src/main/java/org/example/OwnerController.java",
        "src/main/java/org/example/PetController.java",
        "src/main/java/org/example/VisitController.java",
    }
    errors = check_against_index([query(gold_files=[typo])], indexed)

    assert any("not indexed" in error for error in errors)
    suggestions = [error for error in errors if "did you mean" in error]
    assert len(suggestions) == 3
    assert "OwnerController.java" in suggestions[0], "closest match should rank first"


def test_empty_collection_is_a_single_clear_error() -> None:
    errors = check_against_index([query()], set())
    assert errors == ["collection is empty — ingest the repository before validating"]


# --------------------------------------------------------------------------- #
# the real gold set
# --------------------------------------------------------------------------- #


def test_the_committed_goldset_is_structurally_valid() -> None:
    """Guards the file itself, without needing Qdrant."""
    queries = load_goldset(REAL_GOLDSET)
    assert len(queries) == 30
    assert check_structure(queries) == []


def test_load_goldset_reports_the_bad_line_number(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps({"id": "a"}) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"broken\.jsonl:2"):
        load_goldset(path)


def test_missing_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not found"):
        load_goldset(tmp_path / "nope.jsonl")


# --------------------------------------------------------------------------- #
# report rendering
# --------------------------------------------------------------------------- #


def test_category_table_renders_one_separator_row() -> None:
    """Guards a precedence bug: `"a" "b" * n` repeats the concatenation."""
    from evaluation.metrics import MetricSummary
    from evaluation.retrieval_eval import category_table

    queries = [
        GoldQuery.from_raw(
            {"id": "bug-01", "category": "bug", "question": "q", "gold_files": ["a"]}
        ),
        GoldQuery.from_raw(
            {"id": "loc-01", "category": "locator", "question": "q", "gold_files": ["b"]}
        ),
    ]
    summary = MetricSummary(
        configuration="dense_only", queries=2, per_query={"bug-01": 1.0, "loc-01": 0.5}
    )
    lines = category_table([summary], queries).strip().splitlines()

    assert len(lines) == 3, f"header, one separator, one data row; got {lines}"
    assert set(lines[1]) <= {"|", "-"}
    assert lines[1].count("|") == 4, "two categories plus the configuration column"
    assert lines[2].startswith("| dense_only |")


def test_scale_caveat_states_corpus_size_and_coverage() -> None:
    from evaluation.retrieval_eval import scale_caveat

    text = scale_caveat(files=25, chunks=99, final_k=10)
    assert "25 files / 99 chunks" in text
    assert "~10%" in text
    assert "delta" in text


# --------------------------------------------------------------------------- #
# expansion budget composition
# --------------------------------------------------------------------------- #


def scored_chunk(symbol: str) -> object:
    from codemind.core.types import CodeChunk, Language
    from codemind.retrieval.schemas import ScoredChunk

    return ScoredChunk(
        chunk=CodeChunk(
            relative_path=f"src/{symbol}.java",
            language=Language.JAVA,
            symbol_name=symbol,
            kind="method",
            start_line=1,
            end_line=5,
            body=f"void {symbol}() {{}}",
        ),
        score=1.0,
    )


def test_backfill_tops_up_a_short_list_to_final_k() -> None:
    """A symbol with no callers must not yield a 6-chunk configuration."""
    from evaluation.retrieval_eval import backfill

    seeds = [scored_chunk(f"seed{i}") for i in range(5)]
    remainder = [scored_chunk(f"tail{i}") for i in range(5)]
    out = backfill(seeds, remainder, 10)

    assert len(out) == 10
    assert [c.chunk.symbol_name for c in out[:5]] == [f"seed{i}" for i in range(5)]


def test_backfill_leaves_a_full_list_untouched() -> None:
    """When expansion found enough neighbours, nothing is backfilled."""
    from evaluation.retrieval_eval import backfill

    full = [scored_chunk(f"c{i}") for i in range(10)]
    remainder = [scored_chunk("unused")]
    out = backfill(full, remainder, 10)

    assert [c.chunk.symbol_name for c in out] == [f"c{i}" for i in range(10)]


def test_backfill_skips_chunks_already_present() -> None:
    """A neighbour may already sit in the displaced reranked tail."""
    from evaluation.retrieval_eval import backfill

    shared = scored_chunk("shared")
    out = backfill([scored_chunk("seed"), shared], [shared, scored_chunk("fresh")], 3)

    names = [c.chunk.symbol_name for c in out]
    assert names == ["seed", "shared", "fresh"]
    assert len(names) == len(set(names))


def test_backfill_truncates_an_overlong_list() -> None:
    from evaluation.retrieval_eval import backfill

    assert len(backfill([scored_chunk(f"c{i}") for i in range(12)], [], 10)) == 10


def test_backfill_cannot_exceed_available_chunks() -> None:
    """Fewer chunks than k in total is a valid state, not an error."""
    from evaluation.retrieval_eval import backfill

    assert len(backfill([scored_chunk("a")], [scored_chunk("b")], 10)) == 2


# --------------------------------------------------------------------------- #
# significance wiring
# --------------------------------------------------------------------------- #


def test_significance_pairs_covers_consecutive_and_baseline() -> None:
    """Consecutive answers "did this stage help"; baseline answers Gate 1's question."""
    from evaluation.retrieval_eval import significance_pairs

    pairs = significance_pairs(["dense_only", "sparse_only", "hybrid_rrf", "hybrid_dbsf"])

    assert ("dense_only", "sparse_only") in pairs
    assert ("sparse_only", "hybrid_rrf") in pairs
    assert ("hybrid_rrf", "hybrid_dbsf") in pairs
    assert ("dense_only", "hybrid_rrf") in pairs
    assert ("dense_only", "hybrid_dbsf") in pairs
    assert len(pairs) == len(set(pairs)), "dense_only -> sparse_only must not be duplicated"


def test_significance_pairs_handles_a_single_configuration() -> None:
    from evaluation.retrieval_eval import significance_pairs

    assert significance_pairs(["dense_only"]) == []


def test_significance_section_reports_both_metrics_and_no_false_signal() -> None:
    """Identical retrieval must read as 'not distinguishable', never as a win."""
    from evaluation.retrieval_eval import significance_section

    details = [
        {
            "configuration": name,
            "query_id": f"q{i}",
            "gold_files": ["src/A.java"],
            "retrieved_chunks": ["src/A.java", "src/B.java"],
        }
        for name in ("dense_only", "hybrid_rrf")
        for i in range(5)
    ]
    text = significance_section(details)

    assert "### Recall@5" in text
    assert "### Recall@10" in text
    assert "dense_only → hybrid_rrf" in text
    assert "not distinguishable" in text
    assert "improvement" not in text


def test_significance_section_detects_a_real_improvement() -> None:
    from evaluation.retrieval_eval import significance_section

    details = []
    for i in range(20):
        details.append(
            {
                "configuration": "dense_only",
                "query_id": f"q{i}",
                "gold_files": ["src/A.java"],
                "retrieved_chunks": ["src/Z.java"],
            }
        )
        details.append(
            {
                "configuration": "hybrid_rrf",
                "query_id": f"q{i}",
                "gold_files": ["src/A.java"],
                "retrieved_chunks": ["src/A.java"],
            }
        )
    assert "improvement" in significance_section(details)


# --------------------------------------------------------------------------- #
# stale-cache guard and splicing
# --------------------------------------------------------------------------- #


def detail(config: str, query_id: str, question: str, chunks: list[str]) -> dict[str, object]:
    return {
        "configuration": config,
        "query_id": query_id,
        "question": question,
        "gold_files": ["src/A.java"],
        "retrieved_chunks": chunks,
    }


def test_assert_questions_match_accepts_unchanged_wording() -> None:
    from evaluation.retrieval_eval import assert_questions_match

    queries = [query(id="bug-01", question="Why does saving an owner fail?")]
    rows = [detail("dense_only", "bug-01", "Why does saving an owner fail?", ["src/A.java"])]
    assert_questions_match(rows, queries)  # must not raise


def test_assert_questions_match_rejects_reworded_question() -> None:
    """The trap: a cached retrieval scored against new gold reads as plausible."""
    from evaluation.retrieval_eval import assert_questions_match

    queries = [query(id="dep-03", question="Which Java classes use the pet type repository?")]
    rows = [detail("dense_only", "dep-03", "What depends on PetTypeFormatter?", ["src/A.java"])]

    with pytest.raises(CodeMindError, match=r"dep-03") as exc_info:
        assert_questions_match(rows, queries)
    assert "--only dep-03" in str(exc_info.value), "the message must name the fix"


def test_merge_details_replaces_by_configuration_and_query_id() -> None:
    from evaluation.retrieval_eval import merge_details

    cached = [
        detail("dense_only", "dep-03", "old wording", ["src/Stale.java"]),
        detail("dense_only", "bug-01", "unchanged", ["src/Keep.java"]),
    ]
    fresh = [detail("dense_only", "dep-03", "new wording", ["src/Fresh.java"])]
    merged = {(r["configuration"], r["query_id"]): r for r in merge_details(cached, fresh)}

    assert len(merged) == 2
    assert merged[("dense_only", "dep-03")]["retrieved_chunks"] == ["src/Fresh.java"]
    assert merged[("dense_only", "bug-01")]["retrieved_chunks"] == ["src/Keep.java"]


def test_merge_details_keeps_configurations_separate() -> None:
    from evaluation.retrieval_eval import merge_details

    cached = [detail("dense_only", "dep-03", "q", ["src/A.java"])]
    fresh = [detail("hybrid_rrf", "dep-03", "q", ["src/B.java"])]
    assert len(merge_details(cached, fresh)) == 2


def test_summaries_from_details_rebuilds_metrics_from_cache() -> None:
    from evaluation.retrieval_eval import summaries_from_details

    queries = [query(id="bug-01", gold_files=["src/A.java"])]
    rows = [
        detail("dense_only", "bug-01", queries[0].question, ["src/A.java"]),
        detail("hybrid_rrf", "bug-01", queries[0].question, ["src/Z.java"]),
    ]
    summaries = {s.configuration: s for s in summaries_from_details(rows, queries)}

    assert summaries["dense_only"].recall_at_10 == 1.0
    assert summaries["hybrid_rrf"].recall_at_10 == 0.0
    assert summaries["hybrid_rrf"].zero_recall == 1


def test_summaries_from_details_drops_queries_no_longer_in_the_goldset() -> None:
    from evaluation.retrieval_eval import summaries_from_details

    queries = [query(id="bug-01", gold_files=["src/A.java"])]
    rows = [
        detail("dense_only", "bug-01", queries[0].question, ["src/A.java"]),
        detail("dense_only", "removed-99", "gone", ["src/A.java"]),
    ]
    assert summaries_from_details(rows, queries)[0].queries == 1
