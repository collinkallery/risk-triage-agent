"""Tests for the keyword-vs-vector comparator. Uses synthetic scored dicts."""

from __future__ import annotations

from eval.compare_retrieval import (
    invoked_case_ids,
    render_report,
    subset_category_metrics,
)


def _case(case_id, *, correct, normalized, called_search):
    tool_calls = [{"name": "lookup_rubric"}]
    if called_search:
        tool_calls.append({"name": "search_reference"})
    return {
        "case_id": case_id,
        "trace": {"tool_calls": tool_calls},
        "scores": {"category": {"correct": correct, "normalized": normalized}},
    }


def _run(cases, *, retrieval):
    return {
        "metadata": {"retrieval": retrieval, "model": "mock", "eval_set_size": len(cases)},
        "results": cases,
        "aggregates": {
            "headline": {"weighted_accuracy": 0.8, "raw_accuracy": 0.5, "mean_weighted_score": 0.6},
            "per_dimension": {
                "category_raw_accuracy": 0.5,
                "category_weighted_accuracy": 0.8,
                "sub_pattern_accuracy_cat1_only": 1.0,
                "subject_frame_accuracy": 0.9,
                "criteria_f1_mean": 0.7,
                "tool_use_f1_mean": 0.6,
            },
            "slices": {
                "by_tag": {"calibration": {"n": 2, "weighted_accuracy": 0.75}},
                "by_category": {"1": {"n": 3, "weighted_accuracy": 0.9}},
            },
        },
    }


def test_invoked_case_ids_detects_search_calls():
    run = _run(
        [
            _case("C1", correct=True, normalized=1.0, called_search=True),
            _case("C2", correct=False, normalized=0.8, called_search=False),
            _case("C3", correct=True, normalized=1.0, called_search=True),
        ],
        retrieval="keyword",
    )
    assert invoked_case_ids(run) == {"C1", "C3"}


def test_subset_category_metrics_computes_accuracy_and_mean():
    run = _run(
        [
            _case("C1", correct=True, normalized=1.0, called_search=True),
            _case("C2", correct=False, normalized=0.6, called_search=True),
        ],
        retrieval="vector",
    )
    m = subset_category_metrics(run, {"C1", "C2"})
    assert m["n"] == 2
    assert m["raw_accuracy"] == 0.5
    assert abs(m["mean_normalized"] - 0.8) < 1e-9


def test_subset_metrics_empty_subset():
    run = _run([_case("C1", correct=True, normalized=1.0, called_search=False)], retrieval="keyword")
    m = subset_category_metrics(run, set())
    assert m["n"] == 0 and m["raw_accuracy"] is None


def test_render_report_runs_and_has_sections():
    kw = _run([_case("C1", correct=True, normalized=1.0, called_search=True)], retrieval="keyword")
    vec = _run([_case("C1", correct=False, normalized=0.8, called_search=True)], retrieval="vector")
    md = render_report(kw, vec)
    assert "# Retrieval Comparison" in md
    assert "Retrieval-Invoked Segment" in md
    assert "Interpretation" in md
    assert "Keyword" in md and "Vector" in md
