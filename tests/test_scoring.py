"""Tests for the deterministic scoring layer."""

from eval.scoring import (
    compute_aggregates,
    score_case_deterministic,
    score_category,
    score_criteria,
    score_sub_pattern,
)


def test_score_category_correct():
    out = score_category(1, 1)
    assert out["correct"] is True
    assert out["cost"] == 0
    assert out["normalized"] == 1.0


def test_score_category_none_is_worst_case():
    out = score_category(None, 1)
    assert out["correct"] is False
    assert out["normalized"] == 0.0


def test_score_criteria_precision_recall():
    out = score_criteria(["1a"], ["1a", "1b"])
    assert out["precision"] == 1.0
    assert out["recall"] == 0.5
    assert out["false_negatives"] == ["1b"]


def test_score_sub_pattern_not_applicable_for_non_cat1():
    out = score_sub_pattern(None, None, predicted_category=4)
    assert out["applicable"] is False
    assert out["correct"] is True


def _perfect_result(case_id, category):
    gt = {
        "category": category,
        "sub_pattern": "1-deliberate" if category == 1 else None,
        "subject_frame": "first_person",
        "criteria_expected": ["1a", "1b"],
        "expected_tools": ["lookup_rubric", "resolve_escalation"],
    }
    final = {
        "category": category,
        "sub_pattern": gt["sub_pattern"],
        "subject_frame": "first_person",
        "criteria_cited": ["1a", "1b"],
        "action": "immediate_escalation",
        "reasoning": "x",
    }
    return {
        "case_id": case_id,
        "ground_truth": gt,
        "case_metadata": {"voice": "v", "difficulty": "easy", "tags": ["t"]},
        "trace": {
            "final_output": final,
            "tool_calls": [
                {"name": "lookup_rubric", "arguments": {}, "result": {}, "iteration": 1},
                {"name": "resolve_escalation", "arguments": {}, "result": {}, "iteration": 2},
            ],
        },
    }


def test_score_case_deterministic_perfect():
    result = _perfect_result("C1", 1)
    scores = score_case_deterministic(result)
    assert scores["category"]["correct"] is True
    assert scores["subject_frame"]["correct"] is True
    assert scores["criteria"]["f1"] == 1.0
    assert scores["tool_use"]["f1"] == 1.0
    assert scores["weighted_score"] > 0.99


def test_compute_aggregates():
    results = [_perfect_result("C1", 1), _perfect_result("C2", 2)]
    for r in results:
        r["scores"] = score_case_deterministic(r)
    agg = compute_aggregates(results)
    assert agg["n"] == 2
    assert agg["headline"]["raw_accuracy"] == 1.0
    assert "confusion_matrix" in agg
    assert agg["confusion_matrix"]["1"]["1"] == 1
