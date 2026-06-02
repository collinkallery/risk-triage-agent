"""
Deterministic scoring for the behavioral risk triage eval.

Five dimensions, four deterministic:

  1. Category cost — asymmetric cost matrix from rubric. 0 = correct,
     higher = worse. Maps to weighted accuracy (1 - cost/max_cost).
  2. Sub-pattern accuracy — Cat 1 only. Binary.
  3. Subject frame cost — smaller matrix from rubric. 0–2 per case.
  4. Criteria precision/recall — set comparison between cited and expected.
  5. Tool use precision/recall — set comparison between tools used and
     expected.

The LLM judge handles the fifth dimension (reasoning quality) separately
in judge.py.

Output shape: each case gets a `scores` dict with all dimensions plus a
weighted_score in [0, 1]. Aggregates are computed across the run by the
orchestrator in score.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Rubric data needed for scoring (cost matrices)
# -----------------------------------------------------------------------------


def load_rubric() -> dict[str, Any]:
    with open(PROJECT_ROOT / "data" / "rubric.yaml") as f:
        return yaml.safe_load(f)


_RUBRIC = load_rubric()
_CATEGORY_COST: dict[str, dict[str, int]] = _RUBRIC["confusion_cost"]
_FRAME_COST: dict[str, dict[str, int]] = _RUBRIC["subject_frame_cost"]

# Max possible category cost = worst cell in the matrix. Used to normalize.
_MAX_CATEGORY_COST = max(v for row in _CATEGORY_COST.values() for v in row.values())
_MAX_FRAME_COST = max(v for row in _FRAME_COST.values() for v in row.values())


# -----------------------------------------------------------------------------
# Per-dimension scoring functions
# -----------------------------------------------------------------------------


def score_category(predicted: int | None, expected: int) -> dict[str, Any]:
    """Score category prediction via the asymmetric cost matrix.

    Returns:
      {
        "predicted": int | None,
        "expected": int,
        "cost": int (0 = correct, higher = worse, _MAX_CATEGORY_COST = worst),
        "correct": bool,
        "normalized": float in [0, 1] where 1 = correct, 0 = worst possible
      }
    """
    if predicted is None:
        # No prediction (agent errored out). Treat as worst-case.
        return {
            "predicted": None,
            "expected": expected,
            "cost": _MAX_CATEGORY_COST,
            "correct": False,
            "normalized": 0.0,
        }

    cost = _CATEGORY_COST.get(str(expected), {}).get(str(predicted))
    if cost is None:
        # Defensive: predicted category outside 1–5. Treat as worst-case.
        return {
            "predicted": predicted,
            "expected": expected,
            "cost": _MAX_CATEGORY_COST,
            "correct": False,
            "normalized": 0.0,
            "error": f"Predicted category {predicted} not in cost matrix.",
        }

    return {
        "predicted": predicted,
        "expected": expected,
        "cost": cost,
        "correct": cost == 0,
        "normalized": round(1.0 - (cost / _MAX_CATEGORY_COST), 3),
    }


def score_sub_pattern(predicted: str | None, expected: str | None, predicted_category: int | None) -> dict[str, Any]:
    """Score sub-pattern. Only applies to Cat 1 cases.

    Returns N/A for non-Cat-1 ground truth. For Cat 1, binary correct/incorrect.
    """
    # Ground truth is non-Cat-1: sub_pattern should be null in both.
    if expected is None:
        return {
            "predicted": predicted,
            "expected": None,
            "applicable": False,
            "correct": True,  # vacuously
            "note": "Not Cat 1 — sub_pattern not applicable.",
        }

    # Ground truth is Cat 1 but agent didn't predict Cat 1.
    if predicted_category != 1:
        return {
            "predicted": predicted,
            "expected": expected,
            "applicable": True,
            "correct": False,
            "note": "Agent did not predict Cat 1, so sub_pattern not assessable on its terms.",
        }

    return {
        "predicted": predicted,
        "expected": expected,
        "applicable": True,
        "correct": predicted == expected,
    }


def score_subject_frame(predicted: str | None, expected: str) -> dict[str, Any]:
    """Score subject frame via the smaller frame cost matrix."""
    valid = {"first_person", "third_party_clear", "third_party_ambiguous"}
    if predicted is None or predicted not in valid:
        return {
            "predicted": predicted,
            "expected": expected,
            "cost": _MAX_FRAME_COST,
            "correct": False,
            "normalized": 0.0,
        }

    cost = _FRAME_COST.get(expected, {}).get(predicted)
    if cost is None:
        return {
            "predicted": predicted,
            "expected": expected,
            "cost": _MAX_FRAME_COST,
            "correct": False,
            "normalized": 0.0,
            "error": f"Predicted frame {predicted} not in cost matrix.",
        }

    return {
        "predicted": predicted,
        "expected": expected,
        "cost": cost,
        "correct": cost == 0,
        "normalized": round(1.0 - (cost / _MAX_FRAME_COST), 3),
    }


def _set_precision_recall(predicted: list[str], expected: list[str]) -> dict[str, Any]:
    """Set-based precision and recall with safe handling of empty sets."""
    pred_set = set(predicted or [])
    exp_set = set(expected or [])
    tp = len(pred_set & exp_set)

    # Precision: of what was predicted, how much was expected?
    if not pred_set:
        # No predictions. Precision is 1.0 by convention if expected is also empty,
        # else 0.0 (the agent failed to predict anything correct).
        precision = 1.0 if not exp_set else 0.0
    else:
        precision = tp / len(pred_set)

    # Recall: of what was expected, how much was predicted?
    if not exp_set:
        # Nothing was expected. Recall is 1.0 by convention.
        recall = 1.0
    else:
        recall = tp / len(exp_set)

    # F1 with safe division.
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "predicted": sorted(pred_set),
        "expected": sorted(exp_set),
        "true_positives": sorted(pred_set & exp_set),
        "false_positives": sorted(pred_set - exp_set),
        "false_negatives": sorted(exp_set - pred_set),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def score_criteria(predicted: list[str] | None, expected: list[str]) -> dict[str, Any]:
    return _set_precision_recall(predicted or [], expected)


def score_tool_use(actual_tool_names: list[str], expected: list[str]) -> dict[str, Any]:
    """Score tool use via set comparison.

    actual_tool_names is the list of tool names from the trace's tool_calls,
    deduplicated (we count each tool as either used or not, not how many
    times — multiple lookups of the same category are fine).
    """
    return _set_precision_recall(list(set(actual_tool_names)), expected)


# -----------------------------------------------------------------------------
# Per-case orchestration (deterministic dimensions only)
# -----------------------------------------------------------------------------


def score_case_deterministic(result: dict[str, Any]) -> dict[str, Any]:
    """Score a single result on the four deterministic dimensions.

    `result` is one element of the runner's `results` list. Returns a
    `scores` dict to attach to that result.
    """
    ground_truth = result["ground_truth"]
    trace = result["trace"]
    final_output = trace.get("final_output") or {}

    predicted_category = final_output.get("category")
    predicted_sub_pattern = final_output.get("sub_pattern")
    predicted_frame = final_output.get("subject_frame")
    predicted_criteria = final_output.get("criteria_cited") or []

    actual_tool_names = [tc["name"] for tc in trace.get("tool_calls", [])]

    cat_score = score_category(predicted_category, ground_truth["category"])
    sub_score = score_sub_pattern(
        predicted_sub_pattern,
        ground_truth.get("sub_pattern"),
        predicted_category,
    )
    frame_score = score_subject_frame(predicted_frame, ground_truth["subject_frame"])
    crit_score = score_criteria(predicted_criteria, ground_truth["criteria_expected"])
    tool_score = score_tool_use(actual_tool_names, ground_truth["expected_tools"])

    # Per-case weighted score: normalized category × 0.5, normalized frame × 0.2,
    # criteria F1 × 0.2, tool F1 × 0.1. Category dominates because it's the
    # primary triage decision; frame and criteria are secondary; tool use is
    # process not outcome.
    weighted = (
        cat_score["normalized"] * 0.5
        + frame_score["normalized"] * 0.2
        + crit_score["f1"] * 0.2
        + tool_score["f1"] * 0.1
    )

    return {
        "category": cat_score,
        "sub_pattern": sub_score,
        "subject_frame": frame_score,
        "criteria": crit_score,
        "tool_use": tool_score,
        "weighted_score": round(weighted, 3),
        # judge score added later by judge.py if enabled
        "reasoning_judge": None,
    }


# -----------------------------------------------------------------------------
# Aggregates across a run
# -----------------------------------------------------------------------------


def compute_aggregates(scored_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up per-case scores into run-level aggregates with all four slices."""
    n = len(scored_results)
    if n == 0:
        return {"n": 0, "note": "No results to aggregate."}

    # Headline: weighted accuracy across all cases.
    total_cost = sum(r["scores"]["category"]["cost"] for r in scored_results)
    max_possible_cost = n * _MAX_CATEGORY_COST
    weighted_accuracy = round(1.0 - (total_cost / max_possible_cost), 3) if max_possible_cost else 0.0

    # Raw accuracy (ignoring cost matrix).
    n_correct = sum(1 for r in scored_results if r["scores"]["category"]["correct"])
    raw_accuracy = round(n_correct / n, 3)

    # Per-dimension means.
    frame_correct = sum(1 for r in scored_results if r["scores"]["subject_frame"]["correct"])
    crit_f1_mean = round(sum(r["scores"]["criteria"]["f1"] for r in scored_results) / n, 3)
    tool_f1_mean = round(sum(r["scores"]["tool_use"]["f1"] for r in scored_results) / n, 3)
    weighted_mean = round(sum(r["scores"]["weighted_score"] for r in scored_results) / n, 3)

    # Sub-pattern accuracy (Cat 1 cases only).
    cat1_cases = [r for r in scored_results if r["scores"]["sub_pattern"]["applicable"]]
    if cat1_cases:
        sub_correct = sum(1 for r in cat1_cases if r["scores"]["sub_pattern"]["correct"])
        sub_accuracy = round(sub_correct / len(cat1_cases), 3)
    else:
        sub_accuracy = None

    # Confusion matrix.
    confusion: dict[str, dict[str, int]] = {str(i): {str(j): 0 for j in range(1, 6)} for i in range(1, 6)}
    for r in scored_results:
        e = r["scores"]["category"]["expected"]
        p = r["scores"]["category"]["predicted"]
        if p in {1, 2, 3, 4, 5}:
            confusion[str(e)][str(p)] += 1

    # Slices.
    def _slice_by(key_fn):
        buckets: dict[str, list[dict[str, Any]]] = {}
        for r in scored_results:
            keys = key_fn(r)
            keys = [keys] if isinstance(keys, str) else keys
            for k in keys:
                buckets.setdefault(k, []).append(r)
        return {
            k: {
                "n": len(rs),
                "raw_accuracy": round(sum(1 for r in rs if r["scores"]["category"]["correct"]) / len(rs), 3),
                "weighted_accuracy": round(
                    1.0 - (sum(r["scores"]["category"]["cost"] for r in rs) / (len(rs) * _MAX_CATEGORY_COST)),
                    3,
                ),
                "mean_weighted_score": round(sum(r["scores"]["weighted_score"] for r in rs) / len(rs), 3),
            }
            for k, rs in sorted(buckets.items())
        }

    by_category = _slice_by(lambda r: f"Cat{r['ground_truth']['category']}")
    by_difficulty = _slice_by(lambda r: r["case_metadata"].get("difficulty", "unknown"))
    by_frame = _slice_by(lambda r: r["ground_truth"]["subject_frame"])
    by_tag = _slice_by(lambda r: r["case_metadata"].get("tags") or ["_no_tag"])

    # Worst failures — sort by category cost descending, then by weighted score asc.
    worst = sorted(
        scored_results,
        key=lambda r: (-r["scores"]["category"]["cost"], r["scores"]["weighted_score"]),
    )[:5]
    worst_summaries = [
        {
            "case_id": r["case_id"],
            "expected_category": r["scores"]["category"]["expected"],
            "predicted_category": r["scores"]["category"]["predicted"],
            "cost": r["scores"]["category"]["cost"],
            "weighted_score": r["scores"]["weighted_score"],
            "tags": r["case_metadata"].get("tags", []),
        }
        for r in worst
    ]

    return {
        "n": n,
        "headline": {
            "weighted_accuracy": weighted_accuracy,
            "raw_accuracy": raw_accuracy,
            "mean_weighted_score": weighted_mean,
        },
        "per_dimension": {
            "category_raw_accuracy": raw_accuracy,
            "category_weighted_accuracy": weighted_accuracy,
            "sub_pattern_accuracy_cat1_only": sub_accuracy,
            "subject_frame_accuracy": round(frame_correct / n, 3),
            "criteria_f1_mean": crit_f1_mean,
            "tool_use_f1_mean": tool_f1_mean,
        },
        "confusion_matrix": confusion,
        "slices": {
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "by_subject_frame": by_frame,
            "by_tag": by_tag,
        },
        "worst_failures": worst_summaries,
    }