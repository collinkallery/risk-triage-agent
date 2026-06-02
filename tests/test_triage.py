"""Tests for the ad-hoc triage CLI. Mock-based; no credentials or network."""

from __future__ import annotations

import io
import json

import pytest

from agent.agent import MockClient, run_agent
from agent.triage import (
    build_json_payload,
    render_human,
    resolve_narrative,
    score_against_expected,
)


def _cat1_script() -> list[dict]:
    """A scripted 3-turn Cat 1 interaction (mirrors smoke_test)."""
    final = {
        "category": 1,
        "sub_pattern": "1-deliberate",
        "subject_frame": "first_person",
        "criteria_cited": ["1a", "1b", "1c"],
        "action": "immediate_escalation",
        "reasoning": "Means, timeframe, and goodbye framing present. Meets 1a/1b/1c.",
    }
    return [
        {
            "text": "Checking the rubric.",
            "tool_calls": [{"name": "lookup_rubric", "arguments": {"category_id": 1}}],
            "input_tokens": 3200,
            "output_tokens": 80,
        },
        {
            "text": "Confirming action.",
            "tool_calls": [
                {"name": "resolve_escalation", "arguments": {"category_id": 1, "subject_frame": "first_person"}}
            ],
            "input_tokens": 3800,
            "output_tokens": 50,
        },
        {
            "text": "```json\n" + json.dumps(final) + "\n```",
            "tool_calls": [],
            "input_tokens": 4100,
            "output_tokens": 180,
        },
    ]


def _run_cat1():
    return run_agent("placeholder narrative", MockClient(script=_cat1_script()), case_id="adhoc")


# -----------------------------------------------------------------------------
# Input resolution
# -----------------------------------------------------------------------------


def test_resolve_narrative_prefers_text():
    assert resolve_narrative("hello", None) == "hello"


def test_resolve_narrative_reads_file(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("from a file", encoding="utf-8")
    assert resolve_narrative(None, p) == "from a file"


def test_resolve_narrative_errors_without_input(monkeypatch):
    # Simulate a tty (no piped stdin) so it should raise.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    with pytest.raises(ValueError):
        resolve_narrative(None, None)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def test_render_human_includes_core_fields():
    trace = _run_cat1()
    out = render_human(trace, scores=None, show_trace=False)
    assert "Category:" in out
    assert "Imminent Risk" in out          # name enrichment via lookup_rubric
    assert "immediate_escalation" in out
    assert "1a, 1b, 1c" in out


def test_render_human_show_trace_includes_tool_calls():
    trace = _run_cat1()
    out = render_human(trace, scores=None, show_trace=True)
    assert "--- trace" in out
    assert "lookup_rubric" in out
    assert "resolve_escalation" in out


def test_render_human_handles_no_final_output():
    # Empty script → MockClient placeholder still produces output, so force a
    # failure trace by exhausting iterations against a degenerate client.
    bad_client = MockClient(script=[{"text": None, "tool_calls": [], "input_tokens": 0, "output_tokens": 0}])
    trace = run_agent("x", bad_client, case_id="adhoc", max_iterations=1)
    out = render_human(trace, scores=None, show_trace=False)
    assert "No classification produced." in out


def test_json_payload_shape():
    trace = _run_cat1()
    payload = build_json_payload(trace, scores=None, show_trace=True)
    assert payload["classification"]["category"] == 1
    assert payload["meta"]["terminated_reason"] == "completed"
    assert "trace" in payload
    assert payload["trace"]["tool_calls"][0]["name"] == "lookup_rubric"


# -----------------------------------------------------------------------------
# Optional single-case scoring
# -----------------------------------------------------------------------------


def test_score_against_expected_returns_none_when_unset():
    assert score_against_expected(_run_cat1(), None, None) is None


def test_score_against_expected_correct_category_zero_cost():
    scores = score_against_expected(_run_cat1(), expected_category=1, expected_frame=None)
    assert scores["category"]["correct"] is True
    assert scores["category"]["cost"] == 0


def test_score_against_expected_wrong_category_has_cost():
    # Predicted Cat 1, expected Cat 4 → a false-positive escalation, nonzero cost.
    scores = score_against_expected(_run_cat1(), expected_category=4, expected_frame=None)
    assert scores["category"]["correct"] is False
    assert scores["category"]["cost"] > 0


def test_score_against_expected_frame():
    scores = score_against_expected(_run_cat1(), expected_category=None, expected_frame="first_person")
    assert scores["subject_frame"]["correct"] is True
