"""Tests for the three agent tools and the dispatch layer."""

from agent.tools import dispatch, lookup_rubric, resolve_escalation, search_reference


def test_lookup_rubric_valid():
    out = lookup_rubric(1)
    assert out["id"] == 1
    assert "criteria" in out and out["criteria"]


def test_lookup_rubric_invalid_id():
    assert "error" in lookup_rubric(99)
    assert "error" in lookup_rubric("1")  # wrong type


def test_search_reference_returns_matches():
    out = search_reference("euphemistic finality language")
    assert "results" in out
    assert len(out["results"]) >= 1
    assert all("content" in r for r in out["results"])


def test_search_reference_no_match():
    out = search_reference("zzzqqq nonexistenttoken")
    assert out["results"] == []
    assert "note" in out


def test_search_reference_empty_query():
    assert "error" in search_reference("   ")


def test_resolve_escalation_valid():
    out = resolve_escalation(1, "first_person")
    assert out["category_id"] == 1
    assert out["action"] == "immediate_escalation"


def test_resolve_escalation_bad_frame():
    assert "error" in resolve_escalation(1, "not_a_frame")


def test_dispatch_unknown_tool():
    assert "error" in dispatch("nonexistent_tool", {})


def test_dispatch_bad_arguments():
    # Wrong kwarg name → TypeError caught and returned as structured error.
    out = dispatch("lookup_rubric", {"wrong_kwarg": 1})
    assert "error" in out


def test_dispatch_routes_correctly():
    out = dispatch("resolve_escalation", {"category_id": 2, "subject_frame": "first_person"})
    assert out["action"] == "warm_handoff"
