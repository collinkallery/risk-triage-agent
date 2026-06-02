"""Tests for the LLM-as-judge layer (mock provider only — no API calls)."""

import pytest

from eval.judge import (
    MockJudgeClient,
    _parse_judge_response,
    judge_case,
    make_judge_client,
)


def test_parse_clean_json():
    out = _parse_judge_response('{"grounded": true, "explanation": "ok"}')
    assert out["grounded"] is True
    assert out["explanation"] == "ok"


def test_parse_embedded_json():
    out = _parse_judge_response('Verdict: {"grounded": false, "explanation": "no"} end.')
    assert out["grounded"] is False


def test_parse_garbage_marks_parse_error():
    out = _parse_judge_response("not json at all")
    assert out["grounded"] is None
    assert out.get("parse_error") is True


def test_make_judge_client_mock():
    client = make_judge_client("mock")
    assert isinstance(client, MockJudgeClient)


def test_make_judge_client_bad_provider():
    with pytest.raises(ValueError):
        make_judge_client("bogus")


def test_make_judge_client_defaults_to_anthropic():
    # We don't construct a live client here (no key/SDK); just confirm the
    # default provider string routes to the Anthropic branch by checking that
    # an explicit mock is required to avoid it.
    assert make_judge_client.__defaults__[0] == "anthropic"


def test_judge_case_with_mock():
    result = {
        "ground_truth": {
            "category": 1,
            "sub_pattern": "1-deliberate",
            "subject_frame": "first_person",
            "criteria_expected": ["1a"],
        },
        "trace": {
            "narrative": "I have the pills counted out for tonight.",
            "final_output": {
                "category": 1,
                "criteria_cited": ["1a"],
                "reasoning": "Explicit means and timeframe.",
            },
        },
    }
    verdict = judge_case(result, MockJudgeClient(default_grounded=True))
    assert verdict["grounded"] is True
    assert "input_tokens" in verdict


def test_judge_case_no_final_output_is_skipped():
    result = {"ground_truth": {}, "trace": {"narrative": "x", "final_output": None}}
    verdict = judge_case(result, MockJudgeClient())
    assert verdict.get("skipped") is True
