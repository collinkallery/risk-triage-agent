"""Tests for the agent's final-output JSON extraction and validation."""

from agent.agent import _extract_json_block, _is_valid_final_output


def test_extract_fenced_json():
    text = 'prose\n```json\n{"category": 1}\n```\nmore'
    assert _extract_json_block(text) == {"category": 1}


def test_extract_balanced_braces_without_fence():
    text = 'The answer is {"category": 2, "x": [1, 2]} done.'
    assert _extract_json_block(text) == {"category": 2, "x": [1, 2]}


def test_extract_returns_none_when_no_json():
    assert _extract_json_block("no json here") is None
    assert _extract_json_block("") is None


def test_valid_final_output_requires_all_fields():
    complete = {
        "category": 1,
        "sub_pattern": "1-deliberate",
        "subject_frame": "first_person",
        "criteria_cited": ["1a"],
        "action": "immediate_escalation",
        "reasoning": "x",
    }
    assert _is_valid_final_output(complete) is True


def test_invalid_final_output_missing_field():
    assert _is_valid_final_output({"category": 1}) is False
    assert _is_valid_final_output("not a dict") is False
