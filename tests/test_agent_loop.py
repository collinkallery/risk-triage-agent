"""Agent-loop tests using the deterministic MockClient — no API calls."""

from agent.agent import MockClient, run_agent


def _final_output_script():
    """A scripted 3-turn interaction ending in a valid Cat 1 classification."""
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
            "text": (
                '```json\n{"category": 1, "sub_pattern": "1-deliberate", '
                '"subject_frame": "first_person", "criteria_cited": ["1a", "1b", "1c"], '
                '"action": "immediate_escalation", "reasoning": "Means, timeframe, finality."}\n```'
            ),
            "tool_calls": [],
            "input_tokens": 4100,
            "output_tokens": 180,
        },
    ]


def test_loop_reaches_valid_final_output():
    trace = run_agent("I have the pills counted out for tonight.", MockClient(_final_output_script()), case_id="T1")
    assert trace.terminated_reason == "completed"
    assert trace.final_output["category"] == 1
    assert trace.final_output["sub_pattern"] == "1-deliberate"
    assert [tc.name for tc in trace.tool_calls] == ["lookup_rubric", "resolve_escalation"]
    assert trace.iterations == 3


def test_trace_serializes_to_dict():
    trace = run_agent("narrative", MockClient(_final_output_script()), case_id="T1")
    d = trace.to_dict()
    assert d["case_id"] == "T1"
    assert isinstance(d["tool_calls"], list)


def test_max_iterations_termination():
    # A client that always asks for a tool call never commits — should hit the cap.
    looping = [
        {"text": "again", "tool_calls": [{"name": "lookup_rubric", "arguments": {"category_id": 1}}],
         "input_tokens": 10, "output_tokens": 5}
        for _ in range(20)
    ]
    trace = run_agent("n", MockClient(looping), case_id="T2", max_iterations=3)
    assert trace.terminated_reason == "max_iterations"
    assert trace.iterations == 3


def test_budget_exceeded_termination():
    big = [
        {"text": "thinking", "tool_calls": [{"name": "lookup_rubric", "arguments": {"category_id": 1}}],
         "input_tokens": 10_000, "output_tokens": 5}
        for _ in range(10)
    ]
    trace = run_agent("n", MockClient(big), case_id="T3", max_input_tokens=5_000)
    assert trace.terminated_reason == "budget_exceeded"
    assert "budget" in trace.error.lower()


def test_exhausted_script_falls_back_to_placeholder():
    # Empty script → MockClient emits a terminal placeholder final output.
    trace = run_agent("n", MockClient([]), case_id="T4")
    assert trace.final_output is not None
    assert trace.final_output["category"] == 5
