"""Smoke test: run the agent loop end-to-end with a scripted MockClient."""

import json
from agent.agent import MockClient, run_agent

# Script a realistic 3-turn interaction:
#   Turn 1: model calls lookup_rubric(1)
#   Turn 2: model calls resolve_escalation(1, first_person)
#   Turn 3: model emits final structured output
script = [
    {
        "text": "Reading the narrative. Specific means (pills), timeframe (tonight), goodbye framing. Looks like Cat 1 deliberate. Let me check the rubric.",
        "tool_calls": [{"name": "lookup_rubric", "arguments": {"category_id": 1}}],
        "input_tokens": 3200,
        "output_tokens": 80,
    },
    {
        "text": "Criteria 1a, 1b, 1c met. Sub-pattern 1-deliberate. First-person. Confirming action.",
        "tool_calls": [
            {"name": "resolve_escalation", "arguments": {"category_id": 1, "subject_frame": "first_person"}}
        ],
        "input_tokens": 3800,
        "output_tokens": 50,
    },
    {
        "text": """```json
{
  "category": 1,
  "sub_pattern": "1-deliberate",
  "subject_frame": "first_person",
  "criteria_cited": ["1a", "1b", "1c"],
  "action": "immediate_escalation",
  "reasoning": "Narrative contains explicit means (pills counted out), explicit timeframe (tonight), and implicit goodbye framing. Meets 1a, 1b, 1c. Deliberate sub-pattern — committed and specific. First-person throughout."
}
```""",
        "tool_calls": [],
        "input_tokens": 4100,
        "output_tokens": 180,
    },
]

narrative = (
    "I have the pills counted out on the counter. I just wanted someone to "
    "know before tonight. Thank you for everything you've tried to do."
)

client = MockClient(script=script)
trace = run_agent(narrative, client, case_id="C001")

print(f"Case: {trace.case_id}")
print(f"Iterations: {trace.iterations}")
print(f"Terminated: {trace.terminated_reason}")
print(f"Latency: {trace.latency_seconds}s")
print(f"Input tokens: {trace.input_tokens}, Output tokens: {trace.output_tokens}")
print(f"Tool calls: {len(trace.tool_calls)}")
for tc in trace.tool_calls:
    print(f"  [iter {tc.iteration}] {tc.name}({tc.arguments}) → keys: {list(tc.result.keys())[:5]}")
print()
print("Final output:")
print(json.dumps(trace.final_output, indent=2))
print()
print("Error:", trace.error)

# Also dump full trace as JSON to confirm serialization works
trace_dict = trace.to_dict()
serialized = json.dumps(trace_dict, indent=2, default=str)
print(f"\nFull trace JSON length: {len(serialized)} chars (serializes cleanly)")