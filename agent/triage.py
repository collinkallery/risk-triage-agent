"""
Ad-hoc triage CLI — classify a single narrative from the command line.

This is the inference counterpart to the eval runner. The runner iterates the
labeled eval set and scores every case; this classifies one arbitrary narrative
and prints the result. It reuses the same `run_agent` loop, the same config, and
the same tools — nothing about the agent changes.

Input (one of, in this precedence):
  --text "..."          inline string
  --file PATH           read the narrative from a file
  (stdin)               piped input, e.g.  cat note.txt | python -m agent.triage

Usage:
  python -m agent.triage --text "I can't keep doing this anymore."
  python -m agent.triage --file note.txt --show-trace
  echo "five years ago I was in a dark place; I'm okay now" | python -m agent.triage
  python -m agent.triage --text "..." --json
  python -m agent.triage --text "..." --mock          # plumbing only, no API call

Optional single-case scoring (no eval set required):
  --expected N          score the predicted category against expected (1-5)
                        using the rubric's asymmetric cost matrix
  --expected-frame F    score the predicted subject frame against expected

Scoring here is intentionally limited to the two cost-matrix dimensions
(category and subject frame). Criteria/tool F1 and the reasoning judge require
the full eval harness — that is what the eval set is for. This flag exists for
quick "did it get this one right, and how wrong was it?" checks on ad-hoc text.

Result goes to stdout; diagnostics go to stderr, so the output stays pipeable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from agent.agent import GeminiClient, MockClient, Trace, run_agent
from agent.tools import lookup_rubric

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_FRAMES = {"first_person", "third_party_clear", "third_party_ambiguous"}


def load_config() -> dict[str, Any]:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Input resolution
# -----------------------------------------------------------------------------


def resolve_narrative(text: str | None, file: Path | None) -> str:
    """Pick the narrative from --text, --file, or stdin (in that order).

    Raises ValueError with a usage hint if no input is available.
    """
    if text is not None:
        return text
    if file is not None:
        return file.read_text(encoding="utf-8").strip()
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    raise ValueError(
        "No narrative provided. Use --text, --file, or pipe text via stdin."
    )


# -----------------------------------------------------------------------------
# Client construction
# -----------------------------------------------------------------------------


def _mock_client() -> MockClient:
    """A MockClient that returns a single, clearly-labeled placeholder.

    For testing CLI plumbing without GCP credentials or API cost. The output
    is a fixed placeholder — it does NOT reflect the input narrative. Never
    treat a --mock result as a real classification.
    """
    placeholder = {
        "category": 5,
        "sub_pattern": None,
        "subject_frame": "first_person",
        "criteria_cited": ["5a", "5b", "5c"],
        "action": "clarify_or_conservative_escalation",
        "reasoning": "[mock mode — fixed placeholder, not a real classification]",
    }
    return MockClient(
        script=[
            {
                "text": "```json\n" + json.dumps(placeholder) + "\n```",
                "tool_calls": [],
                "input_tokens": 0,
                "output_tokens": 0,
            }
        ]
    )


def build_client(mock: bool, config: dict[str, Any]):
    """Return a live Gemini client, or a mock client for plumbing tests."""
    if mock:
        return _mock_client()
    return GeminiClient(
        model=config["model"]["name"],
        temperature=config["model"]["temperature"],
    )


# -----------------------------------------------------------------------------
# Optional single-case scoring (category + frame only)
# -----------------------------------------------------------------------------


def score_against_expected(
    trace: Trace,
    expected_category: int | None,
    expected_frame: str | None,
) -> dict[str, Any] | None:
    """Score the prediction against expected values via the rubric cost matrices.

    Lazy-imports eval.scoring so the agent package stays usable without the
    eval layer unless this path is actually exercised. Returns None if no
    expected values were supplied.
    """
    if expected_category is None and expected_frame is None:
        return None

    from eval.scoring import score_category, score_subject_frame

    final = trace.final_output or {}
    out: dict[str, Any] = {}
    if expected_category is not None:
        out["category"] = score_category(final.get("category"), expected_category)
    if expected_frame is not None:
        out["subject_frame"] = score_subject_frame(final.get("subject_frame"), expected_frame)
    return out


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def _category_meta(category: Any) -> dict[str, Any]:
    """Look up the human-readable name + canonical action for a category id."""
    if isinstance(category, int) and category in {1, 2, 3, 4, 5}:
        rub = lookup_rubric(category)
        if "error" not in rub:
            return {"name": rub.get("name"), "action": rub.get("action")}
    return {"name": None, "action": None}


def render_human(trace: Trace, scores: dict[str, Any] | None, show_trace: bool) -> str:
    """Human-readable rendering of a triage result."""
    final = trace.final_output
    lines: list[str] = []

    if final is None:
        lines.append("No classification produced.")
        lines.append(f"  reason: {trace.terminated_reason}")
        if trace.error:
            lines.append(f"  error:  {trace.error}")
        return "\n".join(lines)

    cat = final.get("category")
    meta = _category_meta(cat)
    cat_label = f"{cat}  ({meta['name']})" if meta["name"] else f"{cat}"
    criteria = final.get("criteria_cited") or []

    lines.append(f"Category:      {cat_label}")
    if final.get("sub_pattern"):
        lines.append(f"Sub-pattern:   {final['sub_pattern']}")
    lines.append(f"Subject frame: {final.get('subject_frame')}")
    lines.append(f"Action:        {final.get('action')}")
    lines.append(f"Criteria:      {', '.join(criteria) if criteria else '(none cited)'}")
    lines.append(f"Reasoning:     {final.get('reasoning', '').strip()}")

    if scores:
        lines.append("")
        lines.append("Scoring vs expected:")
        if "category" in scores:
            c = scores["category"]
            verdict = "correct" if c["correct"] else "incorrect"
            lines.append(
                f"  category:      expected {c['expected']}, got {c['predicted']} "
                f"— {verdict} (cost {c['cost']}, normalized {c['normalized']})"
            )
        if "subject_frame" in scores:
            f = scores["subject_frame"]
            verdict = "correct" if f["correct"] else "incorrect"
            lines.append(
                f"  subject_frame: expected {f['expected']}, got {f['predicted']} "
                f"— {verdict} (cost {f['cost']}, normalized {f['normalized']})"
            )

    if show_trace:
        lines.append("")
        lines.append(f"--- trace ({trace.iterations} iteration(s), {trace.terminated_reason}) ---")
        for i, thought in enumerate(trace.thoughts, start=1):
            lines.append(f"[thought {i}] {thought.strip()}")
        for tc in trace.tool_calls:
            lines.append(f"[tool · iter {tc.iteration}] {tc.name}({json.dumps(tc.arguments)})")

    return "\n".join(lines)


def build_json_payload(trace: Trace, scores: dict[str, Any] | None, show_trace: bool) -> dict[str, Any]:
    """Machine-readable payload for --json."""
    payload: dict[str, Any] = {
        "classification": trace.final_output,
        "meta": {
            "iterations": trace.iterations,
            "terminated_reason": trace.terminated_reason,
            "error": trace.error,
            "latency_seconds": trace.latency_seconds,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
        },
    }
    if scores is not None:
        payload["scores"] = scores
    if show_trace:
        payload["trace"] = {
            "thoughts": trace.thoughts,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "iteration": tc.iteration}
                for tc in trace.tool_calls
            ],
        }
    return payload


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify a single narrative with the behavioral risk triage agent."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--text", default=None, help="Narrative text to classify.")
    src.add_argument("--file", type=Path, default=None, help="Path to a file containing the narrative.")
    parser.add_argument("--mock", action="store_true", help="Use a placeholder MockClient (no API call). Plumbing only.")
    parser.add_argument("--show-trace", action="store_true", help="Include the reasoning trace in the output.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON payload instead of human-readable text.")
    parser.add_argument("--expected", type=int, choices=[1, 2, 3, 4, 5], default=None, help="Expected category to score against.")
    parser.add_argument("--expected-frame", choices=sorted(VALID_FRAMES), default=None, help="Expected subject frame to score against.")
    args = parser.parse_args(argv)

    config = load_config()

    try:
        narrative = resolve_narrative(args.text, args.file)
    except (ValueError, OSError) as e:
        print(f"Input error: {e}", file=sys.stderr)
        return 2

    if args.mock:
        print("MOCK MODE: output is a fixed placeholder, not a real classification.", file=sys.stderr)

    try:
        client = build_client(args.mock, config)
    except Exception as e:
        print(f"Could not initialize the model client: {e}", file=sys.stderr)
        print(
            "For a live run, set GOOGLE_CLOUD_PROJECT and run "
            "`gcloud auth application-default login`. Or pass --mock to test plumbing.",
            file=sys.stderr,
        )
        return 1

    print(f"Classifying narrative ({len(narrative)} chars)...", file=sys.stderr)
    trace = run_agent(
        narrative=narrative,
        client=client,
        case_id="adhoc",
        max_iterations=config["model"]["max_iterations"],
        max_input_tokens=config["model"]["max_input_token_budget"],
    )

    scores = score_against_expected(trace, args.expected, args.expected_frame)

    if args.json:
        print(json.dumps(build_json_payload(trace, scores, args.show_trace), indent=2, default=str))
    else:
        print(render_human(trace, scores, args.show_trace))

    # Non-zero exit if the agent failed to produce a classification.
    return 0 if trace.final_output is not None else 1


if __name__ == "__main__":
    sys.exit(main())
