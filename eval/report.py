"""
Report generator — turns a scored results JSON into a markdown report.

Usage:
  python -m eval.report results/run_X_scored.json
  python -m eval.report results/run_X_scored.json --output=reports/run_X.md

The report is structured so a reviewer can stop reading at any depth and
still have useful signal:
  1. Headline (3 numbers)
  2. Per-dimension breakdown
  3. Confusion matrix
  4. Slices: by category, difficulty, subject_frame, tag
  5. Worst failures with full traces inline
  6. Ops data: tokens, latency, cost
  7. Run metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Gemini 2.5 Pro pricing (approximate, USD per 1M tokens) — used for
# rough cost estimates only. Update if pricing changes.
INPUT_PRICE_PER_M = 1.25
OUTPUT_PRICE_PER_M = 10.00


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%"


def _f(x: float | None, places: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{places}f}"


def _render_headline(agg: dict[str, Any]) -> str:
    h = agg["headline"]
    return f"""## Headline

| Metric | Value |
|---|---|
| Weighted accuracy | **{_pct(h['weighted_accuracy'])}** |
| Raw accuracy | {_pct(h['raw_accuracy'])} |
| Mean per-case weighted score | {_f(h['mean_weighted_score'])} |

Weighted accuracy applies the asymmetric cost matrix from the rubric \
(false negatives on Cat 1 penalized most heavily). Raw accuracy is the \
fraction of cases where the predicted category exactly matched ground \
truth. The gap between them is informative: a small gap means failures \
were catastrophic, a large gap means failures were mostly adjacent-category \
misses.
"""


def _render_per_dimension(agg: dict[str, Any]) -> str:
    d = agg["per_dimension"]
    sub_acc = _pct(d["sub_pattern_accuracy_cat1_only"]) if d["sub_pattern_accuracy_cat1_only"] is not None else "n/a (no Cat 1 cases)"
    return f"""## Per-Dimension Breakdown

| Dimension | Score |
|---|---|
| Category raw accuracy | {_pct(d['category_raw_accuracy'])} |
| Category weighted accuracy | {_pct(d['category_weighted_accuracy'])} |
| Sub-pattern accuracy (Cat 1 only) | {sub_acc} |
| Subject frame accuracy | {_pct(d['subject_frame_accuracy'])} |
| Criteria citation F1 | {_f(d['criteria_f1_mean'])} |
| Tool use F1 | {_f(d['tool_use_f1_mean'])} |
"""


def _render_confusion_matrix(agg: dict[str, Any]) -> str:
    cm = agg["confusion_matrix"]
    lines = ["## Confusion Matrix", ""]
    lines.append("Rows: ground truth. Columns: predicted. Cell shows count.")
    lines.append("")
    lines.append("|       | Pred Cat1 | Pred Cat2 | Pred Cat3 | Pred Cat4 | Pred Cat5 |")
    lines.append("|---|---|---|---|---|---|")
    for true_cat in ["1", "2", "3", "4", "5"]:
        row = cm[true_cat]
        cells = []
        for pred_cat in ["1", "2", "3", "4", "5"]:
            count = row[pred_cat]
            if true_cat == pred_cat:
                cells.append(f"**{count}**" if count > 0 else "0")
            else:
                cells.append(str(count))
        lines.append(f"| **True Cat{true_cat}** | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _render_slice(title: str, slice_data: dict[str, Any], note: str = "") -> str:
    lines = [f"### {title}", ""]
    if note:
        lines.append(note)
        lines.append("")
    lines.append("| Bucket | n | Raw accuracy | Weighted accuracy | Mean weighted score |")
    lines.append("|---|---|---|---|---|")
    for bucket, stats in slice_data.items():
        lines.append(
            f"| {bucket} | {stats['n']} | {_pct(stats['raw_accuracy'])} | "
            f"{_pct(stats['weighted_accuracy'])} | {_f(stats['mean_weighted_score'])} |"
        )
    return "\n".join(lines) + "\n"


def _render_slices(agg: dict[str, Any]) -> str:
    slices = agg["slices"]
    parts = ["## Slices\n"]
    parts.append(_render_slice("By Category", slices["by_category"]))
    parts.append(_render_slice("By Difficulty", slices["by_difficulty"]))
    parts.append(_render_slice("By Subject Frame", slices["by_subject_frame"]))
    parts.append(
        _render_slice(
            "By Tag",
            slices["by_tag"],
            note=(
                "Tags mark structural properties of cases — over-flagging traps, "
                "calibration cases, etc. Per-tag accuracy surfaces failure-mode-aware "
                "performance information that aggregate accuracy alone hides."
            ),
        )
    )
    return "\n".join(parts)


def _render_judge_stats(agg: dict[str, Any]) -> str:
    if "judge" not in agg:
        return ""
    j = agg["judge"]
    total = j["n_grounded"] + j["n_ungrounded"] + j["n_unparseable"]
    if total == 0:
        return ""
    provider = j.get("provider", "unknown")
    model = j.get("model", "")
    if j.get("cross_model"):
        family_note = (
            f"Cross-model: judged by **{provider}** ({model}), a different model family "
            f"from the Gemini agent, which removes same-model shared bias."
        )
    else:
        family_note = (
            f"Judged by **{provider}** ({model}). This is the same model family as the "
            f"agent, so verdicts may share the agent's biases — a weaker configuration "
            f"than cross-model judging."
        )
    return f"""## Reasoning Judge

LLM-as-judge evaluation of whether the agent's reasoning correctly applied \
its cited criteria, with ground truth visible to the judge. {family_note} \
See README for methodology limitations.

| Verdict | Count |
|---|---|
| Grounded (reasoning correctly applied criteria) | {j['n_grounded']} |
| Ungrounded (reasoning did not correctly apply criteria) | {j['n_ungrounded']} |
| Unparseable judge response | {j['n_unparseable']} |

Cross-tab against category correctness:

| | Category correct | Category incorrect |
|---|---|---|
| Judge said grounded | {j['grounded_and_correct']} | {j['grounded_and_incorrect']} |
| Judge said ungrounded | {j['ungrounded_and_correct']} | (n/a) |

The "grounded and incorrect" cell is the interesting one — those are cases \
where the agent reasoned well from valid criteria but landed on the wrong \
category. They typically indicate either rubric ambiguity or a genuinely \
hard case.

Judge token cost this run: {j['judge_token_cost']['input']:,} input + {j['judge_token_cost']['output']:,} output.
"""


def _render_worst_failures(run: dict[str, Any]) -> str:
    agg = run["aggregates"]
    results_by_id = {r["case_id"]: r for r in run["results"]}

    lines = [
        "## Worst Failures (Top 5)",
        "",
        "Sorted by category cost (descending), then weighted score (ascending). "
        "Full trace included inline — this is where the agent's reasoning "
        "patterns become legible.",
        "",
    ]

    for w in agg["worst_failures"]:
        case_id = w["case_id"]
        full = results_by_id[case_id]
        trace = full["trace"]
        gt = full["ground_truth"]
        final = trace.get("final_output") or {}
        scores = full["scores"]

        lines.append(f"### {case_id} (cost {w['cost']}, weighted {_f(w['weighted_score'])})")
        lines.append("")
        lines.append(f"**Narrative:** {trace['narrative']}")
        lines.append("")
        lines.append(f"**Ground truth:** Cat {gt['category']} · sub_pattern={gt.get('sub_pattern')} · frame={gt['subject_frame']} · expected criteria {gt['criteria_expected']}")
        lines.append("")
        lines.append(f"**Agent output:** Cat {final.get('category')} · sub_pattern={final.get('sub_pattern')} · frame={final.get('subject_frame')} · cited criteria {final.get('criteria_cited')}")
        lines.append("")
        lines.append(f"**Tags:** {', '.join(w['tags']) if w['tags'] else '_no_tag'}")
        lines.append("")

        # Tool call summary
        tool_call_summary = [f"{tc['name']}({tc['arguments']})" for tc in trace.get("tool_calls", [])]
        if tool_call_summary:
            lines.append(f"**Tool calls ({len(tool_call_summary)}):**")
            for tc in tool_call_summary:
                lines.append(f"  - `{tc}`")
            lines.append("")

        if final.get("reasoning"):
            lines.append(f"**Agent reasoning:** {final['reasoning']}")
            lines.append("")

        # Judge verdict if present
        judge = scores.get("reasoning_judge") or {}
        if judge and not judge.get("skipped"):
            grounded = judge.get("grounded")
            verdict = "grounded" if grounded is True else ("ungrounded" if grounded is False else "unparseable")
            lines.append(f"**Judge verdict:** {verdict} — _{judge.get('explanation', '')}_")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _render_ops(run: dict[str, Any]) -> str:
    results = run["results"]
    n = len(results)

    latencies = [r["trace"].get("latency_seconds", 0.0) for r in results]
    latencies.sort()
    if not latencies:
        return ""

    p50 = latencies[n // 2] if n else 0
    p95 = latencies[min(n - 1, int(n * 0.95))] if n else 0

    total_input = sum(r["trace"].get("input_tokens", 0) for r in results)
    total_output = sum(r["trace"].get("output_tokens", 0) for r in results)

    judge_input = 0
    judge_output = 0
    if "judge" in run.get("aggregates", {}):
        judge_input = run["aggregates"]["judge"]["judge_token_cost"]["input"]
        judge_output = run["aggregates"]["judge"]["judge_token_cost"]["output"]

    agent_cost = total_input / 1_000_000 * INPUT_PRICE_PER_M + total_output / 1_000_000 * OUTPUT_PRICE_PER_M
    judge_cost = judge_input / 1_000_000 * INPUT_PRICE_PER_M + judge_output / 1_000_000 * OUTPUT_PRICE_PER_M

    n_terminated_ok = sum(1 for r in results if r["trace"].get("terminated_reason") == "completed")
    n_max_iter = sum(1 for r in results if r["trace"].get("terminated_reason") == "max_iterations")
    n_error = sum(1 for r in results if r["trace"].get("terminated_reason") == "error")
    n_budget = sum(1 for r in results if r["trace"].get("terminated_reason") == "budget_exceeded")

    return f"""## Operations

| Metric | Value |
|---|---|
| Cases run | {n} |
| Completed | {n_terminated_ok} |
| Hit iteration cap | {n_max_iter} |
| Errored | {n_error} |
| Exceeded token budget | {n_budget} |
| Total agent tokens | {total_input:,} input + {total_output:,} output |
| Total judge tokens | {judge_input:,} input + {judge_output:,} output |
| Estimated agent cost (USD) | ${agent_cost:.4f} |
| Estimated judge cost (USD) | ${judge_cost:.4f} |
| **Estimated total cost (USD)** | **${agent_cost + judge_cost:.4f}** |
| Latency p50 | {p50:.2f}s |
| Latency p95 | {p95:.2f}s |
| Mean iterations per case | {sum(r['trace'].get('iterations', 0) for r in results) / max(n, 1):.2f} |

Cost estimates use approximate Gemini 2.5 Pro pricing (${INPUT_PRICE_PER_M}/M input, ${OUTPUT_PRICE_PER_M}/M output).
"""


def _render_metadata(run: dict[str, Any]) -> str:
    m = run["metadata"]
    judge_state = "enabled" if m.get("judge_enabled") else "disabled"
    if m.get("judge_enabled"):
        provider = m.get("judge_provider") or "?"
        model = m.get("judge_model")
        judge_state += f" — {provider}" + (f" ({model})" if model else "")
    return f"""## Run Metadata

- **Timestamp:** {m.get('timestamp')}
- **Agent model:** {m.get('model')} (temperature {m.get('temperature')})
- **Ablation mode:** {m.get('ablation')}
- **Tools available:** {', '.join(m.get('tools_available', []))}
- **Eval set:** {m.get('eval_set_path')} (n={m.get('eval_set_size')})
- **Rubric version:** {m.get('rubric_version')}
- **Judge:** {judge_state}
- **Scored at:** {m.get('scored_at')}
- **Total elapsed:** {m.get('total_elapsed_seconds')}s
"""


# -----------------------------------------------------------------------------
# Top-level
# -----------------------------------------------------------------------------


def render_report(run: dict[str, Any]) -> str:
    agg = run.get("aggregates")
    if not agg:
        return "# Report\n\nNo aggregates found in input. Was this file scored?\n"

    meta = run["metadata"]
    title = f"# Behavioral Risk Triage Eval — {meta.get('ablation', 'unknown')} ablation"

    sections = [
        title,
        f"\n*Run on {meta.get('timestamp')} against {meta.get('eval_set_size')} cases (rubric {meta.get('rubric_version')}).*\n",
        _render_headline(agg),
        _render_per_dimension(agg),
        _render_confusion_matrix(agg),
        _render_slices(agg),
        _render_judge_stats(agg),
        _render_worst_failures(run),
        _render_ops(run),
        _render_metadata(run),
    ]
    return "\n".join(s for s in sections if s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a scored run as markdown.")
    parser.add_argument("scored_path", type=Path, help="Path to a scored JSON file.")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown path.")
    args = parser.parse_args()

    with open(args.scored_path) as f:
        run = json.load(f)

    if args.output is None:
        stem = args.scored_path.stem
        if stem.endswith("_scored"):
            stem = stem[: -len("_scored")]
        args.output = PROJECT_ROOT / "reports" / f"{stem}.md"

    md = render_report(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(md)
    print(f"Report written to: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())