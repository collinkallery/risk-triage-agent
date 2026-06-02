"""
Ablation analysis — compares scored runs across tool configurations.

Usage:
  python -m eval.ablation \\
      --all=results/run_X_all_scored.json \\
      --rubric-only=results/run_X_rubric_only_scored.json \\
      --none=results/run_X_none_scored.json \\
      --output=reports/ablation.md

Treats `--all` as the baseline; deltas are computed against it. If a run
is missing, the corresponding column is omitted gracefully.

The output is a single markdown report:
  - Headline comparison (weighted accuracy, raw accuracy, mean weighted)
  - Per-dimension comparison
  - Per-category and per-tag comparisons (where ablation differences
    typically show up)
  - Ops comparison (cost, latency, tokens) — meaningful because no-tools
    runs are often much cheaper but worse
  - Notes section interpreting the deltas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Loaders and helpers
# -----------------------------------------------------------------------------


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%"


def _f(x: float | None, places: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{places}f}"


def _delta(value: float | None, baseline: float | None) -> str:
    """Format a delta vs baseline. Empty string if either is None."""
    if value is None or baseline is None:
        return ""
    diff = value - baseline
    if abs(diff) < 0.001:
        return "  (—)"
    sign = "+" if diff > 0 else ""
    return f"  ({sign}{diff * 100:.1f}pp)"


def _delta_raw(value: float | None, baseline: float | None) -> str:
    """Format a non-percentage delta."""
    if value is None or baseline is None:
        return ""
    diff = value - baseline
    if abs(diff) < 0.001:
        return "  (—)"
    sign = "+" if diff > 0 else ""
    return f"  ({sign}{diff:.3f})"


# -----------------------------------------------------------------------------
# Section renderers
# -----------------------------------------------------------------------------


def _render_headline_compare(runs: dict[str, dict[str, Any] | None]) -> str:
    """Compare headline numbers across runs."""
    rows = ["## Headline Comparison", ""]
    rows.append("| Metric | Full tools | Rubric only | No tools |")
    rows.append("|---|---|---|---|")

    baseline = runs.get("all")
    base_h = baseline["aggregates"]["headline"] if baseline else None

    def cell(run, key, is_pct=True):
        if not run:
            return "—"
        v = run["aggregates"]["headline"][key]
        base_v = base_h[key] if base_h else None
        delta = _delta(v, base_v) if is_pct else _delta_raw(v, base_v)
        return f"{_pct(v) if is_pct else _f(v)}{delta}"

    rows.append(
        f"| Weighted accuracy | {cell(runs.get('all'), 'weighted_accuracy')} | "
        f"{cell(runs.get('rubric_only'), 'weighted_accuracy')} | "
        f"{cell(runs.get('none'), 'weighted_accuracy')} |"
    )
    rows.append(
        f"| Raw accuracy | {cell(runs.get('all'), 'raw_accuracy')} | "
        f"{cell(runs.get('rubric_only'), 'raw_accuracy')} | "
        f"{cell(runs.get('none'), 'raw_accuracy')} |"
    )
    rows.append(
        f"| Mean weighted score | {cell(runs.get('all'), 'mean_weighted_score', is_pct=False)} | "
        f"{cell(runs.get('rubric_only'), 'mean_weighted_score', is_pct=False)} | "
        f"{cell(runs.get('none'), 'mean_weighted_score', is_pct=False)} |"
    )
    rows.append("")
    rows.append("*Deltas are vs the full-tools baseline. `pp` = percentage points.*")
    return "\n".join(rows) + "\n"


def _render_per_dimension_compare(runs: dict[str, dict[str, Any] | None]) -> str:
    rows = ["## Per-Dimension Comparison", ""]
    rows.append("| Dimension | Full tools | Rubric only | No tools |")
    rows.append("|---|---|---|---|")

    baseline = runs.get("all")
    base_d = baseline["aggregates"]["per_dimension"] if baseline else None

    def cell(run, key, is_pct=True):
        if not run:
            return "—"
        v = run["aggregates"]["per_dimension"].get(key)
        base_v = base_d.get(key) if base_d else None
        delta = _delta(v, base_v) if is_pct else _delta_raw(v, base_v)
        return f"{_pct(v) if is_pct else _f(v)}{delta}"

    metrics = [
        ("category_raw_accuracy", "Category raw accuracy", True),
        ("category_weighted_accuracy", "Category weighted accuracy", True),
        ("sub_pattern_accuracy_cat1_only", "Sub-pattern accuracy (Cat 1)", True),
        ("subject_frame_accuracy", "Subject frame accuracy", True),
        ("criteria_f1_mean", "Criteria F1", False),
        ("tool_use_f1_mean", "Tool use F1", False),
    ]
    for key, label, is_pct in metrics:
        rows.append(
            f"| {label} | "
            f"{cell(runs.get('all'), key, is_pct)} | "
            f"{cell(runs.get('rubric_only'), key, is_pct)} | "
            f"{cell(runs.get('none'), key, is_pct)} |"
        )
    return "\n".join(rows) + "\n"


def _render_slice_compare(runs: dict[str, dict[str, Any] | None], slice_key: str, title: str) -> str:
    """Compare a slice across runs. Slice_key is 'by_category', 'by_tag', etc."""
    rows = [f"### {title}", ""]
    rows.append("| Bucket | n | Full tools weighted | Rubric only weighted | No tools weighted |")
    rows.append("|---|---|---|---|---|")

    # Build a union of all bucket keys present across runs.
    all_buckets: dict[str, int] = {}
    for run in runs.values():
        if not run:
            continue
        for k, stats in run["aggregates"]["slices"][slice_key].items():
            all_buckets[k] = stats["n"]

    baseline = runs.get("all")

    def bucket_val(run, bucket):
        if not run:
            return None
        return run["aggregates"]["slices"][slice_key].get(bucket, {}).get("weighted_accuracy")

    for bucket in sorted(all_buckets):
        n = all_buckets[bucket]
        base_v = bucket_val(baseline, bucket)
        all_v = bucket_val(runs.get("all"), bucket)
        rub_v = bucket_val(runs.get("rubric_only"), bucket)
        none_v = bucket_val(runs.get("none"), bucket)

        def fmt(v):
            if v is None:
                return "—"
            return f"{_pct(v)}{_delta(v, base_v) if base_v is not None else ''}"

        rows.append(f"| {bucket} | {n} | {fmt(all_v)} | {fmt(rub_v)} | {fmt(none_v)} |")
    return "\n".join(rows) + "\n"


def _render_ops_compare(runs: dict[str, dict[str, Any] | None]) -> str:
    rows = ["## Operations Comparison", ""]
    rows.append("| Metric | Full tools | Rubric only | No tools |")
    rows.append("|---|---|---|---|")

    def cell(run, fn, fmt=lambda x: str(x)):
        if not run:
            return "—"
        return fmt(fn(run))

    def total_input(run):
        return sum(r["trace"].get("input_tokens", 0) for r in run["results"])

    def total_output(run):
        return sum(r["trace"].get("output_tokens", 0) for r in run["results"])

    def mean_iter(run):
        n = len(run["results"]) or 1
        return sum(r["trace"].get("iterations", 0) for r in run["results"]) / n

    def mean_tool_calls(run):
        n = len(run["results"]) or 1
        return sum(len(r["trace"].get("tool_calls", [])) for r in run["results"]) / n

    def p50(run):
        ls = sorted(r["trace"].get("latency_seconds", 0.0) for r in run["results"])
        return ls[len(ls) // 2] if ls else 0

    rows.append(
        f"| Total input tokens | "
        f"{cell(runs.get('all'), total_input, lambda x: f'{x:,}')} | "
        f"{cell(runs.get('rubric_only'), total_input, lambda x: f'{x:,}')} | "
        f"{cell(runs.get('none'), total_input, lambda x: f'{x:,}')} |"
    )
    rows.append(
        f"| Total output tokens | "
        f"{cell(runs.get('all'), total_output, lambda x: f'{x:,}')} | "
        f"{cell(runs.get('rubric_only'), total_output, lambda x: f'{x:,}')} | "
        f"{cell(runs.get('none'), total_output, lambda x: f'{x:,}')} |"
    )
    rows.append(
        f"| Mean iterations / case | "
        f"{cell(runs.get('all'), mean_iter, lambda x: f'{x:.2f}')} | "
        f"{cell(runs.get('rubric_only'), mean_iter, lambda x: f'{x:.2f}')} | "
        f"{cell(runs.get('none'), mean_iter, lambda x: f'{x:.2f}')} |"
    )
    rows.append(
        f"| Mean tool calls / case | "
        f"{cell(runs.get('all'), mean_tool_calls, lambda x: f'{x:.2f}')} | "
        f"{cell(runs.get('rubric_only'), mean_tool_calls, lambda x: f'{x:.2f}')} | "
        f"{cell(runs.get('none'), mean_tool_calls, lambda x: f'{x:.2f}')} |"
    )
    rows.append(
        f"| Latency p50 | "
        f"{cell(runs.get('all'), p50, lambda x: f'{x:.2f}s')} | "
        f"{cell(runs.get('rubric_only'), p50, lambda x: f'{x:.2f}s')} | "
        f"{cell(runs.get('none'), p50, lambda x: f'{x:.2f}s')} |"
    )
    return "\n".join(rows) + "\n"


def _render_interpretation(runs: dict[str, dict[str, Any] | None]) -> str:
    """Auto-generate a short interpretive section based on the deltas."""
    rows = ["## Interpretation", ""]

    baseline = runs.get("all")
    if not baseline:
        rows.append("_Baseline (full-tools) run missing — cannot interpret deltas._")
        return "\n".join(rows) + "\n"

    base_wa = baseline["aggregates"]["headline"]["weighted_accuracy"]
    observations = []

    for label, run in [("rubric only", runs.get("rubric_only")), ("no tools", runs.get("none"))]:
        if not run:
            continue
        wa = run["aggregates"]["headline"]["weighted_accuracy"]
        delta = (wa - base_wa) * 100
        if abs(delta) < 1:
            observations.append(
                f"- The **{label}** run matched the full-tools baseline within 1pp on weighted accuracy "
                f"({_pct(wa)} vs {_pct(base_wa)}). The omitted tool(s) did not meaningfully help on this eval set."
            )
        elif delta < 0:
            observations.append(
                f"- The **{label}** run was {abs(delta):.1f}pp worse than the full-tools baseline "
                f"({_pct(wa)} vs {_pct(base_wa)}). The omitted tool(s) contributed measurable signal."
            )
        else:
            observations.append(
                f"- Notably, the **{label}** run beat the full-tools baseline by {delta:.1f}pp "
                f"({_pct(wa)} vs {_pct(base_wa)}). This may indicate that the omitted tool(s) introduced "
                f"distractor information or that the agent over-trusted retrieved content."
            )

    # Look at the by_tag slice for the kinds of cases that benefit most/least from tools.
    if "rubric_only" in runs and runs["rubric_only"]:
        rub = runs["rubric_only"]["aggregates"]["slices"]["by_tag"]
        base_tags = baseline["aggregates"]["slices"]["by_tag"]
        biggest_drops = []
        for tag, stats in rub.items():
            base_stats = base_tags.get(tag)
            if base_stats and base_stats["n"] >= 2:
                drop = stats["weighted_accuracy"] - base_stats["weighted_accuracy"]
                biggest_drops.append((drop, tag, base_stats["n"], base_stats["weighted_accuracy"], stats["weighted_accuracy"]))
        biggest_drops.sort()  # most negative first
        if biggest_drops and biggest_drops[0][0] < -0.05:
            d, tag, n, base_v, rub_v = biggest_drops[0]
            observations.append(
                f"- Without `search_reference`, the agent's performance on **{tag}** cases dropped from "
                f"{_pct(base_v)} to {_pct(rub_v)} (n={n}). This suggests the reference corpus was "
                f"contributing specifically to {tag.replace('_', ' ')} reasoning."
            )

    if not observations:
        observations.append("- No notable deltas to interpret.")

    rows.extend(observations)
    return "\n".join(rows) + "\n"


def _render_metadata_compare(runs: dict[str, dict[str, Any] | None]) -> str:
    rows = ["## Run Metadata", ""]
    for label, key in [("Full tools", "all"), ("Rubric only", "rubric_only"), ("No tools", "none")]:
        run = runs.get(key)
        if not run:
            rows.append(f"- **{label}:** _not provided_")
            continue
        m = run["metadata"]
        rows.append(
            f"- **{label}:** {m.get('timestamp', '?')} · model {m.get('model', '?')} · "
            f"rubric {m.get('rubric_version', '?')} · n={m.get('eval_set_size', '?')}"
        )
    return "\n".join(rows) + "\n"


# -----------------------------------------------------------------------------
# Top-level
# -----------------------------------------------------------------------------


def render_ablation_report(runs: dict[str, dict[str, Any] | None]) -> str:
    sections = [
        "# Ablation Analysis: Tool Configurations\n",
        "_Three configurations compared: full tools (lookup_rubric, search_reference, "
        "resolve_escalation), rubric-only (lookup_rubric only), and no tools. "
        "The full-tools run is the baseline; deltas are computed against it._\n",
        _render_headline_compare(runs),
        _render_per_dimension_compare(runs),
        "## Slices\n",
        _render_slice_compare(runs, "by_category", "By Category (weighted accuracy)"),
        _render_slice_compare(runs, "by_difficulty", "By Difficulty (weighted accuracy)"),
        _render_slice_compare(runs, "by_tag", "By Tag (weighted accuracy)"),
        _render_ops_compare(runs),
        _render_interpretation(runs),
        _render_metadata_compare(runs),
    ]
    return "\n".join(s for s in sections if s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare scored runs across ablation modes.")
    parser.add_argument("--all", type=Path, default=None, help="Full-tools scored JSON.")
    parser.add_argument("--rubric-only", type=Path, default=None, help="Rubric-only scored JSON.")
    parser.add_argument("--none", type=Path, default=None, help="No-tools scored JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown path.")
    args = parser.parse_args()

    runs = {
        "all": _load(args.all),
        "rubric_only": _load(getattr(args, "rubric_only")),
        "none": _load(args.none),
    }
    if not any(runs.values()):
        print("Error: no run files provided.", file=sys.stderr)
        return 1

    if args.output is None:
        args.output = PROJECT_ROOT / "reports" / "ablation.md"

    md = render_ablation_report(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(md)
    print(f"Ablation report written to: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())