"""
Retrieval comparison — keyword vs. vector backend for `search_reference`.

Answers this project's core RAG question: does semantic (vector) retrieval beat
keyword retrieval on this eval, and where? Run the same eval set twice — once with
`--retrieval keyword`, once with `--retrieval vector` — score both, then compare:

  python -m eval.runner --ablation all --retrieval keyword
  python -m eval.runner --ablation all --retrieval vector
  # score both runs, then:
  python -m eval.compare_retrieval \\
      --keyword results/run_X_all_scored.json \\
      --vector  results/run_Y_all_vector_scored.json \\
      --output  reports/retrieval_compare.md

The report contains:
  - Headline + per-dimension comparison (keyword | vector | delta)
  - Slices by tag and category — where semantic retrieval is most likely to help
  - A *retrieval-invoked segment*: metrics restricted to the cases where the agent
    actually called search_reference. This is the sharpest lens on the backend's
    effect, because cases that never query retrieval cannot be affected by which
    backend was installed — including them only dilutes the signal.
  - An auto-generated interpretation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Loaders and formatting helpers
# -----------------------------------------------------------------------------


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _f(x: float | None, places: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{places}f}"


def _delta_pp(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return ""
    diff = value - baseline
    if abs(diff) < 0.001:
        return "  (—)"
    sign = "+" if diff > 0 else ""
    return f"  ({sign}{diff * 100:.1f}pp)"


def _delta_raw(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return ""
    diff = value - baseline
    if abs(diff) < 0.001:
        return "  (—)"
    sign = "+" if diff > 0 else ""
    return f"  ({sign}{diff:.3f})"


# -----------------------------------------------------------------------------
# Retrieval-invoked segment
# -----------------------------------------------------------------------------


def invoked_case_ids(run: dict[str, Any]) -> set[str]:
    """Case IDs where the agent actually called search_reference."""
    ids: set[str] = set()
    for r in run["results"]:
        names = [tc.get("name") for tc in r["trace"].get("tool_calls", [])]
        if "search_reference" in names:
            ids.add(r["case_id"])
    return ids


def subset_category_metrics(run: dict[str, Any], case_ids: set[str]) -> dict[str, Any]:
    """Category metrics over a subset of cases.

    Reports raw accuracy (fraction with the correct category) and mean normalized
    category score (mean of per-case `1 - cost/max_cost`). Mean-normalized is used
    rather than the headline cost-weighted accuracy because it is well-defined
    per-case without recovering each case's max cost, and is directly comparable
    across the two runs over the same case set.
    """
    rows = [r for r in run["results"] if r["case_id"] in case_ids]
    n = len(rows)
    if n == 0:
        return {"n": 0, "raw_accuracy": None, "mean_normalized": None}
    correct = sum(1 for r in rows if r["scores"]["category"]["correct"])
    norm = sum(r["scores"]["category"]["normalized"] for r in rows)
    return {"n": n, "raw_accuracy": correct / n, "mean_normalized": norm / n}


# -----------------------------------------------------------------------------
# Section renderers
# -----------------------------------------------------------------------------


def _render_headline(kw: dict[str, Any], vec: dict[str, Any]) -> str:
    kh = kw["aggregates"]["headline"]
    vh = vec["aggregates"]["headline"]
    rows = ["## Headline", "", "| Metric | Keyword | Vector | Δ (vector − keyword) |", "|---|---|---|---|"]
    rows.append(f"| Weighted accuracy | {_pct(kh['weighted_accuracy'])} | {_pct(vh['weighted_accuracy'])} | {_delta_pp(vh['weighted_accuracy'], kh['weighted_accuracy']).strip()} |")
    rows.append(f"| Raw accuracy | {_pct(kh['raw_accuracy'])} | {_pct(vh['raw_accuracy'])} | {_delta_pp(vh['raw_accuracy'], kh['raw_accuracy']).strip()} |")
    rows.append(f"| Mean weighted score | {_f(kh['mean_weighted_score'])} | {_f(vh['mean_weighted_score'])} | {_delta_raw(vh['mean_weighted_score'], kh['mean_weighted_score']).strip()} |")
    return "\n".join(rows) + "\n"


def _render_per_dimension(kw: dict[str, Any], vec: dict[str, Any]) -> str:
    kd = kw["aggregates"]["per_dimension"]
    vd = vec["aggregates"]["per_dimension"]
    metrics = [
        ("category_raw_accuracy", "Category raw accuracy", True),
        ("category_weighted_accuracy", "Category weighted accuracy", True),
        ("sub_pattern_accuracy_cat1_only", "Sub-pattern accuracy (Cat 1)", True),
        ("subject_frame_accuracy", "Subject frame accuracy", True),
        ("criteria_f1_mean", "Criteria F1", False),
        ("tool_use_f1_mean", "Tool use F1", False),
    ]
    rows = ["## Per-Dimension", "", "| Dimension | Keyword | Vector | Δ |", "|---|---|---|---|"]
    for key, label, is_pct in metrics:
        kv, vv = kd.get(key), vd.get(key)
        if is_pct:
            rows.append(f"| {label} | {_pct(kv)} | {_pct(vv)} | {_delta_pp(vv, kv).strip()} |")
        else:
            rows.append(f"| {label} | {_f(kv)} | {_f(vv)} | {_delta_raw(vv, kv).strip()} |")
    return "\n".join(rows) + "\n"


def _render_slice(kw: dict[str, Any], vec: dict[str, Any], slice_key: str, title: str) -> str:
    kslice = kw["aggregates"]["slices"][slice_key]
    vslice = vec["aggregates"]["slices"][slice_key]
    buckets: dict[str, int] = {}
    for s in (kslice, vslice):
        for k, stats in s.items():
            buckets[k] = stats["n"]
    rows = [f"### {title}", "", "| Bucket | n | Keyword | Vector | Δ |", "|---|---|---|---|---|"]
    for bucket in sorted(buckets):
        n = buckets[bucket]
        kv = kslice.get(bucket, {}).get("weighted_accuracy")
        vv = vslice.get(bucket, {}).get("weighted_accuracy")
        rows.append(f"| {bucket} | {n} | {_pct(kv)} | {_pct(vv)} | {_delta_pp(vv, kv).strip()} |")
    return "\n".join(rows) + "\n"


def _render_invoked_segment(kw: dict[str, Any], vec: dict[str, Any]) -> str:
    kw_ids = invoked_case_ids(kw)
    vec_ids = invoked_case_ids(vec)
    union = kw_ids | vec_ids

    km = subset_category_metrics(kw, union)
    vm = subset_category_metrics(vec, union)

    rows = ["## Retrieval-Invoked Segment", ""]
    rows.append(
        f"search_reference was called on {len(kw_ids)} case(s) in the keyword run and "
        f"{len(vec_ids)} in the vector run ({len(union)} in their union). Metrics below are "
        f"restricted to that union — the only cases where the retrieval backend could matter."
    )
    rows.append("")
    if not union:
        rows.append("_The agent never called search_reference in either run, so the backend had no effect on this eval._")
        return "\n".join(rows) + "\n"
    rows.append("| Metric (category, invoked cases only) | Keyword | Vector | Δ |")
    rows.append("|---|---|---|---|")
    rows.append(f"| Raw accuracy | {_pct(km['raw_accuracy'])} | {_pct(vm['raw_accuracy'])} | {_delta_pp(vm['raw_accuracy'], km['raw_accuracy']).strip()} |")
    rows.append(f"| Mean normalized score | {_f(km['mean_normalized'])} | {_f(vm['mean_normalized'])} | {_delta_raw(vm['mean_normalized'], km['mean_normalized']).strip()} |")
    return "\n".join(rows) + "\n"


def _render_interpretation(kw: dict[str, Any], vec: dict[str, Any]) -> str:
    rows = ["## Interpretation", ""]
    kwa = kw["aggregates"]["headline"]["weighted_accuracy"]
    vwa = vec["aggregates"]["headline"]["weighted_accuracy"]
    delta = (vwa - kwa) * 100

    if abs(delta) < 1:
        rows.append(
            f"- Overall, vector retrieval was within 1pp of keyword on weighted accuracy "
            f"({_pct(vwa)} vs {_pct(kwa)}). On this corpus, semantic retrieval did not move the "
            f"headline metric — unsurprising given a small, keyword-tagged reference corpus where "
            f"lexical overlap is already a strong signal."
        )
    elif delta > 0:
        rows.append(
            f"- Vector retrieval beat keyword by {delta:.1f}pp on weighted accuracy "
            f"({_pct(vwa)} vs {_pct(kwa)}). The cases driving the gain are most visible in the "
            f"slice and invoked-segment tables above."
        )
    else:
        rows.append(
            f"- Vector retrieval was {abs(delta):.1f}pp *worse* than keyword on weighted accuracy "
            f"({_pct(vwa)} vs {_pct(kwa)}). Worth checking whether vector retrieval surfaced "
            f"loosely-related distractor docs that keyword's exact-overlap filter excluded."
        )

    union = invoked_case_ids(kw) | invoked_case_ids(vec)
    if union:
        km = subset_category_metrics(kw, union)
        vm = subset_category_metrics(vec, union)
        seg_delta = (vm["mean_normalized"] - km["mean_normalized"])
        rows.append(
            f"- On the {len(union)} case(s) where retrieval was actually invoked, mean normalized "
            f"category score went from {_f(km['mean_normalized'])} (keyword) to {_f(vm['mean_normalized'])} "
            f"(vector), a {seg_delta:+.3f} change. This is the effect with non-retrieval cases removed."
        )
    else:
        rows.append(
            "- The agent never invoked search_reference in either run, so this comparison reflects "
            "no actual retrieval. Consider prompting that encourages tool use, or harder cases."
        )
    rows.append(
        "- Reminder: the value of vector retrieval scales with corpus size and lexical diversity. "
        "A null or negative result here is itself a finding about when RAG is worth its cost, not a bug."
    )
    return "\n".join(rows) + "\n"


def _render_metadata(kw: dict[str, Any], vec: dict[str, Any]) -> str:
    def line(label: str, run: dict[str, Any]) -> str:
        m = run["metadata"]
        return (
            f"- **{label}:** {m.get('timestamp', '?')} · model {m.get('model', '?')} · "
            f"retrieval {m.get('retrieval', '?')} · ablation {m.get('ablation', '?')} · "
            f"n={m.get('eval_set_size', '?')}"
        )
    return "\n".join(["## Run Metadata", "", line("Keyword", kw), line("Vector", vec)]) + "\n"


# -----------------------------------------------------------------------------
# Top-level
# -----------------------------------------------------------------------------


def render_report(kw: dict[str, Any], vec: dict[str, Any]) -> str:
    sections = [
        "# Retrieval Comparison: Keyword vs. Vector\n",
        "_Same agent, same eval set, same tools — only the `search_reference` backend differs. "
        "Deltas are vector minus keyword; positive means vector did better._\n",
        _render_headline(kw, vec),
        _render_per_dimension(kw, vec),
        "## Slices\n",
        _render_slice(kw, vec, "by_tag", "By Tag (weighted accuracy)"),
        _render_slice(kw, vec, "by_category", "By Category (weighted accuracy)"),
        _render_invoked_segment(kw, vec),
        _render_interpretation(kw, vec),
        _render_metadata(kw, vec),
    ]
    return "\n".join(s for s in sections if s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare keyword vs. vector retrieval on scored runs.")
    parser.add_argument("--keyword", type=Path, required=True, help="Scored JSON from a --retrieval keyword run.")
    parser.add_argument("--vector", type=Path, required=True, help="Scored JSON from a --retrieval vector run.")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown path.")
    args = parser.parse_args()

    kw = _load(args.keyword)
    vec = _load(args.vector)
    if kw is None or vec is None:
        print("Error: both --keyword and --vector scored files are required.", file=sys.stderr)
        return 1

    if args.output is None:
        args.output = PROJECT_ROOT / "reports" / "retrieval_compare.md"

    md = render_report(kw, vec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(md)
    print(f"Retrieval comparison written to: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
