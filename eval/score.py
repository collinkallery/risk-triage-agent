"""
Score orchestrator — reads a results file, applies deterministic scoring
plus optional LLM judging, writes a scored results file.

Usage:
  python -m eval.score results/run_20260601_143000_all.json
  python -m eval.score results/run_X.json --output=results/run_X_scored.json
  python -m eval.score results/run_X.json --no-judge           # skip LLM judge
  python -m eval.score results/run_X.json --mock-judge         # use MockJudgeClient

Output structure: same as input plus per-case `scores` and a top-level
`aggregates` block.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from eval.judge import judge_case, make_judge_client
from eval.scoring import compute_aggregates, score_case_deterministic

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def score_run(
    results_path: Path,
    output_path: Path,
    *,
    use_judge: bool = True,
    mock_judge: bool = False,
    judge_provider: str = "anthropic",
    judge_model: str | None = None,
) -> dict:
    """Read a results file, score every case, write scored file."""
    with open(results_path) as f:
        run = json.load(f)

    results = run["results"]
    n = len(results)
    print(f"Scoring {n} case(s) from {results_path.name}", file=sys.stderr)

    # Deterministic pass first.
    for r in results:
        r["scores"] = score_case_deterministic(r)

    # Judge pass.
    judge_total_input = 0
    judge_total_output = 0
    active_provider = "mock" if mock_judge else judge_provider
    if use_judge:
        model_note = f", model={judge_model}" if (judge_model and active_provider != "mock") else ""
        print(f"Initializing judge: provider={active_provider}{model_note}", file=sys.stderr)
        judge_client = make_judge_client(provider=active_provider, model=judge_model)

        for idx, r in enumerate(results, start=1):
            t0 = time.monotonic()
            try:
                verdict = judge_case(r, judge_client)
            except Exception as e:
                verdict = {
                    "grounded": None,
                    "explanation": f"Judge error: {type(e).__name__}: {e}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "skipped": True,
                }
            elapsed = round(time.monotonic() - t0, 2)
            r["scores"]["reasoning_judge"] = verdict
            judge_total_input += verdict.get("input_tokens", 0)
            judge_total_output += verdict.get("output_tokens", 0)
            grounded = verdict.get("grounded")
            grounded_str = "✓" if grounded is True else ("✗" if grounded is False else "?")
            print(f"  [{idx:>2}/{n}] {r['case_id']} judge={grounded_str} ({elapsed}s)", file=sys.stderr)
    else:
        for r in results:
            r["scores"]["reasoning_judge"] = {"skipped": True, "reason": "judge disabled"}

    # Aggregates.
    aggregates = compute_aggregates(results)

    # Judge agreement stats — fraction of cases where judge said grounded=true
    # AND category was correct (a useful sanity check).
    if use_judge:
        grounded_cases = [r for r in results if r["scores"]["reasoning_judge"].get("grounded") is True]
        ungrounded_cases = [r for r in results if r["scores"]["reasoning_judge"].get("grounded") is False]
        aggregates["judge"] = {
            "provider": active_provider,
            "model": (judge_model or "(provider default)") if active_provider != "mock" else "mock",
            "cross_model": active_provider not in {"gemini", "mock"},
            "n_grounded": len(grounded_cases),
            "n_ungrounded": len(ungrounded_cases),
            "n_unparseable": sum(
                1 for r in results if r["scores"]["reasoning_judge"].get("grounded") is None
            ),
            "grounded_and_correct": sum(
                1 for r in grounded_cases if r["scores"]["category"]["correct"]
            ),
            "grounded_and_incorrect": sum(
                1 for r in grounded_cases if not r["scores"]["category"]["correct"]
            ),
            "ungrounded_and_correct": sum(
                1 for r in ungrounded_cases if r["scores"]["category"]["correct"]
            ),
            "judge_token_cost": {
                "input": judge_total_input,
                "output": judge_total_output,
            },
        }

    run["aggregates"] = aggregates
    run["metadata"]["scored_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run["metadata"]["judge_enabled"] = use_judge
    run["metadata"]["judge_mock"] = bool(mock_judge)
    run["metadata"]["judge_provider"] = active_provider if use_judge else None
    run["metadata"]["judge_model"] = (
        (judge_model or "(provider default)") if (use_judge and active_provider != "mock") else None
    )

    # Write atomically.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(run, f, indent=2, default=str)
    tmp.replace(output_path)

    # Headline summary.
    h = aggregates["headline"]
    print(
        f"\nScored. Weighted accuracy: {h['weighted_accuracy']:.3f} | "
        f"Raw accuracy: {h['raw_accuracy']:.3f} | "
        f"Mean weighted: {h['mean_weighted_score']:.3f}",
        file=sys.stderr,
    )
    print(f"Scored file written to: {output_path}", file=sys.stderr)
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a runner output file.")
    parser.add_argument("results_path", type=Path, help="Path to runner output JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Path for scored output file.")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge.")
    parser.add_argument("--mock-judge", action="store_true", help="Use the offline mock judge.")
    parser.add_argument(
        "--judge-provider",
        choices=["anthropic", "gemini", "mock"],
        default=None,
        help="Judge provider. Overrides config.yaml. Default: anthropic (cross-model).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model string. Overrides config.yaml; defaults to the provider's default.",
    )
    args = parser.parse_args()

    config = load_config()
    judge_cfg = config.get("judge", {})
    judge_provider = args.judge_provider or judge_cfg.get("provider", "anthropic")
    judge_model = args.judge_model or judge_cfg.get("model")

    if args.output is None:
        stem = args.results_path.stem
        args.output = args.results_path.with_name(f"{stem}_scored.json")

    score_run(
        args.results_path,
        args.output,
        use_judge=not args.no_judge,
        mock_judge=args.mock_judge,
        judge_provider=judge_provider,
        judge_model=judge_model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())