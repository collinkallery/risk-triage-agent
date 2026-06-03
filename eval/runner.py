"""
Eval runner — iterates the eval set, runs the agent on each case,
writes a single timestamped results JSON file.
 
Usage:
  python -m eval.runner --ablation=all
  python -m eval.runner --ablation=rubric_only --output=results/custom.json
  python -m eval.runner --dry-run                # uses MockClient, no API calls
  python -m eval.runner --limit=5                # only first 5 cases (smoke test)
 
The results file is the single artifact the scorer consumes. Format:
 
  {
    "metadata": {
      "timestamp": "...",
      "model": "...",
      "ablation": "...",
      "eval_set_version": "...",
      "config_snapshot": {...}
    },
    "results": [
      {"case_id": "C001", "ground_truth": {...}, "trace": {...}},
      ...
    ]
  }
"""
 
from __future__ import annotations
 
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
 
import yaml
 
from agent.agent import GeminiClient, MockClient, Trace, run_agent
from agent.retrieval import MockEmbedder, build_retriever
from agent.tools import TOOL_REGISTRY, set_retriever
 
# -----------------------------------------------------------------------------
# Paths and config
# -----------------------------------------------------------------------------
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
 
 
def load_config() -> dict[str, Any]:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)
 
 
def load_eval_set(path: Path) -> list[dict[str, Any]]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases
 
 
# -----------------------------------------------------------------------------
# Ablation mode → filtered tool schemas
# -----------------------------------------------------------------------------
 
 
def get_tool_schemas_for_ablation(ablation_mode: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only the tool schemas allowed by the ablation mode."""
    allowed = config["ablation_modes"].get(ablation_mode)
    if allowed is None:
        raise ValueError(f"Unknown ablation mode: {ablation_mode!r}. Options: {list(config['ablation_modes'])}")
    return [TOOL_REGISTRY[name]["schema"] for name in allowed if name in TOOL_REGISTRY]
 
 
# -----------------------------------------------------------------------------
# Mock client for dry-run mode
# -----------------------------------------------------------------------------
 
 
def _make_dry_run_client() -> MockClient:
    """A MockClient that returns a placeholder Cat 5 output every time.
 
    Useful for testing the runner's plumbing without API costs. The
    placeholder is intentionally always Cat 5 so the scorer downstream
    will register lots of misclassifications — which is fine, it
    confirms the score calculation works on imperfect data.
    """
    placeholder = {
        "category": 5,
        "sub_pattern": None,
        "subject_frame": "first_person",
        "criteria_cited": ["5a", "5b", "5c"],
        "action": "clarify_or_conservative_escalation",
        "reasoning": "[dry-run placeholder]",
    }
    response = {
        "text": "```json\n" + json.dumps(placeholder) + "\n```",
        "tool_calls": [],
        "input_tokens": 3000,
        "output_tokens": 100,
    }
    # Single-shot response repeated for any number of cases.
    return MockClient(script=[response] * 1000)
 
 
# -----------------------------------------------------------------------------
# The runner
# -----------------------------------------------------------------------------
 
 
def run_eval(
    ablation_mode: str,
    output_path: Path,
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    limit: int | None = None,
    retrieval_mode: str = "keyword",
) -> dict[str, Any]:
    """Execute the eval. Returns the results dict (also written to disk)."""
    eval_set_path = PROJECT_ROOT / config["paths"]["eval_set"]
    cases = load_eval_set(eval_set_path)
 
    if limit:
        cases = cases[:limit]
 
    # Build client. Live or mock.
    if dry_run:
        print(f"[dry-run] Using MockClient (no API calls)", file=sys.stderr)
        client = _make_dry_run_client()
    else:
        print(f"Initializing Gemini client (model={config['model']['name']})", file=sys.stderr)
        client = GeminiClient(
            model=config["model"]["name"],
            temperature=config["model"]["temperature"],
        )
 
    # Install the retrieval backend for search_reference. Keyword is the default;
    # vector uses Vertex embeddings live, or a MockEmbedder under --dry-run so the
    # wiring can be exercised without credentials. Restored in the finally block.
    import agent.tools as tools_module
 
    original_retriever = tools_module.get_retriever()
    docs = tools_module._REFERENCE_CORPUS["docs"]
    if retrieval_mode == "vector":
        embedder = MockEmbedder() if dry_run else None
        set_retriever(build_retriever("vector", config, docs, embedder=embedder))
    else:
        set_retriever(build_retriever("keyword", config, docs))
    print(f"Retrieval backend: {retrieval_mode}", file=sys.stderr)
 
    # Determine which tools the agent sees this run.
    tool_schemas = get_tool_schemas_for_ablation(ablation_mode, config)
    allowed_tool_names = [s["name"] for s in tool_schemas]
    print(f"Ablation mode: {ablation_mode} | tools available: {allowed_tool_names}", file=sys.stderr)
    print(f"Running {len(cases)} case(s)...", file=sys.stderr)
 
    # Monkey-patch the agent's ALL_SCHEMAS for this run so it only sees the
    # allowed tools. We modify the module-level list in agent.agent — restore
    # it after the run completes so subsequent runs aren't affected.
    import agent.agent as agent_module
    original_schemas = agent_module.ALL_SCHEMAS
    agent_module.ALL_SCHEMAS = tool_schemas
 
    results: list[dict[str, Any]] = []
    run_start = time.monotonic()
 
    try:
        for idx, case in enumerate(cases, start=1):
            case_id = case["case_id"]
            narrative = case["narrative"]
            t0 = time.monotonic()
            try:
                trace = run_agent(
                    narrative=narrative,
                    client=client,
                    case_id=case_id,
                    max_iterations=config["model"]["max_iterations"],
                    max_input_tokens=config["model"]["max_input_token_budget"],
                )
                elapsed = round(time.monotonic() - t0, 2)
                final_cat = trace.final_output.get("category") if trace.final_output else None
                status = "ok" if trace.terminated_reason == "completed" else trace.terminated_reason
                print(
                    f"  [{idx:>2}/{len(cases)}] {case_id} "
                    f"({elapsed}s, {trace.iterations} iter, pred=Cat{final_cat}, {status})",
                    file=sys.stderr,
                )
            except Exception as e:
                elapsed = round(time.monotonic() - t0, 2)
                # Construct a failure trace so the result is still structured.
                trace = Trace(case_id=case_id, narrative=narrative, iterations=0)
                trace.error = f"{type(e).__name__}: {e}"
                trace.terminated_reason = "error"
                trace.latency_seconds = elapsed
                print(
                    f"  [{idx:>2}/{len(cases)}] {case_id} FAILED ({elapsed}s): {trace.error}",
                    file=sys.stderr,
                )
 
            results.append(
                {
                    "case_id": case_id,
                    "ground_truth": case["ground_truth"],
                    "case_metadata": {
                        "voice": case.get("voice"),
                        "difficulty": case.get("difficulty"),
                        "tags": case.get("tags", []),
                    },
                    "trace": trace.to_dict(),
                }
            )
    finally:
        # Restore the full schema list and the original retrieval backend.
        agent_module.ALL_SCHEMAS = original_schemas
        set_retriever(original_retriever)
 
    total_elapsed = round(time.monotonic() - run_start, 2)
 
    # Load rubric version for metadata.
    with open(PROJECT_ROOT / config["paths"]["rubric"]) as f:
        rubric_meta = yaml.safe_load(f)
 
    payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": config["model"]["name"] if not dry_run else "mock",
            "temperature": config["model"]["temperature"],
            "ablation": ablation_mode,
            "tools_available": allowed_tool_names,
            "retrieval": retrieval_mode,
            "eval_set_path": str(eval_set_path.relative_to(PROJECT_ROOT)),
            "eval_set_size": len(cases),
            "rubric_version": rubric_meta.get("taxonomy_version"),
            "total_elapsed_seconds": total_elapsed,
            "dry_run": dry_run,
        },
        "results": results,
    }
 
    # Write atomically — temp file then rename.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp_path.replace(output_path)
 
    # Summary to stderr.
    n_complete = sum(1 for r in results if r["trace"].get("terminated_reason") == "completed")
    n_error = sum(1 for r in results if r["trace"].get("terminated_reason") == "error")
    n_other = len(results) - n_complete - n_error
    total_input = sum(r["trace"].get("input_tokens", 0) for r in results)
    total_output = sum(r["trace"].get("output_tokens", 0) for r in results)
    print(
        f"\nRun complete: {n_complete} ok, {n_error} errors, {n_other} other "
        f"({total_elapsed}s total, {total_input} input + {total_output} output tokens)",
        file=sys.stderr,
    )
    print(f"Results written to: {output_path}", file=sys.stderr)
    return payload
 
 
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description="Run the behavioral risk triage eval.")
    parser.add_argument(
        "--ablation",
        choices=["all", "rubric_only", "none"],
        default="all",
        help="Tool ablation mode. Default: all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Default: results/run_{timestamp}_{ablation}.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use MockClient, no API calls. For testing the runner itself.",
    )
    parser.add_argument(
        "--retrieval",
        choices=["keyword", "vector"],
        default="keyword",
        help="Retrieval backend for search_reference. Default: keyword.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases. For smoke testing.",
    )
    args = parser.parse_args()
 
    config = load_config()
 
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        retr = f"_{args.retrieval}" if args.retrieval != "keyword" else ""
        suffix = "_dry" if args.dry_run else ""
        args.output = PROJECT_ROOT / config["paths"]["results_dir"] / f"run_{timestamp}_{args.ablation}{retr}{suffix}.json"
 
    run_eval(
        ablation_mode=args.ablation,
        output_path=args.output,
        config=config,
        dry_run=args.dry_run,
        limit=args.limit,
        retrieval_mode=args.retrieval,
    )
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
