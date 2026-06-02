# Behavioral Risk Triage Agent

A single-agent LLM system that classifies written narratives into behavioral risk
categories, paired with a full evaluation harness for measuring how well it does so.

The agent reads a narrative — a journal entry, support-chat message, clinical case
note, or social-media post — and produces a structured triage classification: a risk
category, a subject frame (who the content is about), the rubric criteria it applied,
and a recommended action path. It reasons in an explicit ReAct loop (thought → tool
call → observation → …) so that every decision leaves an auditable trace.

> **Scope and intent.** This is a demonstration of **evaluation methodology** for a
> safety-adjacent classification task — not a deployed or clinically validated tool. It
> does not assess people, render diagnoses, or replace professional judgment. The
> taxonomy is deliberately narrow (see *Limitations*). Its purpose is to show how one
> would design, instrument, and rigorously measure an agent for a high-stakes,
> asymmetric-cost decision.

## Why this project

The interesting engineering problem here isn't the classifier — it's the *evaluation*.
For a triage decision, not all errors are equal: confidently dismissing a real crisis is
far worse than over-escalating an ambiguous one. That asymmetry has to be baked into how
the system is scored, not bolted on afterward. This repo treats the eval as a
first-class artifact: deterministic cost-weighted scoring, an LLM-as-judge layer for
reasoning quality, slice-level breakdowns, and a tool-ablation framework to measure what
each component actually contributes.

## The taxonomy

Five categories, each mapping to a distinct downstream action:

| # | Category | Action |
|---|---|---|
| 1 | Imminent Risk | `immediate_escalation` |
| 2 | Active Ideation, Non-Imminent | `warm_handoff` |
| 3 | Historical / Recovery Context | `standard_engagement` |
| 4 | Distress Without Risk Indicators | `supportive_engagement` |
| 5 | Ambiguous / Insufficient Information | `clarify_or_conservative_escalation` |

Category 1 additionally carries a **sub-pattern** (`1-deliberate` vs. `1-euphemistic`)
because the two presentations indicate different states of mind even though both escalate.

Crucially, **category and subject frame are independent dimensions.** The category
describes *what* the content is; the subject frame (`first_person`,
`third_party_clear`, `third_party_ambiguous`) describes *who* it is about. Content that
meets Cat 1 criteria is Cat 1 even when framed as being about a friend — the frame is
reported separately and modulates the action path. The full rubric, including criterion
definitions, near-miss negatives, and the asymmetric confusion-cost matrix, lives in
[`data/rubric.yaml`](data/rubric.yaml).

## Architecture

```
agent/
  agent.py          ReAct loop; Trace data structures; GeminiClient + MockClient
  system_prompt.py  The triage system prompt (stance, protocol, anti-patterns, worked example)
  tools.py          Three tools: lookup_rubric, search_reference, resolve_escalation
  triage.py         Ad-hoc CLI: classify a single narrative (inference counterpart to the runner)
eval/
  runner.py         Runs the agent over the eval set; writes a results JSON
  scoring.py        Deterministic scoring (cost matrices, precision/recall, aggregates)
  judge.py          LLM-as-judge: Anthropic (default), Gemini, and mock clients + factory
  score.py          Orchestrates deterministic + judge scoring into a scored JSON
  report.py         Renders a scored run as a markdown report
  ablation.py       Compares scored runs across tool configurations
data/
  rubric.yaml            Source of truth: criteria, examples, cost matrices
  reference_corpus.yaml  Small corpus for the search_reference tool
  eval_set.jsonl         Labeled cases with ground truth and rationale
tests/              Pytest suite (mock-based, no credentials required)
config.yaml         Model, judge, paths, and ablation-mode defaults
smoke_test.py       End-to-end loop check using a scripted MockClient
.github/workflows/  CI: runs the test suite on push / PR
```

The agent is **provider-agnostic**: `run_agent()` accepts anything conforming to a
minimal `Client` protocol. `GeminiClient` is the live Vertex AI implementation;
`MockClient` is a deterministic scripted stand-in, so the loop, trace shape, and all
downstream scoring code can be exercised with zero API calls or credentials.

## Quick start (no credentials required)

Everything except a live model run works offline.

```bash
pip install -r requirements.txt

# 1. Confirm the agent loop works end to end (scripted mock client)
python smoke_test.py

# 2. Run the full eval with the mock client (no API calls)
python -m eval.runner --dry-run

# 3. Score that run with the offline mock judge (no API key needed)
python -m eval.score results/run_<timestamp>_all_dry.json --mock-judge

# 4. Render a markdown report
python -m eval.report results/run_<timestamp>_all_dry_scored.json
```

Reports land in `reports/`; raw and scored run data land in `results/`. Both directories
are git-ignored.

## Ad-hoc classification

The eval harness runs the agent over the *labeled* eval set and scores it. To classify a
single arbitrary narrative instead — no eval set, no ground truth required — use the
`agent.triage` CLI. It is the inference counterpart to the runner: it calls the same
`run_agent` loop on one narrative and prints the classification.

```bash
# Inline text
python -m agent.triage --text "I can't keep doing this anymore."

# From a file, with the full reasoning trace
python -m agent.triage --file note.txt --show-trace

# Piped via stdin
echo "five years ago I was in a dark place; I'm okay now" | python -m agent.triage

# Machine-readable output
python -m agent.triage --text "..." --json
```

Classifying real text requires a live Gemini call (see *Running against a live model*
below for credentials). Pass `--mock` to exercise the CLI plumbing with a fixed
placeholder and no API call — useful for checking wiring, but the placeholder does **not**
reflect the input, so never read a `--mock` result as a real classification.

Because ad-hoc input has no ground-truth label, this path is pure inference: no scoring
and no judge. As a convenience for quick "did it get this one right?" checks, you can pass
an expected category and/or frame, which scores just that single case against the rubric's
cost matrices (category and subject-frame dimensions only — criteria/tool F1 and the
reasoning judge require the full harness):

```bash
python -m agent.triage --text "..." --expected 1 --expected-frame first_person
```

Output goes to stdout and diagnostics to stderr, so `--json` pipes cleanly. The exit code
is `0` on a successful classification, `1` if the agent produced none (or the live client
failed to initialize), and `2` on missing/invalid input.

## Tests

The suite is fully mock-based — it runs with no credentials and no network — and is
exercised in CI on every push and pull request.

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

## Running against a live model

The agent runs on Gemini (Vertex AI, via `google-genai`); the judge defaults to Anthropic
Claude (via `anthropic`), so grading is cross-model.

```bash
# Agent (Gemini on Vertex AI)
export GOOGLE_CLOUD_PROJECT=your-gcp-project
export GOOGLE_CLOUD_LOCATION=us-central1   # optional; defaults to us-central1
gcloud auth application-default login

# Judge (Anthropic Claude — the default cross-model judge)
export ANTHROPIC_API_KEY=sk-ant-...

# Classify a single narrative live
python -m agent.triage --text "I have the pills counted out for tonight." --show-trace

# Or run the full eval
python -m eval.runner --ablation=all
python -m eval.score   results/run_<timestamp>_all.json
python -m eval.report  results/run_<timestamp>_all_scored.json
```

Useful runner flags: `--limit=5` (first N cases), `--output=...` (custom path).
Useful scorer flags: `--no-judge` (skip the judge entirely), `--mock-judge` (offline
judge), `--judge-provider {anthropic,gemini,mock}` and `--judge-model <id>` (override the
`judge:` block in `config.yaml`).

## Scoring

Each case is scored on several dimensions:

- **Category cost** — via the rubric's asymmetric confusion-cost matrix. False negatives
  on Cat 1 are penalized most heavily. The headline metric is *weighted accuracy*
  (`1 − cost/max_cost`), reported alongside raw accuracy; the gap between the two reveals
  whether failures were catastrophic or merely adjacent-category misses.
- **Sub-pattern accuracy** — Cat 1 only.
- **Subject-frame cost** — a smaller cost matrix.
- **Criteria precision/recall/F1** — set comparison of cited vs. expected criterion IDs.
- **Tool-use precision/recall/F1** — set comparison of tools used vs. expected.
- **Reasoning quality** — an LLM-as-judge verdict (see below).

Aggregates include a confusion matrix, slices (by category, difficulty, subject frame,
and tag), and the top failures with full traces inline.

## LLM-as-judge methodology and limitations

The judge answers exactly one question per case: *does the agent's reasoning correctly
apply the criteria it cited, in a way consistent with the correct classification?* It is
deliberately scoped narrowly so it doesn't double-count what the deterministic scorer
already measures.

The judge is **cross-model by default**: the agent runs on Gemini and the judge runs on
Anthropic Claude, so the grader does not share the agent's failure modes. The provider is
configurable (`config.yaml` `judge:` block, or `--judge-provider` / `--judge-model`) —
`anthropic` (default), `gemini` (same family as the agent), or `mock` (offline). The
report labels each run as cross-model or same-family accordingly.

Known limitations, stated plainly:

- Selecting the **Gemini** judge reintroduces same-model shared bias and should be treated
  as a weaker configuration than the default.
- The judge **sees ground truth.** This is intentional — it evaluates consistency with
  correct criterion application, not quality in a vacuum — but it is a limitation worth naming.
- Judge verdicts should be **spot-checked by hand.** The recommended practice is to
  manually review ~5 cases and report the human/judge agreement rate.

## Tool ablation

`eval/ablation.py` compares scored runs across three tool configurations — full tools,
rubric-only, and no tools — to quantify what each tool contributes:

```bash
python -m eval.ablation \
  --all=results/run_X_all_scored.json \
  --rubric-only=results/run_X_rubric_only_scored.json \
  --none=results/run_X_none_scored.json
```

The full-tools run is treated as the baseline and deltas are computed against it.

## Possible next steps

- Expand the taxonomy toward fuller clinical coverage (harm-to-others, substance-acute,
  minors), each with its own criteria and cost rows.
- Add a second human rater to a slice of the eval set and report inter-rater agreement
  alongside the model's scores.
- Replace the keyword `search_reference` stub with real vector retrieval.
- Add a panel/ensemble judge (multiple providers voting) on top of the existing
  cross-model judge.

## Limitations

The taxonomy is intentionally narrow and calibrated to demonstrate eval methodology, not
full clinical coverage. The agent pattern-matches on text from a single narrative with no
conversational history, provider context, or follow-up capability; it cannot determine
whether a third party described in a narrative is real or a disclosure scaffold, and it
does not try to. Its output is one input a human would weigh — never a final decision.

## License

[MIT](LICENSE)
