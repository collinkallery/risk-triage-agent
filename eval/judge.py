"""
LLM-as-judge for reasoning quality.

Scope: ONE question per case. "Does the agent's reasoning correctly apply
the criteria it cited, to land at the right classification for this
narrative?"

Inputs to the judge:
  - the narrative
  - ground truth (category, sub_pattern, subject_frame, criteria_expected)
  - agent's final output (criteria_cited, reasoning, category)

Output:
  {"grounded": true|false, "explanation": "..."}

Judge provider is configurable (see make_judge_client). The default is
Anthropic Claude — a different model family from the Gemini agent — so the
judge does not share the agent's failure modes. GeminiJudgeClient (same
family as the agent) and MockJudgeClient (offline, deterministic) are also
available.

Methodology limitations (documented in README):
  - Cross-model by default (Claude judging a Gemini agent), which removes
    the shared-bias problem of same-model judging. Selecting the Gemini
    judge reintroduces it and should be treated as a weaker configuration.
  - Judge sees ground truth. This is intentional — the question is
    whether reasoning is consistent with correct application of criteria,
    not whether it's good in a vacuum.
  - Spot-check 5 cases by hand and report agreement rate in the README.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

# Default model per provider. Override via config.yaml or the CLI.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

JUDGE_SYSTEM_PROMPT = """\
You are evaluating the reasoning quality of an automated behavioral risk \
triage system. For each case, you are shown:

1. The narrative the agent classified.
2. The ground-truth label (category, sub_pattern, subject_frame, and \
the criteria that should have been cited).
3. The agent's final output (the criteria it cited, its reasoning text, \
and the category it predicted).

Your job is to answer ONE question: does the agent's reasoning correctly \
apply the criteria it cited to this specific narrative, in a way that is \
consistent with arriving at the correct classification?

You are NOT scoring whether the agent's category was correct (that is \
measured separately). You are NOT scoring whether the agent cited the \
right criteria (that is also measured separately). You are scoring \
whether the reasoning USES the cited criteria correctly.

Examples:
- Agent cites criterion 1e (euphemistic finality) but its reasoning \
treats the language as direct/explicit statement of intent. → grounded = false.
- Agent cites 3a and 3b and reasons explicitly about past struggle \
contrasting with present recovery state. → grounded = true.
- Agent cites 4d and reasons that intensity language is present but \
lacks the unantecedented-metaphor pattern. → grounded = true even if \
the final category was wrong (you are not judging category correctness).
- Agent cites criteria but the reasoning is generic ("seems concerning") \
without actually applying the criteria to the narrative's text. → \
grounded = false.

Respond with a single JSON object, no other text:
{"grounded": true|false, "explanation": "<one sentence>"}
"""


JUDGE_USER_TEMPLATE = """\
NARRATIVE:
{narrative}

GROUND TRUTH:
- Category: {gt_category}
- Sub-pattern: {gt_sub_pattern}
- Subject frame: {gt_subject_frame}
- Criteria that should have been cited: {gt_criteria}

AGENT OUTPUT:
- Predicted category: {pred_category}
- Criteria the agent cited: {pred_criteria}
- Agent's reasoning: {pred_reasoning}

Evaluate whether the agent's reasoning correctly applies its cited \
criteria to this narrative. Respond with a JSON object only.
"""


# -----------------------------------------------------------------------------
# Judge client protocol — same shape as agent.Client but simpler
# -----------------------------------------------------------------------------


class JudgeClient(Protocol):
    def judge(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        """Returns {'text': str, 'input_tokens': int, 'output_tokens': int}."""
        ...


class GeminiJudgeClient:
    """Vertex AI judge using google-genai. Mirrors GeminiClient but without tools."""

    def __init__(self, model: str = "gemini-2.5-pro", temperature: float = 0.0):
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as e:
            raise RuntimeError("google-genai required for live judge.") from e

        import os

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable is required.")

        self._types = types
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._model = model
        self._temperature = temperature

    def judge(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
        )
        contents = [types.Content(role="user", parts=[{"text": user_message}])]
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        text = ""
        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.content and candidate.content.parts:
            text_parts = [p.text for p in candidate.content.parts if hasattr(p, "text") and p.text]
            text = "\n".join(text_parts)

        usage = response.usage_metadata
        return {
            "text": text,
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        }


class MockJudgeClient:
    """Returns scripted responses for testing. Defaults to grounded=true."""

    def __init__(self, default_grounded: bool = True, script: list[dict[str, Any]] | None = None):
        self._default = default_grounded
        self._script = list(script) if script else []

    def judge(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        if self._script:
            return self._script.pop(0)
        return {
            "text": json.dumps(
                {"grounded": self._default, "explanation": "[mock judge default]"}
            ),
            "input_tokens": 1200,
            "output_tokens": 30,
        }


class AnthropicJudgeClient:
    """Cross-model judge using Anthropic's Claude via the anthropic SDK.

    This is the default judge. Because the agent runs on Gemini, judging
    with a different model family removes the shared-bias problem inherent
    in same-model evaluation. Requires ANTHROPIC_API_KEY in the environment.
    """

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, temperature: float = 0.0, max_tokens: int = 1024):
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "anthropic is not installed. Install with: pip install anthropic"
            ) from e

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is required.")

        self._client = anthropic.Anthropic()
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def judge(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        # Concatenate all text blocks in the response content.
        text_parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        usage = response.usage
        return {
            "text": "\n".join(text_parts),
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        }


# -----------------------------------------------------------------------------
# Provider factory
# -----------------------------------------------------------------------------


def make_judge_client(
    provider: str = "anthropic",
    model: str | None = None,
    *,
    temperature: float = 0.0,
    mock_grounded: bool = True,
) -> JudgeClient:
    """Construct a judge client for the named provider.

    provider: "anthropic" (default, cross-model), "gemini" (same family as
    the agent), or "mock" (offline). model overrides the provider default.
    """
    provider = (provider or "anthropic").lower()
    if provider == "anthropic":
        return AnthropicJudgeClient(model=model or DEFAULT_ANTHROPIC_MODEL, temperature=temperature)
    if provider == "gemini":
        return GeminiJudgeClient(model=model or DEFAULT_GEMINI_MODEL, temperature=temperature)
    if provider == "mock":
        return MockJudgeClient(default_grounded=mock_grounded)
    raise ValueError(
        f"Unknown judge provider {provider!r}. Options: 'anthropic', 'gemini', 'mock'."
    )


# -----------------------------------------------------------------------------
# Judge orchestration
# -----------------------------------------------------------------------------


def _parse_judge_response(text: str) -> dict[str, Any]:
    """Pull the JSON verdict out of the judge's text."""
    if not text:
        return {"grounded": None, "explanation": "Judge returned empty response.", "parse_error": True}

    # Try direct parse.
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "grounded" in obj:
            return {
                "grounded": bool(obj["grounded"]),
                "explanation": obj.get("explanation", ""),
            }
    except json.JSONDecodeError:
        pass

    # Try first balanced object.
    first = text.find("{")
    if first != -1:
        depth = 0
        for i in range(first, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[first : i + 1])
                        if isinstance(obj, dict) and "grounded" in obj:
                            return {
                                "grounded": bool(obj["grounded"]),
                                "explanation": obj.get("explanation", ""),
                            }
                    except json.JSONDecodeError:
                        break

    return {"grounded": None, "explanation": text[:200], "parse_error": True}


def judge_case(result: dict[str, Any], client: JudgeClient) -> dict[str, Any]:
    """Judge the reasoning quality of a single scored result."""
    final_output = result["trace"].get("final_output") or {}
    if not final_output:
        return {
            "grounded": None,
            "explanation": "No final output to judge.",
            "skipped": True,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    ground_truth = result["ground_truth"]
    user_msg = JUDGE_USER_TEMPLATE.format(
        narrative=result["trace"]["narrative"],
        gt_category=ground_truth["category"],
        gt_sub_pattern=ground_truth.get("sub_pattern"),
        gt_subject_frame=ground_truth["subject_frame"],
        gt_criteria=ground_truth["criteria_expected"],
        pred_category=final_output.get("category"),
        pred_criteria=final_output.get("criteria_cited"),
        pred_reasoning=final_output.get("reasoning", ""),
    )

    response = client.judge(JUDGE_SYSTEM_PROMPT, user_msg)
    parsed = _parse_judge_response(response.get("text", ""))
    parsed["input_tokens"] = response.get("input_tokens", 0)
    parsed["output_tokens"] = response.get("output_tokens", 0)
    return parsed