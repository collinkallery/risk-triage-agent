"""
Behavioral Risk Triage Agent — ReAct-style classification loop.

Single-agent loop. Sends narrative + system prompt to Gemini. Handles
function-call responses by dispatching to local tools. Loops until the
model emits a final structured classification (or hits the iteration cap).

Architecture:
  - GeminiClient is the live Vertex AI implementation.
  - MockClient is a deterministic stand-in for local testing without GCP
    credentials. Useful for iterating on the loop, the trace shape, and
    downstream scoring code.
  - run_agent() is provider-agnostic — it takes any client conforming to
    the minimal Client protocol below.

The loop emits a structured Trace object capturing every thought, tool
call, observation, and the final classification. The scorer consumes
Trace objects; the runner persists them as JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from agent.system_prompt import SYSTEM_PROMPT
from agent.tools import ALL_SCHEMAS, dispatch

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MAX_ITERATIONS = 6  # Cap on tool-call rounds before forcing termination.
MAX_INPUT_TOKEN_BUDGET = 50_000  # Per-case cost guardrail.


# -----------------------------------------------------------------------------
# Trace data structures
# -----------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    iteration: int


@dataclass
class Trace:
    case_id: str | None
    narrative: str
    iterations: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    final_output: dict[str, Any] | None = None
    error: str | None = None
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    terminated_reason: str = "completed"  # completed | max_iterations | error | budget_exceeded

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Client protocol — both the real and mock client conform to this
# -----------------------------------------------------------------------------


class Client(Protocol):
    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send a generation request. Returns a dict with keys:
        - 'text': str | None — assistant prose, if any
        - 'tool_calls': list[{'name': str, 'arguments': dict}] — pending tool calls
        - 'input_tokens': int
        - 'output_tokens': int
        """
        ...


# -----------------------------------------------------------------------------
# GeminiClient — live Vertex AI implementation
# -----------------------------------------------------------------------------


class GeminiClient:
    """Vertex AI client using the google-genai SDK.

    Requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION env vars, and
    application-default credentials set up (gcloud auth application-default
    login). Will lazy-import google-genai so the module can be loaded in
    environments without the dep — useful for the mock-only test path.
    """

    def __init__(self, model: str = "gemini-2.5-pro", temperature: float = 0.0):
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "google-genai is not installed. Install with: pip install google-genai"
            ) from e

        import os

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable is required.")

        self._genai = genai
        self._types = types
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._model = model
        self._temperature = temperature

    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        types = self._types
        # Convert our internal tool schemas to Gemini's FunctionDeclaration format.
        function_declarations = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"],
            )
            for t in tools
        ]
        tool_config = types.Tool(function_declarations=function_declarations)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
            tools=[tool_config],
        )

        # Build Content list from messages. Messages are {role, parts} dicts.
        contents = [
            types.Content(role=m["role"], parts=m["parts"]) for m in messages
        ]

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        # Parse response — Gemini returns either text or function_call parts.
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_calls.append(
                        {
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args or {}),
                        }
                    )
                elif hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

        usage = response.usage_metadata
        return {
            "text": "\n".join(text_parts) if text_parts else None,
            "tool_calls": tool_calls,
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        }


# -----------------------------------------------------------------------------
# MockClient — deterministic stand-in for local testing
# -----------------------------------------------------------------------------


class MockClient:
    """A scripted client that walks through a fixed sequence of responses.

    Useful for testing the agent loop without GCP credentials. The script
    is a list of response dicts; the client returns them in order on each
    .generate() call. If the script runs out, returns a terminal 'I'm done'
    response with a placeholder classification.
    """

    def __init__(self, script: list[dict[str, Any]]):
        self._script = list(script)
        self._call_count = 0

    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._call_count += 1
        if self._script:
            return self._script.pop(0)
        # Fallback: terminate with a placeholder final output.
        placeholder = {
            "category": 5,
            "sub_pattern": None,
            "subject_frame": "first_person",
            "criteria_cited": ["5a", "5b", "5c"],
            "action": "clarify_or_conservative_escalation",
            "reasoning": "Mock client exhausted script; emitting placeholder.",
        }
        return {
            "text": "```json\n" + json.dumps(placeholder) + "\n```",
            "tool_calls": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }


# -----------------------------------------------------------------------------
# Output parsing
# -----------------------------------------------------------------------------


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of model text.

    Looks for fenced ```json blocks first; falls back to the first { ... }
    block in the text. Returns None if no parseable JSON is found.
    """
    if not text:
        return None

    # Try fenced block.
    fence_start = text.find("```json")
    if fence_start != -1:
        body_start = text.find("\n", fence_start) + 1
        fence_end = text.find("```", body_start)
        if fence_end != -1:
            try:
                return json.loads(text[body_start:fence_end].strip())
            except json.JSONDecodeError:
                pass

    # Try first balanced { ... } in text.
    first_brace = text.find("{")
    if first_brace != -1:
        depth = 0
        for i in range(first_brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[first_brace : i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


REQUIRED_OUTPUT_FIELDS = {"category", "sub_pattern", "subject_frame", "criteria_cited", "action", "reasoning"}


def _is_valid_final_output(obj: dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    return REQUIRED_OUTPUT_FIELDS.issubset(obj.keys())


# -----------------------------------------------------------------------------
# The agent loop
# -----------------------------------------------------------------------------


def run_agent(
    narrative: str,
    client: Client,
    case_id: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
    max_input_tokens: int = MAX_INPUT_TOKEN_BUDGET,
) -> Trace:
    """Run the agent loop on a single narrative. Returns a full Trace.

    max_input_tokens is the per-case cumulative input-token guardrail; the
    loop terminates with `budget_exceeded` once it is crossed. Defaults to
    the module constant but the runner passes the value from config.yaml.
    """
    trace = Trace(case_id=case_id, narrative=narrative, iterations=0)
    start = time.monotonic()

    messages: list[dict[str, Any]] = [
        {"role": "user", "parts": [{"text": narrative}]}
    ]

    try:
        for iteration in range(1, max_iterations + 1):
            trace.iterations = iteration

            response = client.generate(
                system_prompt=SYSTEM_PROMPT,
                messages=messages,
                tools=ALL_SCHEMAS,
            )

            trace.input_tokens += response.get("input_tokens", 0)
            trace.output_tokens += response.get("output_tokens", 0)

            if trace.input_tokens > max_input_tokens:
                trace.terminated_reason = "budget_exceeded"
                trace.error = f"Input token budget {max_input_tokens} exceeded."
                break

            text = response.get("text")
            tool_calls = response.get("tool_calls") or []

            if text:
                trace.thoughts.append(text)

            # If the model produced text AND no tool calls, look for a final output.
            if text and not tool_calls:
                parsed = _extract_json_block(text)
                if parsed and _is_valid_final_output(parsed):
                    trace.final_output = parsed
                    trace.terminated_reason = "completed"
                    break
                # Otherwise the model is rambling without committing — push it.
                messages.append({"role": "model", "parts": [{"text": text}]})
                messages.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "You did not produce a valid structured output. "
                                    "Emit a JSON object with fields: category, sub_pattern, "
                                    "subject_frame, criteria_cited, action, reasoning."
                                )
                            }
                        ],
                    }
                )
                continue

            # Dispatch any tool calls and append results.
            if tool_calls:
                # Record the model turn (text + tool calls) in history.
                model_parts: list[dict[str, Any]] = []
                if text:
                    model_parts.append({"text": text})
                for call in tool_calls:
                    model_parts.append(
                        {"function_call": {"name": call["name"], "args": call["arguments"]}}
                    )
                messages.append({"role": "model", "parts": model_parts})

                # Execute each tool, record on trace, append responses.
                user_parts: list[dict[str, Any]] = []
                for call in tool_calls:
                    result = dispatch(call["name"], call["arguments"])
                    trace.tool_calls.append(
                        ToolCall(
                            name=call["name"],
                            arguments=call["arguments"],
                            result=result,
                            iteration=iteration,
                        )
                    )
                    user_parts.append(
                        {"function_response": {"name": call["name"], "response": result}}
                    )
                messages.append({"role": "user", "parts": user_parts})
                continue

            # No text and no tool calls — degenerate response. Stop.
            trace.terminated_reason = "error"
            trace.error = "Model returned no text and no tool calls."
            break
        else:
            # Loop exhausted without break.
            trace.terminated_reason = "max_iterations"

    except Exception as e:
        trace.terminated_reason = "error"
        trace.error = f"{type(e).__name__}: {e}"

    trace.latency_seconds = round(time.monotonic() - start, 3)
    return trace