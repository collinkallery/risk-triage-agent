
"""
Tools for the Behavioral Risk Triage Agent.
 
Three tools, each a plain Python function plus a JSON schema declaration
for Gemini's function-calling API:
 
  - lookup_rubric(category_id): returns the rubric definition and criteria
    for a given category. Forces the agent to ground its reasoning in the
    rubric rather than its priors.
 
  - search_reference(query): retrieval over a small in-memory reference
    corpus (~12 docs). Delegates ranking to the active Retriever (see
    agent/retrieval.py): KeywordRetriever by default, or a VectorRetriever
    (embedding similarity) when one is installed via set_retriever().
 
  - resolve_escalation(category_id, subject_frame): returns the
    category-and-frame-aware action path. Forces the agent to commit to
    a classification before stating the action.
 
Schemas are written in Gemini's function-calling format (subset of
JSON Schema). Tool execution is local and synchronous.
"""
 
from __future__ import annotations
 
from pathlib import Path
from typing import Any
 
import yaml
 
from agent.retrieval import KeywordRetriever, Retriever
 
# -----------------------------------------------------------------------------
# Data loading (cached at module import)
# -----------------------------------------------------------------------------
 
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
 
 
def _load_yaml(filename: str) -> dict[str, Any]:
    with open(_DATA_DIR / filename) as f:
        return yaml.safe_load(f)
 
 
_RUBRIC = _load_yaml("rubric.yaml")
_REFERENCE_CORPUS = _load_yaml("reference_corpus.yaml")
 
 
# -----------------------------------------------------------------------------
# Tool: lookup_rubric
# -----------------------------------------------------------------------------
 
 
def lookup_rubric(category_id: int) -> dict[str, Any]:
    """Return the rubric definition for a single category.
 
    Includes criteria, sub_patterns (if any), positive examples, near-miss
    negatives, and the mapped action. The agent should call this whenever
    it needs to apply criteria to a specific case — do not rely on memory.
    """
    if not isinstance(category_id, int) or category_id not in {1, 2, 3, 4, 5}:
        return {"error": f"Invalid category_id {category_id!r}; must be one of 1, 2, 3, 4, 5."}
 
    for cat in _RUBRIC["categories"]:
        if cat["id"] == category_id:
            return {
                "id": cat["id"],
                "name": cat["name"],
                "action": cat["action"],
                "description": cat["description"],
                "criteria": cat["criteria"],
                "sub_patterns": cat.get("sub_patterns"),
                "positive_examples": cat.get("positive_examples", []),
                "near_miss_negatives": cat.get("near_miss_negatives", []),
            }
    return {"error": f"Category {category_id} not found in rubric."}
 
 
LOOKUP_RUBRIC_SCHEMA = {
    "name": "lookup_rubric",
    "description": (
        "Retrieve the full rubric definition for a single triage category, "
        "including criteria, sub-patterns, examples, and near-miss negatives. "
        "Call this whenever applying criteria to a case."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category_id": {
                "type": "integer",
                "description": "Category to look up. One of 1, 2, 3, 4, 5.",
            }
        },
        "required": ["category_id"],
    },
}
 
 
# -----------------------------------------------------------------------------
# Tool: search_reference
# -----------------------------------------------------------------------------
 
# The active retrieval backend. Defaults to keyword ranking so the tool needs no
# credentials and behaves identically out of the box. The runner / triage CLIs
# can swap in a VectorRetriever via set_retriever() for a live run.
_ACTIVE_RETRIEVER: Retriever = KeywordRetriever(_REFERENCE_CORPUS["docs"])
 
 
def set_retriever(retriever: Retriever) -> None:
    """Install the retrieval backend used by search_reference."""
    global _ACTIVE_RETRIEVER
    _ACTIVE_RETRIEVER = retriever
 
 
def get_retriever() -> Retriever:
    return _ACTIVE_RETRIEVER
 
 
def search_reference(query: str, top_k: int = 3) -> dict[str, Any]:
    """Retrieve relevant docs from the reference corpus.
 
    Ranking is delegated to the active Retriever (keyword by default, or vector
    similarity if one is installed). The query is validated here; the return
    shape is identical regardless of backend.
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "Query must be a non-empty string."}
 
    k = top_k if isinstance(top_k, int) and top_k > 0 else 3
    results = _ACTIVE_RETRIEVER.search(query, k)
 
    if not results:
        return {
            "query": query,
            "results": [],
            "note": "No reference docs matched. Consider rephrasing the query or proceeding with the rubric alone.",
        }
 
    return {"query": query, "results": results}
 
 
SEARCH_REFERENCE_SCHEMA = {
    "name": "search_reference",
    "description": (
        "Search a small reference corpus of clinical principles, category "
        "boundaries, and common misclassifications. Use when you want "
        "additional context beyond the rubric — e.g., guidance on a "
        "specific boundary case or a known anti-pattern."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query describing what you're looking for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of docs to return. Defaults to 3.",
            },
        },
        "required": ["query"],
    },
}
 
 
# -----------------------------------------------------------------------------
# Tool: resolve_escalation
# -----------------------------------------------------------------------------
 
 
def resolve_escalation(category_id: int, subject_frame: str) -> dict[str, Any]:
    """Return the action path for a category-and-frame combination.
 
    Forces the agent to commit to a category and frame before stating
    the action. The scorer can check whether the right tool was called
    for the right classification.
    """
    valid_frames = {"first_person", "third_party_clear", "third_party_ambiguous"}
    if category_id not in {1, 2, 3, 4, 5}:
        return {"error": f"Invalid category_id {category_id!r}."}
    if subject_frame not in valid_frames:
        return {"error": f"Invalid subject_frame {subject_frame!r}. Must be one of {sorted(valid_frames)}."}
 
    for cat in _RUBRIC["categories"]:
        if cat["id"] == category_id:
            base_action = cat["action"]
            action_def = _RUBRIC["actions"].get(base_action, {})
            frame_variants = action_def.get("frame_variants", {})
            frame_specific = frame_variants.get(subject_frame)
            return {
                "category_id": category_id,
                "subject_frame": subject_frame,
                "action": base_action,
                "action_description": action_def.get("description"),
                "frame_specific_guidance": frame_specific,
                "expected_tools": action_def.get("expected_tools", []),
            }
    return {"error": f"Category {category_id} not found."}
 
 
RESOLVE_ESCALATION_SCHEMA = {
    "name": "resolve_escalation",
    "description": (
        "Resolve the action path for a given category and subject frame. "
        "Call this once you have committed to a category and frame to "
        "confirm the exact action string and any frame-specific guidance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category_id": {
                "type": "integer",
                "description": "Committed category. One of 1, 2, 3, 4, 5.",
            },
            "subject_frame": {
                "type": "string",
                "description": "Committed subject frame.",
                "enum": ["first_person", "third_party_clear", "third_party_ambiguous"],
            },
        },
        "required": ["category_id", "subject_frame"],
    },
}
 
 
# -----------------------------------------------------------------------------
# Registry — used by the agent loop to dispatch tool calls
# -----------------------------------------------------------------------------
 
TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "lookup_rubric": {"fn": lookup_rubric, "schema": LOOKUP_RUBRIC_SCHEMA},
    "search_reference": {"fn": search_reference, "schema": SEARCH_REFERENCE_SCHEMA},
    "resolve_escalation": {"fn": resolve_escalation, "schema": RESOLVE_ESCALATION_SCHEMA},
}
 
ALL_SCHEMAS = [t["schema"] for t in TOOL_REGISTRY.values()]
 
 
def dispatch(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Look up a tool by name and execute it with the given arguments.
 
    Returns a dict on success, or {"error": "..."} on failure. Never
    raises — errors become structured tool outputs the agent can react to.
    """
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool {tool_name!r}. Available: {sorted(TOOL_REGISTRY)}."}
 
    try:
        return TOOL_REGISTRY[tool_name]["fn"](**arguments)
    except TypeError as e:
        return {"error": f"Bad arguments to {tool_name}: {e}"}
    except Exception as e:
        return {"error": f"Tool {tool_name} raised {type(e).__name__}: {e}"}
