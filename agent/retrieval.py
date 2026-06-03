"""
Retrieval backends for the `search_reference` tool.

Two interchangeable implementations behind a common `Retriever` protocol:

  - KeywordRetriever — token-overlap ranking. No credentials, deterministic,
    the default. (This is the original `search_reference` ranking, lifted here
    unchanged so the tool's behavior is identical when nothing is swapped.)
  - VectorRetriever — embedding cosine-similarity ranking. Semantic; needs an
    Embedder. Corpus vectors are computed once and cached to disk, keyed by a
    fingerprint of (embedding model + corpus content) so the cache self-invalidates
    when either changes. Only the query is embedded per call after that.

This mirrors the GeminiClient / MockClient split in `agent.py`: the live vector
path uses Vertex embeddings via the same `google-genai` SDK and the same GCP
credentials as the agent, while a deterministic `MockEmbedder` lets the vector
path be exercised offline in tests with no network or credentials.

Both retrievers return the same shape — a list of result dicts, highest score
first — so `search_reference` can wrap either identically:

    [{"id": str, "title": str, "content": str, "match_score": float}, ...]

NOTE on the live embedder: `VertexEmbedder.embed` targets the unified google-genai
SDK (`client.models.embed_content`). Exact method/response shape and the current
Vertex embedding model id can vary by SDK version — verify against your installed
version on the first live run. Nothing else in the codebase depends on the live
embedder being correct: the keyword default and all tests use no embeddings or the
mock embedder.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


class Retriever(Protocol):
    """Ranks reference docs against a query. Returns result dicts, best first."""

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]: ...


class Embedder(Protocol):
    """Maps texts to dense vectors. `model_id` participates in the cache key."""

    @property
    def model_id(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _result(doc: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "id": doc["id"],
        "title": doc["title"],
        "content": doc["content"],
        "match_score": round(float(score), 3),
    }


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _doc_text(doc: dict[str, Any]) -> str:
    """The text representation of a doc that gets embedded / searched."""
    return doc["title"] + "\n" + " ".join(doc["keywords"]) + "\n" + doc["content"]


# -----------------------------------------------------------------------------
# Keyword retriever (the default — original search_reference ranking)
# -----------------------------------------------------------------------------


class KeywordRetriever:
    """Rank by token overlap between the query and each doc's title + keywords.

    Normalized by query length so longer queries don't dominate. Returns only
    docs with nonzero overlap (an empty list signals "no match" to the caller).
    """

    def __init__(self, docs: list[dict[str, Any]]):
        self._docs = docs

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self._docs:
            haystack = doc["title"] + " " + " ".join(doc["keywords"])
            overlap = len(query_tokens & _tokenize(haystack))
            if overlap > 0:
                scored.append((overlap / len(query_tokens), doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_result(doc, score) for score, doc in scored[: max(1, top_k)]]


# -----------------------------------------------------------------------------
# Vector retriever (semantic — embedding cosine similarity)
# -----------------------------------------------------------------------------


class VectorRetriever:
    """Rank by cosine similarity between query and doc embeddings.

    Corpus embeddings are built once and cached to `cache_path` (JSON), keyed by
    a fingerprint of the embedding model id plus corpus content. A fingerprint
    mismatch (model changed or corpus edited) triggers a rebuild.
    """

    def __init__(
        self,
        docs: list[dict[str, Any]],
        embedder: Embedder,
        cache_path: str | Path | None = None,
    ):
        self._docs = docs
        self._embedder = embedder
        self._cache_path = Path(cache_path) if cache_path else None
        if self._cache_path and not self._cache_path.is_absolute():
            self._cache_path = PROJECT_ROOT / self._cache_path
        self._matrix: list[list[float]] = []
        self._load_or_build()

    def _fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self._embedder.model_id.encode())
        for doc in self._docs:
            h.update(doc["id"].encode())
            h.update(_doc_text(doc).encode())
        return h.hexdigest()

    def _load_or_build(self) -> None:
        fp = self._fingerprint()
        if self._cache_path and self._cache_path.exists():
            try:
                cached = json.loads(self._cache_path.read_text())
                if cached.get("fingerprint") == fp and len(cached.get("vectors", [])) == len(self._docs):
                    self._matrix = cached["vectors"]
                    return
            except (json.JSONDecodeError, OSError):
                pass  # fall through to rebuild
        # Build embeddings for the corpus.
        self._matrix = self._embedder.embed([_doc_text(d) for d in self._docs])
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(
                    {"fingerprint": fp, "model": self._embedder.model_id, "vectors": self._matrix}
                )
            )

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_vec = self._embedder.embed([query])[0]
        scored = [
            (_cosine(query_vec, doc_vec), doc)
            for doc_vec, doc in zip(self._matrix, self._docs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_result(doc, score) for score, doc in scored[: max(1, top_k)]]


# -----------------------------------------------------------------------------
# Embedders
# -----------------------------------------------------------------------------


class VertexEmbedder:
    """Live embedder using Vertex AI via the google-genai SDK.

    Reuses the same client construction and credentials as GeminiClient: requires
    GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_LOCATION) plus application-
    default credentials. Lazy-imports google-genai so this module loads fine in
    mock-only / no-dependency environments.
    """

    def __init__(self, model: str = "text-embedding-004", dimensions: int | None = None):
        try:
            from google import genai  # type: ignore
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
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._model = model
        self._dimensions = dimensions

    @property
    def model_id(self) -> str:
        # Dimensions affect the vectors, so fold them into the cache identity.
        return f"{self._model}@{self._dimensions}" if self._dimensions else self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types  # type: ignore

        config = None
        if self._dimensions:
            config = types.EmbedContentConfig(output_dimensionality=self._dimensions)
        resp = self._client.models.embed_content(
            model=self._model, contents=texts, config=config
        )
        # Unified SDK returns resp.embeddings (list), each with .values.
        return [list(e.values) for e in resp.embeddings]


class MockEmbedder:
    """Deterministic, offline embedder for tests — lexical, NOT semantic.

    Hashes tokens into a fixed-dimension bag-of-words vector, so texts that share
    tokens get positive cosine similarity. Enough to exercise VectorRetriever's
    ranking and caching with no network or credentials. It does not model meaning,
    so it is not a stand-in for real embedding quality.
    """

    def __init__(self, dim: int = 64, model_id: str = "mock-embedder-v1"):
        self._dim = dim
        self._model = model_id

    @property
    def model_id(self) -> str:
        return f"{self._model}@{self._dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for tok in _tokenize(text):
                idx = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % self._dim
                vec[idx] += 1.0
            out.append(vec)
        return out


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def build_retriever(
    mode: str,
    config: dict[str, Any],
    docs: list[dict[str, Any]],
    *,
    embedder: Embedder | None = None,
) -> Retriever:
    """Construct a retriever by mode.

    mode="keyword" → KeywordRetriever (no embedder needed).
    mode="vector"  → VectorRetriever. Uses the provided `embedder` if given
                     (tests pass a MockEmbedder); otherwise builds a VertexEmbedder
                     from the `retrieval:` block in config.
    """
    if mode == "keyword":
        return KeywordRetriever(docs)
    if mode == "vector":
        retr_cfg = config.get("retrieval", {}) or {}
        if embedder is None:
            embedder = VertexEmbedder(
                model=retr_cfg.get("embedding_model", "text-embedding-004"),
                dimensions=retr_cfg.get("embedding_dimensions"),
            )
        return VectorRetriever(docs, embedder, cache_path=retr_cfg.get("cache_path"))
    raise ValueError(f"Unknown retrieval mode: {mode!r}. Options: 'keyword', 'vector'.")
