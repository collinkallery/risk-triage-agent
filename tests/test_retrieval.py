"""Tests for retrieval backends, embedding cache, and search_reference delegation.

All offline: keyword needs nothing, vector uses the deterministic MockEmbedder.
"""

from __future__ import annotations

import json

import pytest

import agent.tools as tools
from agent.retrieval import (
    KeywordRetriever,
    MockEmbedder,
    VectorRetriever,
    build_retriever,
)

DOCS = [
    {"id": "d1", "title": "euphemistic finality language", "keywords": ["euphemistic", "1e"], "content": "About euphemistic phrasing."},
    {"id": "d2", "title": "temporal contrast recovery", "keywords": ["historical", "recovery", "past"], "content": "About recovery context."},
    {"id": "d3", "title": "imminent risk means and timeframe", "keywords": ["imminent", "means", "timeframe"], "content": "About imminent risk."},
]


# -----------------------------------------------------------------------------
# KeywordRetriever (original behavior preserved)
# -----------------------------------------------------------------------------


def test_keyword_ranks_overlap_first():
    r = KeywordRetriever(DOCS).search("euphemistic finality", 3)
    assert r[0]["id"] == "d1"
    assert r[0]["match_score"] > 0


def test_keyword_no_overlap_returns_empty():
    assert KeywordRetriever(DOCS).search("zzzqqq nonexistenttoken", 3) == []


def test_keyword_tokenless_query_returns_empty():
    assert KeywordRetriever(DOCS).search("!!! ???", 3) == []


def test_keyword_result_shape():
    r = KeywordRetriever(DOCS).search("imminent timeframe", 3)
    assert set(r[0].keys()) == {"id", "title", "content", "match_score"}


# -----------------------------------------------------------------------------
# VectorRetriever with MockEmbedder
# -----------------------------------------------------------------------------


def test_vector_returns_top_k_ranked():
    r = VectorRetriever(DOCS, MockEmbedder()).search("recovery past historical", 2)
    assert len(r) == 2
    # Scores are sorted descending.
    assert r[0]["match_score"] >= r[1]["match_score"]


def test_vector_overlap_ranks_relevant_doc_top():
    # Query shares all tokens with d2; MockEmbedder is lexical, so d2 should win.
    r = VectorRetriever(DOCS, MockEmbedder()).search("historical recovery past", 3)
    assert r[0]["id"] == "d2"


def test_vector_deterministic():
    a = VectorRetriever(DOCS, MockEmbedder()).search("imminent means", 3)
    b = VectorRetriever(DOCS, MockEmbedder()).search("imminent means", 3)
    assert [(x["id"], x["match_score"]) for x in a] == [(x["id"], x["match_score"]) for x in b]


# -----------------------------------------------------------------------------
# Embedding cache
# -----------------------------------------------------------------------------


def test_cache_is_written_and_reused(tmp_path):
    cache = tmp_path / "emb.json"
    VectorRetriever(DOCS, MockEmbedder(), cache_path=cache)
    assert cache.exists()
    payload = json.loads(cache.read_text())
    assert len(payload["vectors"]) == len(DOCS)
    assert "fingerprint" in payload

    # A second retriever with a counting embedder must NOT re-embed (cache hit).
    class CountingEmbedder(MockEmbedder):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return super().embed(texts)

    ce = CountingEmbedder()
    VectorRetriever(DOCS, ce, cache_path=cache)
    assert ce.calls == 0  # corpus loaded from cache, not rebuilt


def test_cache_invalidates_on_model_change(tmp_path):
    cache = tmp_path / "emb.json"
    VectorRetriever(DOCS, MockEmbedder(model_id="model-a"), cache_path=cache)
    fp_a = json.loads(cache.read_text())["fingerprint"]
    # Different model id → different fingerprint → rebuild on next construction.
    VectorRetriever(DOCS, MockEmbedder(model_id="model-b"), cache_path=cache)
    fp_b = json.loads(cache.read_text())["fingerprint"]
    assert fp_a != fp_b


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def test_build_retriever_keyword():
    assert isinstance(build_retriever("keyword", {}, DOCS), KeywordRetriever)


def test_build_retriever_vector_with_injected_embedder():
    r = build_retriever("vector", {"retrieval": {}}, DOCS, embedder=MockEmbedder())
    assert isinstance(r, VectorRetriever)


def test_build_retriever_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_retriever("semantic", {}, DOCS)


# -----------------------------------------------------------------------------
# search_reference delegates to the active retriever
# -----------------------------------------------------------------------------


def test_set_retriever_swaps_backend():
    original = tools.get_retriever()
    try:
        tools.set_retriever(VectorRetriever(DOCS, MockEmbedder()))
        out = tools.search_reference("imminent means timeframe")
        assert "results" in out and out["results"]
        # Vector backend always returns up to top_k, even on loose matches.
        assert len(out["results"]) >= 1
    finally:
        tools.set_retriever(original)


def test_search_reference_empty_query_still_validated():
    # Validation lives in search_reference, independent of backend.
    assert "error" in tools.search_reference("   ")
