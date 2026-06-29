"""Tests for the CadQuery API RAG knowledge base (offline keyword path)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cad_knowledge import CADKnowledgeBase, KNOWLEDGE_BASE, retrieve_api_docs


def test_retrieve_hole_intent():
    hits = KNOWLEDGE_BASE.retrieve("drill a centered hole through the plate", k=3)
    names = {h.name for h in hits}
    assert any("hole" in n for n in names), names


def test_retrieve_fillet_intent():
    hits = KNOWLEDGE_BASE.retrieve("round the vertical edges of the box", k=4)
    names = {h.name for h in hits}
    assert "fillet" in names or "selectors" in names, names


def test_retrieve_shell_intent():
    hits = KNOWLEDGE_BASE.retrieve("hollow out the box into a thin container", k=3)
    assert any(h.name == "shell" for h in hits), [h.name for h in hits]


def test_block_format_contains_signature_and_gotcha():
    block = retrieve_api_docs("counterbored hole for a cap screw", k=2)
    assert "cboreHole" in block
    assert "⚠️" in block  # gotcha line present
    assert "example:" in block


def test_empty_query_returns_nothing():
    assert KNOWLEDGE_BASE.retrieve("", k=3) == []


def test_lookup_fallback_string_for_no_match():
    kb = CADKnowledgeBase()
    out = kb.lookup("zzzz qqqq vvvv nomatch", k=3)
    # Either a relevant block or the graceful fallback sentence.
    assert isinstance(out, str) and len(out) > 0


def test_deterministic_order():
    a = [h.name for h in KNOWLEDGE_BASE.retrieve("extrude a cylinder", k=5)]
    b = [h.name for h in KNOWLEDGE_BASE.retrieve("extrude a cylinder", k=5)]
    assert a == b


def test_embedding_none_does_not_poison_cache():
    # An embed_fn that yields no vectors must not lock the cache, and retrieval
    # must still succeed via the keyword fallback.
    kb = CADKnowledgeBase()
    hits = kb.retrieve("drill a hole", k=3, embed_fn=lambda text, query=False: None)
    assert any("hole" in h.name for h in hits)
    assert kb._embedded is False  # retryable, not permanently downgraded


def test_embedding_exception_falls_back_to_keyword():
    kb = CADKnowledgeBase()

    def boom(text, query=False):
        raise RuntimeError("embeddings unavailable")

    hits = kb.retrieve("hollow shell container", k=3, embed_fn=boom)
    assert any(h.name == "shell" for h in hits)
    assert kb._embedded is False


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run()
