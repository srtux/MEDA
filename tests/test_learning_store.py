"""Unit tests for the learning store and lesson-signature normalizer.

These run fully offline (no genai client / no API key): embeddings are absent,
so the store exercises its keyword + signature fallback paths.
Run: ``python tests/test_learning_store.py`` or via pytest.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.learning_store import LearningStore
from core.lesson_signature import to_signature


def _store():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return LearningStore(genai_client=None, db_path=tmp.name)


def test_signature_mapping():
    assert to_signature("NameError: name 'model'") == "compile:name_error"
    assert to_signature("num_faces (expected: 6, got: 7)") == "topology:num_faces_mismatch"
    assert to_signature("volume (expected: 10, got: 9)") == "topology:volume_mismatch"
    assert to_signature("Visual validation failure: detached") == "visual:mismatch"
    assert to_signature("[EXPORT_ERROR] boom") == "export:error"
    assert to_signature("") == "unknown"


def test_record_and_retrieve_lesson():
    s = _store()
    s.record_lesson("compile:name_error", "NameError", "must bind model",
                    "model = cq.Workplane('XY').box(1,1,1)", "make a box")
    hits = s.retrieve_lessons("box model name error", k=3, signature="compile:name_error")
    assert hits and hits[0]["error_signature"] == "compile:name_error"
    s.close()


def test_lesson_dedup_merges():
    s = _store()
    a = s.record_lesson("compile:name_error", "NameError", "bind model",
                        "fix v1", "make a widget plate")
    b = s.record_lesson("compile:name_error", "NameError", "bind model",
                        "fix v2", "make a widget plate")
    assert a == b  # merged, not duplicated
    assert s.counts()["lessons"] == 1
    s.close()


def test_confidence_feedback_moves():
    s = _store()
    lid = s.record_lesson("topology:volume_mismatch", "vol", "scale up",
                          "set_parameter", "a cube of 1000mm3")
    # New lessons start at 0.5 confidence (below retrieval floor until proven).
    for _ in range(3):
        s.feedback(lid, helped=True)
    row = s.con.execute("SELECT confidence FROM lessons WHERE id=?", (lid,)).fetchone()
    assert row[0] > 0.5  # positive feedback raises confidence
    s.close()


def test_retrieve_skills_keyword_fallback():
    s = _store()
    s.record_skill("centered_through_hole", "Drill a centered through-hole in a plate",
                   "(diameter)", "model = model.faces('>Z').workplane().hole(diameter)")
    s.record_skill("fillet_all_edges", "Round all edges of the solid",
                   "(radius)", "model = model.edges().fillet(radius)")
    hits = s.retrieve_skills("I want to drill a hole through the plate", k=1)
    assert hits and hits[0]["name"] == "centered_through_hole"
    s.close()


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run()
