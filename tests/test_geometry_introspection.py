"""Tests for B-Rep introspection parsing/formatting (no cadquery needed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry_introspection import (
    parse_report,
    format_report,
    _classify_edge,
    _classify_face,
    _MARKER,
)


# A synthetic measured report resembling a 10x10x10 box.
_BOX_REPORT = {
    "num_faces": 6,
    "num_edges": 12,
    "faces": [
        {"index": 0, "center": [0, 0, 5], "normal": [0, 0, 1], "area": 100, "type": "PLANE"},
        {"index": 1, "center": [0, 0, -5], "normal": [0, 0, -1], "area": 100, "type": "PLANE"},
        {"index": 2, "center": [5, 0, 0], "normal": [1, 0, 0], "area": 100, "type": "PLANE"},
    ],
    "edges": (
        [{"index": i, "length": 10, "center": [0, 0, 0], "type": "LINE", "dir": [0, 0, 10]} for i in range(4)]
        + [{"index": i, "length": 10, "center": [0, 0, 0], "type": "LINE", "dir": [10, 0, 0]} for i in range(4)]
        + [{"index": i, "length": 10, "center": [0, 0, 0], "type": "LINE", "dir": [0, 10, 0]} for i in range(4)]
    ),
}


def test_parse_report_from_stdout():
    import json
    line = f"{_MARKER} {json.dumps(_BOX_REPORT)}"
    stdout = "[MODEL_BUILT]\n" + line + "\nother log\n"
    report = parse_report(stdout)
    assert report is not None and report["num_faces"] == 6


def test_parse_report_missing_returns_none():
    assert parse_report("no marker here") is None


def test_classify_vertical_edge():
    assert _classify_edge({"type": "LINE", "dir": [0, 0, 10]}) == "vertical (|Z)"


def test_classify_curved_edge():
    assert _classify_edge({"type": "CIRCLE", "dir": None}) == "curved"


def test_classify_top_face():
    assert _classify_face({"normal": [0, 0, 1], "type": "PLANE"}) == "top (>Z)"


def test_classify_cylinder_face():
    out = _classify_face({"normal": [0.7, 0.7, 0], "type": "CYLINDER"})
    assert "cylinder" in out or out == "angled"


def test_format_report_mentions_counts_and_hints():
    text = format_report(_BOX_REPORT)
    assert "6 faces" in text and "12 edges" in text
    assert "vertical (|Z)" in text
    assert "|Z" in text  # selector hint
    assert "fillet/chamfer" in text  # safe-size advice


def test_format_report_handles_error():
    assert "error" in format_report({"error": "model not defined"}).lower()


def test_format_report_handles_none():
    assert "unavailable" in format_report(None).lower()


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run()
