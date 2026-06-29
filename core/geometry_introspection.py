"""B-Rep selector grounding: let the modeler *see* real faces and edges.

The competitive roadmap (``docs/text_to_cad_landscape_2026.md`` §8.3) flags
**advanced-feature grounding** — resolving a natural-language geometric query to
specific faces/edges before applying fillet/chamfer/shell — as the field's
explicit frontier (Text2CAD-Bench, the BRepGround paper arXiv:2603.11831). Every
system currently fails here because the LLM picks selectors *blind*: it guesses
``.edges('|Z')`` without knowing how many vertical edges exist or where they
are, so fillets land on the wrong edges or overflow the kernel.

Instead of a trained BERT+UV-Net resolver, MEDA exposes the **actual kernel
measurements** (à la CADSmith's exact OpenCASCADE validation): we run the
current timeline and report each face's normal/center/area and each edge's
length/orientation, plus ready-to-use CadQuery selector hints. The agent then
chooses selectors grounded in geometry that really exists.

``INTROSPECTION_SNIPPET`` runs *inside the sandbox* (where cadquery is
available); the parse/format helpers run in the host process and need no heavy
deps, so this module imports cleanly anywhere.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

_MARKER = "[GEOMETRY_JSON]"

# Executed in the sandbox after the model is built. Pure cadquery/OCP; every
# measurement is wrapped so a single odd face/edge never aborts the report.
INTROSPECTION_SNIPPET = r'''
# === GEOMETRY INTROSPECTION ===
import json as _gj
def _meda_introspect(_m):
    _solid = _m.val() if hasattr(_m, 'val') else _m
    _faces, _edges = [], []
    try:
        _all_faces = _solid.Faces()
    except Exception:
        _all_faces = []
    for _i, _f in enumerate(_all_faces):
        _entry = {"index": _i}
        try:
            _c = _f.Center();  _entry["center"] = [round(_c.x, 3), round(_c.y, 3), round(_c.z, 3)]
        except Exception:
            _entry["center"] = None
        try:
            _n = _f.normalAt();  _entry["normal"] = [round(_n.x, 3), round(_n.y, 3), round(_n.z, 3)]
        except Exception:
            _entry["normal"] = None
        try:
            _entry["area"] = round(float(_f.Area()), 3)
        except Exception:
            _entry["area"] = None
        try:
            _entry["type"] = str(_f.geomType())
        except Exception:
            _entry["type"] = None
        _faces.append(_entry)
    try:
        _all_edges = _solid.Edges()
    except Exception:
        _all_edges = []
    for _i, _e in enumerate(_all_edges):
        _entry = {"index": _i}
        try:
            _entry["length"] = round(float(_e.Length()), 3)
        except Exception:
            _entry["length"] = None
        try:
            _c = _e.Center();  _entry["center"] = [round(_c.x, 3), round(_c.y, 3), round(_c.z, 3)]
        except Exception:
            _entry["center"] = None
        try:
            _entry["type"] = str(_e.geomType())
        except Exception:
            _entry["type"] = None
        try:
            _sp = _e.startPoint();  _ep = _e.endPoint()
            _dx, _dy, _dz = _ep.x - _sp.x, _ep.y - _sp.y, _ep.z - _sp.z
            _entry["dir"] = [round(_dx, 3), round(_dy, 3), round(_dz, 3)]
        except Exception:
            _entry["dir"] = None
        _edges.append(_entry)
    return {"faces": _faces, "edges": _edges, "num_faces": len(_faces), "num_edges": len(_edges)}
try:
    if 'model' in dir():
        print("[GEOMETRY_JSON] " + _gj.dumps(_meda_introspect(model)))
    else:
        print("[GEOMETRY_JSON] " + _gj.dumps({"error": "model not defined"}))
except Exception as _ge:
    print("[GEOMETRY_JSON] " + _gj.dumps({"error": str(_ge)}))
'''


def parse_report(stdout: str) -> Optional[Dict[str, Any]]:
    """Extract the geometry report dict from sandbox stdout, or ``None``."""
    if not stdout:
        return None
    for line in stdout.splitlines():
        if line.startswith(_MARKER):
            try:
                return json.loads(line[len(_MARKER):].strip())
            except Exception:
                return None
    return None


def _classify_edge(edge: Dict[str, Any]) -> str:
    etype = (edge.get("type") or "").upper()
    if "CIRCLE" in etype or "ARC" in etype or "ELLIPSE" in etype or "BSPLINE" in etype:
        return "curved"
    d = edge.get("dir")
    if not d:
        return "other"
    ax, ay, az = abs(d[0]), abs(d[1]), abs(d[2])
    m = max(ax, ay, az)
    if m == 0:
        return "other"
    if az == m and ax < 1e-6 and ay < 1e-6:
        return "vertical (|Z)"
    if ax == m and ay < 1e-6 and az < 1e-6:
        return "horizontal-X"
    if ay == m and ax < 1e-6 and az < 1e-6:
        return "horizontal-Y"
    return "diagonal"


def _classify_face(face: Dict[str, Any]) -> str:
    n = face.get("normal")
    ftype = (face.get("type") or "").upper()
    if n is None:
        return ftype.lower() or "unknown"
    nx, ny, nz = n
    if abs(nz - 1) < 1e-3:
        return "top (>Z)"
    if abs(nz + 1) < 1e-3:
        return "bottom (<Z)"
    if abs(nx - 1) < 1e-3:
        return "right (>X)"
    if abs(nx + 1) < 1e-3:
        return "left (<X)"
    if abs(ny - 1) < 1e-3:
        return "front (>Y)"
    if abs(ny + 1) < 1e-3:
        return "back (<Y)"
    if "CYLINDER" in ftype or "CONE" in ftype or "SPHERE" in ftype:
        return ftype.lower()
    return "angled"


def format_report(report: Optional[Dict[str, Any]], max_items: int = 24) -> str:
    """Render a measured geometry report into an LLM-friendly selector guide."""
    if not report:
        return "Geometry inspection unavailable (model did not build)."
    if report.get("error"):
        return f"Geometry inspection error: {report['error']}"

    faces = report.get("faces", [])
    edges = report.get("edges", [])
    lines = [
        f"Measured geometry of the current model: "
        f"{report.get('num_faces', len(faces))} faces, {report.get('num_edges', len(edges))} edges.",
    ]

    # Face groups by orientation.
    face_groups: Dict[str, List[Dict[str, Any]]] = {}
    for f in faces:
        face_groups.setdefault(_classify_face(f), []).append(f)
    lines.append("\nFaces by orientation:")
    for label, fs in sorted(face_groups.items(), key=lambda kv: -len(kv[1])):
        sample = fs[0]
        lines.append(
            f"  - {len(fs)}x {label} (e.g. center {sample.get('center')}, area {sample.get('area')})"
        )

    # Edge groups by orientation — the actionable part for fillet/chamfer.
    edge_groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in edges:
        edge_groups.setdefault(_classify_edge(e), []).append(e)
    lines.append("\nEdges by orientation:")
    for label, es in sorted(edge_groups.items(), key=lambda kv: -len(kv[1])):
        lengths = [e.get("length") for e in es if e.get("length") is not None]
        min_len = min(lengths) if lengths else None
        lines.append(f"  - {len(es)}x {label} (min length {min_len})")

    # Selector hints.
    hints = []
    if edge_groups.get("vertical (|Z)"):
        hints.append("vertical edges -> model.edges('|Z')")
    if edge_groups.get("horizontal-X"):
        hints.append("X-axis edges -> model.edges('|X')")
    if edge_groups.get("horizontal-Y"):
        hints.append("Y-axis edges -> model.edges('|Y')")
    if face_groups.get("top (>Z)"):
        hints.append("top face -> model.faces('>Z')")
    if face_groups.get("bottom (<Z)"):
        hints.append("bottom face -> model.faces('<Z')")
    if hints:
        lines.append("\nSelector hints:")
        for h in hints:
            lines.append(f"  - {h}")

    # Safe fillet/chamfer ceiling = a fraction of the shortest edge.
    all_lengths = [e.get("length") for e in edges if e.get("length")]
    if all_lengths:
        smallest = min(all_lengths)
        lines.append(
            f"\nKeep any fillet/chamfer size below ~{round(smallest * 0.45, 3)} "
            f"(45% of the shortest edge, {smallest}) to avoid kernel failures."
        )

    return "\n".join(lines)
