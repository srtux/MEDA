"""Retrieval-grounded CadQuery API reference for the modeler agent.

This is MEDA's implementation of the single highest-leverage item on the
competitive roadmap (``docs/text_to_cad_landscape_2026.md`` §8.1): **RAG-ground
the CadQuery API into the modeler**. The open-source peer **CADSmith**
(arXiv:2603.26512) gets much of its reliability from a Coder agent that
retrieves over CadQuery API docs before writing code; ungrounded LLMs otherwise
hallucinate plausible-but-nonexistent methods (``.roundEdges()``,
``.drill()``...) or misuse selectors.

Rather than scrape live docs at runtime, we ship a **curated, kernel-verified**
reference: every entry's ``example`` uses only CadQuery 2.x API that executes in
the OpenCASCADE kernel (the same constraint enforced for seed skills in
``tests/test_seed_skills.py``). Each entry pairs a signature with the *gotchas*
that actually cause failures (selection context, radius limits, closed-profile
requirements) — the knowledge the LLM most often lacks.

Retrieval is keyword-based by default (no dependencies, works offline). When an
``embed_fn`` is supplied (e.g. ``LearningStore._embed``) it upgrades to cosine
similarity over precomputed entry embeddings, lazily cached on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional
import re

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "the", "to", "with", "make",
    "create", "want", "need", "model", "cad", "using", "use", "build", "i",
}


def _tokenize(text: str) -> set:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS and len(t) > 1}


@dataclass
class APIEntry:
    """One retrievable CadQuery operation: signature + summary + gotchas + example."""

    name: str
    category: str
    signature: str
    summary: str
    gotchas: str
    example: str
    keywords: str = ""
    embedding: object = field(default=None, repr=False)

    @property
    def _doc_text(self) -> str:
        return f"{self.name} {self.category} {self.summary} {self.gotchas} {self.keywords}"

    @property
    def tokens(self) -> set:
        return _tokenize(self._doc_text)

    def to_prompt_block(self) -> str:
        return (
            f"### {self.name}{self.signature}\n"
            f"- {self.summary}\n"
            f"- ⚠️ {self.gotchas}\n"
            f"- example: `{self.example}`"
        )


# ----------------------------------------------------------------------------
# Curated, kernel-verified CadQuery 2.x reference.
# ----------------------------------------------------------------------------
_ENTRIES: List[APIEntry] = [
    APIEntry(
        "Workplane", "(plane)", "(\"XY\" | \"XZ\" | \"YZ\")",
        "Start every model on a named base plane; the first feature must bind the solid to `model`.",
        "Without a workplane there is no drawing context. Use 'XY' unless the part is naturally upright on another plane.",
        "model = cq.Workplane('XY')",
        "start begin plane base origin sketch",
    ),
    APIEntry(
        "box", ".box(length, width, height)",
        "(length, width, height)",
        "Create a rectangular block/plate centered on the origin in one call.",
        "Box is centered at the origin, so it spans -h/2..+h/2 in Z. Account for that when placing holes/features by Z.",
        "model = cq.Workplane('XY').box(length, width, height)",
        "block plate cuboid rectangular base brick slab",
    ),
    APIEntry(
        "cylinder", ".cylinder(height, radius)",
        "(height, radius)",
        "Create a cylinder centered on the origin, axis along Z.",
        "Equivalent to `circle(radius).extrude(height)` but centered in Z. Diameter = 2*radius — do not pass diameter.",
        "model = cq.Workplane('XY').cylinder(height, radius)",
        "cylinder rod shaft pillar round bar tube post",
    ),
    APIEntry(
        "circle+extrude", ".circle(radius).extrude(height)",
        "(radius, height)",
        "Sketch a circle then extrude it into a cylinder rising from the base plane (Z=0..height).",
        "Sits on the base plane (not centered in Z) — preferred when you need the base at Z=0.",
        "model = cq.Workplane('XY').circle(radius).extrude(height)",
        "cylinder extrude profile disc puck rod",
    ),
    APIEntry(
        "sphere", ".sphere(radius)",
        "(radius)",
        "Create a sphere centered on the origin.",
        "Filleting/chamfering a sphere is meaningless (no edges); shelling needs a face selector.",
        "model = cq.Workplane('XY').sphere(radius)",
        "sphere ball dome orb round",
    ),
    APIEntry(
        "polygon", ".polygon(nSides, diameter).extrude(height)",
        "(nSides, diameter, height)",
        "Extrude a regular n-gon (hex/oct stock, nut bodies).",
        "The 2nd argument is the CIRCUMSCRIBED diameter (across corners), NOT radius and NOT across-flats.",
        "model = cq.Workplane('XY').polygon(6, 2*circumradius).extrude(height)",
        "hexagon hex octagon polygon nut prism stock flats",
    ),
    APIEntry(
        "hollow_tube", ".circle(outer/2).circle(inner/2).extrude(h)",
        "(outer_d, inner_d, height)",
        "Extrude an annulus (two concentric circles) into a pipe/tube/washer.",
        "The inner circle must be strictly smaller than the outer; two circles before one extrude cuts the bore automatically.",
        "model = cq.Workplane('XY').circle(outer_d/2).circle(inner_d/2).extrude(height)",
        "tube pipe ring washer annulus hollow bore sleeve bushing",
    ),
    APIEntry(
        "rect+extrude", ".rect(xLen, yLen).extrude(height)",
        "(xLen, yLen, height)",
        "Sketch a centered rectangle and extrude it (base at Z=0).",
        "Like box() but rises from the base plane instead of being Z-centered; good as a sketch starting point.",
        "model = cq.Workplane('XY').rect(xLen, yLen).extrude(height)",
        "rectangle plate slab base extrude profile",
    ),
    APIEntry(
        "hole", ".faces('>Z').workplane().hole(diameter)",
        "(diameter, depth=None)",
        "Drill a hole into the currently selected face; centered on that face's workplane.",
        "REQUIRES an active face+workplane selection first. `.hole()` on a raw solid fails. Omit depth for a through-hole.",
        "model = model.faces('>Z').workplane().hole(diameter)",
        "hole drill bore through pierce centered opening",
    ),
    APIEntry(
        "cboreHole", ".faces('>Z').workplane().cboreHole(d, cbore_d, cbore_depth)",
        "(diameter, cboreDiameter, cboreDepth, depth=None)",
        "Counterbored hole: flat recess for a socket-head cap screw.",
        "Needs a face+workplane selection. cboreDiameter must exceed diameter; cboreDepth < part thickness.",
        "model = model.faces('>Z').workplane().cboreHole(hole_d, cbore_d, cbore_depth)",
        "counterbore cbore socket screw recess pocket fastener",
    ),
    APIEntry(
        "cskHole", ".faces('>Z').workplane().cskHole(d, csk_d, angle)",
        "(diameter, cskDiameter, cskAngle, depth=None)",
        "Countersunk hole: conical recess for a flat-head screw (angle typically 82 or 90).",
        "Needs a face+workplane selection. cskDiameter > diameter; angle is the included cone angle in degrees.",
        "model = model.faces('>Z').workplane().cskHole(hole_d, csink_d, 82)",
        "countersink csk flathead conical screw chamfered hole",
    ),
    APIEntry(
        "rarray_holes", ".rarray(xS, yS, xN, yN).hole(d)",
        "(xSpacing, ySpacing, xCount, yCount, diameter)",
        "Punch a rectangular grid of holes through the selected face.",
        "Open a face workplane first. rarray centers the grid; spacing is center-to-center.",
        "model = model.faces('>Z').workplane().rarray(x_spacing, y_spacing, x_count, y_count).hole(diameter)",
        "grid pattern array holes matrix perforation repeated bolt-pattern",
    ),
    APIEntry(
        "polarArray", ".polarArray(radius, startAngle, angle, count)",
        "(radius, startAngle, angle, count)",
        "Lay out points/features evenly around a circle (bolt circles, spokes).",
        "Combine with `.hole()` or `.cutEach()`; place on a face workplane. `angle` is the total sweep (360 for a full circle).",
        "model = model.faces('>Z').workplane().polarArray(bolt_radius, 0, 360, n_bolts).hole(diameter)",
        "bolt circle polar radial spokes evenly around ring pattern",
    ),
    APIEntry(
        "fillet", ".edges(selector).fillet(radius)",
        "(radius)",
        "Round the selected edges. With no selector, `.edges().fillet(r)` rounds ALL edges.",
        "radius MUST be < half the smallest adjacent edge/wall; oversize radius raises a kernel StdFail. Select edges first to fillet only some (e.g. '|Z' for vertical edges).",
        "model = model.edges('|Z').fillet(radius)",
        "fillet round rounded edges smooth corner radius blend",
    ),
    APIEntry(
        "chamfer", ".edges(selector).chamfer(length)",
        "(length)",
        "Bevel the selected edges by a 45° chamfer of the given length.",
        "Same size limit as fillet. `.faces('>Z').edges().chamfer(d)` bevels only the top rim.",
        "model = model.faces('>Z').edges().chamfer(distance)",
        "chamfer bevel edge break angled corner deburr",
    ),
    APIEntry(
        "shell", ".faces(selector).shell(-thickness)",
        "(thickness)",
        "Hollow a solid into a thin-walled shell, removing the selected face(s) as the opening.",
        "Use NEGATIVE thickness to hollow inward. Thickness must be < half the smallest dimension. Select the face to open first ('>Z' = open top).",
        "model = model.faces('>Z').shell(-thickness)",
        "shell hollow wall thin container cup box enclosure cavity",
    ),
    APIEntry(
        "revolve", ".revolve(angleDegrees)",
        "(angleDegrees, axisStart=None, axisEnd=None)",
        "Revolve a closed 2D profile around an axis to make a solid of revolution.",
        "The profile wire must be CLOSED (.close()) and must NOT cross the rotation axis. Default axis is the workplane's local Y axis (axisStart=(0,0,0) -> axisEnd=(0,1,0)); pass axisStart/axisEnd to revolve about a different axis.",
        "model = (cq.Workplane('XZ').lineTo(radius,0).lineTo(radius,height).lineTo(0,height).close().revolve(360))",
        "revolve lathe turned axisymmetric vase bottle wheel pulley spin",
    ),
    APIEntry(
        "loft", ".loft(combine=True)",
        "(ruled=False, combine=True)",
        "Blend two or more profiles on different workplanes into a smooth transition.",
        "Needs >=2 closed wires on offset workplanes BEFORE calling loft. Use `combine=True`. Common for tapers/funnels.",
        "model = (cq.Workplane('XY').rect(length,width).workplane(offset=height).circle(radius).loft(combine=True))",
        "loft blend taper transition funnel morph adapter cone",
    ),
    APIEntry(
        "sweep", ".sweep(path)",
        "(path, ...)",
        "Sweep a profile along a path wire (pipes, handles, cables).",
        "Build the path as a separate Workplane wire; the profile is drawn on a plane perpendicular to the path start.",
        "path = cq.Workplane('XZ').moveTo(0,0).lineTo(0,height); model = cq.Workplane('XY').circle(r).sweep(path)",
        "sweep path pipe handle bend tube along curve extrude-along",
    ),
    APIEntry(
        "extrude_cut", ".faces(sel).workplane().rect(a,b).cutBlind(-depth)",
        "(depth)",
        "Cut a pocket/slot into a face by sketching on it and cutting blind (negative = into the solid).",
        "cutBlind needs a sketch on a face workplane. Use `cutThruAll()` to cut all the way through.",
        "model = model.faces('>Z').workplane().rect(slot_w, slot_l).cutBlind(-depth)",
        "pocket slot cut groove recess notch cavity remove subtract",
    ),
    APIEntry(
        "union", ".union(other)",
        "(other_solid)",
        "Boolean-combine the current model with another solid.",
        "The two solids should genuinely overlap (share volume); barely-touching faces yield non-manifold geometry the exporter rejects.",
        "model = model.union(other_solid)",
        "union combine join add merge fuse weld attach boolean",
    ),
    APIEntry(
        "cut", ".cut(other)",
        "(other_solid)",
        "Boolean-subtract another solid from the current model.",
        "The tool solid must intersect the target; build it with translate() to position it.",
        "model = model.cut(tool_solid)",
        "cut subtract remove difference carve hollow boolean drill",
    ),
    APIEntry(
        "translate", ".translate((dx, dy, dz))",
        "((dx, dy, dz))",
        "Move a solid by a vector (positioning a part before union/cut).",
        "Operates on the whole solid; takes a 3-tuple. Use to position secondary solids relative to the main body.",
        "other_solid = cq.Workplane('XY').box(5,5,5).translate((10,0,0))",
        "translate move offset position shift place locate",
    ),
    APIEntry(
        "rotate", ".rotate(axisStart, axisEnd, angleDegrees)",
        "(axisStartPoint, axisEndPoint, angleDegrees)",
        "Rotate a solid about an arbitrary axis given by two points.",
        "Both axis endpoints are 3-tuples; angle is degrees. e.g. rotate about Z: (0,0,0)->(0,0,1).",
        "model = model.rotate((0,0,0), (0,0,1), 45)",
        "rotate spin turn orient angle tilt about axis",
    ),
    APIEntry(
        "mirror", ".mirror(plane, union=True)",
        "(mirrorPlane, basePointVector=None, union=False)",
        "Mirror a solid across a plane; `union=True` keeps both halves (symmetric parts).",
        "Plane is 'XY'/'XZ'/'YZ'. Without union=True you only get the mirrored copy, not both halves.",
        "model = model.mirror('XZ', union=True)",
        "mirror symmetric reflect flip both halves duplicate",
    ),
    APIEntry(
        "selectors", ".faces('>Z') / .edges('|Z') / .edges('#Z')",
        "(string selector)",
        "String selectors pick faces/edges by direction: '>Z' top face, '<Z' bottom, '|Z' edges PARALLEL to Z (vertical), '#Z' edges PERPENDICULAR to Z, '+Z'/'-Z' normal direction.",
        "Selecting the wrong set is the #1 cause of mis-placed holes/fillets. When unsure WHICH edges exist, call the inspect_current_model tool to list real faces/edges before choosing a selector.",
        "vertical_edges = model.edges('|Z')   # the four upright edges of a box",
        "selector faces edges pick choose direction top bottom vertical which",
    ),
    APIEntry(
        "text", ".text(txt, fontsize, distance)",
        "(txt, fontsize, distance)",
        "Emboss/engrave text on a face (distance>0 raised, <0 engraved).",
        "Sketch on a face workplane first; engraving needs distance negative and the face thick enough.",
        "model = model.faces('>Z').workplane().text('MEDA', 5, 1)",
        "text label engrave emboss letters lettering stamp",
    ),
]


class CADKnowledgeBase:
    """Keyword/embedding retrieval over the curated CadQuery reference."""

    def __init__(self, entries: Optional[List[APIEntry]] = None):
        self.entries = entries if entries is not None else _ENTRIES
        self._embedded = False

    # -- retrieval ----------------------------------------------------------
    def retrieve(
        self, query: str, k: int = 5, embed_fn: Optional[Callable[..., object]] = None,
    ) -> List[APIEntry]:
        """Return up to ``k`` entries most relevant to ``query``.

        Uses cosine similarity when ``embed_fn`` is provided and produces
        vectors; otherwise falls back to Jaccard keyword overlap. Always returns
        deterministic order so two equal-scoring entries don't flap.
        """
        if not query:
            return []

        scored = []
        q_vec = self._maybe_embed_query(query, embed_fn)
        q_tokens = _tokenize(query)
        for entry in self.entries:
            sim = None
            if q_vec is not None and entry.embedding is not None:
                sim = _cosine(q_vec, entry.embedding)
            if sim is None:
                inter = q_tokens & entry.tokens
                union = q_tokens | entry.tokens
                sim = len(inter) / len(union) if union else 0.0
            scored.append((sim, entry.name, entry))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [e for s, _, e in scored[:k] if s > 0]

    def format_for_prompt(self, entries: List[APIEntry]) -> str:
        if not entries:
            return ""
        blocks = "\n".join(e.to_prompt_block() for e in entries)
        return (
            "## Relevant CadQuery API reference (use these exact methods — do NOT invent others)\n"
            + blocks
        )

    def retrieve_block(
        self, query: str, k: int = 5, embed_fn: Optional[Callable[..., object]] = None,
    ) -> str:
        return self.format_for_prompt(self.retrieve(query, k=k, embed_fn=embed_fn))

    def lookup(self, query: str, k: int = 4) -> str:
        """Human/tool-facing lookup returning a formatted reference block."""
        block = self.retrieve_block(query, k=k)
        return block or "No matching CadQuery API entry found; rely on standard cadquery.Workplane methods."

    # -- embeddings (optional) ---------------------------------------------
    def _maybe_embed_query(self, query: str, embed_fn):
        if embed_fn is None:
            return None
        try:
            self._ensure_doc_embeddings(embed_fn)
            return embed_fn(query, query=True)
        except Exception:
            return None

    def _ensure_doc_embeddings(self, embed_fn) -> None:
        if self._embedded:
            return
        any_ok = False
        for entry in self.entries:
            if entry.embedding is None:
                try:
                    entry.embedding = embed_fn(entry._doc_text)
                except Exception:
                    entry.embedding = None
            if entry.embedding is not None:
                any_ok = True
        # Only lock the cache once embeddings are actually usable, so a transient
        # failure (embed_fn returning None for every entry) can be retried later
        # instead of permanently downgrading retrieval to keyword-only.
        if any_ok:
            self._embedded = True


def _cosine(a, b) -> Optional[float]:
    try:
        import numpy as np

        a = np.asarray(a, dtype="float32")
        b = np.asarray(b, dtype="float32")
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return None
        return float((a @ b) / (na * nb))
    except Exception:
        return None


# Module-level singleton so embeddings are cached across runs in a process.
KNOWLEDGE_BASE = CADKnowledgeBase()


def retrieve_api_docs(query: str, k: int = 5, embed_fn=None) -> str:
    """Convenience wrapper returning a formatted reference block for ``query``."""
    return KNOWLEDGE_BASE.retrieve_block(query, k=k, embed_fn=embed_fn)
