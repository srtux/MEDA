"""Seed the MEDA learning store with curated starter lessons and skills.

Run offline (no API key needed): embeddings are left NULL and back-filled
lazily at runtime when a genai client is available. The resulting
``memory/meda_memory.db`` is committed so a fresh clone starts with baseline
CAD knowledge instead of an empty store.

Usage:
    python memory/seed_memory.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.learning_store import LearningStore

# (error_signature, error_detail, root_cause, corrective_fix, prompt_context)
SEED_LESSONS = [
    (
        "compile:name_error",
        "NameError: name 'model' is not defined",
        "The first feature step must bind the main solid to the variable `model`; "
        "later steps then chain off `model`.",
        "Start the timeline with `model = cq.Workplane('XY').box(w, l, h)` (or another "
        "primitive) before any modifying feature.",
        "generic CAD model creation",
    ),
    (
        "compile:attribute_error",
        "AttributeError: 'Workplane' object has no attribute 'hole' on a non-face context",
        "`.hole()` / `.cboreHole()` require an active face+workplane selection; calling "
        "them on the raw solid fails.",
        "Select a face and open a workplane first: "
        "`model = model.faces('>Z').workplane().hole(diameter)`.",
        "drilling a hole into a solid",
    ),
    (
        "topology:num_faces_mismatch",
        "num_faces (expected: 6, got: 7)",
        "Exact face/edge counts are fragile: fillets, chamfers and holes each change "
        "the count, so a guessed integer target is usually wrong.",
        "Do not hard-target exact face/edge counts; rely on volume/center-of-mass with "
        "tolerance and visual critique instead.",
        "primitive solid with secondary features",
    ),
    (
        "export:error",
        "[EXPORT_ERROR] BRep_API: command not done (non-manifold geometry)",
        "Boolean unions of barely-touching or coincident solids can yield non-manifold "
        "geometry that the STL/STEP exporter rejects.",
        "Ensure overlapping solids actually intersect (share volume, not just a face) "
        "before `.union()`, or fuse with a small overlap.",
        "combining multiple solids with boolean union",
    ),
    (
        "visual:mismatch",
        "Visual validation failure: secondary part detached from the main body",
        "Features placed with absolute coordinates can float away from the body when "
        "the body's origin/extent is assumed wrong.",
        "Anchor secondary features to the body using face selectors and workplane "
        "offsets (e.g. `.faces('>Z').workplane(offset=h)`) instead of absolute Z.",
        "model with stems, handles or attached parts",
    ),
    (
        "compile:kernel_error",
        "StdFail_NotDone raised during fillet",
        "A fillet/chamfer radius larger than the adjacent edge length or wall thickness "
        "makes the kernel operation fail.",
        "Keep fillet radius well below half the smallest adjacent dimension; "
        "parameterize it relative to wall thickness.",
        "applying fillets or chamfers to edges",
    ),
]

# (name, goal_description, signature, code_template)
# Every snippet below has been executed against CadQuery 2.8.0 (see
# tests/test_seed_skills.py) so the seeds are known-valid, not hallucinated.
SEED_SKILLS = [
    # --- base primitives -------------------------------------------------
    (
        "base_box",
        "Create a rectangular base solid (plate or block) bound to `model`.",
        "(length, width, height)",
        "model = cq.Workplane('XY').box(length, width, height)",
    ),
    (
        "base_cylinder",
        "Create a cylindrical base solid bound to `model`.",
        "(radius, height)",
        "model = cq.Workplane('XY').circle(radius).extrude(height)",
    ),
    (
        "base_sphere",
        "Create a spherical base solid bound to `model`.",
        "(radius)",
        "model = cq.Workplane('XY').sphere(radius)",
    ),
    (
        "regular_polygon_prism",
        "Extrude a regular n-sided polygon into a prism (hex/oct stock, nuts).",
        "(n_sides, circumradius, height)",
        "model = cq.Workplane('XY').polygon(n_sides, 2*circumradius).extrude(height)",
    ),
    (
        "hollow_tube",
        "Create a hollow tube/pipe by extruding an annulus (outer minus inner circle).",
        "(outer_d, inner_d, height)",
        "model = cq.Workplane('XY').circle(outer_d/2).circle(inner_d/2).extrude(height)",
    ),
    # --- holes -----------------------------------------------------------
    (
        "centered_through_hole",
        "Drill a centered through-hole through the top face of the current solid.",
        "(diameter)",
        "model = model.faces('>Z').workplane().hole(diameter)",
    ),
    (
        "counterbore_hole",
        "Drill a counterbored hole (flat-bottom recess for a socket-head cap screw).",
        "(hole_d, cbore_d, cbore_depth)",
        "model = model.faces('>Z').workplane().cboreHole(hole_d, cbore_d, cbore_depth)",
    ),
    (
        "countersink_hole",
        "Drill a countersunk hole (conical recess for a flat-head screw).",
        "(hole_d, csink_d, angle)",
        "model = model.faces('>Z').workplane().cskHole(hole_d, csink_d, angle)",
    ),
    (
        "rect_hole_pattern",
        "Punch a rectangular grid of holes through the top face.",
        "(x_count, y_count, x_spacing, y_spacing, diameter)",
        "model = (model.faces('>Z').workplane()\n"
        "         .rarray(x_spacing, y_spacing, x_count, y_count)\n"
        "         .hole(diameter))",
    ),
    # --- edge treatments -------------------------------------------------
    (
        "fillet_all_edges",
        "Round (fillet) all edges of the current solid by a given radius.",
        "(radius)",
        "model = model.edges().fillet(radius)",
    ),
    (
        "chamfer_all_edges",
        "Chamfer all edges of the current solid by a given distance.",
        "(distance)",
        "model = model.edges().chamfer(distance)",
    ),
    (
        "chamfer_top_edges",
        "Chamfer only the edges of the top face of the current solid.",
        "(distance)",
        "model = model.faces('>Z').edges().chamfer(distance)",
    ),
    # --- shells & advanced features -------------------------------------
    (
        "shell_open_top",
        "Hollow the solid into a shell (container), removing the top face.",
        "(thickness)",
        "model = model.faces('>Z').shell(-thickness)",
    ),
    (
        "revolve_profile",
        "Revolve an L-shaped profile 360 degrees into a solid of revolution (disc/ring).",
        "(radius, height)",
        "model = (cq.Workplane('XZ')\n"
        "         .lineTo(radius, 0).lineTo(radius, height).lineTo(0, height)\n"
        "         .close().revolve(360))",
    ),
    (
        "loft_taper",
        "Loft from a rectangular base to a circular top (tapered transition/funnel).",
        "(length, width, height, radius)",
        "model = (cq.Workplane('XY').rect(length, width)\n"
        "         .workplane(offset=height).circle(radius)\n"
        "         .loft(combine=True))",
    ),
    (
        "union_two_solids",
        "Boolean-union a second solid onto the current model (combine parts).",
        "(other_solid)",
        "model = model.union(other_solid)",
    ),
    (
        "mirror_about_plane",
        "Mirror the current solid across a plane and keep both halves (symmetric parts).",
        "(plane)  # e.g. 'XZ', 'YZ', 'XY'",
        "model = model.mirror('XZ', union=True)",
    ),
]


def main():
    db_path = Path(__file__).resolve().parent / "meda_memory.db"
    # Start clean so re-seeding is idempotent.
    if db_path.exists():
        db_path.unlink()
    store = LearningStore(genai_client=None, db_path=db_path)
    for lesson in SEED_LESSONS:
        store.record_lesson(*lesson)
    for skill in SEED_SKILLS:
        store.record_skill(*skill)
    print(f"Seeded {store.counts()} into {db_path}")
    store.close()


if __name__ == "__main__":
    main()
