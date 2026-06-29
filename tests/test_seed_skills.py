"""Validate that every seeded CadQuery skill actually executes in the kernel.

Skipped automatically when CadQuery is not installed (e.g. a bare CI image), so
this never fails for environment reasons. Run in the synced env with:
    python tests/test_seed_skills.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cadquery as cq
    _HAVE_CQ = True
except Exception:
    _HAVE_CQ = False

from memory.seed_memory import SEED_SKILLS

# Default parameter values covering every signature used by the seed skills.
_PARAMS = dict(
    length=30, width=20, height=10, radius=8, diameter=6, distance=2,
    thickness=2, n_sides=6, circumradius=10, outer_d=16, inner_d=6,
    hole_d=4, cbore_d=8, cbore_depth=3, csink_d=8, angle=82,
    x_count=2, y_count=3, x_spacing=8, y_spacing=8,
)


def test_seed_skills_execute():
    if not _HAVE_CQ:
        print("SKIP: cadquery not installed in this environment")
        return
    failures = []
    for name, _goal, _sig, code in SEED_SKILLS:
        g = {"cq": cq}
        g.update(_PARAMS)
        # Provide a base solid + a second solid so both "create" and "modify"
        # skills (including union/mirror templates) have what they reference.
        g["model"] = cq.Workplane("XY").box(30, 20, 10)
        g["other_solid"] = cq.Workplane("XY").box(10, 10, 10)
        g["plane"] = "XZ"
        try:
            exec(code, g)
            val = g["model"].val() if hasattr(g["model"], "val") else g["model"]
            vol = val.Volume()
            assert vol > 0, f"{name}: non-positive volume"
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")
    assert not failures, "Invalid seed skills:\n" + "\n".join(failures)
    print(f"All {len(SEED_SKILLS)} seed skills executed in CadQuery.")


if __name__ == "__main__":
    test_seed_skills_execute()
