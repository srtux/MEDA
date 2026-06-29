"""Headless STL rendering helpers.

Two entry points are exposed:

* ``capture_stl_screenshot(stl_path, png_path)`` — CLI-style helper that writes a
  2x2 orthographic collage and ``sys.exit``s on error (kept for backwards
  compatibility / standalone invocation).
* ``capture_orthographic_collage(stl_path, png_path)`` — **loop-safe** variant
  used by the live reasoning loop: it writes the same collage *and returns the
  PNG bytes*, returning ``None`` on any failure instead of killing the process.
  ``core.reasoning_core`` calls this every successful compile so the agent gets
  per-iteration visual feedback; a missing/throwing renderer must never abort
  the whole design run (this was the C1 regression — the function was imported
  but never actually defined here).

All heavy imports (numpy/open3d/matplotlib/PIL) stay module-level because this
file is only imported on the rendering path, which already requires them.
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

# (elevation, azimuth, label) for the four standard engineering views.
_VIEWS = [
    (30, -45, "iso"),    # Isometric
    (90, -90, "top"),    # Top
    (0, -90, "front"),   # Front
    (0, 0, "right"),     # Right
]


def _render_collage(stl_path: Path, png_path: Path) -> bytes:
    """Render a 2x2 orthographic collage of ``stl_path`` to ``png_path``.

    Returns the PNG bytes. Raises on any failure (callers decide how to handle
    it). Kept import-light and free of ``sys.exit`` so it is safe to call from
    inside the agent loop.
    """
    stl_path = Path(stl_path).resolve()
    png_path = Path(png_path).resolve()

    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found at {stl_path}")

    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if mesh.is_empty():
        raise ValueError("Loaded mesh is empty.")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Bounding box -> equal aspect ratio across all four views.
    x_lim = [vertices[:, 0].min(), vertices[:, 0].max()]
    y_lim = [vertices[:, 1].min(), vertices[:, 1].max()]
    z_lim = [vertices[:, 2].min(), vertices[:, 2].max()]
    max_range = max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0])
    if max_range == 0:
        max_range = 1.0
    x_center, y_center, z_center = np.mean(x_lim), np.mean(y_lim), np.mean(z_lim)

    temp_dir = png_path.parent
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = []
    try:
        for elev, azim, label in _VIEWS:
            path = temp_dir / f"temp_{label}.png"
            temp_paths.append(path)
            fig = plt.figure(figsize=(4, 3), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            poly3d = Poly3DCollection(
                vertices[triangles],
                facecolors='#0ea5e9',
                edgecolors='#0284c7',
                alpha=0.95,
                linewidths=0.2,
            )
            ax.add_collection3d(poly3d)
            ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
            ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
            ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            plt.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)

        imgs = [Image.open(p) for p in temp_paths]
        width, height = imgs[0].size
        grid = Image.new("RGBA", (width * 2, height * 2))
        grid.paste(imgs[0], (0, 0))           # iso  (top-left)
        grid.paste(imgs[1], (width, 0))       # top  (top-right)
        grid.paste(imgs[2], (0, height))      # front(bottom-left)
        grid.paste(imgs[3], (width, height))  # right(bottom-right)
        grid.save(png_path)
        for im in imgs:
            im.close()
    finally:
        for p in temp_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    return png_path.read_bytes()


def capture_orthographic_collage(stl_path, png_path):
    """Loop-safe collage renderer used by the agent loop.

    Writes the collage to ``png_path`` and returns the PNG bytes, or ``None`` on
    any failure (never raises, never ``sys.exit``s). This guarantees a render
    failure degrades gracefully instead of forcing the visual critic to zero the
    reward and making success unreachable.
    """
    try:
        return _render_collage(Path(stl_path), Path(png_path))
    except Exception as e:  # pragma: no cover - depends on open3d/matplotlib runtime
        print(f"[WARNING] capture_orthographic_collage failed: {e}", file=sys.stderr)
        return None


def capture_stl_screenshot(stl_path, png_path):
    """Standalone collage renderer; ``sys.exit``s on error (CLI compatibility)."""
    try:
        _render_collage(Path(stl_path), Path(png_path))
        print("Multi-view screenshot capture complete!")
    except Exception as e:
        print(f"Exception: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python capture_screenshot.py <stl_path> <png_path>", file=sys.stderr)
        sys.exit(1)
    capture_stl_screenshot(sys.argv[1], sys.argv[2])
