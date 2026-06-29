"""Headless orthographic rendering of CAD STL files into a 4-view collage.

Provides two entry points:

* ``capture_stl_screenshot`` / CLI – original behaviour, calls ``sys.exit`` on
  error (safe for the subprocess/CLI path).
* ``capture_orthographic_collage`` – loop-safe wrapper used by the reasoning
  core. It NEVER calls ``sys.exit`` (a ``SystemExit`` would escape the agent's
  ``except Exception`` handler) and returns the rendered PNG bytes, or ``None``
  on failure.
"""

import sys
from pathlib import Path
from PIL import Image


def _render_collage(stl_path, png_path):
    """Render a 2x2 orthographic collage of an STL to png_path.

    Raises on any failure (caller decides whether to swallow or exit).
    """
    import open3d as o3d

    stl_path = Path(stl_path).resolve()
    png_path = Path(png_path).resolve()

    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found at {stl_path}")

    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if mesh.is_empty():
        raise ValueError("Loaded mesh is empty.")

    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    # 400x300 per quad so the final image is 800x600.
    vis.create_window(window_name="CAD Viewer", width=400, height=300, visible=False)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal

    vis.update_geometry(mesh)
    vis.poll_events()

    temp_dir = png_path.parent
    iso_path = temp_dir / "temp_iso.png"
    top_path = temp_dir / "temp_top.png"
    front_path = temp_dir / "temp_front.png"
    right_path = temp_dir / "temp_right.png"

    # (front vector, up vector, destination path)
    views = [
        ([1.0, -1.0, 1.0], [0.0, 0.0, 1.0], iso_path),    # Isometric
        ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], top_path),    # Top (XY)
        ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], front_path),  # Front (XZ)
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], right_path),  # Right (YZ)
    ]

    try:
        ctr = vis.get_view_control()
        for front, up, path in views:
            ctr.set_front(front)
            ctr.set_up(up)
            ctr.set_zoom(0.85)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(path), do_render=True)
    finally:
        vis.destroy_window()

    img_iso = Image.open(iso_path)
    img_top = Image.open(top_path)
    img_front = Image.open(front_path)
    img_right = Image.open(right_path)

    width, height = img_iso.size
    grid_image = Image.new("RGB", (width * 2, height * 2))
    grid_image.paste(img_iso, (0, 0))
    grid_image.paste(img_top, (width, 0))
    grid_image.paste(img_front, (0, height))
    grid_image.paste(img_right, (width, height))
    grid_image.save(png_path)

    for p in (iso_path, top_path, front_path, right_path):
        p.unlink(missing_ok=True)


def capture_stl_screenshot(stl_path, png_path):
    """CLI/subprocess entry point. Exits the process on error."""
    try:
        _render_collage(stl_path, png_path)
        print("Multi-view screenshot capture complete!")
    except Exception as e:
        print(f"Exception: {e}", file=sys.stderr)
        sys.exit(1)


def capture_orthographic_collage(stl_path, png_path):
    """Loop-safe collage render. Returns PNG bytes on success, else ``None``.

    Never raises and never calls ``sys.exit`` so it is safe to call from inside
    the agent loop's ``try/except`` blocks.
    """
    try:
        _render_collage(stl_path, png_path)
        return Path(png_path).resolve().read_bytes()
    except SystemExit:
        return None
    except Exception as e:
        print(f"[capture_orthographic_collage] render failed: {e}", file=sys.stderr)
        return None


def make_side_by_side(prev_png, curr_png, out_png):
    """Composite previous and current collages side-by-side for visual diffing.

    Returns the output PNG bytes, or ``None`` on failure.
    """
    try:
        a = Image.open(prev_png)
        b = Image.open(curr_png)
        h = max(a.height, b.height)
        canvas = Image.new("RGB", (a.width + b.width, h), "white")
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        canvas.save(out_png)
        return Path(out_png).resolve().read_bytes()
    except Exception:
        return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python capture_screenshot.py <stl_path> <png_path>", file=sys.stderr)
        sys.exit(1)
    capture_stl_screenshot(sys.argv[1], sys.argv[2])
