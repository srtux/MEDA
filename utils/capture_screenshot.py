import sys
import open3d as o3d
from pathlib import Path
from PIL import Image

def capture_stl_screenshot(stl_path, png_path):
    try:
        stl_path = Path(stl_path).resolve()
        png_path = Path(png_path).resolve()
        
        if not stl_path.exists():
            print(f"Error: STL file not found at {stl_path}", file=sys.stderr)
            sys.exit(1)
            
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        if mesh.is_empty():
            print("Error: Loaded mesh is empty.", file=sys.stderr)
            sys.exit(1)
            
        mesh.compute_vertex_normals()
        
        vis = o3d.visualization.Visualizer()
        # Set 400x300 for each quad so final image is 800x600
        vis.create_window(window_name="CAD Viewer", width=400, height=300, visible=False)
        vis.add_geometry(mesh)
        
        opt = vis.get_render_option()
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal
        
        vis.update_geometry(mesh)
        vis.poll_events()
        
        # Temp paths for individual views
        temp_dir = png_path.parent
        iso_path = temp_dir / "temp_iso.png"
        top_path = temp_dir / "temp_top.png"
        front_path = temp_dir / "temp_front.png"
        right_path = temp_dir / "temp_right.png"
        
        # Define views (front vector, up vector, destination path)
        views = [
            ([1.0, -1.0, 1.0], [0.0, 0.0, 1.0], iso_path),     # Isometric
            ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], top_path),     # Top (XY Plane)
            ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], front_path),   # Front (XZ Plane)
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], right_path)    # Right (YZ Plane)
        ]
        
        ctr = vis.get_view_control()
        for front, up, path in views:
            ctr.set_front(front)
            ctr.set_up(up)
            ctr.set_zoom(0.85)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(path), do_render=True)
            
        vis.destroy_window()
        
        # Stitch images into 2x2 grid
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
        
        # Clean up temps
        iso_path.unlink(missing_ok=True)
        top_path.unlink(missing_ok=True)
        front_path.unlink(missing_ok=True)
        right_path.unlink(missing_ok=True)
        
        print("Multi-view screenshot capture complete!")
    except Exception as e:
        print(f"Exception: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python capture_screenshot.py <stl_path> <png_path>", file=sys.stderr)
        sys.exit(1)
    capture_stl_screenshot(sys.argv[1], sys.argv[2])
