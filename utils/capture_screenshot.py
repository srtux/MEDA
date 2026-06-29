import sys
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from pathlib import Path

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
            
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Calculate bounding box bounds to maintain equal aspect ratio
        x_lim = [vertices[:, 0].min(), vertices[:, 0].max()]
        y_lim = [vertices[:, 1].min(), vertices[:, 1].max()]
        z_lim = [vertices[:, 2].min(), vertices[:, 2].max()]
        
        max_range = max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0])
        if max_range == 0:
            max_range = 1.0
            
        x_center = np.mean(x_lim)
        y_center = np.mean(y_lim)
        z_center = np.mean(z_lim)
        
        # Temp paths for individual views
        temp_dir = png_path.parent
        iso_path = temp_dir / "temp_iso.png"
        top_path = temp_dir / "temp_top.png"
        front_path = temp_dir / "temp_front.png"
        right_path = temp_dir / "temp_right.png"
        
        views = [
            (30, -45, iso_path),   # Isometric
            (90, -90, top_path),   # Top
            (0, -90, front_path),  # Front
            (0, 0, right_path)     # Right
        ]
        
        for elev, azim, path in views:
            fig = plt.figure(figsize=(4, 3), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Render mesh with nice shading colors
            poly3d = Poly3DCollection(
                vertices[triangles], 
                facecolors='#0ea5e9', 
                edgecolors='#0284c7', 
                alpha=0.95, 
                linewidths=0.2
            )
            ax.add_collection3d(poly3d)
            
            # Force equal bounds
            ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
            ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
            
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            
            # Save view image
            plt.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
            
        # Stitch images into a 2x2 grid
        img_iso = Image.open(iso_path)
        img_top = Image.open(top_path)
        img_front = Image.open(front_path)
        img_right = Image.open(right_path)
        
        width, height = img_iso.size
        grid_image = Image.new("RGBA", (width * 2, height * 2))
        
        grid_image.paste(img_iso, (0, 0))
        grid_image.paste(img_top, (width, 0))
        grid_image.paste(img_front, (0, height))
        grid_image.paste(img_right, (width, height))
        
        # Save output png
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
