"""
3D Scene Visualization and Geometry Export utilities using Open3D.
"""

from typing import List, Optional
import copy
import os
import open3d as o3d
from src.detection.classifier import INSTANCE_COLORS


def visualize_geometries(
    geometries: List[o3d.geometry.Geometry],
    window_name: str = "Point-Cloud-First 3D Viewer",
    width: int = 1024,
    height: int = 768
) -> None:
    """Launches an interactive Open3D viewer window displaying 3D geometries."""
    if not geometries:
        print("Warning: No geometries to visualize.")
        return
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=width,
        height=height
    )


def save_point_cloud(
    pcd: o3d.geometry.PointCloud,
    file_path: str,
    write_ascii: bool = False
) -> str:
    """Saves an Open3D PointCloud to disk (.ply)."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    o3d.io.write_point_cloud(file_path, pcd, write_ascii=write_ascii)
    return file_path


def create_composite_scene(
    point_clouds: List[o3d.geometry.PointCloud],
    palette: Optional[List[List[float]]] = None
) -> o3d.geometry.PointCloud:
    """Merges multiple point clouds into a single composite cloud with distinct uniform colors."""
    colors = palette or INSTANCE_COLORS
    composite = o3d.geometry.PointCloud()
    for idx, pcd in enumerate(point_clouds):
        color = colors[idx % len(colors)]
        colored = copy.deepcopy(pcd)
        colored.paint_uniform_color(color)
        composite += colored
    return composite
