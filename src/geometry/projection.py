"""
3D-to-2D Projection: Maps 3D Point Clouds onto 2D planes with bi-directional indexing.
"""

from typing import Tuple, Optional
import numpy as np
from src.geometry.camera import CameraIntrinsics


def project_point_cloud_to_2d(
    pts_3d: np.ndarray,
    colors_3d: Optional[np.ndarray],
    intrinsics: CameraIntrinsics,
    T_view: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 2: Projects a 3D Point Cloud onto a 2D canvas plane.

    Math:
      P_view = T_view @ [X, Y, Z, 1]^T
      u = round(fx * X_v / Z_v + cx)
      v = round(fy * Y_v / Z_v + cy)

    Parameters:
      pts_3d: (N, 3) float32 array of 3D point coordinates
      colors_3d: (N, 3) float32 array of RGB colors in [0, 1], or None
      intrinsics: CameraIntrinsics defining the projection geometry
      T_view: Optional 4x4 viewpoint transformation (defaults to Identity)

    Returns:
      proj_image: (H, W, 3) uint8 RGB image rendered from the projected points
      point_to_pixel_map: (N, 2) int array containing projected (u, v) for every 3D point
      valid_proj: (N,) bool array indicating points that land within image boundaries with Z > 0
    """
    h, w = intrinsics.height, intrinsics.width
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    if len(pts_3d) == 0:
        return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((0, 2), dtype=int), np.zeros(0, dtype=bool)

    # 1. Apply viewpoint transformation if non-identity
    if T_view is not None and not np.allclose(T_view, np.eye(4)):
        pts_h = np.concatenate([pts_3d, np.ones((len(pts_3d), 1), dtype=np.float32)], axis=-1)
        pts_view = (T_view @ pts_h.T).T[:, :3]
    else:
        pts_view = pts_3d

    z = pts_view[:, 2]
    valid_z = z > 0.05

    # 2. Pinhole perspective projection onto 2D plane
    u = np.zeros(len(pts_3d), dtype=int)
    v = np.zeros(len(pts_3d), dtype=int)

    u[valid_z] = np.round((pts_view[valid_z, 0] * fx / z[valid_z]) + cx).astype(int)
    v[valid_z] = np.round((pts_view[valid_z, 1] * fy / z[valid_z]) + cy).astype(int)

    # 3. Check canvas boundaries
    valid_proj = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # 4. Render 2D image plane representation with depth-buffering
    # Sort points from farthest to nearest (painter's algorithm) so foreground points occlude background
    proj_image = np.zeros((h, w, 3), dtype=np.uint8)
    if np.any(valid_proj) and colors_3d is not None:
        sort_idx = np.argsort(-z[valid_proj])
        u_sorted = u[valid_proj][sort_idx]
        v_sorted = v[valid_proj][sort_idx]
        colors_sorted = (colors_3d[valid_proj][sort_idx] * 255.0).astype(np.uint8)
        proj_image[v_sorted, u_sorted] = colors_sorted

    point_to_pixel_map = np.stack([u, v], axis=-1)
    return proj_image, point_to_pixel_map, valid_proj
