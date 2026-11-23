#!/usr/bin/env python3
"""
MVP Pipeline: 2D-to-3D Object Detection, Unprojection, and Canonical Point Cloud Aggregation.

Pipeline Stages:
  1. Load RGB-D Frame (RGB image + Depth map + Camera Intrinsics).
  2. Extract 2D Object Mask (via YOLOv8/11-Seg or fallback center-object mask).
  3. Unproject 2D masked pixels into 3D Camera-Space Point Cloud:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = D(u, v)
  4. Filter sensor outliers (Statistical Outlier Removal).
  5. Estimate interframe SE(3) transformation [R | t] using Point-to-Plane ICP.
  6. Accumulate aligned points into a canonical global object point cloud.
  7. Export canonical 3D object to PLY format.
"""

import os
import sys
import argparse
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is required. Run 'pip install open3d'.")
    sys.exit(1)


def get_default_intrinsics():
    """Default camera intrinsics (PrimeSense / RealSense standard: 640x480)."""
    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5
    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return K, (640, 480)


def unproject_depth_to_point_cloud(depth, color, K, mask=None, depth_scale=1000.0, depth_trunc=3.0):
    """
    Unprojects a 2D depth map into a 3D Open3D PointCloud.
    If a 2D binary mask is provided, only pixels inside the mask are unprojected.
    """
    h, w = depth.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Convert depth to meters
    z = depth.astype(np.float32) / depth_scale

    # Valid depth mask
    valid = (z > 0.1) & (z < depth_trunc)
    if mask is not None:
        valid = valid & (mask > 0)

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]

    # 2D to 3D lifting
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    points = np.stack([x, y, z_valid], axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if color is not None:
        rgb = color[valid]
        if rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb[:, :3])

    return pcd


def filter_outliers(pcd, nb_neighbors=25, std_ratio=1.5):
    """Removes depth noise and edge-bleeding 'flying pixels'."""
    if len(pcd.points) < nb_neighbors:
        return pcd
    filtered_pcd, inliers = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return filtered_pcd


def register_frames_icp(source_pcd, target_pcd, init_transform=np.eye(4), max_distance=0.04):
    """
    Estimates SE(3) transformation [R | t] aligning source_pcd to target_pcd
    using Point-to-Plane ICP.
    """
    # Estimate surface normals if not present
    if not source_pcd.has_normals():
        source_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
    if not target_pcd.has_normals():
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )

    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    return result.transformation, result.fitness


def get_yolo_or_fallback_mask(color_img, conf_thresh=0.4):
    """
    Extracts 2D object segmentation mask using YOLO-Seg if installed.
    Otherwise, generates a center region mask for immediate standalone testing.
    """
    h, w = color_img.shape[:2]
    try:
        from ultralytics import YOLO
        # Uses lightweight segmentation model (auto-downloads ~6MB weights on first run if online)
        model = YOLO("yolov8n-seg.pt")
        results = model(color_img, verbose=False, conf=conf_thresh)
        if len(results) > 0 and results[0].masks is not None:
            # Pick largest detected object mask
            masks = results[0].masks.data.cpu().numpy()
            areas = [m.sum() for m in masks]
            best_idx = np.argmax(areas)
            import cv2
            mask = cv2.resize(masks[best_idx].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            return mask
    except Exception:
        pass

    # Fallback heuristic mask: center region (e.g. tabletop object focus)
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    rh, rw = h // 4, w // 4
    mask[cy - rh : cy + rh, cx - rw : cx + rw] = 1
    return mask


def main():
    parser = argparse.ArgumentParser(description="3D Object Tracking & Point Cloud Accumulation MVP")
    parser.add_argument("--visualize", action="store_true", help="Display interactive 3D viewer")
    parser.add_argument("--output", default="outputs/canonical_object.ply", help="Output path for accumulated 3D model")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    print("=" * 70)
    print("🚀 3D Point Cloud Object Tracking & Reconstruction MVP")
    print("=" * 70)

    # 1. Fetch / Prepare Sample RGB-D Data
    print("\n[Step 1/5] Loading sample RGB-D image sequence...")
    sample_data = o3d.data.SampleRedwoodRGBDImages()
    color_paths = sample_data.color_paths[:5]
    depth_paths = sample_data.depth_paths[:5]
    K, _ = get_default_intrinsics()

    canonical_model = o3d.geometry.PointCloud()
    current_pose = np.eye(4)  # Global canonical frame pose

    print(f"[Step 2/5] Processing {len(color_paths)} consecutive frames...")
    prev_frame_pcd = None

    for frame_idx, (c_path, d_path) in enumerate(zip(color_paths, depth_paths)):
        color = np.asarray(o3d.io.read_image(c_path))
        depth = np.asarray(o3d.io.read_image(d_path))

        # 2. Extract 2D Mask
        mask = get_yolo_or_fallback_mask(color)

        # 3. 2D to 3D Unprojection
        frame_pcd = unproject_depth_to_point_cloud(depth, color, K, mask=mask)

        # 4. Outlier Removal
        frame_pcd = filter_outliers(frame_pcd)
        print(f"  Frame {frame_idx + 1}: Unprojected {len(frame_pcd.points)} clean 3D object points")

        if len(frame_pcd.points) < 50:
            print("  Warning: Insufficient points in frame, skipping...")
            continue

        # 5. Interframe Motion & SE(3) Registration
        if prev_frame_pcd is None:
            # First frame initializes the canonical model
            canonical_model += frame_pcd
            prev_frame_pcd = frame_pcd
        else:
            # Estimate relative transformation T from previous frame to current frame
            rel_transform, fitness = register_frames_icp(frame_pcd, prev_frame_pcd)
            current_pose = current_pose @ np.linalg.inv(rel_transform)

            # Transform current frame points into canonical object coordinates
            transformed_pcd = frame_pcd.transform(current_pose)

            # Accumulate into global canonical representation
            canonical_model += transformed_pcd
            prev_frame_pcd = frame_pcd
            print(f"  Frame {frame_idx + 1} registered with ICP fitness: {fitness:.3f}")

    # 6. Downsample canonical point cloud to clean overlap
    print("\n[Step 3/5] Consolidating and downsampling canonical point cloud...")
    canonical_model = canonical_model.voxel_down_sample(voxel_size=0.005)
    canonical_model = filter_outliers(canonical_model, nb_neighbors=20, std_ratio=1.5)
    print(f"  Total accumulated canonical points: {len(canonical_model.points)}")

    # 7. Save output
    print(f"\n[Step 4/5] Saving canonical 3D object to '{args.output}'...")
    o3d.io.write_point_cloud(args.output, canonical_model)
    print("  Successfully saved.")

    # 8. Interactive 3D visualization
    if args.visualize:
        print("\n[Step 5/5] Launching interactive 3D viewer (close window to exit)...")
        o3d.visualization.draw_geometries(
            [canonical_model],
            window_name="Canonical Accumulated 3D Object",
            width=1024,
            height=768
        )
    else:
        print("\nTip: Pass '--visualize' to view the 3D object interactively on your display.")

    print("\n✅ Done! Ready to run on your GPU rig.")


if __name__ == "__main__":
    main()
