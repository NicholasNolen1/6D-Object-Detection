#!/usr/bin/env python3
"""
MVP Pipeline: Point-Cloud-First Multi-Object 3D Detection, Tracking, and Canonical Fusion.

Executes on sample Redwood RGB-D indoor sequence provided with Open3D.
Demonstrates all 5 core stages using modular functions from the 'src' package:
  1. Full 3D Scene Lifting (src.geometry)
  2. 3D-to-2D Perspective Projection (src.geometry)
  3. 2D Instance Detection on Projected Canvas (src.detection)
  4. 3D Point Classification from 2D Masks (src.detection)
  5. Interframe Multi-Object SE(3) ICP Tracking & Fusion (src.tracking)
"""

import os
import argparse
import open3d as o3d

from src.geometry.pointcloud import rgbd_to_scene_point_cloud
from src.geometry.projection import project_point_cloud_to_2d
from src.detection.detector import YOLODetector
from src.detection.classifier import classify_point_cloud_from_masks
from src.tracking.tracker import MultiObjectTracker
from src.visualization.scene import visualize_geometries, save_point_cloud
from src.datasets.redwood import RedwoodSampleSequence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Point-Cloud-First Multi-Object 3D Tracking & Reconstruction (MVP Demo)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display interactive Open3D 3D viewer",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for accumulated 3D models",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.20,
        help="YOLO detection confidence threshold",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of consecutive frames to process",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 Point-Cloud-First Multi-Object 3D Detection, Tracking & Reconstruction")
    print("   Dataset: Sample Redwood RGB-D Indoor Sequence")
    print("=" * 80)

    # 1. Load Dataset Sequence
    print("\n[Step 1/5] Loading sample Redwood RGB-D image sequence...")
    dataset = RedwoodSampleSequence(max_frames=args.num_frames)
    print(f"Loaded {len(dataset)} frames.")

    # 2. Initialize Models & Tracker
    detector = YOLODetector("yolov8n-seg.pt")
    tracker = MultiObjectTracker(iou_thresh=0.15)

    print(f"\n[Step 2/5] Processing {len(dataset)} consecutive frames...")

    for frame_idx in range(len(dataset)):
        print(f"\n--- Frame {frame_idx + 1}/{len(dataset)} ---")
        color_rgb, depth, intrinsics = dataset.get_frame(frame_idx)

        # -------------------------------------------------------------------------
        # STAGE 1: Convert entire RGB-D sensor frame into Full 3D Point Cloud FIRST!
        # -------------------------------------------------------------------------
        pts_3d, colors_3d, _ = rgbd_to_scene_point_cloud(
            depth, color_rgb, intrinsics, depth_scale=1000.0, z_min=0.1, z_max=3.0
        )
        print(f"  Stage 1: Lifted {len(pts_3d)} 3D points into Full Scene Point Cloud")

        # -------------------------------------------------------------------------
        # STAGE 2: Project 3D Point Cloud onto 2D plane
        # -------------------------------------------------------------------------
        proj_image, pt2px_map, valid_proj = project_point_cloud_to_2d(
            pts_3d, colors_3d, intrinsics
        )
        print("  Stage 2: Projected 3D Point Cloud onto 2D plane")

        # -------------------------------------------------------------------------
        # STAGE 3: Run 2D detection on the 2D projected plane
        # -------------------------------------------------------------------------
        detections = detector.detect(
            proj_image,
            conf_thresh=args.conf_thresh,
            filter_background=False,
            min_pixels=80
        )
        if not detections:
            # Fallback regional masks if YOLO detects no objects in scene
            detections = YOLODetector.fallback_regions(proj_image.shape)

        det_names = [d.class_name for d in detections]
        print(f"  Stage 3: 2D detector found {len(detections)} mask(s) on projected plane: {det_names}")

        # -------------------------------------------------------------------------
        # STAGE 4: Classify the individual 3D points of the point cloud!
        # -------------------------------------------------------------------------
        classified_instances, _ = classify_point_cloud_from_masks(
            pts_3d, colors_3d, pt2px_map, valid_proj, detections, min_points=50
        )
        summary = [(inst.class_name, inst.num_points) for inst in classified_instances]
        print(f"  Stage 4: Classified {len(classified_instances)} 3D object point clouds: {summary}")

        # -------------------------------------------------------------------------
        # STAGE 5: Multi-Object 3D Tracking & Registration
        # -------------------------------------------------------------------------
        updates = tracker.update(classified_instances)
        for tid, cls_name, status in updates:
            print(f"  -> Track {tid} [{cls_name}]: {status}")

    # Consolidate and clean canonical point clouds
    print("\n[Step 3/5] Consolidating canonical point clouds for all tracked objects...")
    tracker.consolidate(voxel_size=0.005)

    for tid, track in tracker.tracks.items():
        n_pts = len(track.canonical_model.points)
        print(f"  Track {tid} [{track.class_name}]: {n_pts} accumulated canonical points (observed {track.frames_seen} frames)")
        out_ply = os.path.join(args.output_dir, f"canonical_track_{tid}_{track.class_name}.ply")
        save_point_cloud(track.canonical_model, out_ply)
        print(f"    Saved: {out_ply}")

    scene_composite = tracker.get_composite_scene()
    scene_ply = os.path.join(args.output_dir, "canonical_scene_all_objects.ply")
    save_point_cloud(scene_composite, scene_ply)
    print(f"\n[Step 4/5] Saved aggregated multi-object scene to '{scene_ply}'.")

    if args.visualize:
        print("\n[Step 5/5] Launching interactive 3D multi-object viewer (close window to exit)...")
        visualize_geometries([scene_composite], window_name="Point-Cloud-First Canonical Scene")
    else:
        print("\nTip: Pass '--visualize' to view the 3D multi-object models interactively.")

    print("\n✅ Done! Point-Cloud-First pipeline successfully executed.")


if __name__ == "__main__":
    main()
