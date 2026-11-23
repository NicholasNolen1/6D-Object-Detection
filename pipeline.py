#!/usr/bin/env python3
"""
Point-Cloud-First Real-World 3D Object Detection, 6D Tracking, and Reconstruction Pipeline.

Core Paradigm:
  1. RGB-D Ingestion: Load depth map + RGB image + camera intrinsics.
  2. Full 3D Scene Lifting: Convert the ENTIRE sensor frame into a Full 3D Scene
     Point Cloud P = {(X_i, Y_i, Z_i, R_i, G_i, B_i)} BEFORE any 2D segmentation.
  3. 3D-to-2D Projection: Project the 3D Point Cloud onto a 2D plane, establishing an explicit
     bi-directional mapping: pi: 3D Point Index i <--> 2D Coordinate (u_i, v_i).
  4. 2D Instance Detection on Projected Plane: Run YOLOv8-Seg (or dataset ground truth) on the
     projected 2D representation to obtain bounding boxes and instance masks for all objects.
  5. 3D Point Classification: Directly classify and segment the individual points of the 3D point
     cloud using the 2D projected masks: P_obj = { p_i in P | pi(p_i) in Mask_2D }.
  6. Interframe Multi-Object SE(3) Tracking: Track each isolated 3D object point cloud across
     frames using 2D bipartite IoU association and Point-to-Plane ICP.
  7. Canonical 3D Fusion & CAD Evaluation: Accumulate aligned multi-view points into canonical
     point clouds and benchmark against target YCB CAD models (ADD-S, Chamfer Distance, F-Score).
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import open3d as o3d

from src.geometry.camera import CameraIntrinsics
from src.geometry.pointcloud import (
    rgbd_to_scene_point_cloud,
    points_to_open3d,
    filter_outliers,
    downsample_point_cloud,
)
from src.geometry.projection import project_point_cloud_to_2d
from src.detection.detector import YOLODetector, DetectionResult
from src.detection.classifier import (
    SegmentedObject,
    classify_point_cloud_from_masks,
    INSTANCE_COLORS,
)
from src.tracking.tracker import MultiObjectTracker
from src.evaluation.metrics import (
    compute_chamfer_distance,
    compute_grasp_success_rate,
)
from src.visualization.overlay import render_mask_overlays
from src.visualization.scene import visualize_geometries, save_point_cloud
from src.datasets.ycb import YCBMultiTrackSequence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Point-Cloud-First Real-World 3D Detection, Tracking & Reconstruction"
    )
    parser.add_argument(
        "--sequence-dir",
        default="data/YCBMultiTrack-Real/006_mustard_bottle_010_potted_meat_can_easy",
        help="Path to sequence directory containing rgb/, depth/, cam_K.txt",
    )
    parser.add_argument(
        "--mask-source",
        choices=["yolo", "gt"],
        default="yolo",
        help="Mask extraction source: 'yolo' for automatic YOLOv8-Seg, 'gt' for dataset ground truth",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.10,
        help="YOLO detection confidence threshold",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include background furniture (chair, table) rather than filtering",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to process",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Number of consecutive frames to process",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        default=True,
        help="Save 2D multi-mask overlay images with bounding boxes",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save canonical 3D models and visualization overlays",
    )
    parser.add_argument(
        "--models-dir",
        default="data/models/YCBMultiTrack-Models",
        help="Base directory containing YCB CAD models",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch interactive Open3D 3D visualizer at end of run",
    )
    return parser.parse_args()


def match_det_to_gt(class_name: str, gt_object_names: list) -> str:
    """Matches a generic detection class label to a known YCB ground truth object name."""
    cls_lower = class_name.lower()
    for gt in gt_object_names:
        gt_lower = gt.lower()
        if cls_lower in gt_lower:
            return gt
        if "bottle" in cls_lower and "bottle" in gt_lower:
            return gt
        if ("cup" in cls_lower or "can" in cls_lower) and "can" in gt_lower:
            return gt
        if "box" in cls_lower and "box" in gt_lower:
            return gt
    return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seq_name = os.path.basename(os.path.normpath(args.sequence_dir))
    vis_dir = os.path.join(args.output_dir, f"vis_{seq_name}")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    print("=" * 85)
    print("🚀 Point-Cloud-First 3D Detection, 6D Tracking & Reconstruction Pipeline")
    print(f"Sequence:           {args.sequence_dir}")
    print(f"Mask Mode:          {args.mask_source.upper()} (All Object Instances)")
    print(f"Confidence Thresh:  {args.conf_thresh}")
    print(f"Filter Background:  {not args.include_background}")
    print(f"Frames:             {args.start_frame} to {args.start_frame + args.num_frames - 1}")
    print("=" * 85)

    # 1. Load Dataset Sequence
    sequence = YCBMultiTrackSequence(args.sequence_dir)
    print(f"Loaded sequence with {len(sequence)} total frames.")
    print(f"Dataset Ground Truth Object(s): {sequence.object_names if sequence.object_names else 'None'}")

    # 2. Load CAD Models for Benchmarking
    cad_points_dict = {}
    cad_mesh_dict = {}
    for obj_name in sequence.object_names:
        mesh = YCBMultiTrackSequence.load_cad_mesh(args.models_dir, obj_name)
        if mesh is not None:
            cad_mesh_dict[obj_name] = mesh
            pcd = mesh.sample_points_uniformly(number_of_points=1000)
            cad_points_dict[obj_name] = np.asarray(pcd.points)

    # 3. Initialize YOLO Detector
    detector = None
    if args.mask_source == "yolo":
        detector = YOLODetector("yolov8n-seg.pt")

    # 4. Multi-Object Tracker
    tracker = MultiObjectTracker(iou_thresh=0.15)
    fps_times = []

    # Frame processing range
    end_frame = min(len(sequence), args.start_frame + args.num_frames)
    total_to_process = end_frame - args.start_frame

    if total_to_process <= 0:
        print(f"Error: Invalid frame range [{args.start_frame}, {end_frame}).")
        return

    print(f"\nProcessing {total_to_process} consecutive real-world frames...\n")

    for idx in range(args.start_frame, end_frame):
        t_start = time.time()
        frame = sequence.get_frame(idx)
        fid = frame.frame_id

        # -------------------------------------------------------------------------
        # STAGE 1: Convert entire RGB-D sensor frame into Full 3D Scene Point Cloud FIRST!
        # -------------------------------------------------------------------------
        pts_3d, colors_3d, _ = rgbd_to_scene_point_cloud(
            frame.depth, frame.color_rgb, frame.intrinsics, depth_scale=1000.0, z_min=0.1, z_max=3.0
        )
        total_scene_points = len(pts_3d)

        # -------------------------------------------------------------------------
        # STAGE 2: Project 3D Point Cloud onto 2D plane
        # -------------------------------------------------------------------------
        proj_image, pt2px_map, valid_proj = project_point_cloud_to_2d(
            pts_3d, colors_3d, frame.intrinsics
        )

        # -------------------------------------------------------------------------
        # STAGE 3: 2D Instance Detection on the Projected Plane
        # -------------------------------------------------------------------------
        detections = []
        if args.mask_source == "gt":
            for obj_name, mask in frame.gt_masks.items():
                if mask.sum() > 80:
                    ys, xs = np.where(mask > 0)
                    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                    detections.append(DetectionResult(
                        mask=mask,
                        box=box,
                        class_name=obj_name,
                        class_id=0,
                        conf=1.0
                    ))
        else:
            filter_bg = not args.include_background
            yolo_dets = detector.detect(
                proj_image,
                conf_thresh=args.conf_thresh,
                filter_background=filter_bg,
                min_pixels=80
            )
            for det in yolo_dets:
                matched_gt = match_det_to_gt(det.class_name, sequence.object_names)
                detections.append(det)

        # -------------------------------------------------------------------------
        # STAGE 4: Classify the individual 3D points of the Point Cloud from 2D masks!
        # -------------------------------------------------------------------------
        segmented_instances, point_labels = classify_point_cloud_from_masks(
            pts_3d, colors_3d, pt2px_map, valid_proj, detections, min_points=50
        )

        # Attach matched ground-truth names
        for inst in segmented_instances:
            inst.gt_name = match_det_to_gt(inst.class_name, sequence.object_names)

        # Save 2D Mask Overlay
        if args.save_vis:
            vis_overlay = render_mask_overlays(frame.color_bgr, segmented_instances)
            cv2.imwrite(os.path.join(vis_dir, f"{fid}_classified_points.png"), vis_overlay)

        print(f"[{fid}] Full Point Cloud: {total_scene_points} pts | Classified {len(segmented_instances)} object instance(s):")
        for i, inst in enumerate(segmented_instances):
            print(f"   ├─ Instance {i}: {inst.class_name} ({inst.num_points} 3D points, conf={inst.conf:.2f}, box={[int(x) for x in inst.box]})")

        # -------------------------------------------------------------------------
        # STAGE 5: Multi-Object 3D Tracking & Registration
        # -------------------------------------------------------------------------
        updates = tracker.update(
            segmented_instances,
            gt_poses_dict=frame.gt_poses,
            cad_points_dict=cad_points_dict
        )
        for tid, cls_name, status in updates:
            print(f"   └─ Track {tid} [{cls_name}]: {status}")

        t_elapsed = time.time() - t_start
        fps_times.append(1.0 / t_elapsed if t_elapsed > 0 else 0)

    # -------------------------------------------------------------------------
    # STAGE 6: Consolidate and Benchmark Canonical Point Clouds
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("📊 Point-Cloud-First Multi-Object Reconstruction & Benchmark Summary")
    print("=" * 85)

    tracker.consolidate(voxel_size=0.003)
    scene_composite = o3d.geometry.PointCloud()

    for idx, (tid, track) in enumerate(tracker.tracks.items()):
        print(f"\n--- Track {tid}: {track.class_name} (GT: {track.gt_name}) ---")
        print(f"  Accumulated canonical points: {len(track.canonical_model.points)} (Observed {track.frames_seen} frames)")

        if track.trans_errors:
            print(f"  Mean Translation Error: {np.mean(track.trans_errors):.2f} mm (Median: {np.median(track.trans_errors):.2f} mm)")
            print(f"  Mean Rotation Error:    {np.mean(track.rot_errors):.2f}° (Median: {np.median(track.rot_errors):.2f}°)")
        if track.adds_errors:
            print(f"  Mean ADD-S Pose Error:  {np.mean(track.adds_errors):.2f} mm")
            succ = compute_grasp_success_rate(track.adds_errors, threshold_mm=20.0)
            print(f"  < 2cm Viable Grasp Rate: {succ:.1f}%")

        if track.gt_name and track.gt_name in cad_mesh_dict:
            cad_mesh = cad_mesh_dict[track.gt_name]
            first_frame_data = sequence.get_frame(args.start_frame)
            if track.gt_name in first_frame_data.gt_poses:
                init_gt_pose = first_frame_data.gt_poses[track.gt_name]
                eval_pcd = o3d.geometry.PointCloud(track.canonical_model)
                eval_pcd.transform(np.linalg.inv(init_gt_pose))

                chamfer_stats = compute_chamfer_distance(eval_pcd, cad_mesh, num_eval_points=10000)
                print(f"  CAD Surface Chamfer Dist: {chamfer_stats['mean_chamfer_mm']:.2f} mm (Median: {chamfer_stats['median_chamfer_mm']:.2f} mm)")
                print(f"  F-Score (< 5mm threshold): {chamfer_stats['f_score_5mm']:.1f}%")
                print(f"  F-Score (< 10mm threshold): {chamfer_stats['f_score_10mm']:.1f}%")

                obj_mesh_ply = os.path.join(args.output_dir, f"{seq_name}_track_{tid}_{track.class_name}_mesh_frame.ply")
                save_point_cloud(eval_pcd, obj_mesh_ply)

        obj_ply = os.path.join(args.output_dir, f"{seq_name}_track_{tid}_{track.class_name}_canonical.ply")
        save_point_cloud(track.canonical_model, obj_ply)
        print(f"  Saved object model: {obj_ply}")

    scene_composite = tracker.get_composite_scene()
    scene_ply = os.path.join(args.output_dir, f"{seq_name}_multi_object_scene.ply")
    save_point_cloud(scene_composite, scene_ply)
    print(f"\nSaved aggregated multi-object scene model to '{scene_ply}'.")

    if args.save_vis:
        print(f"Saved 2D multi-mask visualization overlays to '{vis_dir}/'.")

    if fps_times:
        print(f"Average Pipeline Speed: {np.mean(fps_times):.1f} FPS")

    if args.visualize:
        print("\nLaunching Open3D Interactive Visualizer for Multi-Object Scene...")
        visualize_geometries([scene_composite], window_name="Point-Cloud-First Multi-Object Scene")


if __name__ == "__main__":
    main()
