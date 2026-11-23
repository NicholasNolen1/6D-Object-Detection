#!/usr/bin/env python3
"""
Replay isolated 3D object reconstructions through their tracked 6D trajectories in 3D space.

Supports:
  1. Multi-Object Simultaneous Replay: Replays multiple objects moving and rotating together.
  2. Trajectory Extraction: Genuine interframe SE(3) poses solved during tracking.
  3. Continuous Lie-Algebra / SLERP Interpolation: Interpolates smoothly between past and present frames.
  4. Isolated Projection Rendering: Renders only the isolated objects without background clutter.
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

from src.geometry.camera import CameraIntrinsics
from src.geometry.projection import project_point_cloud_to_2d
from src.datasets.ycb import YCBMultiTrackSequence
from pipeline import match_det_to_gt
from src.detection.detector import DetectionResult, YOLODetector
from src.detection.classifier import classify_point_cloud_from_masks, INSTANCE_COLORS
from src.tracking.tracker import MultiObjectTracker
from src.geometry.pointcloud import rgbd_to_scene_point_cloud


def interpolate_se3_trajectory(poses: list, interp_factor: int = 2) -> list:
    """
    Interpolates SE(3) trajectory between consecutive discrete frames using:
      - Spherical Linear Interpolation (SLERP) for rotation SO(3)
      - Linear/Spline interpolation for translation R^3
    """
    if len(poses) < 2:
        return poses

    times = np.arange(len(poses))
    rotations = Rotation.from_matrix([P[:3, :3] for P in poses])
    translations = np.array([P[:3, 3] for P in poses])

    slerp = Slerp(times, rotations)

    query_times = np.linspace(0, len(poses) - 1, (len(poses) - 1) * interp_factor + 1)
    interp_rots = slerp(query_times)

    interp_trans = np.zeros((len(query_times), 3))
    for dim in range(3):
        interp_trans[:, dim] = np.interp(query_times, times, translations[:, dim])

    interpolated_poses = []
    for R, t in zip(interp_rots.as_matrix(), interp_trans):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        interpolated_poses.append(T)

    return interpolated_poses


def main():
    parser = argparse.ArgumentParser(description="Replay isolated 3D objects along their tracked 3D trajectories")
    parser.add_argument("--sequence-dir", default="data/YCBMultiTrack-Real/021_bleach_cleanser_005_tomato_soup_can_008_pudding_box")
    parser.add_argument("--output-dir", default="outputs/replay_multi_object")
    parser.add_argument("--target-class", default="all", help="Target object class to replay, or 'all' for multi-object replay")
    parser.add_argument("--mask-source", choices=["gt", "yolo"], default="gt", help="Detection/mask source")
    parser.add_argument("--start-frame", type=int, default=1010, help="First frame index to process")
    parser.add_argument("--num-frames", type=int, default=30, help="Number of sequence frames to track")
    parser.add_argument("--interp-factor", type=int, default=3, help="Interpolation factor between frames (e.g. 3x sub-frame smoothness)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("=" * 85)
    print("🎬 Isolated 3D Multi-Object Trajectory Replay Engine")
    print(f"Sequence:      {args.sequence_dir}")
    print(f"Target:        {args.target_class.upper()}")
    print(f"Mask Mode:     {args.mask_source.upper()}")
    print(f"Frames:        {args.start_frame} to {args.start_frame + args.num_frames - 1} (Interp: {args.interp_factor}x)")
    print("=" * 85)

    # 1. Load sequence
    seq = YCBMultiTrackSequence(args.sequence_dir)
    intrinsics = seq.intrinsics

    # 2. Track objects across frames
    tracker = MultiObjectTracker(iou_thresh=0.15)
    detector = YOLODetector("yolov8n-seg.pt") if args.mask_source == "yolo" else None

    # History of camera poses per track_id: {tid: [T_cam_from_curr_frame, ...]}
    track_pose_histories = {}
    track_frame_indices = {}

    end_frame = min(args.start_frame + args.num_frames, len(seq))
    print(f"\n[Stage 1/3] Tracking objects across frames {args.start_frame}..{end_frame - 1}...")

    for idx in range(args.start_frame, end_frame):
        frame = seq.get_frame(idx)
        pts_3d, colors_3d, _ = rgbd_to_scene_point_cloud(frame.depth, frame.color_rgb, intrinsics)
        proj_img, pt2px, valid_proj = project_point_cloud_to_2d(pts_3d, colors_3d, intrinsics)

        detections = []
        if args.mask_source == "gt":
            for obj_name, mask in frame.gt_masks.items():
                if mask.sum() > 80:
                    ys, xs = np.where(mask > 0)
                    detections.append(DetectionResult(
                        mask=mask,
                        box=[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                        class_name=obj_name,
                        class_id=0,
                        conf=1.0
                    ))
        else:
            yolo_dets = detector.detect(proj_img, conf_thresh=0.10, filter_background=True)
            for det in yolo_dets:
                det.class_name = det.class_name.lower()
                detections.append(det)

        segmented, _ = classify_point_cloud_from_masks(pts_3d, colors_3d, pt2px, valid_proj, detections)
        for s in segmented:
            s.gt_name = match_det_to_gt(s.class_name, seq.object_names)

        tracker.update(segmented, gt_poses_dict=frame.gt_poses)

        # Record instantaneous camera pose for each active track
        for tid, tr in tracker.tracks.items():
            if tr.frames_seen >= 1:
                if tid not in track_pose_histories:
                    track_pose_histories[tid] = []
                    track_frame_indices[tid] = []
                T_cam_from_curr = np.linalg.inv(tr.T_canonical_from_curr)
                track_pose_histories[tid].append(T_cam_from_curr.copy())
                track_frame_indices[tid].append(idx - args.start_frame)

    tracker.consolidate(voxel_size=0.003)

    # 3. Filter tracks according to target-class
    replayed_tracks = {}
    for tid, tr in tracker.tracks.items():
        if len(track_pose_histories.get(tid, [])) < 3:
            continue
        if args.target_class != "all" and args.target_class.lower() not in tr.class_name.lower():
            continue
        replayed_tracks[tid] = tr

    if not replayed_tracks:
        print("Error: No qualifying tracks found for replay.")
        return

    print(f"\nDiscovered {len(replayed_tracks)} distinct object tracks for 3D replay:")
    for tid, tr in replayed_tracks.items():
        n_pts = len(tr.canonical_model.points)
        n_poses = len(track_pose_histories[tid])
        print(f"  ├─ Track {tid} [{tr.class_name}]: {n_pts} canonical points | {n_poses} keyframe poses")

    # 4. Continuous Lie-Algebra / SLERP trajectory interpolation
    print(f"\n[Stage 2/3] Performing continuous SE(3) trajectory interpolation ({args.interp_factor}x)...")
    interpolated_trajectories = {}
    for tid in replayed_tracks.keys():
        poses = track_pose_histories[tid]
        interp_poses = interpolate_se3_trajectory(poses, interp_factor=args.interp_factor)
        interpolated_trajectories[tid] = interp_poses
        print(f"  ├─ Track {tid}: {len(poses)} discrete keyframes -> {len(interp_poses)} continuous poses")

    max_steps = max(len(p) for p in interpolated_trajectories.values())

    # 5. Render isolated multi-object replay
    print(f"\n[Stage 3/3] Replaying isolated multi-object scene (saving to {frames_dir}/)...")
    video_path = os.path.join(args.output_dir, f"isolated_{args.target_class}_replay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(video_path, fourcc, 24.0, (intrinsics.width, intrinsics.height))

    for step in range(max_steps):
        # Merge all active objects at this sub-frame step into one scene
        all_scene_pts = []
        all_scene_colors = []

        active_labels = []
        for obj_idx, (tid, tr) in enumerate(replayed_tracks.items()):
            poses = interpolated_trajectories[tid]
            if step >= len(poses):
                continue
            T_pose = poses[step]

            pts_canonical = np.asarray(tr.canonical_model.points)
            colors_canonical = np.asarray(tr.canonical_model.colors)

            # Transform into camera frame at this time
            pts_h = np.concatenate([pts_canonical, np.ones((len(pts_canonical), 1))], axis=-1)
            pts_replayed = (T_pose @ pts_h.T).T[:, :3]

            all_scene_pts.append(pts_replayed)
            all_scene_colors.append(colors_canonical)

            t_pos = T_pose[:3, 3] * 1000.0  # mm
            active_labels.append(f"{tr.class_name[:14]}: [{t_pos[0]:.0f}, {t_pos[1]:.0f}, {t_pos[2]:.0f}]")

        if not all_scene_pts:
            continue

        merged_pts = np.vstack(all_scene_pts).astype(np.float32)
        merged_colors = np.vstack(all_scene_colors).astype(np.float32)

        # Render onto camera canvas with depth-buffering
        proj_img, _, _ = project_point_cloud_to_2d(merged_pts, merged_colors, intrinsics)

        # On-screen HUD annotations
        cv2.putText(proj_img, "ISOLATED MULTI-OBJECT 4D REPLAY", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(proj_img, f"Frame: {step + 1}/{max_steps} | Objects: {len(active_labels)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        for idx, lbl in enumerate(active_labels):
            color_rgb = INSTANCE_COLORS[idx % len(INSTANCE_COLORS)]
            color_bgr = (int(color_rgb[2] * 255), int(color_rgb[1] * 255), int(color_rgb[0] * 255))
            cv2.putText(proj_img, lbl, (20, 85 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1, cv2.LINE_AA)

        frame_bgr = cv2.cvtColor(proj_img, cv2.COLOR_RGB2BGR)
        out_video.write(frame_bgr)

        if step % args.interp_factor == 0:
            sample_img_path = os.path.join(frames_dir, f"frame_{step:04d}.png")
            cv2.imwrite(sample_img_path, frame_bgr)

    out_video.release()
    print(f"\n✅ Video replay saved to: {video_path}")
    print(f"   Sample keyframe images saved to: {frames_dir}/")


if __name__ == "__main__":
    main()
