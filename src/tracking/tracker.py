"""
Multi-Object 3D Tracking and Canonical Model Accumulation Engine.
"""

from typing import List, Dict, Optional, Tuple
import copy
import numpy as np
import open3d as o3d

from src.detection.classifier import SegmentedObject, INSTANCE_COLORS
from src.tracking.association import match_detections_to_tracks
from src.tracking.registration import register_point_clouds_icp
from src.geometry.pointcloud import downsample_point_cloud, filter_outliers


def compute_rotation_angle_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Computes angular rotation error in degrees between two 3x3 rotation matrices."""
    R_err = R_pred @ R_gt.T
    trace = np.clip(np.trace(R_err), -1.0, 3.0)
    angle_rad = np.arccos((trace - 1.0) / 2.0)
    return float(np.degrees(angle_rad))


def compute_adds_error(model_points: np.ndarray, pred_pose: np.ndarray, gt_pose: np.ndarray) -> float:
    """Computes ADD-S (Average Distance for Symmetric objects) between ground-truth and predicted 6D poses."""
    pts_gt = (gt_pose[:3, :3] @ model_points.T + gt_pose[:3, 3:4]).T
    pts_pred = (pred_pose[:3, :3] @ model_points.T + pred_pose[:3, 3:4]).T

    from scipy.spatial import cKDTree
    tree = cKDTree(pts_gt)
    dists, _ = tree.query(pts_pred, k=1)
    return float(np.mean(dists))


class TrackedObject:
    """Manages the 3D trajectory, velocity prior, and canonical point cloud for a single object instance."""
    def __init__(
        self,
        track_id: int,
        class_name: str,
        init_box: List[float],
        init_pcd: o3d.geometry.PointCloud,
        gt_name: Optional[str] = None,
        init_gt_pose: Optional[np.ndarray] = None
    ):
        self.track_id = track_id
        self.class_name = class_name
        self.gt_name = gt_name
        self.last_box = init_box
        self.canonical_model = copy.deepcopy(init_pcd)
        self.prev_pcd = copy.deepcopy(init_pcd)
        self.T_canonical_from_curr = np.eye(4)
        self.T_est_mesh2cam = init_gt_pose.copy() if init_gt_pose is not None else None
        self.curr_velocity = np.eye(4)
        self.frames_seen = 1

        self.trans_errors: List[float] = []
        self.rot_errors: List[float] = []
        self.adds_errors: List[float] = []

    def update(
        self,
        new_box: List[float],
        frame_pcd: o3d.geometry.PointCloud,
        gt_pose: Optional[np.ndarray] = None,
        cad_points: Optional[np.ndarray] = None
    ) -> str:
        """
        Fuses a new frame observation into this object track:
          1. Solves relative SE(3) transformation T_curr_to_prev aligning frame_pcd to prev_pcd.
          2. Updates cumulative canonical transformation.
          3. Transforms and accumulates points into canonical model.
          4. Updates velocity prior and evaluates metrics if ground truth is supplied.
        """
        self.last_box = new_box
        self.frames_seen += 1

        # Interframe ICP registration using velocity prior as initial guess
        init_guess = self.curr_velocity
        T_curr_to_prev, fitness, rmse = register_point_clouds_icp(
            frame_pcd,
            self.prev_pcd,
            init_transform=init_guess,
            max_distance=0.04,
            method="point_to_plane"
        )

        self.curr_velocity = T_curr_to_prev.copy()

        # Update absolute 6D pose: T_mesh2cam(t) = inv(T_curr_to_prev) * T_mesh2cam(t-1)
        if self.T_est_mesh2cam is not None:
            self.T_est_mesh2cam = np.linalg.inv(T_curr_to_prev) @ self.T_est_mesh2cam

        # Update canonical accumulation: T_0_to_t = T_0_to_prev @ T_curr_to_prev
        self.T_canonical_from_curr = self.T_canonical_from_curr @ T_curr_to_prev
        frame_in_canonical = copy.deepcopy(frame_pcd).transform(self.T_canonical_from_curr)
        self.canonical_model += frame_in_canonical

        self.prev_pcd = copy.deepcopy(frame_pcd)

        metrics_str = f"ICP Fit: {fitness:.2f}"
        if gt_pose is not None and self.T_est_mesh2cam is not None:
            t_err = float(np.linalg.norm(self.T_est_mesh2cam[:3, 3] - gt_pose[:3, 3]) * 1000.0)
            r_err = compute_rotation_angle_deg(self.T_est_mesh2cam[:3, :3], gt_pose[:3, :3])
            self.trans_errors.append(t_err)
            self.rot_errors.append(r_err)
            metrics_str += f" | Trans Err: {t_err:.1f} mm | Rot Err: {r_err:.1f}°"

            if cad_points is not None:
                adds = compute_adds_error(cad_points, self.T_est_mesh2cam, gt_pose) * 1000.0
                self.adds_errors.append(adds)
                metrics_str += f" | ADD-S: {adds:.1f} mm"

        return metrics_str


class MultiObjectTracker:
    """Manages all active 3D object tracks across video sequences."""
    def __init__(self, iou_thresh: float = 0.15):
        self.iou_thresh = iou_thresh
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 0

    def update(
        self,
        segmented_instances: List[SegmentedObject],
        gt_poses_dict: Optional[Dict[str, np.ndarray]] = None,
        cad_points_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> List[Tuple[int, str, str]]:
        """
        Updates multi-object tracks from new segmented 3D instances.
        Returns:
          updates_log: List of (track_id, class_name, status_message)
        """
        updates_log = []
        if len(segmented_instances) == 0:
            return updates_log

        detection_boxes = [inst.box for inst in segmented_instances]
        track_boxes = {tid: track.last_box for tid, track in self.tracks.items()}

        matches, unmatched_dets, _ = match_detections_to_tracks(
            detection_boxes, track_boxes, iou_thresh=self.iou_thresh
        )

        # 1. Update matched tracks
        for det_idx, track_id in matches:
            inst = segmented_instances[det_idx]
            gt_pose = None
            cad_pts = None

            gt_name = self.tracks[track_id].gt_name or inst.gt_name
            if gt_poses_dict and gt_name in gt_poses_dict:
                gt_pose = gt_poses_dict[gt_name]
            if cad_points_dict and gt_name in cad_points_dict:
                cad_pts = cad_points_dict[gt_name]

            msg = self.tracks[track_id].update(
                inst.box, inst.pcd, gt_pose=gt_pose, cad_points=cad_pts
            )
            updates_log.append((track_id, self.tracks[track_id].class_name, f"Matched ({msg})"))

        # 2. Initialize new tracks for unmatched instances
        for det_idx in unmatched_dets:
            inst = segmented_instances[det_idx]
            gt_pose = None
            if gt_poses_dict and inst.gt_name in gt_poses_dict:
                gt_pose = gt_poses_dict[inst.gt_name]

            new_track = TrackedObject(
                track_id=self.next_track_id,
                class_name=inst.class_name,
                init_box=inst.box,
                init_pcd=inst.pcd,
                gt_name=inst.gt_name,
                init_gt_pose=gt_pose
            )
            self.tracks[self.next_track_id] = new_track
            updates_log.append((
                self.next_track_id,
                inst.class_name,
                f"Initialized with {inst.num_points} 3D points (GT: {inst.gt_name})"
            ))
            self.next_track_id += 1

        return updates_log

    def consolidate(self, voxel_size: float = 0.003) -> None:
        """Voxel downsamples and filters canonical point clouds for all tracks."""
        for track in self.tracks.values():
            track.canonical_model = downsample_point_cloud(track.canonical_model, voxel_size=voxel_size)
            track.canonical_model = filter_outliers(track.canonical_model, nb_neighbors=25, std_ratio=1.5)

    def get_composite_scene(self) -> o3d.geometry.PointCloud:
        """Assembles a single composite PointCloud where each object has a distinct visual color."""
        scene = o3d.geometry.PointCloud()
        for idx, track in enumerate(self.tracks.values()):
            color = INSTANCE_COLORS[idx % len(INSTANCE_COLORS)]
            colored_cloud = copy.deepcopy(track.canonical_model)
            colored_cloud.paint_uniform_color(color)
            scene += colored_cloud
        return scene
