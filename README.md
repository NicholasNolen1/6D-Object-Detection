# Next-Gen 3D Object Detection, 6D Pose Tracking, and Reconstruction

**Authors:** Nicholas Nolen, Gan-Orgil Gantumur  
**Target Hardware:** Consumer-grade GPU (NVIDIA RTX 40/50 Series)  

---

## 📖 Overview

This repository implements a real-time, point-cloud-first pipeline that:
1. Ingests RGB-D sensor streams and lifts the entire frame into a **Full 3D Scene Point Cloud** $\mathcal{P} = \{(X_i, Y_i, Z_i, R_i, G_i, B_i)\}$.
2. **Projects the 3D Point Cloud onto a 2D plane**, establishing an explicit bi-directional coordinate map $\pi: \text{3D Point } i \longleftrightarrow \text{2D Coordinate } (u_i, v_i)$.
3. Runs 2D instance segmentation (YOLOv8-Seg) on the projected 2D representation to isolate bounding boxes and masks for all objects in the scene.
4. **Classifies the individual points of the 3D point cloud**, directly segmenting object-level 3D point clusters without losing geometric context.
5. Tracks interframe $SE(3)$ motion per object using Point-to-Plane ICP and fuses multi-view observations into clean canonical 3D object representations over time.

For full research details, refer to [proposal.md](proposal.md) and the original proposal [Proposal.pdf](Proposal.pdf).

---

## 🚀 Quickstart on Your GPU Rig

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd project

# Run the automated setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source .venv/bin/activate
```

> **Note for NVIDIA GPU users:** If you want CUDA-accelerated PyTorch:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

### 2. Run on Real-World Benchmark RGB-D Clips

Download targeted clips from the official **YCBMultiTrack-Real** dataset (Intel RealSense D435i + OptiTrack ground truth) using our partial streaming utility:

```bash
# Download targeted real-world video clips (only ~30-50 MB each, no 9GB download needed)
python scripts/download_clip.py --sequence 006_mustard_bottle --start-frame 80 --num-frames 100
python scripts/download_clip.py --sequence 006_mustard_bottle_010_potted_meat_can_easy --start-frame 0 --num-frames 60

# Run the real-time 3D tracking & reconstruction pipeline with YOLOv8-Seg
python pipeline.py --sequence-dir data/real/006_mustard_bottle --object-name 006_mustard_bottle --start-frame 80 --num-frames 30 --mask-source yolo

# Run on the multi-object clip
python pipeline.py --sequence-dir data/real/006_mustard_bottle_010_potted_meat_can_easy --object-name 006_mustard_bottle --start-frame 0 --num-frames 30 --mask-source yolo --output-ply outputs/real_multi_obj_canonical.ply

# Optional: Visualize accumulated canonical point clouds interactively
python pipeline.py --sequence-dir data/real/006_mustard_bottle --visualize
```

### 3. Run the Synthetic MVP Pipeline Demo

The `demo.py` script runs on sample Redwood RGB-D frames using the modular `src` library:

```bash
python demo.py
python demo.py --visualize
```

### 4. Run the Full Test Suite

The codebase includes an extensive automated test suite covering all modules with 100% genuine math and integration checks:

```bash
pytest -v
```

---

## 📂 Project Structure

```text
├── src/                          # Reusable core library modules
│   ├── geometry/                 # Camera models, unprojection, and 3D-to-2D projection
│   │   ├── camera.py             # CameraIntrinsics pinhole parameters & matrix math
│   │   ├── pointcloud.py         # RGB-D to 3D scene point cloud lifting & filtering
│   │   └── projection.py         # 3D-to-2D projection with bi-directional point indexing
│   ├── detection/                # 2D detection & 3D point cloud classification
│   │   ├── detector.py           # YOLOv8-Seg detector wrapper & regional fallbacks
│   │   └── classifier.py         # Point-cloud classification from 2D projected masks
│   ├── tracking/                 # Multi-object association & ICP registration
│   │   ├── association.py        # 2D Bipartite IoU matching
│   │   ├── registration.py       # Point-to-Plane ICP & normal estimation
│   │   └── tracker.py            # TrackedObject state & MultiObjectTracker
│   ├── evaluation/               # Quantitative benchmark metrics
│   │   └── metrics.py            # ADD-S, Chamfer distance, F-Score, translation/rotation error
│   ├── datasets/                 # RGB-D sequence loaders
│   │   ├── ycb.py                # YCBMultiTrack RGB-D sequence & CAD mesh loader
│   │   └── redwood.py            # Redwood sample sequence loader
│   └── visualization/            # Visual overlay & 3D scene rendering
│       ├── overlay.py            # 2D mask overlays and bounding box drawing
│       └── scene.py              # Open3D 3D viewer & composite scene assembly
├── tests/                        # Comprehensive unit & integration tests
│   ├── test_geometry.py
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_evaluation.py
│   ├── test_datasets.py
│   └── test_integration.py       # End-to-end integration test on real sequences
├── pipeline.py                   # Real-world 3D detection, 6D tracking & reconstruction pipeline
├── demo.py                       # MVP demo on sample Redwood RGB-D sequences
├── scripts/
│   └── download_clip.py          # Targeted streaming downloader for YCBMultiTrack clips
├── setup.sh                      # One-command environment installer
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Core Python dependencies
├── Proposal.pdf                  # Original project proposal
├── proposal.md                   # Refined technical proposal with detailed methodology
└── outputs/                      # Generated 3D point cloud models (.ply) and overlays
```

---

## 📊 Target Datasets

* **YCBMultiTrack (Lin et al., ECCV 2024):** Multi-object sequences with mutual occlusion and re-identification (baseline for Point2Pose).
* **YCB-Video / YCBInEOAT:** Benchmark 6D pose dataset with ground-truth CAD models.
* **HO3D:** Hand-object interaction sequences featuring severe occlusions.
