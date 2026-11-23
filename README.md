# Next-Gen 3D Object Detection, 6D Pose Tracking, and Reconstruction

**Authors:** Nicholas Nolen, Gan-Orgil Gantumur  
**Target Hardware:** Consumer-grade GPU (NVIDIA RTX 40/50 Series)  

---

## 📖 Overview

This repository implements a real-time, model-free pipeline that lifts 2D video object detections/segmentations into 3D camera space, tracks object movements across consecutive frames using standard interframe motion and point correspondence paradigms, and fuses observed points into a clean canonical 3D object representation over time.

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

### 2. Run the MVP Pipeline Demo

The included `demo.py` script automatically runs the full 2D-to-3D lifting, interframe registration, and canonical point cloud accumulation:

```bash
# Run pipeline and save output point cloud to outputs/canonical_object.ply
python demo.py

# Run with interactive 3D viewer enabled
python demo.py --visualize
```

---

## 📂 Project Structure

```text
├── Proposal.pdf          # Original project proposal
├── proposal.md           # Refined technical proposal with detailed methodology
├── demo.py               # Working MVP demo: 2D-to-3D unprojection & ICP registration
├── setup.sh              # One-command environment installer
├── requirements.txt      # Core Python dependencies (Open3D, PyTorch, Ultralytics, etc.)
├── .gitignore            # Git ignore rules for datasets, weights, and point clouds
└── outputs/              # Generated 3D point cloud models (.ply)
```

---

## 📊 Target Datasets

* **YCBMultiTrack (Lin et al., ECCV 2024):** Multi-object sequences with mutual occlusion and re-identification (baseline for Point2Pose).
* **YCB-Video / YCBInEOAT:** Benchmark 6D pose dataset with ground-truth CAD models.
* **HO3D:** Hand-object interaction sequences featuring severe occlusions.
