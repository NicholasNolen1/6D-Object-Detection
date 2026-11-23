#!/usr/bin/env bash
# Quickstart Setup Script for 3D Point Cloud Object Tracking & Reconstruction

set -e

echo "================================================================="
echo "⚡ Setting up 3D Object Tracking & Reconstruction Environment"
echo "================================================================="

# 1. Create Python virtual environment if not already present
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# 5. Make demo script executable
chmod +x demo.py

echo ""
echo "================================================================="
echo "✅ Environment setup complete!"
echo "To run the demo on your rig:"
echo "   source .venv/bin/activate"
echo "   python demo.py --visualize"
echo "================================================================="
