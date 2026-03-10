#!/bin/bash
set -e

echo "--- 1. CLEANING ---"
rm -rf dist/ build/ *.egg-info
# Remove old env if it exists to start fresh
conda env remove -n pupil_ci_test -y || true

echo "--- 2. CREATING MINIMAL ENV ---"
# Only install Python and Pip via Conda to avoid solver conflicts
conda create -y -n pupil_ci_test python=3.10 pip -c conda-forge

# Activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pupil_ci_test

echo "--- 3. INSTALLING BINARIES VIA PIP ---"
# We use 'opencv-python-headless' because it bundles its own 
# GUI libraries, often avoiding the need for system libGL
pip install opencv-python-headless imageio[ffmpeg]

echo "--- 4. INSTALLING PYTORCH & DETECTRON2 ---"
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

echo "--- 5. INSTALLING PROJECT ---"
pip install build
python -m build
pip install dist/*.tar.gz

echo "--- 6. VERIFICATION ---"
python -c "import detectron2; import cv2; print('✅ Setup Complete!')"
pytest