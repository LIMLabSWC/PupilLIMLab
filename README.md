[![PyPI version](https://img.shields.io/pypi/v/Pupil_LIMlab.svg)](https://pypi.org/project/Pupil_LIMlab/)

# PupilToolKit

*A lightweight toolkit for pupil tracking and processing used by LIM Lab.*

## Overview

PupilToolKit provides utilities and pipelines for pupil extraction, processing, and inference. The repository includes two main components:

- `PupilProcessing`: processing pipelines and utility functions.
- `PupilSense`: integration with PupilSense inference tools and example scripts.

Example configuration files for experiments and devices are stored in the `configs/` directory.

## Features

- Extract and process pupil data
- Run inference with PupilSense models
- Example scripts and test coverage

## Requirements

- Python 3.10+ (recommended)
- Typical scientific packages: `numpy`, `scipy`, etc. (see `pyproject.toml` / `setup.py`)

Create and activate a virtual environment before installing:

```bash
conda create -n process_pupil python=3.10 -y
conda activate process_pupil
pip install -e .
```

Or using pip/venv:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -e .
```

## Installation

Recommended: create and activate a virtual environment (conda or venv) before installing packages.

- **PyTorch & TorchVision (CUDA 12.1)** — install the CUDA 12.1 builds of PyTorch 2.2 and TorchVision 0.17.0:

```bash
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

If you do not have CUDA 12.1 or need a CPU-only build, install the matching CPU/compatible wheel instead (omit `+cu121` or follow the official PyTorch install selector at https://pytorch.org).

- **Detectron2** — platform-specific instructions:

- Windows (pre-built wheels):

```bash
pip install detectron2 --extra-index-url https://myhloli.github.io/wheels/
```

- Linux (install from upstream):

```bash
pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git
```

- **PupilLIMlab package** — install the project package from PyPI (or your registry):

```bash
pip install PupilLIMlabSWC
```

After these steps you should be able to run the example scripts and the PupilSense inference tools.


## Quick start

Run the example script:

```bash
python example/run_script.py
```

Run tests:

```bash
pytest -q
```

## Repository structure

- `PupilProcessing/` — core processing modules and utilities
- `PupilSense/` — PupilSense integration and inference scripts
- `configs/` — YAML configuration files for experiments and devices
- `example/` — runnable example scripts
- `tests/` — unit tests

## Contributing

Contributions welcome. Please open issues or pull requests and follow existing code style. Add tests for any behavior changes.
