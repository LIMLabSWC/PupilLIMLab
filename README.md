# PupilToolKit

*A lightweight toolkit for pupil tracking and processing used by LIM Lab.*

## Overview

PupilToolKit provides utilities and pipelines for pupil extraction, processing, and inference. The repository includes two main components:

- `PupilProcessing`: processing pipelines and utility functions.
- `PupilSense`: integration with PupilSense inference tools and example scripts.

Configuration files for experiments and devices are stored in the `configs/` directory.

## Features

- Extract and process pupil data
- Run inference with PupilSense models
- Example scripts and test coverage

## Requirements

- Python 3.8+ (recommended)
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

## License

This repository does not include a license file. Add a `LICENSE` if you intend to publish.
