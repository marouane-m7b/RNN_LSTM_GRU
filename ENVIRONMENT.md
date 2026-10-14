# RNN / LSTM / GRU — ready-to-use Python environment

The TensorFlow/GPU dependencies from TP2 have been reused through uv's local
cache. No package downloads were needed. This project has its own virtual
environment in the local `.venv` folder, copied from cached package files.

## Existing setup

- Windows 11 + Ubuntu-24.04 in WSL2.
- Python 3.12.3, TensorFlow 2.21.0, Keras 3.15.1.
- NVIDIA RTX 5070, approximately 12 GB VRAM.
- NumPy, pandas, matplotlib, scikit-learn, notebook execution tools, and PDF tools.
- Virtual environment: `.venv/` inside this project folder.
- This is a Linux/WSL environment: it contains `bin/`, not Windows `Scripts/`.
- Exact dependency versions: `uv.lock`.

## Run from PowerShell in this folder

```powershell
.\run_wsl.ps1 script.py
.\run_wsl.ps1 -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
```

The PowerShell helper starts WSL and selects this project's environment.
`run_gpu.py` exposes the uv-installed CUDA libraries before starting Python.
It preserves the TensorFlow GPU library-path fix used for TP2.

## Run from WSL

```sh
UV_PROJECT_ENVIRONMENT=.venv \
  ~/.local/bin/uv run --offline --extra gpu python run_gpu.py script.py
```

## Maintain dependencies

Use uv for any dependency changes. Reuse the existing environment and cache:

```sh
UV_PROJECT_ENVIRONMENT=.venv \
  ~/.local/bin/uv sync --offline --extra gpu --link-mode copy
```

Offline mode intentionally refuses missing dependencies instead of downloading
them. If a genuinely new library is needed, use uv with network access explicitly.
Keep Linux and Windows virtual environments separate.

The existing data files and notebooks have not been changed. Comparison
experiments will be implemented after the assignment and requirements are supplied.
