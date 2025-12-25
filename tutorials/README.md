# Tutorials

Here’s a **Step 1 tutorial using Genesis** on **macOS (Apple Silicon / M5)**, with your preferences: **uv + Docker + Git**. I’ll structure it so you can (A) run a viewer on macOS natively (best dev experience), and (B) optionally run headless in Docker for reproducibility/CI.

Sources: Genesis repo + official docs for “Hello, Genesis” and “Control Your Robot”. ([GitHub](https://github.com/Genesis-Embodied-AI/Genesis))

Reference: https://genesis-world.readthedocs.io/en/latest/


## What you will build in this tutorial

1. Run **Hello, Genesis**: load a Franka Panda arm and step physics (with viewer).
2. Run a minimal **control** example: move end-effector above a cube, “attach” it (weld/suction), move, release.
3. Keep the project in a clean Git repo, using uv.

## Prerequisites

- macOS on Apple Silicon
- Python 3.10–3.13 supported by Genesis package. ([GitHub](https://github.com/Genesis-Embodied-AI/Genesis))
- PyTorch installed (Genesis docs explicitly require it). ([GitHub](https://github.com/Genesis-Embodied-AI/Genesis))

Note: some users hit macOS wheel issues on older macOS versions (especially around OMPL). If you get a “no wheel for macosx_…arm64” type error, upgrading macOS usually resolves it. ([Qiita](https://qiita.com/ryosukematsuda/items/16a08301b498d8207ff3?utm_source=chatgpt.com))

---

## Part A: Native macOS setup with uv (recommended)

### 1) Create your project repo

```bash
mkdir genesis-arm-trial
cd genesis-arm-trial
git init
uv init
```

### 2) Pick a Python version (good default: 3.11 or 3.12)

`pyproject.toml`

```jsx
[project]
name = "genesis-arm-trial"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12,<3.14"
dependencies = []
```

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```

### 3) Install dependencies (PyTorch + Genesis)

### 3.1 Pin a stable Python version (important)

Genesis + PyTorch are safest on **Python 3.11 or 3.12** right now.

```bash
uv python pin 3.12
uv lock --refresh
```

This creates `.python-version` so uv always uses it.

### 3.2 Install PyTorch (Apple Silicon)

Genesis **requires PyTorch**, but PyTorch should be installed **explicitly**.

For macOS Apple Silicon (CPU / Metal):

```bash
uv add torch torchvision torchaudio
```

### 3.3 Install Genesis

The PyPI package name is **`genesis-world`**

The import name is **`genesis`**

```bash
uv add genesis-world

```

Optional but recommended:

```bash
uv add numpy
```

### 3.4 What your `pyproject.toml` should look like

Check `pyproject.toml`:

```toml
[project]
name = "genesis-arm-trial"
version = "0.1.0"
requires-python = ">=3.12,<3.14"

dependencies = [
    "torch",
    "torchvision",
    "torchaudio",
    "genesis-world",
    "numpy",
]
```

## Sanity checklist (you’re good if all pass)

```bash
uv run python - << 'EOF'
import genesis as gs
import torch
print("Genesis OK")
print("Torch:", torch.__version__)
print("MPS:", torch.backends.mps.is_available())
EOF

# Genesis OK
# Torch: 2.9.1
# MPS: True
```

### 4) Run the tutorials

- Hello: `tutorials/hello/README.md`
- Control: `tutorials/control/README.md`

---

## Part C: Docker setup (recommended use: headless runs / CI)

Recommendation: use native macOS for interactive viewer work; use Docker only for headless runs or CI. On macOS, Docker typically won’t give you GPU/Metal acceleration or a smooth GUI viewer, so treat it as “headless reproducibility”.

Genesis repo provides a Dockerfile build command. ([GitHub](https://github.com/Genesis-Embodied-AI/Genesis))

### 1) Build the official Genesis image (from their repo)

In a separate folder (or as a git submodule), clone Genesis:

```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
docker build -t genesis -f docker/Dockerfile docker

```

This is the command shown in the repo. ([GitHub](https://github.com/Genesis-Embodied-AI/Genesis))

### 2) Run your scripts headlessly

Mount your project into the container and run without viewer:

- Set `show_viewer=False`
- Use offscreen rendering (later) or just step physics

Example run:

```bash
docker run --rm -it -v "$(pwd)":/workspace -w /workspace genesis:latest \
  python tutorials/hello/hello.py

```

The repo’s example run command is Linux/X11 oriented (uses `xhost`, DISPLAY, `/tmp/.X11-unix`, `--gpus all`). On macOS you usually skip the GUI path. ([GitHub](https://github.com/Genesis-Embodied-AI/Genesis))

### Optional docker-compose.yml

If you want it anyway for convenience:

```yaml
services:
  genesis:
    image: genesis:latest
    volumes:
      - ./:/workspace
    working_dir: /workspace
    command: python tutorials/hello/hello.py

```

## Suggested “done” criteria for Step 1

You’re ready to proceed to the Raspberry Pi + real arm when:

- You can run `tutorials/hello/hello.py` reliably
- You can do a controlled pick/place (even suction-weld) in sim
- You can refactor into a clean loop:
    - observe state
    - decide target pose (script first)
    - execute
    - verify result

If you want, I can tailor the next iteration into a super practical “LLM as high-level planner” layer for this sim (JSON actions like `move_to(x,y,z)`, `attach`, `release`) before you touch hardware.
