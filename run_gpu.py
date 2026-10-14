"""Lance Python avec les bibliothèques CUDA installées par uv visibles sous WSL."""
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit("Usage : uv run --extra gpu python run_gpu.py script.py")
env = os.environ.copy()
if sys.platform == "linux":
    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    paths = [str(p) for p in sorted(nvidia.glob("*/lib"))]
    paths += ["/usr/lib/wsl/lib"]
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(paths)
raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]], env=env))
