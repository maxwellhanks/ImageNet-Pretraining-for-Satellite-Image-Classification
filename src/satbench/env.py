"""Hardware, platform, paths and provenance (git commit, versions)."""
from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import torch
import torchvision


# Paths
def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk up from ``start`` (default: cwd) and from this file to the repo root."""
    for origin in (Path(start or Path.cwd()).resolve(), Path(__file__).resolve()):
        for p in (origin, *origin.parents):
            if (p / "pyproject.toml").exists() and (p / "src" / "satbench").is_dir():
                return p
    return Path.cwd().resolve()


def default_dirs(repo_root: Path) -> dict[str, Path]:
    return {
        "data_dir": repo_root / "data",
        "results_dir": repo_root / "results",
        "splits_dir": repo_root / "splits",
        "checkpoints_dir": repo_root / "checkpoints",
    }


# Device
def resolve_device(preference: str = "auto") -> torch.device:
    preference = preference.lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested but MPS is not available")
    return torch.device(preference)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


# Provenance
def detect_platform() -> str:
    if "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    return "local"


def git_info(repo_root: Path) -> tuple[str, bool]:
    """(commit, dirty). 'dirty' only looks at code + configs, not results/notebooks."""
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no",
                          "--", "src", "configs", "pyproject.toml", "uv.lock"))
        return commit, dirty
    except Exception:
        return "unknown", False


def collect_env(device: torch.device, repo_root: Path) -> dict[str, object]:
    from . import __version__

    commit, dirty = git_info(repo_root)
    return {
        "device": device.type,
        "device_name": device_name(device),
        "platform": detect_platform(),
        "hostname": socket.gethostname(),
        "os": platform.platform(terse=True),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "satbench_version": __version__,
        "git_commit": commit,
        "git_dirty": dirty,
    }
