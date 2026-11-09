"""All sources of randomness are controlled from here.

Design: every run is reproducible *in isolation*. Nothing depends on which runs
happened before it in the same process, so skipping/resuming runs never changes
results. Each random stream gets its own seed derived from the run seed:

* model initialisation      -> torch global RNG, seeded right before building
* training-subset selection -> numpy Generator(derive_seed(seed, "train_subset"))
* mini-batch shuffling      -> torch.Generator(derive_seed(seed, "shuffle"))
"""
from __future__ import annotations

import os

# Must be set before the first cuBLAS call for deterministic CUDA matmuls.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib  # noqa: E402
import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def derive_seed(seed: int, *tags: str) -> int:
    """Stable 32-bit seed for a named random stream (same on every machine)."""
    payload = "|".join([str(int(seed)), *tags]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU, CUDA and MPS) and toggle determinism."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)  # also seeds all CUDA / MPS devices
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: ops without a deterministic kernel warn instead of crashing
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
