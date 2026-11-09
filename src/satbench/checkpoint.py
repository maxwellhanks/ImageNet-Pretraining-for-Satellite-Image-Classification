"""Save / load / discover run checkpoints for later transfer evaluation."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .results import load_runs


@dataclass
class Checkpoint:
    """Stores the weights and metadata needed to reload a finished run for transfer eval."""
    state_dict: dict[str, torch.Tensor]
    config_hash: str
    run_id: str
    model: str
    dataset: str
    num_classes: int
    image_size: int
    experiment: str
    pipeline_version: str
    split_hash: str


def save_checkpoint(ckpt: Checkpoint, path: Path) -> None:
    """Safely saves a Checkpoint to a PyTorch tensor file using a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(asdict(ckpt), tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
    """Loads a Checkpoint written by ``save_checkpoint``."""
    raw = torch.load(path, map_location=map_location, weights_only=True)
    return Checkpoint(**raw)


HEAD_PREFIXES = ("fc.", "heads.")  # torchvision ResNet / ViT classification heads


def load_backbone_state(net: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Loads every non-head tensor of ``state_dict`` into ``net``; the head keeps its fresh init.
    Returns the dropped head keys. Raises if any backbone key is missing or unexpected."""
    backbone = {k: v for k, v in state_dict.items() if not k.startswith(HEAD_PREFIXES)}
    missing, unexpected = net.load_state_dict(backbone, strict=False)
    bad_missing = [k for k in missing if not k.startswith(HEAD_PREFIXES)]
    if bad_missing or unexpected:
        raise ValueError(f"checkpoint does not match the network: missing={bad_missing[:5]} "
                         f"unexpected={list(unexpected)[:5]}")
    return [k for k in state_dict if k.startswith(HEAD_PREFIXES)]


_HASH_RE = re.compile(r"^[0-9a-f]{12}$")


def resolve_init_from(spec: str, *, model: str, seed: int, runs: pd.DataFrame,
                      checkpoints_dir: Path) -> str:
    """Turns an ``init_from`` spec into the config_hash of a finished run with a local checkpoint.

    ``spec`` is a 12-hex config_hash, or ``"<model>@<fraction>[@<seed>]"`` where ``<model>``
    may be empty (= the fine-tune counterpart of ``model``) and ``<seed>`` defaults to ``seed``.
    Debug runs (``eval_limit``) and runs without a local ``.pt`` are ignored; among several
    candidates the newest pipeline version wins, then clean-git over dirty-git, then the latest.
    """
    from .config import backbone_counterpart

    if _HASH_RE.match(spec):
        path = checkpoints_dir / f"{spec}.pt"
        if not path.exists():
            raise FileNotFoundError(f"init_from={spec}: no local checkpoint {path}")
        return spec

    parts = spec.split("@")
    if len(parts) not in (2, 3):
        raise ValueError(f"init_from spec '{spec}' is not <hash> or <model>@<fraction>[@<seed>]")
    src_model = parts[0] or backbone_counterpart(model)
    src_fraction = float(parts[1])
    src_seed = int(parts[2]) if len(parts) == 3 else int(seed)

    if runs is None or runs.empty:
        raise FileNotFoundError(f"init_from={spec}: no runs logged in results/ to resolve it")
    cand = runs[(runs["model"] == src_model)
                & np.isclose(runs["train_fraction"].astype(float), src_fraction)
                & (runs["seed"].astype(int) == src_seed)
                & runs["eval_limit"].isna()]
    cand = cand[cand["config_hash"].map(lambda h: (checkpoints_dir / f"{h}.pt").exists())]
    if cand.empty:
        raise FileNotFoundError(
            f"init_from={spec}: no finished run of {src_model} @ {src_fraction:g} seed {src_seed} "
            f"with a checkpoint in {checkpoints_dir} (run it first, or copy the .pt file)")
    cand = cand.assign(_pv=pd.to_numeric(cand["pipeline_version"]),
                       _clean=cand["git_dirty"].astype(str).str.lower() != "true")
    best = cand.sort_values(["_pv", "_clean", "finished_at"]).iloc[-1]
    if not best["_clean"]:
        print(f"  warning: init_from={spec} resolves to {best['config_hash']}, a run made with "
              f"uncommitted code (git_dirty)")
    return str(best["config_hash"])


def list_checkpoints(
    checkpoints_dir: Path,
    results_dir: Path | None = None,
) -> pd.DataFrame:
    """Summarise ``*.pt`` files, optionally joined with training-run metadata."""
    rows = []
    runs = load_runs(results_dir) if results_dir is not None else pd.DataFrame()
    for path in sorted(checkpoints_dir.glob("*.pt")):
        try:
            ckpt = load_checkpoint(path)
        except Exception as exc:  # noqa: BLE001 - report bad files in the table
            rows.append({
                "path": str(path), "config_hash": path.stem, "error": str(exc),
            })
            continue

        row = {
            "path": str(path),
            "config_hash": ckpt.config_hash,
            "model": ckpt.model,
            "source_dataset": ckpt.dataset,
            "experiment": ckpt.experiment,
            "image_size": ckpt.image_size,
            "num_classes": ckpt.num_classes,
            "run_id": ckpt.run_id,
            "pipeline_version": ckpt.pipeline_version,
            "error": "",
        }
        if not runs.empty and "config_hash" in runs.columns:
            match = runs[runs["config_hash"] == ckpt.config_hash]
            if not match.empty:
                row_metadata = match.iloc[-1]
                row["train_fraction"] = row_metadata.get("train_fraction")
                row["seed"] = row_metadata.get("seed")
                row["test_acc"] = row_metadata.get("test_acc")
                row["user"] = row_metadata.get("user")
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sort_cols = [c for c in ("model", "train_fraction", "pipeline_version") if c in df.columns]
    return df.sort_values(sort_cols, kind="stable", na_position="last").reset_index(drop=True)


def resolve_checkpoint_paths(
    checkpoints_dir: Path,
    *,
    results_dir: Path | None = None,
    checkpoint: Path | None = None,
    config_hash: str | None = None,
    model: str | None = None,
    train_fraction: float | None = None,
    seed: int | None = None,
    experiment: str | None = None,
    source_dataset: str | None = None,
    all_checkpoints: bool = False,
) -> list[Path]:
    """Resolve one or more checkpoint paths without requiring a manual hash.

    Selection (first match wins):
      * ``checkpoint`` - explicit file path
      * ``config_hash`` - ``checkpoints_dir/<hash>.pt``
      * filters (``model`` / ``train_fraction`` / ``seed`` / ...) via results CSV
      * ``all_checkpoints`` - every ``*.pt`` in ``checkpoints_dir``
    """
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return [path]

    if config_hash is not None:
        path = checkpoints_dir / f"{config_hash}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return [path]

    filters = {
        "model": model,
        "train_fraction": train_fraction,
        "seed": seed,
        "experiment": experiment,
        "dataset": source_dataset,
    }
    selected_filters = {k: v for k, v in filters.items() if v is not None}

    if selected_filters:
        if results_dir is None:
            raise ValueError(
                "results_dir is required when filtering by model/fraction/seed/..."
            )
        runs = load_runs(results_dir)
        if runs.empty:
            raise FileNotFoundError(f"no training runs found under {results_dir}")
        mask = pd.Series(True, index=runs.index)
        for col, val in selected_filters.items():
            if col == "train_fraction":
                mask &= np.isclose(runs[col].astype(float), float(val))
            elif col == "seed":
                mask &= runs[col].astype(int) == int(val)
            else:
                mask &= runs[col].astype(str) == str(val)
        matched = runs.loc[mask].drop_duplicates("config_hash")
        if matched.empty:
            raise FileNotFoundError(f"no runs match filters {selected_filters}")
        paths, missing = [], []
        for config_hash_value in matched["config_hash"]:
            path = checkpoints_dir / f"{config_hash_value}.pt"
            if path.is_file():
                paths.append(path)
            else:
                missing.append(str(path))
        if not paths:
            raise FileNotFoundError(
                f"matched runs but no checkpoint files under {checkpoints_dir}: {missing}"
            )
        if missing:
            print(f"[checkpoint] warning: missing checkpoints (skipped): {missing}")
        return paths

    if all_checkpoints:
        paths = sorted(checkpoints_dir.glob("*.pt"))
        if not paths:
            raise FileNotFoundError(f"no checkpoints in {checkpoints_dir}")
        return paths

    raise ValueError(
        "specify --checkpoint, --config-hash, filters (--model / --fraction / --seed), or --all"
    )
