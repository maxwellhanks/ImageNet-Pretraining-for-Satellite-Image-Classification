"""Transfer evaluation: load a EuroSAT (etc.) checkpoint and evaluate on another dataset."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .checkpoint import Checkpoint, load_checkpoint
from .config import RunConfig, resolve_user
from .data import TransformedSubset, load_dataset
from .env import collect_env, find_repo_root, resolve_device
from .models import build_model
from .results import TRANSFER_SCHEMA_VERSION, TransferStore, load_runs
from .train import EvalResult, evaluate


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_config_from_checkpoint(ckpt: Checkpoint) -> RunConfig:
    """Minimal RunConfig so ``build_model`` gets ``image_size`` (other fields unused)."""
    return RunConfig(
        experiment=ckpt.experiment,
        dataset=ckpt.dataset,
        model=ckpt.model,
        train_fraction=1.0,
        seed=0,
        epochs=1,
        batch_size=64,
        lr=1e-4,
        weight_decay=1e-2,
        image_size=ckpt.image_size,
        select_best_on_val=True,
        deterministic=True,
        eval_limit=None,
    )


def transfer_eval(
    checkpoint_path: Path,
    *,
    target_dataset: str,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[Checkpoint, EvalResult, int]:
    """Load ``checkpoint_path`` and evaluate on ``target_dataset`` (all samples).
    Returns ``(checkpoint, metrics, n_eval)``.
    """
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    if ckpt.dataset == target_dataset:
        raise ValueError(
            f"checkpoint source dataset is already '{target_dataset}'; nothing to transfer"
        )

    target = load_dataset(target_dataset, data_dir)
    if target.num_classes != ckpt.num_classes:
        raise ValueError(
            f"class count mismatch: checkpoint has {ckpt.num_classes}, "
            f"{target_dataset} has {target.num_classes}"
        )

    bundle = build_model(ckpt.model, ckpt.num_classes, _run_config_from_checkpoint(ckpt))
    bundle.net.load_state_dict(ckpt.state_dict)
    bundle.net.to(device)

    loader = DataLoader(
        TransformedSubset(target, np.arange(len(target)), bundle.eval_transform),
        batch_size=batch_size,
        shuffle=False,
    )
    return ckpt, evaluate(bundle, loader, target.num_classes, device), len(target)


def run_transfer_evals(
    checkpoint_paths: list[Path],
    *,
    data_dir: Path,
    results_dir: Path,
    target_dataset: str = "uc_merced_rgb",
    user: str | None = None,
    device: str = "auto",
    batch_size: int = 128,
    skip_completed: bool = True,
) -> pd.DataFrame:
    """Evaluate each checkpoint on ``target_dataset`` and append transfer CSV rows."""
    user = resolve_user(user)
    repo = find_repo_root()
    device_t = resolve_device(device)
    env = collect_env(device_t, repo)

    store = TransferStore(results_dir, user)
    done = store.completed_keys() if skip_completed else set()
    runs = load_runs(results_dir)

    new_rows: list[dict] = []
    skipped = 0
    for i, path in enumerate(checkpoint_paths, 1):
        # Peek hash without full weight load when possible (filename = config_hash)
        config_hash = path.stem
        key = (config_hash, target_dataset)
        if key in done:
            skipped += 1
            print(f"[{i}/{len(checkpoint_paths)}] skip (already transferred): {path.name} -> {target_dataset}")
            continue

        print(f"\n[{i}/{len(checkpoint_paths)}] transfer {path.name} -> {target_dataset}")
        started_at = _now()
        transfer_id = uuid.uuid4().hex[:16]
        ckpt, result, n_eval = transfer_eval(
            path, target_dataset=target_dataset, data_dir=data_dir,
            device=device_t, batch_size=batch_size,
        )

        train_fraction = seed = source_test_acc = None
        if not runs.empty:
            match = runs[runs["config_hash"] == ckpt.config_hash]
            if not match.empty:
                row_metadata = match.iloc[-1]
                train_fraction = float(row_metadata["train_fraction"])
                seed = int(row_metadata["seed"])
                source_test_acc = float(row_metadata["test_acc"])

        row = {
            "schema_version": TRANSFER_SCHEMA_VERSION,
            "transfer_id": transfer_id,
            "config_hash": ckpt.config_hash,
            "run_id": ckpt.run_id,
            "user": user,
            "status": "completed",
            "started_at": started_at,
            "finished_at": _now(),
            "source_dataset": ckpt.dataset,
            "target_dataset": target_dataset,
            "model": ckpt.model,
            "experiment": ckpt.experiment,
            "pipeline_version": ckpt.pipeline_version,
            "split_hash": ckpt.split_hash,
            "image_size": ckpt.image_size,
            "num_classes": ckpt.num_classes,
            "train_fraction": train_fraction,
            "seed": seed,
            "source_test_acc": source_test_acc,
            "n_eval": n_eval,
            "transfer_acc": result.acc,
            "transfer_loss": result.loss,
            "transfer_macro_f1": result.macro_f1,
            "transfer_confusion_json": json.dumps(result.confusion),
            "checkpoint_path": str(path),
            "device": env["device"],
            "device_name": env["device_name"],
            "platform": env["platform"],
            "git_commit": env["git_commit"],
            "git_dirty": env["git_dirty"],
        }
        store.append(row)
        new_rows.append(row)
        done.add(key)
        print(f"  -> transfer acc {result.acc:.4f} | macro-F1 {result.macro_f1:.4f} | "
              f"n_eval {n_eval}")

    print(f"\ndone: {len(new_rows)} new, {skipped} skipped -> {store.path}")
    return pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
