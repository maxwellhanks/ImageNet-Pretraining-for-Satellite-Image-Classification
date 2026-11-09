"""Run a whole experiment grid and log every completed run."""
from __future__ import annotations

import dataclasses
import json
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config  # module access: PIPELINE_VERSION must match config_hash() even after autoreload
from .config import ExperimentConfig, RunConfig, resolve_user
from .data import Splits, get_splits, load_dataset
from .env import collect_env, default_dirs, find_repo_root, resolve_device
from .models import MODELS
from .results import SCHEMA_VERSION, ResultStore, load_runs
from .train import RunOutcome, execute_run
from .checkpoint import Checkpoint, load_checkpoint, resolve_init_from, save_checkpoint


def _now() -> str:
    """Returns the current UTC time as an ISO-formatted string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_row(run: RunConfig, outcome: RunOutcome, *, run_id: str, config_hash: str, user: str,
             started_at: str, splits: Splits, env: dict) -> dict:
    """Formats the final run summary as a dictionary for CSV saving."""
    cfg = run.to_dict()
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "config_hash": config_hash,
        "experiment": run.experiment, "user": user, "status": "completed",
        "started_at": started_at, "finished_at": _now(),
        **{k: v for k, v in cfg.items() if k != "experiment"},
        "split_hash": splits.split_hash, "pipeline_version": config.PIPELINE_VERSION,
        "n_train": outcome.n_train, "n_val": outcome.n_val, "n_test": outcome.n_test,
        "n_params_total": outcome.n_params_total, "n_params_trainable": outcome.n_params_trainable,
        "best_epoch": outcome.best_epoch, "best_val_acc": outcome.best_val_acc,
        "final_val_acc": outcome.final_val_acc,
        "test_acc": outcome.test.acc, "test_loss": outcome.test.loss,
        "test_macro_f1": outcome.test.macro_f1,
        "train_time_s": outcome.train_time_s, "eval_time_s": outcome.eval_time_s,
        "total_time_s": outcome.total_time_s,
        "sec_per_epoch": (outcome.train_time_s + outcome.eval_time_s) / max(1, run.epochs),
        **env,
        "test_confusion_json": json.dumps(outcome.test.confusion),
        "config_json": json.dumps({**cfg, "split_hash": splits.split_hash,
                                   "pipeline_version": config.PIPELINE_VERSION}, sort_keys=True),
    }


def _epoch_rows(run: RunConfig, outcome: RunOutcome, *, run_id: str, config_hash: str,
                user: str) -> list[dict]:
    """Formats per-epoch metrics as a list of dictionaries for CSV saving."""
    return [{
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "config_hash": config_hash,
        "experiment": run.experiment, "user": user, "model": run.model,
        "train_fraction": run.train_fraction, "seed": run.seed, "epoch": e.epoch,
        "train_loss": e.train_loss, "train_acc": e.train_acc, "val_loss": e.val_loss,
        "val_acc": e.val_acc, "val_macro_f1": e.val_macro_f1, "lr": e.lr,
        "train_time_s": e.train_time_s, "eval_time_s": e.eval_time_s,
    } for e in outcome.epochs]


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    """Execute every run of ``cfg`` that is not yet completed. Returns new run rows."""
    user = resolve_user(cfg.user)
    
    # Check for unknown models before starting
    unknown = [m for m in cfg.models if m not in MODELS]
    if unknown:
        raise KeyError(f"Unknown models {unknown}. Available: {sorted(MODELS)}")

    # Resolve paths and environment
    repo_root = find_repo_root()
    dirs = default_dirs(repo_root)
    data_dir = Path(cfg.data_dir or dirs["data_dir"])
    results_dir = Path(cfg.results_dir or dirs["results_dir"])
    splits_dir = Path(cfg.splits_dir or dirs["splits_dir"])
    checkpoints_dir = Path(cfg.checkpoints_dir or dirs["checkpoints_dir"])

    device = resolve_device(cfg.device)
    env = collect_env(device, repo_root)

    print("=" * 78)
    print(cfg.summary())
    print(f"device          : {env['device_name']} [{env['platform']}]")
    print(f"torch           : {env['torch_version']}  |  git {env['git_commit'][:10]}"
          f"{' (UNCOMMITTED CODE CHANGES!)' if env['git_dirty'] else ''}")
    print(f"results         : {results_dir}")
    print("=" * 78)

    # Load data and splits once for the entire grid
    data = load_dataset(cfg.dataset, data_dir)
    splits = get_splits(data, splits_dir)
    print(f"[data] {data.name}: {len(data)} images, {data.num_classes} classes | "
          f"train/val/test = {len(splits.train)}/{len(splits.val)}/{len(splits.test)} "
          f"| split_hash {splits.split_hash}")

    # Initialize results storage and identify already completed runs
    store = ResultStore(results_dir, user)
    done = store.completed_keys(cfg.skip_completed)
    runs = list(cfg.iter_runs())
    new_rows, failed, skipped = [], [], 0

    # Resolve init_from specs ("<model>@<fraction>[@<seed>]") to config hashes of local checkpoints,
    # so the hash of an initialised run names its exact source. Unresolvable runs are reported as failed.
    logged = load_runs(results_dir) if any(r.init_from for r in runs) else None
    resolved: list[RunConfig] = []
    for run in runs:
        if run.init_from is None:
            resolved.append(run)
            continue
        try:
            source = resolve_init_from(run.init_from, model=run.model, seed=run.seed,
                                       runs=logged, checkpoints_dir=checkpoints_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"SKIP {run.label()}: {exc}")
            failed.append(run.label())
            continue
        resolved.append(dataclasses.replace(run, init_from=source))
    runs = resolved

    # Iterate through all grid combinations
    for i, run in enumerate(runs, 1):
        config_hash = run.config_hash(splits.split_hash)
        
        # Skip if we already successfully ran this exact config
        if (run.experiment, config_hash) in done:
            skipped += 1
            print(f"[{i}/{len(runs)}] skip (already completed): {run.label()}")
            continue
        print(f"\n[{i}/{len(runs)}] {run.label()}  [{config_hash}]")
        run_id, started_at = uuid.uuid4().hex[:16], _now()
        try:
            init_state = load_checkpoint(checkpoints_dir / f"{run.init_from}.pt").state_dict if run.init_from else None
            outcome = execute_run(run, data, splits, device, cfg.eval_batch_size, cfg.progress,
                                  init_state=init_state)
            save_checkpoint(
                Checkpoint(
                    state_dict=outcome.best_state,
                    config_hash=config_hash,
                    run_id=run_id,
                    model=run.model,
                    dataset=run.dataset,
                    num_classes=data.num_classes,
                    image_size=run.image_size,
                    experiment=run.experiment,
                    pipeline_version=config.PIPELINE_VERSION,
                    split_hash=splits.split_hash,
                ),
                checkpoints_dir / f"{config_hash}.pt",
            )
            outcome.best_state = None
        except KeyboardInterrupt:
            print("Interrupted - completed runs are saved; re-run to resume.")
            raise
        except Exception:
            if not cfg.continue_on_error:
                raise
            traceback.print_exc()
            failed.append(run.label())
            continue

        row = _run_row(run, outcome, run_id=run_id, config_hash=config_hash, user=user,
                       started_at=started_at, splits=splits, env=env)
        store.append(row, _epoch_rows(run, outcome, run_id=run_id, config_hash=config_hash, user=user))
        new_rows.append(row)
        print(f"  -> test acc {outcome.test.acc:.4f} | macro-F1 {outcome.test.macro_f1:.4f} | "
              f"best epoch {outcome.best_epoch} | n_train {outcome.n_train} | "
              f"{outcome.total_time_s:.1f}s")

    print("\n" + "=" * 78)
    print(f"done: {len(new_rows)} new, {skipped} skipped, {len(failed)} failed")
    for f in failed:
        print(f"  FAILED: {f}")
    if new_rows:
        print(f"saved to {store.runs_path} and {store.epochs_path}")
    print("=" * 78)
    return pd.DataFrame(new_rows)
