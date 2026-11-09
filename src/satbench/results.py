"""Append-only CSV result storage.

Layout (one file per user -> no git merge conflicts between group members):

    results/
      runs/<user>.v<SCHEMA>.csv          one row per *completed* run (summary + provenance)
      epochs/<user>.v<SCHEMA>.csv        one row per epoch of every completed run
      transfer/<user>.v<TRANSFER>.csv    one row per completed transfer eval

Rows are only written after a run finishes successfully. Epoch rows are written
first and the run row last, so a row in runs/ is the "commit marker": loaders
ignore epoch rows whose run_id has no run row. Existing rows are never edited.
If you change the columns, bump SCHEMA_VERSION (new files are started
automatically; old ones stay readable). Transfer eval uses its own
TRANSFER_SCHEMA_VERSION so training CSVs stay untouched.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

SCHEMA_VERSION = 2  # v2: + init_from column

RUN_COLUMNS = [
    # identity
    "schema_version", "run_id", "config_hash", "experiment", "user", "status",
    "started_at", "finished_at",
    # run config
    "dataset", "model", "train_fraction", "seed", "epochs", "batch_size", "lr",
    "weight_decay", "image_size", "select_best_on_val",
    "deterministic", "eval_limit", "split_hash", "pipeline_version", "init_from",
    # outcome
    "n_train", "n_val", "n_test", "n_params_total", "n_params_trainable",
    "best_epoch", "best_val_acc", "final_val_acc",
    "test_acc", "test_loss", "test_macro_f1",
    "train_time_s", "eval_time_s", "total_time_s", "sec_per_epoch",
    # provenance
    "device", "device_name", "platform", "hostname", "os", "python_version",
    "torch_version", "torchvision_version", "satbench_version", "git_commit", "git_dirty",
    # free-form
    "test_confusion_json", "config_json",
]

EPOCH_COLUMNS = [
    "schema_version", "run_id", "config_hash", "experiment", "user", "model",
    "train_fraction", "seed", "epoch", "train_loss", "train_acc", "val_loss",
    "val_acc", "val_macro_f1", "lr", "train_time_s", "eval_time_s",
]

# Separate schema for cross-dataset transfer eval (does not bump SCHEMA_VERSION).
TRANSFER_SCHEMA_VERSION = 1

TRANSFER_COLUMNS = [
    "schema_version", "transfer_id", "config_hash", "run_id", "user", "status",
    "started_at", "finished_at",
    "source_dataset", "target_dataset", "model", "experiment",
    "pipeline_version", "split_hash", "image_size", "num_classes",
    "train_fraction", "seed", "source_test_acc",
    "n_eval", "transfer_acc", "transfer_loss", "transfer_macro_f1",
    "transfer_confusion_json", "checkpoint_path",
    "device", "device_name", "platform", "git_commit", "git_dirty",
]


def _append_rows(path: Path, columns: list[str], rows: Iterable[dict]) -> int:
    """Appends rows to a CSV file safely, creating the file and header if missing."""
    rows = list(rows)
    for r in rows:
        if set(r) != set(columns):
            missing, extra = set(columns) - set(r), set(r) - set(columns)
            raise ValueError(f"Row/schema mismatch for {path.name}: missing={missing} extra={extra}")
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with open(path, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        if header != columns:
            raise RuntimeError(f"{path} has a different header than SCHEMA_VERSION={SCHEMA_VERSION}. "
                               "Bump SCHEMA_VERSION when changing columns.")
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        if not exists:
            w.writeheader()
        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())
    return len(rows)


class ResultStore:
    def __init__(self, results_dir: str | Path, user: str):
        self.dir = Path(results_dir)
        self.user = user
        self.runs_path = self.dir / "runs" / f"{user}.v{SCHEMA_VERSION}.csv"
        self.epochs_path = self.dir / "epochs" / f"{user}.v{SCHEMA_VERSION}.csv"

    def completed_keys(self, scope: str = "user") -> set[tuple[str, str]]:
        """{(experiment, config_hash)} already completed (by this user / anyone).

        Runs made with uncommitted code changes (git_dirty) do not count: their
        config_hash does not identify the code that produced them."""
        if scope == "none":
            return set()
        # every schema version of the user's file counts, so a schema bump does not re-run old work
        pattern = f"{self.user}.v*.csv" if scope == "user" else "*.csv"
        files = sorted((self.dir / "runs").glob(pattern))
        keys: set[tuple[str, str]] = set()
        for f in files:
            if f.exists():
                with open(f, newline="", encoding="utf-8") as fh:
                    keys |= {(r["experiment"], r["config_hash"]) for r in csv.DictReader(fh)
                             if r["git_dirty"].lower() != "true"}
        return keys

    def append(self, run_row: dict, epoch_rows: list[dict]) -> None:
        """Appends epoch results, then the final run result as a commit marker."""
        _append_rows(self.epochs_path, EPOCH_COLUMNS, epoch_rows)
        _append_rows(self.runs_path, RUN_COLUMNS, [run_row])


class TransferStore:
    """Append-only transfer-eval CSV per user: ``results/transfer/<user>.vN.csv``."""

    def __init__(self, results_dir: str | Path, user: str):
        self.dir = Path(results_dir)
        self.user = user
        self.path = self.dir / "transfer" / f"{user}.v{TRANSFER_SCHEMA_VERSION}.csv"

    def completed_keys(self) -> set[tuple[str, str]]:
        """{(config_hash, target_dataset)} already completed by this user."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return set()
        with open(self.path, newline="", encoding="utf-8") as fh:
            return {(r["config_hash"], r["target_dataset"]) for r in csv.DictReader(fh)}

    def append(self, row: dict) -> None:
        _append_rows(self.path, TRANSFER_COLUMNS, [row])


# Loading
def _read_all(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    frames = [pd.read_csv(f) for f in files if f.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_runs(results_dir: str | Path) -> pd.DataFrame:
    """Loads all completed run summaries across all users in the results directory."""
    df = _read_all(Path(results_dir) / "runs")
    if df.empty:
        return pd.DataFrame(columns=RUN_COLUMNS)
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df.drop_duplicates("run_id").reset_index(drop=True)


def load_epochs(results_dir: str | Path, runs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Loads all per-epoch metrics for completed runs across all users."""
    df = _read_all(Path(results_dir) / "epochs")
    if df.empty:
        return pd.DataFrame(columns=EPOCH_COLUMNS)
    runs = load_runs(results_dir) if runs is None else runs
    df = df[df["run_id"].isin(runs["run_id"])]  # drop rows of unfinished runs
    return df.drop_duplicates(["run_id", "epoch"]).reset_index(drop=True)


def load_transfers(results_dir: str | Path) -> pd.DataFrame:
    """Loads all transfer-eval rows across all users."""
    df = _read_all(Path(results_dir) / "transfer")
    if df.empty:
        return pd.DataFrame(columns=TRANSFER_COLUMNS)
    for col in ("started_at", "finished_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df.drop_duplicates("transfer_id").reset_index(drop=True)
