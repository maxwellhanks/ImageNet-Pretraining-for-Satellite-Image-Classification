"""Experiment configuration.

Two levels:

* ``ExperimentConfig`` – what you write in a YAML file / notebook: a *grid* of
  models x train fractions x seeds plus shared hyperparameters and execution
  settings (device, paths, ...).
* ``RunConfig`` – one cell of that grid. Contains exactly the fields that
  determine the outcome of a run, so its hash identifies "the same computation"
  across users and machines.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

# Bump when training/evaluation logic changes in a way that changes results.
# It is part of the config hash, so old and new results are never mixed up.
PIPELINE_VERSION = "2"

# Hyperparameters that may be overridden per model via ``model_overrides``.
OVERRIDABLE = ("epochs", "batch_size", "lr", "weight_decay", "image_size")

SKIP_MODES = ("user", "any", "none")

_USER_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,31}")


def resolve_user(user: str | None) -> str:
    """Validates the username (used in file names and to trace results)."""
    user = (user or os.environ.get("SATBENCH_USER", "")).strip().lower()
    if not _USER_RE.fullmatch(user):
        raise ValueError(
            "invalid username, see config.py _USER_RE for regex definition"
        )
    return user


def _coerce_hparams(hp: dict[str, Any]) -> dict[str, Any]:
    # YAML 1.1 reads `1e-4` (no dot) as a *string*; coerce defensively.
    return {
        "epochs": int(hp["epochs"]),
        "batch_size": int(hp["batch_size"]),
        "lr": float(hp["lr"]),
        "weight_decay": float(hp["weight_decay"]),
        "image_size": int(hp["image_size"]),
    }


@dataclass(frozen=True)
class RunConfig:
    """Everything that determines the outcome of a single training run."""

    experiment: str
    dataset: str
    model: str
    train_fraction: float
    seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    image_size: int
    select_best_on_val: bool
    deterministic: bool
    eval_limit: int | None
    # config_hash of a finished run whose backbone weights initialise this run
    # (head re-initialised). None = the model's own init (random / ImageNet / SSL4EO).
    # Written as a spec in YAML (see checkpoint.resolve_init_from), resolved by the pipeline.
    init_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self, split_hash: str) -> str:
        """Identifier of the computation. Excludes the experiment *name* so the
        same run in two experiments (or by two users) can be compared."""
        d = self.to_dict()
        d.pop("experiment")
        if d.get("init_from") is None:
            d.pop("init_from", None)  # keeps hashes of runs made before this field existed
        d["split_hash"] = split_hash
        d["pipeline_version"] = PIPELINE_VERSION
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def label(self) -> str:
        init = f" | init_from={self.init_from}" if self.init_from else ""
        return f"{self.model} | frac={self.train_fraction:g} | seed={self.seed}{init}"


@dataclass
class ExperimentConfig:
    experiment: str
    user: str | None = None

    # The grid
    dataset: str = "eurosat_rgb"
    models: list[str] = field(
        default_factory=lambda: ["linear", "resnet50_random", "resnet50_imagenet"]
    )
    train_fractions: list[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
    )
    seeds: list[int] = field(default_factory=lambda: [0])
    # Backbone initialisations, one run per entry. None = the model's own init; otherwise a
    # 12-hex config_hash or a spec "<model>@<fraction>[@<seed>]" naming a finished run.
    # "@<fraction>" uses the run's own model without "_linear_probe" and its own seed, e.g.
    # for resnet50_imagenet_linear_probe, "@0.1" = resnet50_imagenet fine-tuned on 10 % (same seed).
    # Entries other than None are skipped for models without a backbone ("linear").
    init_from: list[str | None] = field(default_factory=lambda: [None])

    # Shared hyperparameters (part of every RunConfig)
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-2
    image_size: int = 224  # input size for ResNets (EuroSAT is 64x64 natively)
    select_best_on_val: bool = True  # evaluate the best-val-epoch weights on test
    deterministic: bool = True
    eval_limit: int | None = None  # DEBUG ONLY: subsample val/test to this size
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Execution settings (do NOT affect results)
    device: str = "auto"  # auto | cuda | mps | cpu
    eval_batch_size: int = 128
    data_dir: str | None = None
    results_dir: str | None = None
    splits_dir: str | None = None
    checkpoints_dir: str | None = None
    skip_completed: str = "user"  # user | any | none
    continue_on_error: bool = True
    progress: bool = True

    def __post_init__(self) -> None:
        self.models = [str(m) for m in _as_list(self.models)]
        self.train_fractions = [float(f) for f in _as_list(self.train_fractions)]
        self.seeds = [int(s) for s in _as_list(self.seeds)]
        self.init_from = [None if x in (None, "", "none", "null") else str(x) for x in _as_list(self.init_from)]
        if not self.init_from:
            raise ValueError("init_from must be non-empty (use [null] for no checkpoint init)")
        for k, v in _coerce_hparams({k: getattr(self, k) for k in OVERRIDABLE}).items():
            setattr(self, k, v)
        if self.eval_limit is not None:
            self.eval_limit = int(self.eval_limit)

        if not self.experiment or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.experiment):
            raise ValueError("experiment must be a non-empty name of [A-Za-z0-9_.-]")
        if not self.models or not self.train_fractions or not self.seeds:
            raise ValueError("models, train_fractions and seeds must be non-empty")
        if any(not 0 <= f <= 1 for f in self.train_fractions):
            raise ValueError("train_fractions must lie in [0, 1]")
        if self.skip_completed not in SKIP_MODES:
            raise ValueError(f"skip_completed must be one of {SKIP_MODES}")
        for model, ov in self.model_overrides.items():
            bad = set(ov) - set(OVERRIDABLE)
            if bad:
                raise ValueError(f"model_overrides[{model}] has non-overridable keys {bad}")

    # Construction
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "ExperimentConfig":
        """Loads a YAML config and applies keyword overrides."""
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = (set(raw) | set(overrides)) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**raw)

    def replace(self, **overrides: Any) -> "ExperimentConfig":
        return dataclasses.replace(self, **{k: v for k, v in overrides.items() if v is not None})

    # Expansion
    def iter_runs(self) -> Iterator[RunConfig]:
        """Expands the grid into individual run configurations.
        
        Uses seed-major order: a full set of results for seed 0 executes first.
        """
        for seed in self.seeds:
            for fraction in self.train_fractions:
                for model in self.models:
                    hp = {k: getattr(self, k) for k in OVERRIDABLE}
                    hp.update(self.model_overrides.get(model, {}))
                    for init_from in self.init_from:
                        if init_from is not None and not has_backbone(model):
                            continue
                        yield RunConfig(
                            experiment=self.experiment,
                            dataset=self.dataset,
                            model=model,
                            train_fraction=fraction,
                            seed=seed,
                            select_best_on_val=bool(self.select_best_on_val),
                            deterministic=bool(self.deterministic),
                            eval_limit=self.eval_limit,
                            init_from=init_from,  # spec; resolved to a config_hash by the pipeline
                            **_coerce_hparams(hp),
                        )

    def n_runs(self) -> int:
        return sum(1 for _ in self.iter_runs())

    def summary(self) -> str:
        lines = [
            f"experiment      : {self.experiment}",
            f"user            : {self.user}",
            f"dataset         : {self.dataset}",
            f"models          : {self.models}",
            f"train_fractions : {self.train_fractions}",
            f"seeds           : {self.seeds}",
            f"epochs / batch  : {self.epochs} / {self.batch_size}",
            f"optim           : AdamW lr={self.lr:g} wd={self.weight_decay:g}",
            f"image_size      : {self.image_size}",
            f"model_overrides : {self.model_overrides or '-'}",
            f"init_from       : {self.init_from}",
            f"total runs      : {self.n_runs()}",
        ]
        if self.eval_limit:
            lines.append(f"eval_limit      : {self.eval_limit}  (DEBUG: val/test subsampled)")
        return "\n".join(lines)


def _as_list(x: Any) -> list[Any]:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def has_backbone(model: str) -> bool:
    """Whether a model name denotes a network with a backbone that can be initialised from a checkpoint."""
    return model != "linear"


def backbone_counterpart(model: str) -> str:
    """The fully trainable model of the same backbone family (``*_linear_probe`` -> fine-tune)."""
    return model.removesuffix("_linear_probe")
