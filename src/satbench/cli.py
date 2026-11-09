from __future__ import annotations

import argparse
import sys

from .config import ExperimentConfig


def _cmd_run(a: argparse.Namespace) -> None:
    from .pipeline import run_experiment

    cfg = ExperimentConfig.from_yaml(
        a.config, user=a.user, experiment=a.experiment, epochs=a.epochs, seeds=a.seeds,
        models=a.models, train_fractions=a.fractions, batch_size=a.batch_size, lr=a.lr,
        init_from=a.init_from,
        device=a.device,
        skip_completed="none" if a.no_skip else None,
        progress=False if a.no_progress else None,
    )
    run_experiment(cfg)


def _cmd_prepare(a: argparse.Namespace) -> None:
    from .data import get_splits, load_dataset
    from .env import default_dirs, find_repo_root

    dirs = default_dirs(find_repo_root())
    ds = load_dataset(a.dataset, dirs["data_dir"])
    print(f"{ds.name}: {len(ds)} images, classes={ds.classes}")
    if ds.name == "eurosat_rgb":
        sp = get_splits(ds, dirs["splits_dir"])
        print(f"train/val/test = {len(sp.train)}/{len(sp.val)}/{len(sp.test)}  split_hash={sp.split_hash}")


def _cmd_models(_: argparse.Namespace) -> None:
    from .models import MODELS
    print("\n".join(sorted(MODELS)))


def _cmd_list_checkpoints(_: argparse.Namespace) -> None:
    from .checkpoint import list_checkpoints
    from .env import default_dirs, find_repo_root

    dirs = default_dirs(find_repo_root())
    df = list_checkpoints(dirs["checkpoints_dir"], dirs["results_dir"])
    if df.empty:
        print(f"no checkpoints in {dirs['checkpoints_dir']}")
        return
    cols = [c for c in (
        "config_hash", "model", "train_fraction", "seed", "pipeline_version",
        "test_acc", "source_dataset", "experiment", "user", "error",
    ) if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\n{len(df)} checkpoint(s) in {dirs['checkpoints_dir']}")


def _cmd_transfer(a: argparse.Namespace) -> None:
    from .checkpoint import resolve_checkpoint_paths
    from .env import default_dirs, find_repo_root
    from .transfer import run_transfer_evals
    from pathlib import Path

    dirs = default_dirs(find_repo_root())
    paths = resolve_checkpoint_paths(
        dirs["checkpoints_dir"],
        results_dir=dirs["results_dir"],
        checkpoint=Path(a.checkpoint) if a.checkpoint else None,
        config_hash=a.config_hash,
        model=a.model,
        train_fraction=a.fraction,
        seed=a.seed,
        experiment=a.experiment,
        source_dataset=a.source,
        all_checkpoints=a.all,
    )
    run_transfer_evals(
        paths,
        data_dir=dirs["data_dir"],
        results_dir=dirs["results_dir"],
        target_dataset=a.target,
        user=a.user,
        device=a.device,
        batch_size=a.batch_size,
        skip_completed=not a.no_skip,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="satbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an experiment grid from a YAML config")
    r.add_argument("--config", required=True)
    r.add_argument("--user", help="your username (or set SATBENCH_USER)")
    r.add_argument("--experiment")
    r.add_argument("--epochs", type=int)
    r.add_argument("--seeds", type=int, nargs="+")
    r.add_argument("--models", nargs="+")
    r.add_argument("--fractions", type=float, nargs="+")
    r.add_argument("--batch-size", type=int)
    r.add_argument("--lr", type=float)
    r.add_argument("--init-from", dest="init_from", nargs="+",
                   help="backbone inits: 'none', a config_hash, or '<model>@<fraction>[@<seed>]' (e.g. '@0.1')")
    r.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    r.add_argument("--no-skip", action="store_true", help="re-run even if already completed")
    r.add_argument("--no-progress", action="store_true")
    r.set_defaults(fn=_cmd_run)

    d = sub.add_parser("prepare-data", help="download + cache a dataset, create its split file")
    d.add_argument("--dataset", default="eurosat_rgb")
    d.set_defaults(fn=_cmd_prepare)

    sub.add_parser("list-models").set_defaults(fn=_cmd_models)

    sub.add_parser(
        "list-checkpoints",
        help="list local checkpoints joined with training-run metadata",
    ).set_defaults(fn=_cmd_list_checkpoints)

    t = sub.add_parser(
        "transfer-eval",
        help="evaluate checkpointed models on another dataset (e.g. uc_merced_rgb)",
    )
    t.add_argument("--target", default="uc_merced_rgb")
    t.add_argument("--user", help="your username (or set SATBENCH_USER)")
    t.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--no-skip", action="store_true", help="re-eval even if already completed")
    # checkpoint selection (no need to type the hash by hand)
    t.add_argument("--checkpoint", help="explicit path to a .pt file")
    t.add_argument("--config-hash", help="select checkpoints/<hash>.pt")
    # filter checkpoint from results
    t.add_argument("--model", help="filter by model name via results CSV")
    t.add_argument("--fraction", type=float, help="filter by train_fraction")
    t.add_argument("--seed", type=int, help="filter by seed")
    t.add_argument("--experiment", help="filter by experiment name")
    t.add_argument("--source", help="filter by source dataset (e.g. eurosat_rgb)")
    t.add_argument("--all", action="store_true")
    t.set_defaults(fn=_cmd_transfer)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
