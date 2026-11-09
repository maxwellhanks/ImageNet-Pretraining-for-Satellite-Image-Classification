"""Offline transfer-eval tests on synthetic datasets (CPU)."""
from __future__ import annotations

import pytest

from satbench import load_transfers, run_experiment
from satbench.data import ImageDataset, load_dataset, register_dataset
from satbench.checkpoint import resolve_checkpoint_paths
from satbench.transfer import run_transfer_evals


def _cfg(tmp_path, results="results", **kw):
    from satbench import ExperimentConfig
    base = dict(
        experiment="test", user="tester", dataset="fake",
        models=["linear"], train_fractions=[0.5], seeds=[0],
        epochs=2, batch_size=16, image_size=32, device="cpu", progress=False,
        data_dir=str(tmp_path / "data"), results_dir=str(tmp_path / results),
        splits_dir=str(tmp_path / "splits"),
        checkpoints_dir=str(tmp_path / "checkpoints"),
        continue_on_error=False,
    )
    base.update(kw)
    return ExperimentConfig(**base)


def _register_fake_shift() -> None:
    """Second 4-class synthetic target so transfer_eval has something to hit."""
    @register_dataset("fake_shift")
    def _load(data_dir):
        base = load_dataset("fake", data_dir)
        return ImageDataset(
            name="fake_shift",
            images=base.images.flip(-1).contiguous(),  # trivial shift
            labels=base.labels,
            classes=list(base.classes),
            keys=[f"shift/{k}" for k in base.keys],
        )


def test_resolve_and_transfer_by_filters(tmp_path):
    _register_fake_shift()
    run_experiment(_cfg(tmp_path))
    ckpt_dir = tmp_path / "checkpoints"
    results_dir = tmp_path / "results"

    paths = resolve_checkpoint_paths(
        ckpt_dir, results_dir=results_dir,
        model="linear", train_fraction=0.5, seed=0,
    )
    assert len(paths) == 1
    assert paths[0].is_file()

    # check that explicit hash / path work
    h = paths[0].stem
    assert resolve_checkpoint_paths(ckpt_dir, config_hash=h) == paths
    assert resolve_checkpoint_paths(ckpt_dir, checkpoint=paths[0]) == paths

    out = run_transfer_evals(
        paths, target_dataset="fake_shift", user="tester", device="cpu",
        data_dir=tmp_path / "data", results_dir=results_dir,
    )
    assert len(out) == 1
    assert out.iloc[0]["transfer_acc"] >= 0.0
    transfers = load_transfers(results_dir)
    assert len(transfers) == 1
    assert transfers.iloc[0]["target_dataset"] == "fake_shift"
    assert transfers.iloc[0]["source_dataset"] == "fake"

    # skip on second call
    again = run_transfer_evals(
        paths, target_dataset="fake_shift", user="tester", device="cpu",
        data_dir=tmp_path / "data", results_dir=results_dir,
    )
    assert again.empty
    assert len(load_transfers(results_dir)) == 1


def test_resolve_requires_selector(tmp_path):
    with pytest.raises(ValueError, match="specify"):
        resolve_checkpoint_paths(tmp_path / "checkpoints")
