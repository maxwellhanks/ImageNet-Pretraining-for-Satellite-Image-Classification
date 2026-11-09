"""Fast offline tests on the synthetic 'fake' dataset (CPU, ~1 minute).

    uv run pytest
"""
import numpy as np
import pandas as pd
import pytest

from satbench import ExperimentConfig, load_epochs, load_runs, run_experiment
from satbench.data import get_splits, label_fraction_subset, load_dataset
from satbench.checkpoint import load_checkpoint
from satbench.results import SCHEMA_VERSION


def _cfg(tmp_path, results="results", **kw):
    base = dict(experiment="test", user="tester", dataset="fake",
                models=["linear", "resnet50_random"], train_fractions=[0.5], seeds=[0],
                epochs=2, batch_size=16, image_size=32, device="cpu", progress=False,
                data_dir=str(tmp_path / "data"), results_dir=str(tmp_path / results),
                splits_dir=str(tmp_path / "splits"), checkpoints_dir=str(tmp_path / "checkpoints"), continue_on_error=False)
    base.update(kw)
    return ExperimentConfig(**base)


def test_subsets_are_stratified_and_nested(tmp_path):
    ds = load_dataset("fake", tmp_path)
    sp = get_splits(ds, tmp_path)
    labels = ds.labels.numpy()
    small = label_fraction_subset(sp.train, labels, 0.1, seed=3)
    large = label_fraction_subset(sp.train, labels, 0.5, seed=3)
    assert set(small) <= set(large)                       # nested
    assert len(np.unique(labels[small])) == ds.num_classes  # every class present
    assert not set(sp.train) & set(sp.test)
    other = label_fraction_subset(sp.train, labels, 0.5, seed=4)
    assert set(other) != set(large)                       # seed changes the subset


def test_runs_are_logged_skipped_and_reproducible(tmp_path, monkeypatch):
    # dirty runs are never skipped, so pin a clean git state independent of the working tree
    monkeypatch.setattr("satbench.env.git_info", lambda repo_root: ("test", False))
    first = run_experiment(_cfg(tmp_path))
    assert len(first) == 2
    runs = load_runs(tmp_path / "results")
    epochs = load_epochs(tmp_path / "results")
    assert len(runs) == 2 and len(epochs) == 4
    assert (tmp_path / f"results/runs/tester.v{SCHEMA_VERSION}.csv").exists()

    # one checkpoint per completed run, named by config_hash
    ckpts = list((tmp_path / "checkpoints").glob("*.pt"))
    assert len(ckpts) == 2
    for p in ckpts:
        ckpt = load_checkpoint(p)
        assert ckpt.dataset == "fake"
        assert ckpt.model in {"linear", "resnet50_random"}
        assert ckpt.state_dict
        assert p.name == f"{ckpt.config_hash}.pt"

    # same config again -> everything skipped, nothing appended
    assert run_experiment(_cfg(tmp_path)).empty
    assert len(load_runs(tmp_path / "results")) == 2
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 2

    # independent re-run into another folder -> bit-identical metrics on CPU
    run_experiment(_cfg(tmp_path, results="results2", user="other"))
    a = runs.sort_values("model").reset_index(drop=True)
    b = load_runs(tmp_path / "results2").sort_values("model").reset_index(drop=True)
    pd.testing.assert_series_equal(a["config_hash"], b["config_hash"])
    pd.testing.assert_series_equal(a["test_acc"], b["test_acc"])
    pd.testing.assert_series_equal(a["test_loss"], b["test_loss"])


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        _cfg(tmp_path, train_fractions=[-0.1])
    with pytest.raises(ValueError):
        run_experiment(_cfg(tmp_path, user="Bad Name!"))
    cfg = _cfg(tmp_path, model_overrides={"linear": {"lr": "1e-3"}})
    lr = {r.model: r.lr for r in cfg.iter_runs()}
    assert lr == {"linear": 1e-3, "resnet50_random": 1e-4}


def test_init_from_loads_backbone_and_changes_hash(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr("satbench.env.git_info", lambda repo_root: ("test", False))
    # 1. source run: a fine-tuned ResNet on the fake dataset
    src = run_experiment(_cfg(tmp_path, models=["resnet50_random"]))
    src_hash = src.iloc[0]["config_hash"]

    # 2. probe runs of the same backbone family: own init vs. initialised from the source run
    cfg = _cfg(tmp_path, experiment="probe", models=["linear", "resnet50_random_linear_probe"],
               init_from=[None, "@0.5"])
    assert cfg.n_runs() == 3  # 'linear' has no backbone -> only its None entry runs
    out = run_experiment(cfg)
    assert len(out) == 3
    probe = out[out["model"] == "resnet50_random_linear_probe"].set_index("init_from", drop=False)
    assert set(probe.index.fillna("none")) == {"none", src_hash}       # spec resolved to the source hash
    assert probe.loc[src_hash, "config_hash"] != probe[probe["init_from"].isna()].iloc[0]["config_hash"]

    # 3. the initialised probe carries the source backbone, but a fresh head
    source = load_checkpoint(tmp_path / "checkpoints" / f"{src_hash}.pt").state_dict
    adapted = load_checkpoint(tmp_path / "checkpoints" / f"{probe.loc[src_hash, 'config_hash']}.pt").state_dict
    assert torch.equal(source["conv1.weight"], adapted["conv1.weight"])
    assert not torch.equal(source["fc.weight"], adapted["fc.weight"])

    # 4. a spec without a matching checkpoint is reported, not silently trained from scratch
    missing = run_experiment(_cfg(tmp_path, experiment="probe2", models=["resnet50_random_linear_probe"],
                                  init_from=["@0.25"], continue_on_error=True))
    assert missing.empty
