# STAT7007 G3

Pipeline for comparing pretraining strategies (random init, ImageNet, SSL4EO-S12) and architectures (ResNet-50, ViT-B/16) for land-cover classification on EuroSAT RGB.

## Prerequisites

You need to install two core tools to get started:

### 1. Install `uv` (Package Manager)
```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install `just` (Command Runner)
```bash
uv tool install just
```

## Getting Started

### 1. Initialize the Environment
Clone the repository and run the following in the project root:

```bash
git clone https://github.com/maxwellhanks/ImageNet-Pretraining-for-Satellite-Image-Classification.git
cd ImageNet-Pretraining-for-Satellite-Image-Classification
just install                     # syncs all dependencies into a local .venv folder
uv run nbstripout --install      # strips notebook outputs on commit to avoid merge conflicts
uv run pytest                    # offline sanity check that setup works with synthetic data
```

> **Windows + NVIDIA GPU:** PyPI only has CPU torch wheels for Windows. Uncomment the
> `[tool.uv.sources]` block at the bottom of `pyproject.toml`, then run `uv lock && just install`.

### 2. Configure VS Code
- Open a `.ipynb` notebook.
- Click Select Kernel in the top-right corner.
- Choose Python Environments...
- Select the interpreter located at `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Mac/Linux).

## Available Commands (via `just`)

| Command | Usage | Description |
| :--- | :--- | :--- |
| **default** | `just` | Opens the interactive command picker menu. |
| **install** | `just install` | Syncs all libraries into `.venv`. |
| **list-models** | `just list-models` | Prints registered model names. |
| **list-checkpoints** | `just list-checkpoints` | Lists local `checkpoints/*.pt` joined with training-run metadata (model, fraction, seed, test_acc). |
| **prepare** | `just prepare <dataset>` | Download + cache a dataset (`eurosat_rgb`, `uc_merced_rgb`, or `fake`). EuroSAT also writes `splits/eurosat_rgb_split.csv`. |
| **prepare-all** | `just prepare-all` | Runs `prepare` for eurosat_rgb, uc_merced_rgb, and fake. |
| **smoke** | `just smoke <user>` | Short end-to-end run (`configs/smoke_test.yaml`) writing to `results/runs/<user>.v2.csv`. |
| **transfer-all** | `just transfer-all <user> [target]` | Transfer-eval every local checkpoint onto `target` (default `uc_merced_rgb`). |
| **transfer** | `just transfer <user> <model> <fraction> <seed> [target]` | Transfer-eval by training filters (no hash needed). |
| **test** | `just test` | Offline `pytest` on the synthetic `fake` dataset. |
| **topdf** | `just topdf <file>` | Converts a notebook to PDF. |
| **venv** | `just venv` | Creates a fresh virtualenv (default `.venv`). |
| **clean-data** | `just clean-data` | Deletes `data/` caches and downloads (re-run `prepare` after). |
| **clean** | `just clean` | Removes `.venv` and clears the `uv` cache. |



### Running Notebooks

Open `notebooks/01_run_experiments.ipynb` locally (`uv run jupyter lab` or VS Code) or in Colab. Both notebooks locate the repository root automatically, run directly within the codebase, log results to `results/runs/<user>.v2.csv` and `results/epochs/<user>.v2.csv`, and can be safely interrupted and resumed. Set `USER` (or `SATBENCH_USER`) to your own name before a real run so new rows are not appended to Maxwell's, Manuel's, or Felix's files.

## Repository layout

```
configs/                      experiment configurations in YAML format
src/satbench/
  config.py                   ExperimentConfig (grid) -> RunConfig (one run) + config hash
  seeding.py                  all randomness for reproducability: seed_everything, derive_seed
  data.py                     dataset registry, uint8 cache, fixed splits, label subsets
  models.py                   model registry: network + its preprocessing
  train.py                    train / validate / test for one run
  results.py                  append-only CSV storage, loading, merging
  pipeline.py                 runs a grid, skips completed runs, logs provenance
  checkpoint.py               save / load best weights for transfer eval
  transfer.py                 evaluate checkpoints on another dataset
  env.py                      device, platform, git commit, versions
  cli.py                      `satbench` command
notebooks/
  01_run_experiments.ipynb    run locally or on Colab
  02_visualize_results.ipynb  plots + tables from all results
splits/eurosat_rgb_split.csv  fixed train/val/test assignment (commit it!)
results/runs/<user>.v2.csv    one row per completed run (current training schema)
results/epochs/<user>.v2.csv  one row per epoch of each completed run
results/transfer/<user>.v1.csv  one row per transfer eval (transfer schema)
checkpoints/<config_hash>.pt  best weights (gitignored)
tests/                        offline tests on a synthetic dataset
```

Checked-in results are one file per member — the latest copy only. Earlier snapshots under `results/epochs`, `results/runs`, and `results/transfer` were removed.

| Member | Runs and epochs | Transfer |
| --- | --- | --- |
| maxwell | `maxwell.v4final.csv` | `maxwell.v1.csv` |
| manuel | `manuel.v2.csv` | `manuel.v1.csv` |
| felix | `felix.v2.csv` | none yet |

`maxwell.v4final.csv` is the consolidated export (it replaces `maxwell.v1`–`v3`). It has no `init_from` column, so further runs by the username `maxwell` append to `maxwell.v2.csv`. Loaders read every CSV in the folder, so both are included. Manuel's and Felix's earlier schema-v1 rows were copied into their `v2` files with `init_from` left blank; those `v2` files are what the current code appends to. The pipeline-v1 runs that were logged as user `max` are not included: the same cells were rerun after the SSL4EO preprocessing fix, and those reruns are in `maxwell.v4final.csv`.



## Running experiments

```bash
# before the first run on any machine, download the datasets (and, on first use, the model weights)
uv run satbench prepare-data
uv run satbench run --config configs/smoke_test.yaml --user <you>

# the real thing; split the sweep between members, e.g. one seed each
uv run satbench run --config configs/baseline_label_efficiency.yaml --user <you> --seeds 0
uv run satbench run --config configs/vit_label_efficiency.yaml --user <you> --seeds 0
```

Replace `<you>` with your own username (`[a-z0-9][a-z0-9_-]{1,31}`, stored lowercase). `export SATBENCH_USER=<you>` saves typing `--user`. Leave `maxwell`, `manuel`, and `felix` for those members; a new name gets its own CSV.

Useful overrides: `--epochs`, `--seeds`, `--models`, `--fractions`, `--device`, `--no-skip`. The notebook `01_run_experiments.ipynb` does the same thing.

Completed runs are **skipped** automatically, so an interrupted sweep resumes where it stopped. Then commit your results and push them to this repository (`origin` after the clone above). `data/` and `checkpoints/` are gitignored.

```bash
git pull && git add results/ && git commit -m "results: baseline (<you>)" && git push
```

Plots: `notebooks/02_visualize_results.ipynb` reads every member's results and saves figures to `figures/` as PNG and PDF.

After each completed run, weights are saved to `checkpoints/<config_hash>.pt`
(gitignored). Use these for later transfer eval (e.g. UC Merced). Override the
folder with `checkpoints_dir` in the experiment config if needed.

### Few-shot adaptation from checkpoints (`init_from`)

A run can start from the **backbone of a finished run** (head re-initialised). `init_from` is a
grid axis of the experiment config: one run per entry, `null` = the model's own init.

```yaml
# configs/ucm_fewshot_linear_probe.yaml: frozen probes on UC Merced, unmodified backbone vs.
# the same backbone after fine-tuning on 1 % / 10 % / 100 % EuroSAT (same seed)
dataset: uc_merced_rgb
models: [resnet50_imagenet_linear_probe, resnet50_ssl4eo_linear_probe]
init_from: [null, "@0.01", "@0.1", "@1.0"]
```

An entry is a 12-hex `config_hash`, or `<model>@<fraction>[@<seed>]`; an empty model
(`"@0.1"`) means the fine-tune counterpart of the probe model (`resnet50_imagenet` for
`resnet50_imagenet_linear_probe`) and the seed defaults to the run's seed. Specs are resolved
through `results/runs/*.csv` to a local `checkpoints/<hash>.pt`, so the source run (and its
checkpoint) must exist on the machine. The resolved hash becomes part of the new run's
`config_hash` and is logged in the `init_from` column (results schema v2). CLI:
`--init-from none @0.1`.

### Transfer evaluation (e.g. EuroSAT → UC Merced)

Evaluate saved checkpoints on another dataset. Results append to
`results/transfer/<user>.v1.csv` (skipped if that `config_hash` → target is
already logged). You do **not** need to copy hashes by hand — filter by the
same knobs you used when training:

```bash
just prepare uc_merced_rgb          # once
just list-checkpoints               # see model / fraction / seed / test_acc

# every local checkpoint
just transfer-all felix

# one cell of the label-efficiency grid
just transfer felix resnet50_imagenet 1.0 0

# edge cases: hash / path / extra filters via the CLI
uv run satbench transfer-eval --user felix --config-hash abcdef012345
uv run satbench transfer-eval --user felix --checkpoint checkpoints/abcdef012345.pt
```

---



## Pipeline Explanation

See [THEORY.md](THEORY.md) for the theoretical background, ML concepts, and rationale for our approach. The sections below explain how the pipeline itself works.

### Part 2: How the pipeline works

#### The life of an experiment

```
configs/*.yaml  (+ overrides from notebook or command line)
      │
      ▼
ExperimentConfig ── iter_runs() ──► RunConfig, RunConfig, ...   (models × fractions × seeds)
                                         │
             for each run:               ▼
             already completed? ── yes ─► skip
                     │ no
                     ▼
             execute_run():  seed → choose subset → build model → train/validate per epoch
                             → restore best weights → test once
                     │
                     ▼
             append rows to results/epochs/<user>.v2.csv and results/runs/<user>.v2.csv
```

Each module in `src/satbench/` is responsible for one box in this picture.

#### `config.py`: what to run

An `ExperimentConfig` describes a whole grid: a list of models, a list of training fractions, a list of seeds, and shared hyperparameters (epochs, batch size, learning rate, weight decay, image size; AdamW without learning rate decay is used). It also holds execution settings such as the device and folders, which do not affect results. A YAML file in `configs/` is just a list of these fields; `ExperimentConfig.from_yaml` loads it and applies any overrides you pass.

`iter_runs()` expands the grid into individual `RunConfig` objects, one per combination. The baseline config has 3 models × 7 fractions × 3 seeds = 63 runs. A `RunConfig` contains exactly the fields that determine a run's outcome, and nothing else.

`model_overrides` lets one model use different hyperparameters, for example a higher learning rate for the linear baseline, without affecting the others.

Each run gets a **config hash**: the first 12 hex digits of a SHA-256 hash of its `RunConfig` (excluding the experiment name), the hash of the data split, and `PIPELINE_VERSION`. Two runs with the same hash are the same computation, regardless of who ran them or under which experiment name. This is how already-completed runs are skipped and how results from different members are matched up. If you change the training code in a way that changes results, bump `PIPELINE_VERSION` so new runs get new hashes and are not confused with old ones.

#### `seeding.py`: controlling randomness

`seed_everything(seed)` seeds Python's, NumPy's and PyTorch's random number generators (including GPU ones) and switches PyTorch into deterministic mode, which avoids GPU algorithms that give slightly different results on every call.

Different random choices use different, independent seeds derived from the run seed: `derive_seed(seed, "train_subset")` hashes the string `"0|train_subset"` with SHA-256 and takes the first 4 bytes as an integer. The streams are:


| What is random                          | How it is seeded                                           |
| --------------------------------------- | ---------------------------------------------------------- |
| model initialisation                    | global PyTorch RNG, seeded right before the model is built |
| which images are in the training subset | NumPy generator with `derive_seed(seed, "train_subset")`   |
| mini-batch order                        | PyTorch generator with `derive_seed(seed, "shuffle")`      |


Because every run reseeds from scratch, a run's result does not depend on which runs happened before it in the same session. That is what makes skipping and resuming safe.

#### `data.py`: images, splits, subsets

**Loading and caching.** The first time EuroSAT is used, torchvision downloads it (about 90 MB of JPEG files). The pipeline decodes all 27 000 images once and stores them as a single `uint8` tensor of shape $27000 \times 3 \times 64 \times 64$ (about 330 MB) in `data/eurosat_rgb_uint8.pt`. Later runs load this file in a second or two instead of decoding JPEGs every epoch. The dataset object also stores labels, class names, and a stable key per image (its relative file path).

Datasets are registered in a dictionary `DATASETS` by name using the `@register_dataset("name")` decorator. A tiny synthetic dataset called `fake` (4 classes of coloured noise) is registered too; the tests use it so they run offline in seconds.

**Fixed splits.** `get_splits` assigns each image to train, validation or test. EuroSAT has no official split in torchvision, so we use a stratified 60/20/20 train/val/test split with our fixed seed (3007). The assignment was saved to `splits/eurosat_rgb_split.csv` when setting this repo up and running the `prepare-data` command the first time. It is reloaded from there afterwards, so the test set is identical for everyone forever. No changes to the file should be done without discussing in the group. A short hash of the assignment (`split_hash`) is stored with every result.

**Training subsets.** `label_fraction_subset` picks the images used for a given fraction $f$ and seed. For each class $c$ with $n_c$ training images, it draws a random permutation $\pi_c$ of that class's images (using the subset seed) and takes the first

$$
k_c(f) = \max\left(1, \lfloor f \, n_c + \tfrac12 \rfloor\right)
$$

images. Two properties follow directly. The subset is stratified (every class keeps the same proportion and appears at least once). The subsets are nested: the permutation $\pi_c$ does not depend on $f$ and $k_c(f)$ is non-decreasing in $f$, so the 1% subset is contained in the 2% subset, and so on. Nesting makes the label-efficiency curve smoother, since adding data never swaps out images already seen. The subset depends only on the seed and fraction, not on the model, so all models are compared on the same images.

For EuroSAT the training split has 1800 images per class for most classes (1500 for four classes, 1200 for Pasture), giving these subset sizes:


| fraction        | 1%  | 2%  | 5%  | 10%   | 25%   | 50%   | 100%   |
| --------------- | --- | --- | --- | ----- | ----- | ----- | ------ |
| training images | 162 | 324 | 810 | 1 620 | 4 050 | 8 100 | 16 200 |


**Batching.** `iter_batches` yields mini-batches by indexing into the cached tensor, shuffled with the seeded generator during training and in fixed order during evaluation.

#### `models.py`: what to train

Models are registered by name with `@register_model("name")`. Each registered function receives the number of classes and the `RunConfig` and returns a `ModelBundle`, which pairs the network with the preprocessing it expects. Current families:

- **Linear**: `linear` (logistic regression on raw 64×64 pixels).
- **ResNet-50**: `resnet50_{random,imagenet,ssl4eo}` and matching `*_linear_probe` variants. Freezing uses `_resnet50(..., freeze_backbone=True)`; `train_one_epoch` keeps BatchNorm layers with frozen weights in eval mode so their running statistics stay fixed.
- **ViT-B/16**: `vit_b16_{random,imagenet}` and matching `*_linear_probe` variants (head is `heads.head` instead of `fc`). No SSL4EO ViT yet.

ResNet sweeps live in `configs/baseline_label_efficiency.yaml`; ViT sweeps in `configs/vit_label_efficiency.yaml` (same fractions/seeds, smaller default batch size).

`Preprocess` is a small PyTorch module that converts a batch of `uint8` images to floats in $[0,1]$, resizes them if needed, and normalises them per channel. It runs on the GPU right before the network. Keeping it in the bundle means a new model (for example with SSL4EO weights and different normalisation statistics) automatically brings its own correct preprocessing.

To add a model, write one decorated function and add its name to a config; see "Extending" below.

#### `train.py`: one run

`execute_run` performs one run from start to finish:

1. Seed everything, select the training subset, and fix the validation and test indices.
2. Build the model (initialisation uses the seeded RNG), move it to the device, create the optimiser and the shuffle generator.
3. For each epoch, call `train_one_epoch`, then `evaluate` on the validation set, record the metrics, and save a copy of the weights if validation accuracy improved.
4. Restore the best weights, evaluate on the test set once, and return everything as a `RunOutcome`.

`train_one_epoch` is the standard PyTorch training loop. For each mini-batch it preprocesses the images, resets the gradients (`optimizer.zero_grad`), computes the logits (forward pass), computes the cross-entropy loss, computes gradients (`loss.backward()`, the backward pass), and updates the parameters (`optimizer.step()`). The reported training loss and accuracy are averaged over the epoch while the weights are changing, which is why early on the training accuracy can be lower than the validation accuracy measured at the end of the epoch.

`evaluate` runs under `torch.no_grad()` (no gradients needed, which saves memory and time) with the network in evaluation mode. It accumulates the confusion matrix with a small trick: the pair (true class $y$, prediction $\hat y$) is encoded as the single integer $10y + \hat y$, and counting these integers with `bincount` gives the $10 \times 10$ matrix in one operation. Accuracy, loss and macro F1 are computed from it.

GPUs run asynchronously: Python continues while the GPU is still working. Before reading the clock, the code calls `synchronize` so timings measure the actual computation.

#### `results.py`: saving results

Results go into two CSV files per user. `results/runs/<user>.v2.csv` gets one row per completed run with its configuration, outcomes, timings, and provenance. `results/epochs/<user>.v2.csv` gets one row per epoch, used for learning curves. Transfer eval is separate and appends to `results/transfer/<user>.v1.csv`.

The files are **append-only**: rows are added, never edited or deleted. Rows are only written once a run has finished successfully. Epoch rows are written first and the run row last, so the run row acts as a "commit marker": if a crash happens in between, the loaders ignore epoch rows that have no matching run row.

Each member only writes to files named after themselves, so committing results never causes git merge conflicts. The number in the file name is the schema version. Training results use `SCHEMA_VERSION` in `results.py` (currently 2). Transfer eval uses `TRANSFER_SCHEMA_VERSION` (currently 1) and does not follow the training schema. If you add or rename columns, bump the matching version: new rows go into a new file, and the loaders combine every CSV in the folder.

`load_runs` and `load_epochs` read all members' files into pandas DataFrames.

#### `pipeline.py`: putting it together

`run_experiment(cfg)` validates the username and model names, finds the device, prints a summary, loads the dataset and splits, and loops over the runs. For each run it computes the config hash and skips the run if `(experiment, config_hash)` already appears in your results file. Otherwise it calls `execute_run`, builds the result rows, and appends them.

If a run crashes, the error is printed and the loop continues with the next run (set `continue_on_error: false` to stop instead). Pressing stop (KeyboardInterrupt) always stops immediately; completed runs are already saved and are skipped when you run it again.

#### `env.py` and `cli.py`

`env.py` detects the best available device (CUDA, then Apple MPS, then CPU), whether you are on Colab, the repo location, library versions, and the current git commit. The `git_dirty` flag is true if files in `src/`, `configs/`, `pyproject.toml` or `uv.lock` had uncommitted changes when the experiment started. Results from uncommitted code cannot be traced back to an exact version of the code, so commit before running real experiments.

`cli.py` defines the `satbench` command (`run`, `prepare-data`, `list-models`). It does the same thing as the run notebook. Run `uv run satbench run --help` to see all options.

#### The notebooks

`01_run_experiments.ipynb` sets up the environment, lets you set your username, config and overrides, and runs the experiment. `02_visualize_results.ipynb` loads everyone's results and produces the summary table, the label-efficiency curve, learning curves, runtime plots, a reproducibility table and a confusion matrix, saving figures to `figures/`.

### Part 3: Reading the outputs



#### Training logs

A typical run prints:

```
[12/63] resnet50_imagenet | frac=0.05 | seed=0  [3f9a1c2b7d10]
  ep   1/10 | train loss 1.2034 acc 0.6123 | val loss 0.4521 acc 0.8711 * | 38.2s
  ep   2/10 | train loss 0.3310 acc 0.8970 | val loss 0.2987 acc 0.9104 * | 37.9s
  ep   3/10 | train loss 0.1841 acc 0.9432 | val loss 0.3120 acc 0.9050 | 38.0s
  ...
  -> test acc 0.9187 | macro-F1 0.9162 | best epoch 2 | n_train 810 | 392.4s
```

The first line shows the run's position in the grid, its settings, and its config hash. Each epoch line shows training and validation loss and accuracy; a `*` marks a new best validation accuracy, whose weights are kept. The final line reports test results for the best epoch's weights.

Patterns to recognise: training loss falling while validation loss rises is overfitting. Validation loss above about 2.3 means the model is worse than uniform guessing. Training accuracy near 1.0 with much lower validation accuracy is typical at 1–2% subsets.

#### Run CSV columns


| Columns                                                        | Meaning                                                                |
| -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `run_id`                                                       | unique random ID of this run                                           |
| `config_hash`                                                  | ID of the computation; equal hashes should give (nearly) equal results |
| `experiment`, `user`                                           | which experiment, who ran it                                           |
| `started_at`, `finished_at`                                    | UTC timestamps                                                         |
| `dataset` … `eval_limit`                                       | the full run configuration (model, fraction, seed, hyperparameters)    |
| `split_hash`, `pipeline_version`                               | which data split and which version of the training logic               |
| `n_train`, `n_val`, `n_test`                                   | number of images actually used                                         |
| `n_params_total`, `n_params_trainable`                         | parameter counts (differ once backbones are frozen)                    |
| `best_epoch`, `best_val_acc`, `final_val_acc`                  | model selection on the validation set                                  |
| `test_acc`, `test_loss`, `test_macro_f1`                       | **the results to report**                                              |
| `train_time_s`, `eval_time_s`, `total_time_s`, `sec_per_epoch` | wall-clock timings                                                     |
| `device` … `git_dirty`                                         | where and with what software and code version it ran                   |
| `test_confusion_json`                                          | 10 × 10 test confusion matrix (rows = true, columns = predicted)       |
| `config_json`                                                  | the full configuration as JSON                                         |


The epoch CSV has per-epoch `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_macro_f1`, the learning rate used, and timings, plus identifying columns (`run_id`, `model`, `train_fraction`, `seed`) so it can be grouped directly.

### Part 4: Common tasks and pitfalls

**Running only part of a sweep.** Use overrides, for example `--seeds 1` or `--models resnet50_imagenet --fractions 0.01 0.02`. Completed runs are skipped, so overlapping work between members is harmless but wastes compute; agree on who runs what.

**Re-running something on purpose** (for example to check reproducibility): use `--no-skip`, or `skip_completed="none"` in the notebook.

**A quick try with fewer epochs.** Pass `epochs=2` (or `--epochs 2`). Since epochs are part of the config hash, these runs never get mixed up with the 10-epoch runs, but give them a different experiment name (`--experiment trial_lr`) so they don't clutter the main plots.

**Learning rates and other numbers in YAML.** Write `1.0e-4`, not `1e-4`; the YAML parser reads the second form as text. The config converts it anyway, but the dotted form avoids surprises.

`eval_limit` subsamples the validation and test sets and exists only for quick debugging (the smoke test uses it). The visualisation notebook excludes these runs by default. Never report them.

**Before real experiments**, commit your code changes. Results with `git_dirty = True` cannot be traced back to an exact code version.

**After running**, commit your results file and push: `git pull && git add results/ && git commit -m "..." && git push`.

**If you change how training works** (loss, preprocessing, evaluation), bump `PIPELINE_VERSION` in `config.py`. If you add or rename result columns, bump `SCHEMA_VERSION` in `results.py`.

**Out-of-memory errors on a GPU**: lower `eval_batch_size` in the config (it does not affect results). Lowering `batch_size` does change results, so treat it as a different experiment.

## Extending

New model: register a builder in `models.py` and add its name to a config.

```python
@register_model("resnet50_imagenet_linear_probe")
def _resnet50_imagenet_linear_probe(num_classes, run):
    train_tf, eval_tf = _resnet_transforms(run.image_size)
    return ModelBundle(_resnet50(num_classes, ResNet50_Weights.IMAGENET1K_V1, freeze_backbone=True),
                       train_tf, eval_tf,
                       "ResNet-50, ImageNet-1k V1 init, linear probe")
# Frozen BatchNorm layers are kept in eval mode by train_one_epoch automatically.
```

New dataset: register a loader in `data.py` returning an `ImageDataset`.
New result columns: add them to `results.py` and bump `SCHEMA_VERSION`.
Training-logic changes that alter results: bump `PIPELINE_VERSION` in `config.py`.