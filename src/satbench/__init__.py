"""satbench: reproducible pretraining experiments for satellite image classification."""
__version__ = "0.1.0"

from . import seeding  # noqa: F401  (sets CUBLAS_WORKSPACE_CONFIG before CUDA init)
from .config import ExperimentConfig, RunConfig  # noqa: E402
from .models import MODELS, register_model  # noqa: E402
from .data import DATASETS, register_dataset  # noqa: E402
from .pipeline import run_experiment  # noqa: E402
from .results import load_epochs, load_runs, load_transfers  # noqa: E402

__all__ = [
    "ExperimentConfig", "RunConfig", "run_experiment",
    "MODELS", "register_model", "DATASETS", "register_dataset",
    "load_runs", "load_epochs", "load_transfers",
]
