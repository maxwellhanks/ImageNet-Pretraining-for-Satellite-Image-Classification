"""Datasets, fixed splits and label-efficiency subsets.

Images are decoded once and cached as a single uint8 tensor (e.g. EuroSAT RGB:
27000 x 3 x 64 x 64 = ~330 MB). PyTorch DataLoaders are then used to sample from 
this cache and apply CPU-based transforms (augmentations and normalisation).
This makes training fast while still allowing for standard data augmentation.
"""
from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .seeding import derive_seed

SPLIT_SEED = 3007  # fixed forever: the test set must never change
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2} # train/val/test split fractions


@dataclass
class ImageDataset:
    """In-memory cache for a dataset's images and metadata."""
    name: str
    images: torch.Tensor  # Cached images in uint8 format, shape [N, 3, H, W]
    labels: torch.Tensor  # Integer class labels, shape [N]
    classes: list[str]    # String names of classes
    keys: list[str]       # Stable per-sample unique identifiers (e.g., relative file paths)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


@dataclass
class Splits:
    """Indices separating the dataset into training, validation, and testing sets."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    split_hash: str       # Short hash of the assignment used to track reproducibility


# Registry
DatasetLoader = Callable[[Path], ImageDataset]
DATASETS: dict[str, DatasetLoader] = {}


def register_dataset(name: str) -> Callable[[DatasetLoader], DatasetLoader]:
    """Decorator to register a new dataset loading function."""
    def deco(fn: DatasetLoader) -> DatasetLoader:
        DATASETS[name] = fn
        return fn
    return deco


def load_dataset(name: str, data_dir: str | Path) -> ImageDataset:
    """Loads a dataset by name, downloading and caching it if necessary."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return DATASETS[name](data_dir)


def _save_cache(ds: ImageDataset, path: Path) -> None:
    """Safely saves an ImageDataset to a PyTorch tensor file using a temporary file."""
    tmp = path.with_suffix(".tmp")
    torch.save({"name": ds.name, "images": ds.images, "labels": ds.labels,
                "classes": ds.classes, "keys": ds.keys}, tmp)
    os.replace(tmp, path)


@register_dataset("eurosat_rgb")
def _load_eurosat_rgb(data_dir: Path) -> ImageDataset:
    """Loads the EuroSAT RGB dataset. Caches decoded uint8 tensors to disk on first run."""
    cache = data_dir / "eurosat_rgb_uint8.pt"
    if cache.exists():
        # Load the pre-processed cache if it exists to save time
        return ImageDataset(**torch.load(cache, weights_only=True))

    from PIL import Image
    from torchvision.datasets import EuroSAT
    from tqdm.auto import tqdm

    # Download raw images
    raw = EuroSAT(root=str(data_dir), download=True)
    
    # Pre-allocate memory for all images
    images = torch.empty((len(raw.samples), 3, 64, 64), dtype=torch.uint8)
    
    for i, (path, _) in enumerate(tqdm(raw.samples, desc="Caching EuroSAT RGB")):
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        if arr.shape != (64, 64, 3):
            raise ValueError(f"Unexpected image shape {arr.shape} for {path}")
            
        # Convert HWC to CHW format for PyTorch
        images[i] = torch.from_numpy(arr).permute(2, 0, 1)
        
    ds = ImageDataset(
        name="eurosat_rgb",
        images=images,
        labels=torch.tensor([y for _, y in raw.samples], dtype=torch.int64),
        classes=list(raw.classes),
        keys=[Path(os.path.relpath(p, raw.root)).as_posix() for p, _ in raw.samples],
    )
    
    # Save to disk for future runs
    _save_cache(ds, cache)
    return ds

# manual UC Merced -> EuroSAT label map for transfer eval (see THEORY.md)
# only these classes are kept
# edit carefully, then delete uc_merced_rgb_uint8.pt
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
EUROSAT_TO_IDX = {c: i for i, c in enumerate(EUROSAT_CLASSES)}

UC_MERCED_TO_EUROSAT = {
    "forest": "Forest",
    "river": "River",
    "sparseresidential": "Residential",
    "mediumresidential": "Residential",
    "denseresidential": "Residential",
    "freeway": "Highway",
    "intersection": "Highway",
    "overpass": "Highway",
    "agricultural": "AnnualCrop",
    "storagetanks": "Industrial",
    "mobilehomepark": "Residential",
    "chaparral": "HerbaceousVegetation",
}
UC_MERCED_TO_EUROSAT_IDX = {
    uc_merced: EUROSAT_TO_IDX[eurosat] for uc_merced, eurosat in UC_MERCED_TO_EUROSAT.items()
}

@register_dataset("uc_merced_rgb")
def _load_uc_merced_rgb(data_dir: Path) -> ImageDataset:
    """Loads the UC Merced RGB dataset mapped to EuroSAT labels. Caches decoded uint8 tensors to disk on first run."""
    cache = data_dir / "uc_merced_rgb_uint8.pt"
    if cache.exists():
        # Load the pre-processed cache if it exists to save time
        return ImageDataset(**torch.load(cache, weights_only=True))
    
    from torchgeo.datasets import UCMerced
    from tqdm.auto import tqdm
    
    images_list, labels_list, keys = [], [], []
    for split in ("train", "val", "test"):
        # Download raw images
        raw = UCMerced(
            root=str(data_dir / "uc_merced"), 
            split=split, 
            download=(split == "train"))

        for i in tqdm(range(len(raw)), desc=f"Caching UC Merced RGB [{split}]"):
            sample = raw[i]
            uc_merced_label = raw.classes[int(sample["label"])]
            y = UC_MERCED_TO_EUROSAT_IDX.get(uc_merced_label.lower())
            if y is None:
                continue

            img = sample["image"]
            if img.dtype != torch.uint8:
                img = img.clamp(0, 255).round().to(torch.uint8)

            images_list.append(img)
            labels_list.append(y)
            keys.append(Path(os.path.relpath(raw.samples[i][0], data_dir)).as_posix())

    if not images_list:
        raise RuntimeError("No UC Merced samples matched UC_MERCED_TO_EUROSAT")

    ds = ImageDataset(
        name="uc_merced_rgb",
        images=torch.stack(images_list),
        labels=torch.tensor(labels_list, dtype=torch.int64),
        classes=list(EUROSAT_CLASSES),
        keys=keys,
    )
    
    # Save to disk for future runs
    _save_cache(ds, cache)
    return ds

@register_dataset("fake")
def _load_fake(data_dir: Path) -> ImageDataset:
    """Tiny synthetic, learnable dataset for tests / offline smoke runs."""
    n_classes, per_class, size = 4, 60, 32
    g = torch.Generator().manual_seed(0)
    
    # Generate base colors and noise
    colors = torch.randint(40, 215, (n_classes, 3, 1, 1), generator=g)
    labels = torch.arange(n_classes).repeat_interleave(per_class)
    noise = torch.randint(-40, 41, (len(labels), 3, size, size), generator=g)
    
    # Combine to create synthetic images
    images = (colors[labels] + noise).clamp(0, 255).to(torch.uint8)
    
    return ImageDataset(
        name="fake", images=images, labels=labels,
        classes=[f"class{c}" for c in range(n_classes)],
        keys=[f"class{y}/img{i:04d}" for i, y in enumerate(labels.tolist())],
    )


# Splits
def _hash_split(keys: list[str], assignment: list[str]) -> str:
    """Computes a SHA-256 hash of the dataset split assignment for reproducibility tracking."""
    h = hashlib.sha256()
    for k, s in sorted(zip(keys, assignment)):
        h.update(f"{k},{s}\n".encode())
    return h.hexdigest()[:12]


def _generate_split(labels: np.ndarray) -> list[str]:
    """Generates a stratified 60/20/20 train/val/test split using a fixed seed."""
    rng = np.random.default_rng(SPLIT_SEED)
    assignment = np.empty(len(labels), dtype=object)
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_train = int(np.floor(SPLIT_FRACTIONS["train"] * len(idx) + 0.5))
        n_val = int(np.floor(SPLIT_FRACTIONS["val"] * len(idx) + 0.5))
        assignment[idx[:n_train]] = "train"
        assignment[idx[n_train:n_train + n_val]] = "val"
        assignment[idx[n_train + n_val:]] = "test"
    return assignment.tolist()


def get_splits(ds: ImageDataset, splits_dir: str | Path, persist: bool | None = None) -> Splits:
    """Loads dataset split assignments from disk, or generates and saves them if missing.
    
    The split file ensures the test set remains strictly identical across all runs and users.
    """
    persist = ds.name != "fake" if persist is None else persist
    path = Path(splits_dir) / f"{ds.name}_split.csv"
    labels = ds.labels.numpy()

    if path.exists():
        # Load existing split assignment from disk
        key_to_idx = {k: i for i, k in enumerate(ds.keys)}
        assignment = [""] * len(ds)
        
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                i = key_to_idx[row["key"]]
                if int(row["label"]) != labels[i]:
                    raise ValueError(f"Label mismatch for {row['key']} in {path}")
                assignment[i] = row["split"]
                
        if "" in assignment:
            raise ValueError(f"{path} does not cover every sample of {ds.name}")
            
    else:
        # Generate new splits and persist them for future reproducibility
        assignment = _generate_split(labels)
        if persist:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(["key", "label", "class_name", "split"])
                for k, y, s in zip(ds.keys, labels, assignment):
                    w.writerow([k, int(y), ds.classes[y], s])
            print(f"[splits] created {path} - commit this file!")

    arr = np.asarray(assignment)
    return Splits(
        train=np.flatnonzero(arr == "train"),
        val=np.flatnonzero(arr == "val"),
        test=np.flatnonzero(arr == "test"),
        split_hash=_hash_split(ds.keys, assignment),
    )


# Subsets
def label_fraction_subset(train_idx: np.ndarray, labels: np.ndarray,
                          fraction: float, seed: int) -> np.ndarray:
    """Stratified subset of the training split.

    * stratified: every class keeps ``fraction`` of its samples (>= 1),
      so tiny fractions are not accidentally missing classes;
    * nested: for a fixed seed, the 1% subset is contained in the 2% subset,
      which is contained in the 5% subset, ... (smoother label-efficiency curves);
    * independent of the model, so all models see the same images per seed.
    """
    rng = np.random.default_rng(derive_seed(seed, "train_subset"))
    train_labels = labels[train_idx]
    chosen = []
    for c in np.unique(train_labels):
        cls_idx = rng.permutation(train_idx[train_labels == c])
        k = 0 if fraction == 0.0 else max(1, int(np.floor(fraction * len(cls_idx) + 0.5)))
        chosen.append(cls_idx[:k])
    return np.sort(np.concatenate(chosen))


def limit_indices(idx: np.ndarray, limit: int | None) -> np.ndarray:
    """Deterministically subsamples indices, used for quick debugging (``eval_limit``)."""
    if limit is None or limit >= len(idx):
        return idx
    rng = np.random.default_rng(derive_seed(0, "eval_limit"))
    return np.sort(rng.permutation(idx)[:limit])


class TransformedSubset(Dataset):
    """A PyTorch Dataset that applies a transform to a specific subset of the data."""
    def __init__(self, dataset: ImageDataset, indices: np.ndarray, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        idx = self.indices[i]
        x = self.dataset.images[idx]
        y = self.dataset.labels[idx]
        
        if self.transform is not None:
            x = self.transform(x)
            
        return x, int(y)
