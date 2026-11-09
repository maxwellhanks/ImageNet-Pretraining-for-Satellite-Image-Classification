"""Training and evaluation of a single run."""
from __future__ import annotations
from torch.nn.modules.batchnorm import _BatchNorm
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .checkpoint import load_backbone_state
from .config import RunConfig
from torch.utils.data import DataLoader
from .data import ImageDataset, Splits, label_fraction_subset, limit_indices, TransformedSubset
from .env import empty_cache, synchronize
from .models import ModelBundle, build_model
from .seeding import derive_seed, seed_everything


@dataclass
class EvalResult:
    """Stores the aggregated results of evaluating a model on a dataset."""
    loss: float
    acc: float
    macro_f1: float
    confusion: list[list[int]]  # rows = true class, cols = predicted class


@dataclass
class EpochMetrics:
    """Stores the performance metrics recorded at the end of a single epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_macro_f1: float
    lr: float
    train_time_s: float
    eval_time_s: float


@dataclass
class RunOutcome:
    """Stores the final summary of a complete training run, including test results."""
    n_train: int
    n_val: int
    n_test: int
    n_params_total: int
    n_params_trainable: int
    best_epoch: int
    best_val_acc: float
    final_val_acc: float
    test: EvalResult
    train_time_s: float
    eval_time_s: float
    total_time_s: float
    epochs: list[EpochMetrics] = field(default_factory=list)
    best_state: dict[str, torch.Tensor] | None = None


def _make_optimizer(params: list[torch.nn.Parameter], run: RunConfig) -> torch.optim.Optimizer:
    """Creates the optimizer for the run (AdamW)."""
    return torch.optim.AdamW(params, lr=run.lr, weight_decay=run.weight_decay)


def train_one_epoch(bundle: ModelBundle, train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, desc: str = "", progress: bool = True) -> tuple[float, float]:
    """Runs one full pass (epoch) over the training dataset, updating model weights."""
    # Put the network in training mode (activates dropout, updates BatchNorm stats)
    bundle.net.train()

    # set batch norms to eval mode if frozen to stop updating running statistics in linear probe mode
    for m in bundle.net.modules():
        if isinstance(m, _BatchNorm) and m.affine and not m.weight.requires_grad:
            m.eval()

    total_loss, correct, seen = 0.0, 0, 0
    
    # Iterate over mini-batches of images (x) and labels (y)
    for x, y in tqdm(train_loader, desc=desc, leave=False, disable=not progress):
        # Move the data to the target device (GPU/MPS/CPU)
        x, y = x.to(device), y.to(device)
        
        # Reset the gradients from the previous step
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass: compute the model's predictions (logits)
        logits = bundle.net(x)
        
        # Compute the cross-entropy loss between predictions and true labels
        loss = F.cross_entropy(logits, y)
        
        # Backward pass: compute the gradients of the loss with respect to model parameters
        loss.backward()
        
        # Optimization step: update the model parameters using the computed gradients
        optimizer.step()
        
        # Accumulate metrics for reporting
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
        
    if seen == 0:
        return 0.0, 0.0
    return total_loss / seen, correct / seen


@torch.no_grad() # Disable gradient calculation to save memory and speed up evaluation
def evaluate(bundle: ModelBundle, data_loader: DataLoader, num_classes: int, device: torch.device) -> EvalResult:
    """Evaluates the model on a dataset without updating weights, returning accuracy, loss, and F1."""
    # Put the network in evaluation mode (disables dropout, uses frozen BatchNorm stats)
    bundle.net.eval()
    
    c = num_classes
    # Flattened confusion matrix (c * c) to accumulate predictions
    cm = torch.zeros(c * c, dtype=torch.long)
    total_loss = 0.0
    
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        
        # Forward pass only
        logits = bundle.net(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        
        # Trick to compute confusion matrix quickly on CPU: 
        # (true_class * num_classes + predicted_class) maps every pair to a unique 1D index
        preds = logits.argmax(1)
        cm += torch.bincount((y * c + preds).cpu(), minlength=c * c)
        
    # Reshape the flattened counts back into a 2D confusion matrix (rows=true, cols=pred)
    cm = cm.view(c, c)
    cmf = cm.double()
    
    # Extract True Positives (diagonal elements)
    tp = cmf.diag()
    
    # Calculate denominator for F1 score: 2*TP + FP + FN
    # cmf.sum(0) is column sums (all predicted), cmf.sum(1) is row sums (all true)
    denom = 2 * tp + (cmf.sum(0) - tp) + (cmf.sum(1) - tp)
    
    # Calculate per-class F1 score, handling division by zero
    f1 = torch.where(denom > 0, 2 * tp / denom.clamp(min=1), torch.zeros_like(tp))
    
    n = cmf.sum().item()
    return EvalResult(loss=total_loss / n, acc=tp.sum().item() / n,
                      macro_f1=f1.mean().item(), confusion=cm.tolist())


def execute_run(run: RunConfig, data: ImageDataset, splits: Splits, device: torch.device,
                eval_batch_size: int = 128, progress: bool = True,
                init_state: dict[str, torch.Tensor] | None = None) -> RunOutcome:
    """
    Executes a single experiment run from start to finish:
      1. Seeds the random number generators.
      2. Initializes the model and subsets the data.
      3. Trains for ``run.epochs``, evaluating on the validation set after each.
      4. Restores the best weights and evaluates on the test set once.
    """
    t_start = time.perf_counter()

    # Seed all sources of randomness for reproducibility
    seed_everything(run.seed, run.deterministic)
    
    # Sub-sample the training set according to the requested label fraction
    train_idx = label_fraction_subset(splits.train, data.labels.numpy(), run.train_fraction, run.seed)
    
    # Apply optional debug limits to validation and test sets
    val_idx = limit_indices(splits.val, run.eval_limit)
    test_idx = limit_indices(splits.test, run.eval_limit)

    # Initialize the model (init uses the seeded global RNG) and move it to the device
    bundle = build_model(run.model, data.num_classes, run)
    if init_state is not None:
        # Backbone from a finished run (run.init_from); the head keeps its seeded fresh init.
        # Freezing (linear probes) was already applied by the builder and survives loading.
        dropped = load_backbone_state(bundle.net, init_state)
        print(f"  init_from {run.init_from}: loaded backbone, re-initialised head {dropped}")
    bundle.net.to(device)
    n_total, n_trainable = bundle.n_params()
    
    # Setup the optimizer and a seeded generator for the DataLoader's shuffling
    optimizer = _make_optimizer(bundle.trainable_parameters(), run)
    shuffle_gen = torch.Generator().manual_seed(derive_seed(run.seed, "shuffle"))

    # Create PyTorch datasets using TransformedSubset (handles CPU-side transformations)
    train_ds = TransformedSubset(data, train_idx, bundle.train_transform)
    val_ds = TransformedSubset(data, val_idx, bundle.eval_transform)
    test_ds = TransformedSubset(data, test_idx, bundle.eval_transform)
    
    # Create DataLoaders to batch and optionally shuffle the data
    train_loader = DataLoader(train_ds, batch_size=run.batch_size, shuffle=(len(train_ds) > 0), generator=shuffle_gen)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size)

    epochs: list[EpochMetrics] = []
    best_epoch, best_val, best_state = 0, -1.0, None
    train_time = eval_time = 0.0

    # If the train fraction is 0, we don't need to train for multiple epochs.
    # We just evaluate the untrained model once.
    num_epochs = 1 if len(train_idx) == 0 else run.epochs

    # Start the training loop over epochs
    for ep in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        
        # Train for one epoch (skip if 0 samples)
        if len(train_idx) > 0:
            tr_loss, tr_acc = train_one_epoch(bundle, train_loader, optimizer,
                                              device, f"ep {ep}/{num_epochs}", progress)
        else:
            tr_loss, tr_acc = 0.0, 0.0
            
        synchronize(device)
        t1 = time.perf_counter()
        
        # Evaluate on the validation set
        val = evaluate(bundle, val_loader, data.num_classes, device)
        synchronize(device)
        t2 = time.perf_counter()
        
        train_time += t1 - t0
        eval_time += t2 - t1

        lr_used = optimizer.param_groups[0]["lr"]
        epochs.append(EpochMetrics(ep, tr_loss, tr_acc, val.loss, val.acc, val.macro_f1,
                                   lr_used, t1 - t0, t2 - t1))
                                   
        # Model Selection: save model weights if validation accuracy improved
        improved = val.acc > best_val
        if improved:
            best_epoch, best_val = ep, val.acc
            if run.select_best_on_val:
                best_state = {k: v.detach().to("cpu", copy=True)
                              for k, v in bundle.net.state_dict().items()}
                              
        print(f"  ep {ep:>3}/{num_epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {val.loss:.4f} acc {val.acc:.4f}{' *' if improved else ''} | "
              f"{t2 - t0:.1f}s")

    # Final Testing: load the best weights found during training and evaluate on the unseen test set
    if run.select_best_on_val and best_state is not None:
        bundle.net.load_state_dict(best_state)
        
    t3 = time.perf_counter()
    test = evaluate(bundle, test_loader, data.num_classes, device)
    synchronize(device)
    eval_time += time.perf_counter() - t3

    # If select_best_on_val is False, fall back to final weights
    if best_state is None:
        best_state = {k: v.detach().to("cpu", copy=True)
                      for k, v in bundle.net.state_dict().items()}

    # Clean up memory
    del bundle, optimizer
    empty_cache(device)

    return RunOutcome(
        n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
        n_params_total=n_total, n_params_trainable=n_trainable,
        best_epoch=best_epoch, best_val_acc=best_val, final_val_acc=epochs[-1].val_acc,
        test=test, train_time_s=train_time, eval_time_s=eval_time,
        total_time_s=time.perf_counter() - t_start, epochs=epochs, best_state=best_state,
    )
