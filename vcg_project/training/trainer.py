"""
Training loop and evaluation metrics for ECG → VCG models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    """Training configuration."""
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15          # Early stopping patience
    warmup_epochs: int = 0      # Linear LR warmup before cosine decay (0 = off)
    noise_std: float = 0.0      # Gaussian noise added to ECG input during training only (0 = off)
    device: str = "auto"        # "auto", "cuda", "cpu"


@dataclass
class TrainState:
    """Tracks training state across epochs."""
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    total_time: float = 0.0


def _loop_area(xyz: np.ndarray, xi: int, yi: int) -> float:
    """
    Shoelace-formula area enclosed by a beat's trajectory projected onto
    a 2D plane (e.g. XY, YZ, XZ), treating consecutive time samples as
    polygon vertices and closing the loop back to the first point.

    Parameters
    ----------
    xyz : np.ndarray
        Single beat, shape [3, T].
    xi, yi : int
        Which of the 3 channels form the projection plane.
    """
    x, y = xyz[xi], xyz[yi]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return float(0.5 * np.abs(np.sum(x * y2 - x2 * y)))


def compute_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """
    Compute evaluation metrics between predicted and target VCG.

    Parameters
    ----------
    predicted : np.ndarray
        Shape [N, 3, T] or [3, T].
    target : np.ndarray
        Shape [N, 3, T] or [3, T].

    Returns
    -------
    dict with keys:
        'mse': Mean squared error
        'rmse': Root mean squared error
        'mae': Mean absolute error
        'r_squared': Pooled R² (coefficient of determination), per lead
            then averaged across leads — computed over all beats/time
            points at once, consistent with how 'correlation' is pooled
            (previously this was averaged per-beat, which is unstable
            for beats with near-flat ground truth and inconsistent with
            the correlation calculation).
        'correlation': Mean Pearson correlation across XYZ leads, pooled
            across all beats/time per lead before averaging.
        'loop_area_error': Mean relative error in loop area enclosed by
            the XY/YZ/XZ projections (shoelace formula), averaged across
            beats and the 3 planes. Point-wise metrics like MSE/R² can
            look good while the loop is visibly the wrong size/shape —
            this catches that.
        'max_deviation': Mean (across beats) worst-case single-timepoint
            3D Euclidean distance between predicted and target. A cheap
            proxy for Fréchet-style shape deviation — complements MSE's
            average-case view with a worst-case one.
    """
    if predicted.ndim == 2:
        predicted = predicted[np.newaxis, ...]
        target = target[np.newaxis, ...]

    # Flatten for global metrics
    pred_flat = predicted.reshape(-1)
    targ_flat = target.reshape(-1)

    mse = float(np.mean((pred_flat - targ_flat) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred_flat - targ_flat)))

    # R² per lead, pooled over all beats and time (not averaged per-beat)
    ss_res = np.sum((predicted - target) ** 2, axis=(0, 2))
    target_mean = target.mean(axis=(0, 2), keepdims=True)
    ss_tot = np.sum((target - target_mean) ** 2, axis=(0, 2))
    r2_per_lead = 1.0 - ss_res / (ss_tot + 1e-8)
    r_squared = float(np.mean(r2_per_lead))

    # Pearson correlation per lead, pooled across beats/time
    corrs = []
    for i in range(predicted.shape[1]):
        p = predicted[:, i, :].flatten()
        t = target[:, i, :].flatten()
        if np.std(p) > 1e-8 and np.std(t) > 1e-8:
            corrs.append(float(np.corrcoef(p, t)[0, 1]))
    correlation = float(np.mean(corrs)) if corrs else 0.0

    # Loop-shape metrics, computed per beat then averaged
    area_errors = []
    max_devs = []
    for n in range(predicted.shape[0]):
        p_beat, t_beat = predicted[n], target[n]  # [3, T]

        plane_errors = []
        for xi, yi in ((0, 1), (1, 2), (0, 2)):
            a_pred = _loop_area(p_beat, xi, yi)
            a_true = _loop_area(t_beat, xi, yi)
            plane_errors.append(abs(a_pred - a_true) / (a_true + 1e-8))
        area_errors.append(np.mean(plane_errors))

        pointwise_dist = np.sqrt(np.sum((p_beat - t_beat) ** 2, axis=0))  # [T]
        max_devs.append(np.max(pointwise_dist))

    loop_area_error = float(np.mean(area_errors))
    max_deviation = float(np.mean(max_devs))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared,
        "correlation": correlation,
        "loop_area_error": loop_area_error,
        "max_deviation": max_deviation,
    }


def train_model(
    model: nn.Module,
    train_ecg: np.ndarray,
    train_vcg: np.ndarray,
    val_ecg: np.ndarray,
    val_vcg: np.ndarray,
    config: TrainConfig | None = None,
    verbose: bool = True,
) -> TrainState:
    """
    Train an ECG → VCG model.

    Parameters
    ----------
    model : nn.Module
        Model to train (LSTM or Transformer).
    train_ecg : np.ndarray
        Training ECG, shape [N, 12, T].
    train_vcg : np.ndarray
        Training VCG targets, shape [N, 3, T].
    val_ecg : np.ndarray
        Validation ECG.
    val_vcg : np.ndarray
        Validation VCG targets.
    config : TrainConfig, optional
        Training configuration.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    TrainState
        Training history and statistics.
    """
    if config is None:
        config = TrainConfig()

    # Resolve device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    # Both models hit their best val loss at epoch 0-1 then overfit for the
    # rest of training — a warmup ramps the LR up from near-zero instead of
    # taking full-size steps from the first batch, which is one plausible
    # way that early jump lands in a sharp, poorly-generalizing minimum.
    if config.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.epochs - config.warmup_epochs),
            eta_min=config.lr * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )
    criterion = nn.MSELoss()

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.tensor(train_ecg, dtype=torch.float32),
        torch.tensor(train_vcg, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_ecg, dtype=torch.float32),
        torch.tensor(val_vcg, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
    )

    state = TrainState()
    best_state_dict = None
    start_time = time.time()

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        for ecg_batch, vcg_batch in train_loader:
            ecg_batch = ecg_batch.to(device)
            vcg_batch = vcg_batch.to(device)

            # Training-only input noise (simulates electrode/amplifier
            # noise). ECG is already z-scored to ~unit variance per channel
            # at this point, so noise_std is directly "fraction of a
            # channel's own std" without extra per-channel scaling. Applied
            # only here, never in the validation loop or at eval time, and
            # the VCG target is left untouched — the model should learn to
            # recover the same true VCG despite this input perturbation.
            if config.noise_std > 0:
                ecg_batch = ecg_batch + torch.randn_like(ecg_batch) * config.noise_std

            optimizer.zero_grad()
            vcg_pred = model(ecg_batch)
            loss = criterion(vcg_pred, vcg_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches
        state.train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for ecg_batch, vcg_batch in val_loader:
                ecg_batch = ecg_batch.to(device)
                vcg_batch = vcg_batch.to(device)

                vcg_pred = model(ecg_batch)
                loss = criterion(vcg_pred, vcg_batch)
                val_loss += loss.item()
                n_val += 1

        val_loss /= n_val
        state.val_losses.append(val_loss)

        # Early stopping
        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            state.best_epoch = epoch
            state.epochs_without_improvement = 0
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            state.epochs_without_improvement += 1

        scheduler.step()

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == config.epochs - 1):
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f} | "
                f"Best: {state.best_val_loss:.6f} (ep {state.best_epoch}) | "
                f"LR: {lr:.2e}"
            )

        # Early stopping check
        if state.epochs_without_improvement >= config.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no improvement for {config.patience} epochs)")
            break

    state.total_time = time.time() - start_time

    # Restore best-val-loss weights into the model in place — without this,
    # the caller ends up with whichever epoch training happened to stop on
    # (the last one before patience ran out), which is typically worse than
    # the best checkpoint once the model has started overfitting.
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    if verbose:
        print(f"\nTraining complete in {state.total_time:.1f}s")
        print(f"Best val loss: {state.best_val_loss:.6f} at epoch {state.best_epoch}")
        print(f"Restored model weights from best epoch ({state.best_epoch})")

    return state


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    ecg: np.ndarray,
    vcg: np.ndarray,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Evaluate a trained model on test data.

    Returns
    -------
    dict with metrics: mse, rmse, mae, r_squared, correlation.
    """
    model.eval()
    model.to(device)

    ecg_tensor = torch.tensor(ecg, dtype=torch.float32).to(device)
    vcg_tensor = torch.tensor(vcg, dtype=torch.float32).to(device)

    dataset = TensorDataset(ecg_tensor, vcg_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    for ecg_batch, vcg_batch in loader:
        vcg_pred = model(ecg_batch)
        all_preds.append(vcg_pred.cpu().numpy())
        all_targets.append(vcg_batch.cpu().numpy())

    predicted = np.concatenate(all_preds, axis=0)
    target = np.concatenate(all_targets, axis=0)

    return compute_metrics(predicted, target)


def evaluate_kors(
    ecg: np.ndarray,
    vcg: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate Kors regression baseline.

    Parameters
    ----------
    ecg : np.ndarray
        ECG beats, shape [N, 12, T].
    vcg : np.ndarray
        Ground truth VCG, shape [N, 3, T].

    Returns
    -------
    dict with metrics.
    """
    from vcg_project.models.kors import kors_transform_beats

    vcg_kors = kors_transform_beats(ecg)
    return compute_metrics(vcg_kors, vcg)
