"""
Shared train / validation loop with:
  - weighted cross-entropy
  - AdamW optimiser
  - cosine or ReduceLROnPlateau scheduler
  - early stopping on val macro-F1
  - checkpoint saving (best only or every epoch)
  - TensorBoard logging
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

from src.data.sequence_dataset import WaterEventDataset, class_weights
from src.evaluation.metrics import macro_f1, save_eval_results
from src.training.config import TrainConfig


class _NoOpSummaryWriter:
    """Fallback writer used when tensorboard is not installed."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add_scalars(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights.

    L = - alpha_c * (1 - p_c)^gamma * log(p_c)

    ``alpha`` is the class-weight tensor (same semantics as CrossEntropy ``weight``).
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            w = self.weight.to(logits.device)[target]
            loss = loss * w
        return loss.mean()


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float = float("-inf") if mode == "max" else float("inf")
        self.counter: int = 0
        self.improved: bool = False

    def __call__(self, value: float) -> bool:
        """Return True if training should stop."""
        improved = (
            value > self.best if self.mode == "max" else value < self.best
        )
        if improved:
            self.best = value
            self.counter = 0
            self.improved = True
        else:
            self.counter += 1
            self.improved = False
        return self.counter >= self.patience


def _make_optimizer_and_scheduler(
    model: nn.Module, cfg: TrainConfig
) -> tuple[AdamW, Any]:
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched: Any
    if cfg.scheduler == "cosine":
        sched = CosineAnnealingLR(opt, T_max=cfg.cosine_t_max)
    elif cfg.scheduler == "plateau":
        sched = ReduceLROnPlateau(
            opt,
            mode="max",
            patience=cfg.plateau_patience,
            factor=cfg.plateau_factor,
        )
    else:
        sched = None
    return opt, sched


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    device: torch.device,
    train: bool,
    global_step: int,
    writer: SummaryWriter | None,
    log_every: int,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """Run one epoch; return (avg_loss, all_labels, all_preds, global_step)."""
    model.train(train)
    total_loss = 0.0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            if len(batch) == 3:
                x, meta, y = batch
                meta = meta.to(device, non_blocking=True)
            else:
                x, y = batch
                meta = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if meta is not None:
                logits = model(x, meta)
            else:
                logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                if writer and global_step % log_every == 0:
                    writer.add_scalar("train/loss_step", loss.item(), global_step)

            total_loss += loss.item()
            all_labels.append(y.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    avg_loss = total_loss / len(loader)
    return avg_loss, labels, preds, global_step


def train(
    model: nn.Module,
    cfg: TrainConfig,
    train_ds: WaterEventDataset,
    val_ds: WaterEventDataset,
    label_names: list[str],
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Full training loop.

    Returns a history dict with per-epoch train/val losses and macro-F1 values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    ckpt_dir = Path(cfg.checkpoint_dir)
    log_dir = Path(cfg.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if SummaryWriter is None:
        print("[train] tensorboard is not installed; continuing without TensorBoard logging")
        writer = _NoOpSummaryWriter()
    else:
        writer = SummaryWriter(log_dir=str(log_dir))

    # Loss
    if cfg.use_class_weights:
        cw = class_weights(train_ds).to(device)
    else:
        cw = None
    if cfg.loss_type == "focal":
        criterion = FocalLoss(gamma=cfg.focal_gamma, weight=cw)
        print(f"[train] loss: focal (gamma={cfg.focal_gamma}, class_weights={cw is not None})")
    else:
        criterion = nn.CrossEntropyLoss(weight=cw)
        print(f"[train] loss: cross-entropy (class_weights={cw is not None})")

    # DataLoaders
    from src.data.sequence_dataset import make_weighted_sampler

    if cfg.use_weighted_sampler:
        sampler = make_weighted_sampler(train_ds)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
    )

    opt, sched = _make_optimizer_and_scheduler(model, cfg)
    stopper = EarlyStopping(patience=cfg.early_stop_patience, mode="max")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    global_step = 0
    best_val_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss, tr_labels, tr_preds, global_step = _epoch_pass(
            model, train_loader, criterion, opt, device,
            train=True, global_step=global_step,
            writer=writer, log_every=cfg.log_every_n_steps,
        )
        val_loss, val_labels, val_preds, _ = _epoch_pass(
            model, val_loader, criterion, None, device,
            train=False, global_step=global_step,
            writer=None, log_every=cfg.log_every_n_steps,
        )

        train_mf1 = macro_f1(tr_labels, tr_preds)
        val_mf1 = macro_f1(val_labels, val_preds)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_mf1)
        history["val_f1"].append(val_mf1)

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("macro_f1", {"train": train_mf1, "val": val_mf1}, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs}  "
            f"loss {train_loss:.4f}/{val_loss:.4f}  "
            f"F1 {train_mf1:.4f}/{val_mf1:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # Scheduler step
        if sched is not None:
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(val_mf1)
            else:
                sched.step()

        # Checkpoint
        if val_mf1 > best_val_f1:
            best_val_f1 = val_mf1
            best_epoch = epoch
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1": val_mf1,
                    "cfg": cfg.__dict__,
                },
                ckpt_path,
            )

        if not cfg.save_best_only:
            torch.save(
                model.state_dict(),
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

        if stopper(val_mf1):
            print(f"Early stopping at epoch {epoch} (no improvement for {cfg.early_stop_patience} epochs).")
            break

    writer.close()

    # Save training history
    history_path = Path(cfg.log_dir) / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best val macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")

    return {"history": history, "best_val_f1": best_val_f1, "best_epoch": best_epoch}
