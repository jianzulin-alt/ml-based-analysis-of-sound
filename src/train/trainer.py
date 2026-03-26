from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the newly unified metrics function
from src.train.metrics import compute_metrics

# Import utilities from your centralized utils module
from src.utils.train_utils import get_autocast_context, save_checkpoint, write_history_csv

class BCEFocalLoss(nn.Module):
    """
    Focal Loss adapted for Multi-Hot/Binary Targets to handle class imbalance.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class AudioTrainer:
    """
    Core training engine for audio classification (Single & Multi Label).
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: str,
        scaler: Optional[torch.amp.GradScaler],
        config: dict,
        ckpt_dir: Path,
        use_cuda_amp: bool,
        use_mps_amp: bool,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.config = config
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.use_cuda_amp = bool(use_cuda_amp)
        self.use_mps_amp = bool(use_mps_amp)

        # Dynamic Loss Function Selection
        loss_fn_type = str(self.config.get("training", {}).get("loss_function", "cross_entropy")).strip().lower()
        
        if loss_fn_type == "focal":
            print("Trainer: Configured with BCEFocalLoss.")
            self.criterion = BCEFocalLoss()
        elif loss_fn_type == "bce":
            print("Trainer: Configured with BCEWithLogitsLoss (Multi-Label mode).")
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            print("Trainer: Configured with standard CrossEntropyLoss (Single-Label mode).")
            self.criterion = nn.CrossEntropyLoss()

    def _prepare_targets(self, y_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y_raw.to(self.device, non_blocking=True)
        return y, y

    def train_epoch(self, dataloader: Any) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        task_mode = self.config.get("task_mode", "single_label")

        for batch_idx, (x_raw, y_raw) in enumerate(dataloader):
            x = x_raw.to(self.device, non_blocking=True)
            y, y_metrics = self._prepare_targets(y_raw)

            self.optimizer.zero_grad(set_to_none=True)

            with get_autocast_context(self.use_cuda_amp, self.use_mps_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            all_preds.append(logits.detach())
            all_targets.append(y_metrics.detach())

        avg_loss = total_loss / len(dataloader.dataset)
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        metrics = compute_metrics(cat_preds, cat_targets, task_mode=task_mode)
        metrics["loss"] = avg_loss
        return metrics

    @torch.no_grad()
    def val_epoch(self, dataloader: Any) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        task_mode = self.config.get("task_mode", "single_label")

        for x_raw, y_raw in dataloader:
            x = x_raw.to(self.device, non_blocking=True)
            y, y_metrics = self._prepare_targets(y_raw)

            with get_autocast_context(self.use_cuda_amp, self.use_mps_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            all_preds.append(logits.detach())
            all_targets.append(y_metrics.detach())

        avg_loss = total_loss / len(dataloader.dataset)
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        metrics = compute_metrics(cat_preds, cat_targets, task_mode=task_mode)
        metrics["loss"] = avg_loss
        return metrics

    def fit(self, train_loader, val_loader, pin_mem=False, start_epoch=1, history=None, best_val_loss=None, epochs_no_improve=0):
        epochs = int(self.config.get("training", {}).get("epochs", 50))
        patience = int(self.config.get("training", {}).get("patience", 10))

        # FIX: Initialize history as a dictionary, not a list
        if history is None: 
            history = {} 
            
        if best_val_loss is None: 
            best_val_loss = float("inf")

        print(f"Starting training loop from epoch {start_epoch} to {epochs}")
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.val_epoch(val_loader)

            train_loss = train_metrics["loss"]
            val_loss = val_metrics["loss"]

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                is_best = True
            else:
                epochs_no_improve += 1

            # Build the record for this epoch
            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            for k, v in train_metrics.items():
                if k != "loss": record[f"train_{k}"] = v
            for k, v in val_metrics.items():
                if k != "loss": record[f"val_{k}"] = v

            # FIX: Append each metric to the dictionary of lists
            for key, value in record.items():
                if key not in history:
                    history[key] = []
                history[key].append(value)

            payload = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "opt_state": self.optimizer.state_dict(),
                "sched_state": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
                "config": self.config,
                "history": history,
                "best_val_loss": best_val_loss,
                "epochs_no_improve": epochs_no_improve,
            }
            
            save_checkpoint(payload, self.ckpt_dir / "last.pt")
            if is_best:
                save_checkpoint(payload, self.ckpt_dir / "best_val.pt")

            write_history_csv(history, self.ckpt_dir / "history.csv")

            dt = time.time() - t0
            metric_name = "macro_f1"
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_{metric_name}={train_metrics.get(metric_name, 0.0):.4f} val_{metric_name}={val_metrics.get(metric_name, 0.0):.4f} | "
                f"{dt:.1f}s"
            )

            if epochs_no_improve >= patience:
                print(f"Early stopping: no validation improvement for {patience} epochs.")
                break

        return history