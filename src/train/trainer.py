from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.train.metrics import compute_multi_label_metrics, compute_single_label_metrics
from src.train.utils import get_autocast_context, save_checkpoint


class AudioTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
        scaler,
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

        self.task_mode = str(config["task_mode"]).strip().lower()
        self.use_cuda_amp = bool(use_cuda_amp)
        self.use_mps_amp = bool(use_mps_amp)
        self.threshold = float(
            (config.get("multi_label", {}) or {}).get("threshold", 0.5)
        )

        if self.task_mode == "single_label":
            self.criterion = nn.CrossEntropyLoss()
        elif self.task_mode == "multi_label":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task_mode: {self.task_mode}")

    def _prepare_targets(self, y_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert targets for criterion and return a validity mask.
        """
        if self.task_mode == "multi_label":
            valid_mask = torch.ones(y_raw.size(0), dtype=torch.bool, device=y_raw.device)
            return y_raw.float(), valid_mask

        if y_raw.ndim == 1:
            valid_mask = torch.ones(y_raw.size(0), dtype=torch.bool, device=y_raw.device)
            return y_raw.long(), valid_mask

        sums = y_raw.sum(dim=1)
        valid_mask = sums > 0
        y_idx = torch.argmax(y_raw, dim=1).long()
        return y_idx, valid_mask

    def _compute_metrics(
        self, pred_chunks: List[torch.Tensor], target_chunks: List[torch.Tensor]
    ) -> Dict[str, float]:
        if not pred_chunks:
            if self.task_mode == "single_label":
                return {"acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}
            return {"hamming_acc": 0.0, "subset_acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}

        preds = torch.cat(pred_chunks, dim=0).numpy()
        targets = torch.cat(target_chunks, dim=0).numpy()

        if self.task_mode == "single_label":
            return compute_single_label_metrics(targets.astype(np.int64), preds.astype(np.int64))
        return compute_multi_label_metrics(targets.astype(np.int32), preds.astype(np.float32), self.threshold)

    def _run_epoch(self, loader, pin_mem: bool, train: bool) -> Tuple[float, Dict[str, float]]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        loss_sum = 0.0
        total = 0
        pred_chunks: List[torch.Tensor] = []
        target_chunks: List[torch.Tensor] = []

        for x, y_raw in loader:
            x = x.to(self.device, non_blocking=pin_mem)
            y_raw = y_raw.to(self.device, non_blocking=pin_mem)

            y_target, valid_mask = self._prepare_targets(y_raw)
            if not valid_mask.any():
                continue

            x_valid = x[valid_mask]
            y_valid = y_target[valid_mask]

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            autocast_ctx = get_autocast_context(self.use_cuda_amp, self.use_mps_amp)
            grad_ctx = torch.enable_grad() if train else torch.no_grad()
            with grad_ctx:
                with autocast_ctx:
                    logits = self.model(x_valid)
                    loss = self.criterion(logits, y_valid)

            if train:
                if self.use_cuda_amp and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()

            if self.task_mode == "single_label":
                preds = torch.argmax(logits.detach(), dim=1).cpu()
                tgts = y_valid.detach().cpu()
            else:
                preds = torch.sigmoid(logits.detach()).cpu()
                tgts = y_valid.detach().cpu()

            pred_chunks.append(preds)
            target_chunks.append(tgts)
            batch_n = x_valid.size(0)
            loss_sum += float(loss.item()) * batch_n
            total += batch_n

        metrics = self._compute_metrics(pred_chunks, target_chunks)
        return loss_sum / max(1, total), metrics

    def train_epoch(self, loader, pin_mem: bool) -> Tuple[float, Dict[str, float]]:
        return self._run_epoch(loader, pin_mem=pin_mem, train=True)

    def evaluate_epoch(self, loader, pin_mem: bool) -> Tuple[float, Dict[str, float]]:
        return self._run_epoch(loader, pin_mem=pin_mem, train=False)

    def fit(
        self,
        train_loader,
        val_loader,
        pin_mem: bool,
        *,
        start_epoch: int = 1,
        history: Dict[str, List[float]] | None = None,
        best_val_loss: float | None = None,
        epochs_no_improve: int = 0,
    ) -> Dict[str, List[float]]:
        train_cfg = self.config.get("training", {}) or {}
        epochs = int(train_cfg.get("epochs", 50))
        patience = int(train_cfg.get("patience", 10))

        history = {
            k: [float(vv) for vv in v]
            for k, v in (history or {}).items()
            if isinstance(v, list)
        }
        history.setdefault("train_loss", [])
        history.setdefault("val_loss", [])

        if best_val_loss is None:
            best_val_loss = float("inf")

        if start_epoch > epochs:
            print(
                f"Checkpoint already reached epoch {start_epoch - 1}, "
                f"which is >= configured epochs={epochs}. Nothing to resume."
            )
            return history

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_loss, train_metrics = self.train_epoch(train_loader, pin_mem=pin_mem)
            val_loss, val_metrics = self.evaluate_epoch(val_loader, pin_mem=pin_mem)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Persist history.
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            for k, v in train_metrics.items():
                history.setdefault(f"train_{k}", []).append(float(v))
            for k, v in val_metrics.items():
                history.setdefault(f"val_{k}", []).append(float(v))

            is_best = val_loss < best_val_loss - 1e-8
            if is_best:
                best_val_loss = float(val_loss)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

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

            dt = time.time() - t0
            metric_name = "macro_f1"
            train_m = train_metrics.get(metric_name, 0.0)
            val_m = val_metrics.get(metric_name, 0.0)
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_{metric_name}={train_m:.4f} val_{metric_name}={val_m:.4f} | "
                f"{dt:.1f}s"
            )

            if epochs_no_improve >= patience:
                print(
                    f"Early stopping: no val_loss improvement for {patience} epochs. "
                    f"Best val_loss={best_val_loss:.4f}"
                )
                break

        return history
