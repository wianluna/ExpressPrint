import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from expressprint.utils import ComputeMetrics, CosineLRScheduler, add_metrics_dict, divide_metrics, dump_scalar_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        log_path: str,
        num_classes: int = 9,
        top_k_accuracy: int = 1,
        num_epochs: int = 5,
        optimizer_type: str = "adamw",
        lr: float = 0.00001,
        lr_scheduler_type: str = "cosine",
        lr_scheduler_params: dict = {"lr_peak_epoch": 3, "lr_peak": 0.00001},
        verbose: bool = False,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = Path(log_path)
        self.verbose = verbose

        self.model = model
        self.model.to(self.device)
        self.scaler = GradScaler()

        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.top_k_accuracy = top_k_accuracy
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_params = lr_scheduler_params

        self._init_logger()

    def __del__(self) -> None:
        self.writer.close()

    def _init_logger(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        print(f"=> Logging in {self.log_dir}")

    def _save_checkpoint(self) -> None:
        torch.save(self.model.state_dict(), self.log_dir / "model.pth")

    def _init_optimizer(self) -> None:
        if self.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"Unsupported optimizer type: {self.optimizer_type}")

    def _init_lr_scheduler(self) -> None:
        if self.lr_scheduler_type == "cosine":
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                num_epochs=self.num_epochs,
                lr_peak=self.lr_scheduler_params["lr_peak"],
                lr_peak_epoch=self.lr_scheduler_params["lr_peak_epoch"],
                num_steps_per_epoch=len(self.train_loader),
            )
        else:
            self.lr_scheduler = None

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self._prepare_for_training(train_loader, val_loader)
        self._train_loop()

    def _prepare_for_training(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.cur_epoch = 0
        self.end_epoch = self.num_epochs

        self._init_optimizer()
        self._init_lr_scheduler()

        self.loss_computer = nn.CrossEntropyLoss()

        self.metric_computer = ComputeMetrics(top_k=self.top_k_accuracy, num_classes=self.num_classes)

    def _train_loop(self) -> None:
        train_data_len = len(self.train_loader)
        val_data_len = len(self.val_loader)

        while self.cur_epoch < self.end_epoch:
            done_steps = self.cur_epoch * train_data_len
            batch_start_time = time.time()

            for step, (images, labels) in enumerate(self.train_loader):
                metrics = self._train_step(images, labels, step, batch_start_time)

                dump_scalar_metrics(metrics, self.writer, phase="train", global_step=done_steps + step)

                print(
                    f"TRAIN epoch: {self.cur_epoch}, step: {step} of {train_data_len}, "
                    f"total_loss: {metrics['total_loss']}",
                    end="\r",
                )

                batch_start_time = time.time()

            self.model.eval()
            metrics = {}

            for step, (images, labels) in enumerate(self.val_loader):
                metrics_new = self._val_step(images, labels)
                metrics = add_metrics_dict(metrics=metrics, metrics_new=metrics_new)

                print(
                    f"TEST epoch: {self.cur_epoch}, step: {step} of {val_data_len}, "
                    f"loss: {metrics_new['loss_nat']}",
                    end="\r",
                )

            metrics = divide_metrics(metrics, n=len(self.val_loader))
            dump_scalar_metrics(metrics, self.writer, phase="val", global_step=self.cur_epoch)

            self.cur_epoch += 1

        self._save_checkpoint()

    def _train_step(self, images: torch.Tensor, labels: torch.Tensor, step: int, start_time: float) -> Dict[str, Any]:
        images = images.to(self.device)
        labels = labels.to(self.device)

        metrics = {}
        metrics["data_time"] = time.time() - start_time
        if self.lr_scheduler:
            metrics["learning_rate"] = self.lr_scheduler.adjust_learning_rate(
                self.cur_epoch,
                step=step,
            )
        else:
            metrics["learning_rate"] = self.lr

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True, device_type="cuda"):
            model_out = self.model(images)
            loss = self.loss_computer(model_out, labels)

        metrics.update(self.metric_computer(model_out=model_out, target=labels, loss=loss, is_nat=True))

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        metrics["total_loss"] = loss.cpu().detach().numpy()
        metrics["total_time"] = time.time() - start_time
        return metrics

    def _val_step(self, inputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            model_out = self.model(inputs)
            loss = self.loss_computer(model_out, labels)

        return self.metric_computer(model_out=model_out, target=labels, loss=loss, is_nat=True)
