import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from expressprint.models import BaseModel
from expressprint.utils import add_metrics_dict, divide_metrics, dump_scalar_metrics


class ExpressPrintTrainer:
    def __init__(
        self,
        model: BaseModel,
        wm_encoder: nn.Module,
        wm_decoder: nn.Module,
        wm_block_idx: int,
        log_path: str,
        wm_channel_idx: int = 46,
        wm_size: int = 32,
        num_epochs: int = 5,
        optimizer_type: str = "adamw",
        lr: float = 0.00001,
        wm_lr_factor: float = 10,
        wm_threshold: float = 0.5,
        verbose: bool = False,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = Path(log_path)
        self.verbose = verbose

        self.num_epochs = num_epochs
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.wm_lr_factor = wm_lr_factor

        self.model_wrapper = model
        self.model = self.model_wrapper.get_model()
        self.model.to(self.device)
        self.model_copy_wrapper = copy.deepcopy(self.model_wrapper)
        self.model_copy = self.model_copy_wrapper.get_model()
        self.model_copy.to(self.device)
        self.scaler = GradScaler()

        # Prepare watermark models
        self.wm_encoder = wm_encoder
        self.wm_decoder = wm_decoder

        self.wm_encoder.to(self.device)
        self.wm_decoder.to(self.device)

        model_blocks = model.get_blocks()
        self.wm_layer_decode: nn.Module = model_blocks[-1]
        self.wm_layer_encode: nn.Module = model_blocks[wm_block_idx]
        self.wm_channel_idx = wm_channel_idx

        # freeze layers before watermark layer
        for param in self.model.parameters():
            param.requires_grad = False

        for layer in model_blocks[wm_block_idx + 1 : -1]:
            for param in layer.parameters():
                param.requires_grad = True

        self.wm_size = wm_size
        self.wm_threshold = wm_threshold

    def __del__(self) -> None:
        self.writer.close()

    def _hook_encode_wm(self, module: nn.Module, inputs: torch.Tensor, output: torch.Tensor):
        encoded_message = self.wm_encoder(inputs[0][:, self.wm_channel_idx, :], self.binary_message.float())
        self.encoded_message = encoded_message  # for test dataset generation with wm injection into images
        output[:, self.wm_channel_idx, :] = encoded_message

    def _hook_decode_wm(self, module, inputs, output):
        self.decoded_message = self.wm_decoder(inputs[0][:, self.wm_channel_idx, :], threshold=self.wm_threshold)
        self.wm_feats = output

    def _hook_feats(self, module, inputs, output):
        self.feats = output

    def _init_logger(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        print(f"=> Logging in {self.log_dir}")

    def _save_checkpoint(self) -> None:
        torch.save(self.model.state_dict(), self.log_dir / "model.pth")
        torch.save(self.wm_encoder.state_dict(), self.log_dir / "wm_encoder.pth")
        torch.save(self.wm_decoder.state_dict(), self.log_dir / "wm_decoder.pth")

    def _init_optimizer(self) -> None:
        model_trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        param_groups = [
            {
                "params": model_trainable_params,
                "lr": self.lr,
            },
            {
                "params": self.wm_encoder.parameters(),
                "lr": self.lr * self.wm_lr_factor,
            },
            {
                "params": self.wm_decoder.parameters(),
                "lr": self.lr * self.wm_lr_factor,
            },
        ]
        if self.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(param_groups)
        else:
            raise NotImplementedError(f"Unsupported optimizer type: {self.optimizer_type}")

    def _init_lr_scheduler(self) -> None:
        self.lr_scheduler = None

    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> None:
        self._prepare_for_training(train_loader, val_loader)
        self._train_loop()

        self.prepare_testset(test_loader)

        self.wm_encode_hook.remove()
        self.wm_decode_hook.remove()
        self.feats_hook.remove()

        return self.model_wrapper, self.wm_encoder, self.wm_decoder

    def _prepare_for_training(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self._init_logger()
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.wm_encode_hook = self.wm_layer_encode.register_forward_hook(self._hook_encode_wm)
        self.wm_decode_hook = self.wm_layer_decode.register_forward_hook(self._hook_decode_wm)

        model_copy_blocks = self.model_copy_wrapper.get_blocks()
        self.feats_hook = model_copy_blocks[-1].register_forward_hook(self._hook_feats)

        self.cur_epoch = 0
        self.end_epoch = self.num_epochs

        self._init_optimizer()
        self._init_lr_scheduler()

        self.loss_computer = nn.CrossEntropyLoss()

    def _train_loop(self) -> None:
        train_data_len = 500
        val_data_len = len(self.val_loader)

        while self.cur_epoch < self.end_epoch:
            done_steps = self.cur_epoch * train_data_len
            self.model.train()
            self.model_copy.train()
            self.wm_encoder.train()
            self.wm_decoder.train()

            for step, (images, labels) in enumerate(self.train_loader):
                metrics = self._train_step(images, labels)

                dump_scalar_metrics(metrics, self.writer, phase="train", global_step=done_steps + step)

                print(
                    f"TRAIN epoch: {self.cur_epoch}, step: {step} of {train_data_len}, "
                    f"total_loss: {metrics['total_loss']}",
                    end="\r",
                )

                if step == 499:
                    break

            self.model.eval()
            self.model_copy.eval()
            self.wm_encoder.eval()
            self.wm_decoder.eval()
            metrics = {}

            for step, (images, labels) in enumerate(self.val_loader):
                metrics_new = self._val_step(images, labels)
                metrics = add_metrics_dict(metrics=metrics, metrics_new=metrics_new)

                print(
                    f"TEST epoch: {self.cur_epoch}, step: {step} of {val_data_len}, loss: {metrics_new['total_loss']}",
                    end="\r",
                )

                if step == 9:
                    break

            metrics = divide_metrics(metrics, n=100)
            dump_scalar_metrics(metrics, self.writer, phase="val", global_step=self.cur_epoch)

            self.cur_epoch += 1

        self._save_checkpoint()
        # cmd = [sys.executable, "-m", "deepbreach.experiments.eval_wm", "--model-path", self.log_dir]
        # print(cmd)
        # subprocess.run(cmd)

    def _train_step(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        images, labels = images.to(self.device), labels.to(self.device)
        metrics = {}

        with autocast(enabled=True, device_type="cuda"):
            self.binary_message = torch.randint(0, 2, (images.size(0), self.wm_size), device=self.device).float()
            _ = self.model(images)
            _ = self.model_copy(images)

            feats_loss = nn.functional.mse_loss(self.feats, self.wm_feats)
            wm_loss = nn.functional.mse_loss(self.binary_message, self.decoded_message)
            loss = feats_loss + wm_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        metrics.update(
            {
                "loss_feats": feats_loss.item(),
                "loss_wm": wm_loss.item(),
                "total_loss": loss.item(),
            }
        )
        return metrics

    def _val_step(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        images, labels = images.to(self.device), labels.to(self.device)
        metrics = {}

        with torch.no_grad():
            self.binary_message = torch.randint(0, 2, (images.size(0), self.wm_size), device=self.device).float()

            _ = self.model(images)
            _ = self.model_copy(images)

            feats_loss = nn.functional.mse_loss(self.feats, self.wm_feats)
            wm_loss = nn.functional.mse_loss(self.binary_message, self.decoded_message)
            loss = feats_loss + wm_loss

            if self.verbose:
                print("Message:", self.binary_message[0])
                print("Decoded message:", self.decoded_message[0])
                print("Difference:", torch.abs(self.binary_message[0] - self.decoded_message[0]))

        metrics.update(
            {
                "loss_feats": feats_loss.cpu().detach().numpy(),
                "loss_wm": wm_loss.cpu().detach().numpy(),
                "total_loss": loss.cpu().detach().numpy(),
            }
        )
        return metrics

    def prepare_testset(self, test_loader: DataLoader, num_samples: int = 1000) -> None:
        self.model.eval()
        self.wm_encoder.eval()
        self.wm_decoder.eval()

        successful_samples = []
        error_count = 0
        progress_bar = tqdm(total=num_samples, desc="Preparing Test Set", unit="samples")

        with torch.no_grad():
            for images, labels, paths in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                binary_messages = torch.randint(0, 2, (images.size(0), self.wm_size), device=self.device).float()

                self.binary_message = binary_messages
                _ = self.model(images)
                decoded_messages = self.decoded_message

                for img_path, original_msg, decoded_msg, label in zip(paths, binary_messages, decoded_messages, labels):
                    if torch.equal(original_msg, decoded_msg):
                        successful_samples.append(
                            {
                                "image_name": str(Path(img_path).resolve()),
                                "binary_message": original_msg.cpu().tolist(),
                                "target": label.item(),
                            }
                        )

                        progress_bar.update(1)
                    else:
                        if self.verbose:
                            print(
                                f"Error: Original message and decoded message do not match in {torch.abs(original_msg - decoded_msg).sum()} bits"
                            )
                        error_count += 1

                    if len(successful_samples) >= num_samples:
                        progress_bar.close()
                        if self.verbose:
                            print(f"Error count: {error_count}")
                        break

                if len(successful_samples) >= num_samples:
                    break

            output_path = self.log_dir / "wm_test_set.json"
            with open(output_path, "w") as f:
                json.dump(successful_samples, f, indent=4)

        print(f"✅ Saved {len(successful_samples)} samples to: {output_path}")
