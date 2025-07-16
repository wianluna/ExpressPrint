import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from expressprint.datasets import WatermarkDataLoader
from expressprint.methods.watermarking import Watermarker
from expressprint.models import BaseModel, WMDecoder, WMEncoder


class ExpressPrintWatermarker(Watermarker):
    def __init__(
        self,
        model: BaseModel,
        encoder: WMEncoder,
        decoder: WMDecoder,
        wm_block_idx: int = 12,
        wm_channel_idx: int = 46,
        wm_size: int = 32,
        wm_threshold: float = 0.5,
    ) -> None:
        super().__init__(model)
        self.encoder = encoder
        self.decoder = decoder

        self.encoder.eval()
        self.decoder.eval()

        model_blocks = model.get_blocks()
        self.wm_layer_decode: nn.Module = model_blocks[-1]
        self.wm_layer_encode: nn.Module = model_blocks[wm_block_idx]
        self.wm_channel_idx = wm_channel_idx

        self.wm_size = wm_size
        self.wm_threshold = wm_threshold

    def _hook_encode_wm(self, module: nn.Module, inputs: torch.Tensor, output: torch.Tensor):
        encoded_message = self.encoder(inputs[0][:, self.wm_channel_idx, :], self.binary_message.float())
        output[:, self.wm_channel_idx, :] = encoded_message

    def _hook_decode_wm(self, module, inputs, output):
        self.decoded_message = self.decoder(inputs[0][:, self.wm_channel_idx, :], threshold=self.wm_threshold)
        self.wm_feats = output

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Embed a watermark into the given model and return embeded message."""
        self.wm_encode_hook = self.wm_layer_encode.register_forward_hook(self._hook_encode_wm)
        self.wm_decode_hook = self.wm_layer_decode.register_forward_hook(self._hook_decode_wm)

        self.binary_message = torch.randint(0, 2, (images.size(0), self.wm_size)).float().to(self.device)
        _ = self.model.model(images)
        return self.binary_message

    def extract(self) -> torch.Tensor:
        """Extract a watermark from the given model."""
        self.wm_encode_hook.remove()
        self.wm_decode_hook.remove()

        return self.decoded_message

    def evaluate(self, test_loader: WatermarkDataLoader) -> None:
        all_bit_errors = []

        self.wm_encode_hook = self.wm_layer_encode.register_forward_hook(self._hook_encode_wm)
        self.wm_decode_hook = self.wm_layer_decode.register_forward_hook(self._hook_decode_wm)

        with torch.no_grad():
            for images, binary_message, _ in tqdm(test_loader, total=len(test_loader), desc="Testing Watermark"):
                images = images.to(self.device)
                binary_message = binary_message.to(self.device)
                self.binary_message = binary_message

                _ = self.model.model(images)

                decoded_message = self.decoded_message
                similarity = torch.sum((binary_message != decoded_message).float(), dim=1)
                all_bit_errors.extend(similarity.cpu().numpy())

        self.wm_encode_hook.remove()
        self.wm_decode_hook.remove()

        all_bit_errors = np.array(all_bit_errors)

        thresholds = np.arange(8)
        accuracies = [np.mean(all_bit_errors <= t) * 100 for t in thresholds]

        print("\nAccuracy by number of erroneous bits:")
        print("≤ Errors\tAccuracy")
        for t, acc in zip(thresholds, accuracies):
            print(f"≤ {t}\t\t{acc:.2f}%")

        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#323A48")

        ax.plot(
            thresholds,
            accuracies,
            color="mediumorchid",
            linestyle="-",
            marker="o",
            markersize=5,
            markerfacecolor="none",
        )
        acc_min, acc_max = min(accuracies), max(accuracies)

        dynamic_margin = min(2.5, (acc_max - acc_min) * 0.1)
        ax.set_ylim(acc_min - dynamic_margin, acc_max + dynamic_margin)

        ax.set_title("Watermark Decoding Accuracy", fontsize=18, fontweight="bold", color="white")
        ax.set_xlabel("Max Allowed Bit Errors", fontsize=18, labelpad=4.0, color="white")
        ax.set_ylabel("Accuracy (%)", fontsize=18, color="white")

        ax.set_xticks(thresholds)
        ax.set_xticklabels(thresholds, fontsize=16, color="white")
        ax.set_yticklabels(ax.get_yticks(), fontsize=16, color="white")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        ax.tick_params(axis="x", which="major", pad=2.0, colors="white")
        ax.tick_params(axis="y", which="major", pad=0.4, colors="white")

        ax.grid(axis="x", color="gray", linestyle="--", alpha=0.3)
        ax.grid(axis="y", color="gray", linestyle="--", alpha=0.3)

        ax.set_facecolor("#323A48")

        plt.show()
