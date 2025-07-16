import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

from expressprint.models import BaseModel
from expressprint.utils import enable_vit_custom_block, plot_layer_ax_vit


class ActivationsAnalyzer:
    def __init__(self, model: BaseModel, dataset: Dataset, get_blocks_kwargs: dict[str, any] = {}) -> None:
        self.model_wrapper = model
        self.model = model.get_model()
        self.model_blocks = model.get_blocks(**get_blocks_kwargs)
        self.dataset = dataset

    def analyze(self, num_images: int = 100, plot_path: Optional[str] = None) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        for block_id, block in enumerate(self.model_blocks):
            enable_vit_custom_block(block, block_id)

        stats = []

        samples_idx = random.sample(range(len(self.dataset)), num_images)
        for img_idx in tqdm(samples_idx):
            images, _ = self.dataset[img_idx]
            images = images.unsqueeze(0).to(device)

            with torch.no_grad():
                _ = self.model(images)

            layer_stats_np = np.zeros((6, len(self.model_blocks)))
            for layer_id in range(len(self.model_blocks)):
                feat_abs = self.model_blocks[layer_id].feat.abs()
                sort_res = torch.sort(feat_abs.flatten(), descending=True)
                layer_stats_np[:5, layer_id] = sort_res.values.cpu()[:5]
                layer_stats_np[5, layer_id] = torch.median(feat_abs)

            stats.append(layer_stats_np)

        stats_mean = np.mean(stats, axis=0)
        top5_mean = stats_mean[:5, :].mean(axis=0)  # shape [num_blocks]
        diffs = np.diff(top5_mean)
        expressive_block = int(np.argmax(diffs) + 1)
        print(f"📌 Selected expressive block: {expressive_block} (based on top-5 activation change)")

        plot_layer_ax_vit(np.mean(stats, axis=0), self.model_wrapper.get_model_name(), plot_path)

        return expressive_block
