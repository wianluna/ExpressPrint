"""
Dataset and DataLoader for watermarking data.

This module provides a custom PyTorch Dataset (WatermarksDataset) and DataLoader (WatermarkDataLoader)
for loading watermark images with associated binary messages and targets.

The dataset loads image paths, binary messages, and targets from a JSON file, and supports optional
image transformations during data loading.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from expressprint.datasets.dataset_pattern import BaseDataLoader


class WatermarkDataset(Dataset):
    def __init__(self, json_path: Path, transforms: Optional[transforms.Compose] = None) -> None:
        """
        Initialize the WatermarksDataset.

        :param json_path: Path to the JSON file containing dataset metadata
        :param transforms: Image transformations to apply
        """
        self.json_path = json_path
        self.transform = transforms

        with open(self.json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        image_path = item["image_name"]
        binary_message = torch.tensor(item["binary_message"], dtype=torch.float32)
        target = torch.tensor(item["target"], dtype=torch.long)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, binary_message, target


class WatermarkDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, json_path: str) -> None:
        """
        Initialize the WatermarkDataLoader.

        :param data_dir: Directory path for dataset storage
        :param json_path: Path to the JSON file containing dataset metadata
        """
        super().__init__(Path(data_dir))
        self.json_path = Path(json_path)

    def get_train_dataset(self, train_transforms: Optional[transforms.Compose] = None) -> Dataset:
        return WatermarkDataset(self.json_path, transforms=train_transforms)

    def get_val_dataset(self, val_transforms: Optional[transforms.Compose] = None) -> Dataset:
        return WatermarkDataset(self.json_path, transforms=val_transforms)
