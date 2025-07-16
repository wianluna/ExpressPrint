from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from expressprint.datasets.dataset_pattern import BaseDataLoader


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset to return (image, label, image_path) instead of just (image, label)."""

    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]  # self.imgs contains a list of (path, label)
        return original_tuple + (path,)


class ImageNetDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str) -> None:
        super().__init__(Path(data_dir))

    def _make_dataset(self, split: str, transform: Optional[transforms.Compose], return_path: bool) -> Dataset:
        split_path = self.data_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_path}")

        if return_path:
            return ImageFolderWithPaths(split_path, transform=transform)
        else:
            return ImageFolder(split_path, transform=transform)

    def get_train_dataset(
        self, train_transforms: Optional[transforms.Compose] = None, return_path: bool = False
    ) -> Dataset:
        return self._make_dataset("train", train_transforms, return_path=return_path)

    def get_val_dataset(
        self, val_transforms: Optional[transforms.Compose] = None, return_path: bool = False
    ) -> Dataset:
        return self._make_dataset("val", val_transforms, return_path=return_path)
