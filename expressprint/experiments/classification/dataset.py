from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from expressprint.datasets import BaseDataLoader


class DataLoader(BaseDataLoader):
    def __init__(self, data_dir: str) -> None:
        super().__init__(Path(data_dir))

    def get_train_dataset(self, train_transforms: Optional[transforms.Compose] = None) -> Dataset:
        return ImageFolder(self.data_dir / "train", transform=train_transforms)

    def get_val_dataset(self, val_transforms: Optional[transforms.Compose] = None) -> Dataset:
        return ImageFolder(self.data_dir / "val", transform=val_transforms)
