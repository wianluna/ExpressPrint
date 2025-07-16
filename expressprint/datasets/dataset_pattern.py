"""
Base abstract class for creating data loaders with standardized train and validation dataset retrieval methods.

This class provides a template for implementing custom data loaders with consistent interfaces for
obtaining training and validation datasets and data loaders. Subclasses must implement
get_train_dataset() and get_val_dataset() methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BaseDataLoader(ABC):
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        """
        Initialize the BaseDataLoader.

        :param data_dir: Directory path for dataset storage.
        """
        self.data_dir = data_dir

    @abstractmethod
    def get_train_dataset(self, train_transforms: Optional[transforms.Compose] = None, **kwargs) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_val_dataset(self, val_transforms: Optional[transforms.Compose] = None, **kwargs) -> Dataset:
        raise NotImplementedError

    def get_train_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        train_transforms: Optional[transforms.Compose] = None,
        **kwargs,
    ) -> DataLoader:
        return DataLoader(
            self.get_train_dataset(train_transforms=train_transforms, **kwargs),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    def get_val_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = False,
        val_transforms: Optional[transforms.Compose] = None,
        **kwargs,
    ) -> DataLoader:
        return DataLoader(
            self.get_val_dataset(val_transforms=val_transforms, **kwargs),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
