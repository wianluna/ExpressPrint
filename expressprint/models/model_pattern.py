"""Base model class for standardized model construction and serialization."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.transforms import Compose


class BaseModel(ABC):
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.model = self.build_model(**kwargs)

    @abstractmethod
    def build_model(self, **kwargs) -> nn.Module:
        raise NotImplementedError("Subclasses must implement build_model method")

    @abstractmethod
    def get_hidden_size(self) -> int:
        raise NotImplementedError("Subclasses must implement get_hidden_size method")

    @abstractmethod
    def get_blocks(self) -> List[nn.Module]:
        raise NotImplementedError("Subclasses must implement get_blocks method")

    @abstractmethod
    def get_data_transforms(self) -> Tuple[Compose, Compose]:
        raise NotImplementedError("Subclasses must implement get_data_transforms method")

    def load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint)

    def save_model(self, model_path: str) -> None:
        model_path = Path(model_path)
        model_path.parennt.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def get_model(self) -> nn.Module:
        return self.model

    def get_model_name(self) -> str:
        return self.model_name
