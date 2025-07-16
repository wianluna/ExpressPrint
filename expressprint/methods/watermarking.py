from abc import ABC, abstractmethod

import torch

from expressprint.models import BaseModel


class Watermarker(ABC):
    def __init__(self, model: BaseModel) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.model.to(self.device)
        self.model.model.eval()

    @abstractmethod
    def embed(self, **kwargs) -> None:
        """Embed a watermark into the given model."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, **kwargs) -> None:
        """Extract a watermark from the given model."""
        raise NotImplementedError
