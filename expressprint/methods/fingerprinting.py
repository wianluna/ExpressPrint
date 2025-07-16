from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Fingerprinter(ABC):
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def extract(self, model: nn.Module, **kwargs) -> None:
        """Create a fingerprint from the given model."""
        raise NotImplementedError

    @abstractmethod
    def verify(self, model: nn.Module, **kwargs) -> None:
        """Verify the fingerprint of the given model."""
        raise NotImplementedError
