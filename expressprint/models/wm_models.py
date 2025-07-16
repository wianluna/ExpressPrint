from typing import Optional

import torch
import torch.nn as nn


class WMEncoder(nn.Module):
    def __init__(self, watermark_size: int = 32, feature_dim: int = 1024, hidden_dim: int = 1024) -> None:
        """
        :param watermark_size: Size of the binary watermark vector to be embedded.
        :param feature_dim: Dimensionality of the input feature (e.g., ViT hidden size).
        :param hidden_dim: Size of internal hidden representation in the encoder.
        """
        super().__init__()
        self.fc1 = nn.Linear(feature_dim + watermark_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, features: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        :param features: Input feature tensor from expressive layer (B, D)
        :param message: Binary watermark message (B, M)
        :return: Modified feature tensor (B, D)
        """
        combined = torch.cat([features, message], dim=-1)
        x = torch.relu(self.fc1(combined))
        encoded = torch.sigmoid(self.fc2(x))
        return encoded


class WMDecoder(nn.Module):
    def __init__(self, watermark_size: int = 32, feature_dim: int = 1024, hidden_dim: int = 512) -> None:
        """
        :param watermark_size: Length of the binary watermark.
        :param feature_dim: Dimensionality of the input feature (e.g., ViT hidden size).
        :param hidden_dim: Size of internal hidden representation in the decoder.
        """
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, watermark_size)

    def forward(self, encoded_features: torch.Tensor, threshold: Optional[float] = 0.5) -> torch.Tensor:
        """
        :param encoded_features: Output features from downstream layers (B, D)
        :param threshold: If not training, binarize output with this threshold
        :return: Decoded message (float or binary int tensor)
        """
        x = torch.relu(self.fc1(encoded_features))
        decoded_message = torch.sigmoid(self.fc2(x))

        if self.training or threshold is None:
            return decoded_message

        return (decoded_message > threshold).int()
