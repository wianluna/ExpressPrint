from typing import List, Tuple

import timm
import torch.nn as nn
from torchvision.transforms import Compose

from expressprint.models.model_pattern import BaseModel

hidden_size = {
    "base": 768,
    "large": 1024,
    "giant": 1568,
}


class DINOv2Model(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__("DINOv2", **kwargs)

    def get_hidden_size(self, **kwargs) -> int:
        return hidden_size[kwargs["model_size"]]

    def build_model(self, **kwargs) -> None:
        return timm.create_model(
            f"vit_{kwargs['model_size']}_patch14_dinov2.lvd142m", pretrained=True, num_classes=1000
        )

    def get_blocks(self) -> List[nn.Module]:
        if not hasattr(self, "model"):
            self.build_model()
        return self.model.blocks

    def get_data_transforms(self) -> Tuple[Compose, Compose]:
        data_config = timm.data.resolve_model_data_config(self.model)
        train_transform = timm.data.create_transform(**data_config, is_training=True)
        val_transform = timm.data.create_transform(**data_config, is_training=False)
        return train_transform, val_transform
