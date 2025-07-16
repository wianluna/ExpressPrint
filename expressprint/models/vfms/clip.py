from typing import List, Tuple

import timm
import torch.nn as nn
from torchvision.transforms import Compose

from expressprint.models.model_pattern import BaseModel

hidden_size = {
    "large": 1024,
}


class CLIPModel(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__("OpenAI CLIP", **kwargs)
        self.kwargs = kwargs

    def build_model(self, **kwargs) -> None:
        patch_size = 14 if kwargs["model_size"] == "large" else 16
        return timm.create_model(f"vit_{kwargs['model_size']}_patch{patch_size}_clip_224.openai", pretrained=True)

    def get_hidden_size(self) -> int:
        return hidden_size[self.kwargs["model_size"]]

    def get_blocks(self) -> List[nn.Module]:
        return self.model.blocks

    def get_data_transforms(self) -> Tuple[Compose, Compose]:
        data_config = timm.data.resolve_model_data_config(self.model)
        train_transform = timm.data.create_transform(**data_config, is_training=True)
        val_transform = timm.data.create_transform(**data_config, is_training=False)
        return train_transform, val_transform
