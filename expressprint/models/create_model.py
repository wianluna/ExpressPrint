from expressprint.models.model_pattern import BaseModel
from expressprint.models.vfms import CLIPModel, DINOv2Model
from expressprint.models.wm_models import WMDecoder, WMEncoder


def create_vit(model_family: str, **kwargs) -> BaseModel:
    if model_family == "openai_clip":
        return CLIPModel(**kwargs)
    elif model_family == "dinov2":
        return DINOv2Model(**kwargs)
    else:
        raise NotImplementedError


def create_wm_models(
    watermark_size: int, feature_dim: int, encoder_hidden_dim: int = 1024, decoder_hidden_dim: int = 512
) -> tuple[WMEncoder, WMDecoder]:
    """
    Create watermark encoder and decoder models with separate hidden dimensions.

    :param watermark_size: Length of the binary watermark vector (e.g., 32).
    :param feature_dim: Dimensionality of the ViT hidden features (e.g., 768 or 1024).
    :param encoder_hidden_dim: Hidden layer size inside the encoder.
    :param decoder_hidden_dim: Hidden layer size inside the decoder.
    :return: Tuple of (WMEncoder, WMDecoder) instances.
    """
    encoder = WMEncoder(watermark_size=watermark_size, feature_dim=feature_dim, hidden_dim=encoder_hidden_dim)
    decoder = WMDecoder(watermark_size=watermark_size, feature_dim=feature_dim, hidden_dim=decoder_hidden_dim)
    return encoder, decoder
