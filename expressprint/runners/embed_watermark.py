import shutil
from argparse import ArgumentParser
from pathlib import Path

from expressprint.datasets import ImageNetDataLoader
from expressprint.models import WMDecoder, WMEncoder, create_vit
from expressprint.trainers.wm_trainer import ExpressPrintTrainer
from expressprint.utils import load_config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise RuntimeError(f"Cannot find config file {args.config}")
    config = load_config(config_path)

    log_path: Path = Path(config["log"]["directory"]).resolve() / config["log"]["experiment"]
    log_path.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, log_path / "presets.yaml")

    model = create_vit(**config["model"])
    wm_encoder = WMEncoder(**config["watermark"]["encoder"])
    wm_decoder = WMDecoder(**config["watermark"]["decoder"])

    trainer = ExpressPrintTrainer(
        model=model,
        wm_encoder=wm_encoder,
        wm_decoder=wm_decoder,
        wm_block_idx=config["watermark"]["block_idx"],
        wm_channel_idx=config["watermark"]["channel_idx"],
        log_path=log_path,
        verbose=True,
    )

    dataloader = ImageNetDataLoader(**config["data"])
    train_transforms, val_transform = model.get_data_transforms()

    train_loader = dataloader.get_train_loader(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        train_transforms=train_transforms,
    )
    val_loader = dataloader.get_val_loader(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        val_transforms=val_transform,
    )

    test_loader = dataloader.get_val_loader(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        val_transforms=val_transform,
        return_path=True,
    )

    trainer.train(train_loader, val_loader, test_loader)
