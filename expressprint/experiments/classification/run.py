import shutil
from argparse import ArgumentParser
from pathlib import Path

from expressprint.experiments.classification import DataLoader, Trainer, create_model
from expressprint.models import create_vit
from expressprint.utils.config import load_config

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
    if "model_path" in config:
        model.load_model(config["model_path"])

    classifier = create_model(model=model.get_model(), num_classes=config["model"]["num_classes"])

    train_transforms, val_transform = model.get_data_transforms()

    dataloader = DataLoader(**config["data"])
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

    trainer = Trainer(
        classifier,
        log_path=log_path,
        num_classes=config["model"]["num_classes"],
        num_epochs=config["training"]["num_epochs"],
        optimizer_type=config["optimizer"]["type"],
        lr=config["optimizer"]["lr"],
        lr_scheduler_type=config["lr_scheduler"]["type"],
        lr_scheduler_params=config["lr_scheduler"]["params"],
        verbose=True,
    )

    trainer.train(train_loader, val_loader)
