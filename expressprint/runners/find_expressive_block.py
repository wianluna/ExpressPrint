from expressprint.datasets import ImageNetDataLoader
from expressprint.methods import ActivationsAnalyzer
from expressprint.models import create_vit

if __name__ == "__main__":
    model = create_vit(model_family="openai_clip", model_size="large")

    dataloader = ImageNetDataLoader(data_dir="../imagenet/ILSVRC/Data/CLS-LOC")
    train_transforms, val_transform = model.get_data_transforms()
    dataset = dataloader.get_val_dataset(val_transform)

    analyzer = ActivationsAnalyzer(model, dataset)
    expressive_block = analyzer.analyze(plot_path="./openai_clip_plot.png")
