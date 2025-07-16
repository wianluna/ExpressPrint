from typing import Any, Dict, Optional

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import accuracy


def sparse_to_dense_target(sparse_target: Tensor) -> Tensor:
    """
    If to calculate accuracy for a sparse target, then the multi-label approach will be applied.
    This is not what we expect.
    https://github.com/PyTorchLightning/metrics/issues/554
    """
    if len(sparse_target.shape) == 2:
        return sparse_target.argmax(axis=1)
    else:
        return sparse_target


class ComputeMetrics:
    def __init__(self, top_k: int, num_classes: int) -> None:
        self.top_k = top_k
        self.num_classes = num_classes

    def __call__(
        self, model_out: Tensor, target: Tensor, loss: Optional[Tensor] = None, is_nat: bool = True
    ) -> Dict[str, Tensor]:
        metrics = {}
        prefix = "nat" if is_nat else "adv"

        image_prediction = model_out.cpu()

        if loss is not None:
            metrics[f"loss_{prefix}"] = loss.cpu()
        metrics[f"acc_{prefix}"] = accuracy(
            image_prediction,
            sparse_to_dense_target(target).cpu(),
            task="multiclass",
            num_classes=self.num_classes,
            top_k=1,
        )
        if self.top_k > 1:
            metrics[f"acc{self.top_k}_{prefix}"] = accuracy(
                image_prediction,
                sparse_to_dense_target(target).cpu(),
                task="multiclass",
                num_classes=self.num_classes,
                top_k=self.top_k,
            )

        return metrics


def dump_scalar_metrics(metrics: Dict[str, Any], writer: SummaryWriter, phase: str, global_step: int) -> None:
    prefix = phase.lower()
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(
            f"{metric_name}/{prefix}",
            metric_value,
            global_step=global_step,
        )


def add_metrics_dict(metrics: Dict[str, Any], metrics_new: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in metrics_new.items():
        metrics[key] = metrics.get(key, 0) + value
    return metrics


def divide_metrics(metrics: Dict[str, Any], n: int) -> Dict[str, Any]:
    return {key: (value / n).item() for key, value in metrics.items()}
