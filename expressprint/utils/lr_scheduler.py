import math
from typing import Dict, List, Union

import numpy as np
from torch.optim import Optimizer

PointDef = Dict[str, Union[float, int]]


# NOTE: _LRScheduler is a private class in PyTorch, therefore we cannot inherit
# from it and use its functionality.
class InterpolationLRScheduler:
    """A learning rate scheduler based on linear interpolation.

    :param optimizer: Optimizer to control. This class will repeatedly set the
        learning rate of optimizer's parameter groups.
    :param points: List of points for interpolation. Each point specifies the
        learning rate at some completed fraction of the training process, or
        alternatively, at the beginning of some training epoch. See Cifar-10 and
        ImageNet default.yml configs for examples.
    :param num_epochs: Number of training epochs.
    :param num_steps_per_epoch: Number of minibatch steps per each epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        points: List[PointDef],
        num_epochs: int,
        num_steps_per_epoch: int,
    ) -> None:
        self.optimizer = optimizer
        self.num_steps_per_epoch = num_steps_per_epoch

        self.point_to_epoch: List[float] = []
        self.point_to_lr: List[float] = []

        for point in points:
            if "fraction_completed" in point:
                epoch_with_fraction = point["fraction_completed"] * num_epochs
                self.point_to_epoch.append(epoch_with_fraction)
            else:
                self.point_to_epoch.append(point["epoch"])
            self.point_to_lr.append(point["lr"])

    def _interpolate(self, epoch_with_fraction) -> np.ndarray:
        return np.interp(epoch_with_fraction, self.point_to_epoch, self.point_to_lr)

    def adjust_learning_rate(self, epoch: int, step: int, scale: float = 1.0) -> float:
        # Some configurations set LR for the first point to 0.0, so add 1 to the
        # step (otherwise we will be stuck with initial weights forever)
        epoch_with_fraction = epoch + (step + 1) / self.num_steps_per_epoch
        lr = self._interpolate(epoch_with_fraction)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr * scale

        return lr


class CosineLRScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        num_epochs: int,
        lr_peak: float,
        lr_peak_epoch: int,
        num_steps_per_epoch: int,
    ) -> None:
        self.optimizer = optimizer
        self.lr_peak_epoch = lr_peak_epoch
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.lr_peak = lr_peak

    def get_lr(self, epoch) -> float:
        if epoch <= self.lr_peak_epoch:
            xs = [0, self.lr_peak_epoch]
            ys = [1e-4 * self.lr_peak, self.lr_peak]
            lr = np.interp([epoch], xs, ys)[0]
        else:
            lr_min = 5e-6
            lr = lr_min + 0.5 * (self.lr_peak - lr_min) * (
                1 + math.cos(math.pi * (epoch - self.lr_peak_epoch) / (self.num_epochs - self.lr_peak_epoch))
            )
        return lr

    def adjust_learning_rate(self, epoch: int, step: int) -> float:
        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)

        lr = np.interp(step, [0, self.num_steps_per_epoch], [lr_start, lr_end])

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
