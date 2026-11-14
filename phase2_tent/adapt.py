"""Unsupervised test-time adaptation helpers."""

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.optim import SGD, Optimizer

from phase2_tent.entropy import entropy_loss
from phase2_tent.tent import configure_tent


def create_tent_optimizer(
    model: nn.Module,
    learning_rate: float = 0.00025,
    momentum: float = 0.9,
) -> Optimizer:
    """Configure TENT and optimize only BatchNorm affine parameters."""
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be a finite positive number.")
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in the range [0, 1).")

    configure_tent(model)
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("The model has no trainable BatchNorm affine parameters.")

    return SGD(parameters, lr=learning_rate, momentum=momentum)


def adapt_batch(
    model: nn.Module,
    images: torch.Tensor,
    optimizer: Optimizer,
    loss_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[float, torch.Tensor]:
    """Adapt on an unlabeled batch and return loss and pre-update logits.

    The returned logits are captured before the optimizer update, allowing
    evaluation metrics for this batch to follow the same prediction protocol.
    Labels must not be passed to this function.
    """
    if images.ndim != 4 or images.size(0) == 0:
        raise ValueError("images must have shape [batch, channels, height, width].")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = (loss_fn or entropy_loss)(logits)
    if not torch.isfinite(loss):
        raise FloatingPointError("Entropy loss is not finite; skipping TENT update.")

    loss.backward()
    for parameter_group in optimizer.param_groups:
        for parameter in parameter_group["params"]:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "TENT produced non-finite gradients; optimizer update was skipped."
                )

    pre_update_logits = logits.detach()
    loss_value = loss.item()
    optimizer.step()
    return loss_value, pre_update_logits
