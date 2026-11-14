"""Batch adaptation with the marginal-entropy-corrected TENT objective."""

import math

import torch
from torch import nn
from torch.optim import Optimizer

from phase2_tent.entropy import corrected_entropy_loss


def _gradient_norm(gradients: tuple[torch.Tensor | None, ...]) -> float:
    squared_sum = sum(
        gradient.detach().pow(2).sum().item()
        for gradient in gradients
        if gradient is not None
    )
    norm = math.sqrt(squared_sum)
    if not math.isfinite(norm):
        raise FloatingPointError("Corrected TENT produced a non-finite gradient norm.")
    return norm


def adapt_corrected_batch_with_logits(
    model: nn.Module,
    images: torch.Tensor,
    optimizer: Optimizer,
    lambda_value: float = 0.1,
) -> tuple[dict[str, float], torch.Tensor]:
    """Update configured TENT parameters using an unlabeled image batch.

    Create the optimizer with ``create_tent_optimizer`` so only the selected
    BatchNorm affine parameters are trainable. Labels must not be passed here.
    Return pre-update logits for evaluation and separate objective gradient
    norms for diagnosing the marginal entropy correction.
    """
    if images.ndim != 4 or images.size(0) == 0:
        raise ValueError("images must have shape [batch, channels, height, width].")
    if lambda_value < 0 or not math.isfinite(lambda_value):
        raise ValueError("lambda_value must be finite and nonnegative.")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss, conditional, marginal = corrected_entropy_loss(
        logits,
        lambda_value=lambda_value,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Corrected entropy loss is not finite; skipping TENT update."
        )

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("The model has no trainable parameters for corrected TENT.")
    conditional_gradients = torch.autograd.grad(
        conditional,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    marginal_gradients = torch.autograd.grad(
        -lambda_value * marginal,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    conditional_grad_norm = _gradient_norm(conditional_gradients)
    marginal_grad_norm = _gradient_norm(marginal_gradients)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for parameter_group in optimizer.param_groups:
        for parameter in parameter_group["params"]:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "Corrected TENT produced non-finite gradients; "
                    "optimizer update was skipped."
                )

    metrics = {
        "loss": loss.item(),
        "conditional_entropy": conditional.item(),
        "marginal_entropy": marginal.item(),
        "lambda": lambda_value,
        "conditional_grad_norm": conditional_grad_norm,
        "marginal_grad_norm": marginal_grad_norm,
    }
    pre_update_logits = logits.detach()
    optimizer.step()
    return metrics, pre_update_logits


def adapt_corrected_batch(
    model: nn.Module,
    images: torch.Tensor,
    optimizer: Optimizer,
    lambda_value: float = 0.1,
) -> dict[str, float]:
    """Adapt on one batch and return objective and gradient diagnostics."""
    metrics, _ = adapt_corrected_batch_with_logits(
        model=model,
        images=images,
        optimizer=optimizer,
        lambda_value=lambda_value,
    )
    return metrics
