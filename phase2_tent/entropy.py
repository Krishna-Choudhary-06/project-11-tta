"""Entropy calculations for test-time adaptation."""

import math

import torch
import torch.nn.functional as F


def _validate_logits(logits: torch.Tensor, eps: float) -> None:
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits [batch, classes], got {logits.shape}.")
    if logits.size(0) == 0:
        raise ValueError("Cannot calculate entropy for an empty batch.")
    if logits.size(1) < 2:
        raise ValueError("Expected logits for at least two classes.")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must be floating-point values.")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values.")
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and greater than zero.")


def _conditional_entropy(probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    entropy = -(
        probabilities * torch.log(probabilities.clamp_min(eps))
    ).sum(dim=1)
    return entropy.mean()


def _marginal_entropy(probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    marginal_probabilities = probabilities.mean(dim=0)
    return -(
        marginal_probabilities
        * torch.log(marginal_probabilities.clamp_min(eps))
    ).sum()


def calculate_conditional_entropy(
    logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return mean per-example predictive entropy H(Y|X)."""
    _validate_logits(logits, eps)
    probabilities = F.softmax(logits, dim=1)
    return _conditional_entropy(probabilities, eps)


def calculate_marginal_entropy(
    logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return entropy of the batch-mean class probabilities H(mean p(Y|X))."""
    _validate_logits(logits, eps)
    probabilities = F.softmax(logits, dim=1)
    return _marginal_entropy(probabilities, eps)


def corrected_entropy_loss(
    logits: torch.Tensor,
    lambda_value: float = 0.1,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return corrected loss, conditional entropy, and marginal entropy.

    The marginal probabilities remain attached to the computation graph so
    the diversity term contributes gradients to the model.
    """
    _validate_logits(logits, eps)
    if lambda_value < 0 or not math.isfinite(lambda_value):
        raise ValueError("lambda_value must be finite and nonnegative.")

    probabilities = F.softmax(logits, dim=1)
    conditional = _conditional_entropy(probabilities, eps)
    marginal = _marginal_entropy(probabilities, eps)
    loss = conditional - lambda_value * marginal
    return loss, conditional, marginal


def entropy_loss(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return the original TENT mean prediction entropy."""
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits [batch, classes], got {logits.shape}.")
    if eps <= 0:
        raise ValueError("eps must be greater than zero.")

    probabilities = F.softmax(logits, dim=1)
    entropy = -torch.sum(
        probabilities * torch.log(probabilities + eps),
        dim=1,
    )
    return entropy.mean()


if __name__ == "__main__":
    uniform_logits = torch.tensor([[0.0, 0.0]])
    confident_logits = torch.log(torch.tensor([[0.99, 0.01]]))
    uniform_entropy = calculate_conditional_entropy(uniform_logits)
    confident_entropy = calculate_conditional_entropy(confident_logits)
    print(f"Entropy for [0.50, 0.50]: {uniform_entropy.item():.4f}")
    print(f"Entropy for [0.99, 0.01]: {confident_entropy.item():.4f}")
    print(f"Uniform prediction has higher entropy: {uniform_entropy > confident_entropy}")
