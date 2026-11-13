"""Entropy calculations for test-time adaptation."""

import torch
import torch.nn.functional as F


def entropy_loss(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return the mean prediction entropy for a batch of class logits."""
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
    uniform_entropy = entropy_loss(uniform_logits)
    confident_entropy = entropy_loss(confident_logits)
    print(f"Entropy for [0.50, 0.50]: {uniform_entropy.item():.4f}")
    print(f"Entropy for [0.99, 0.01]: {confident_entropy.item():.4f}")
    print(f"Uniform prediction has higher entropy: {uniform_entropy > confident_entropy}")
