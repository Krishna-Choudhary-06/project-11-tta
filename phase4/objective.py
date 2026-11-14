"""Compatibility wrapper for the Phase 4 corrected entropy objective."""

import torch

from phase2_tent.entropy import corrected_entropy_loss as calculate_corrected_entropy_loss


def corrected_entropy_loss(
    logits: torch.Tensor,
    marginal_entropy_weight: float = 1.0,
) -> torch.Tensor:
    """Return scalar H(Y|X) - weight * H(mean prediction probability)."""
    loss, _, _ = calculate_corrected_entropy_loss(
        logits,
        lambda_value=marginal_entropy_weight,
    )
    return loss
