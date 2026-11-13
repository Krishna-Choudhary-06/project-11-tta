"""Batch metrics for classification and test-time adaptation monitoring."""

import math
from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as F


def measure_time_to_collapse(
    records: Iterable[Mapping[str, object]],
    entropy_threshold: float = 0.20,
    dominant_threshold: float = 0.90,
    consecutive_batches: int = 5,
    entropy_key: str = "normalized_marginal_entropy",
    dominant_ratio_key: str = "dominant_class_ratio",
) -> int | None:
    """Return the first batch of a sustained prediction-collapse interval.

    A batch qualifies only when normalized marginal entropy is below
    ``entropy_threshold`` and the dominant class ratio is at least
    ``dominant_threshold``. Both conditions must hold for consecutive batch
    numbers. Return ``None`` if no such interval occurs.
    """
    if not 0 <= entropy_threshold <= 1:
        raise ValueError("entropy_threshold must be between 0 and 1.")
    if not 0 <= dominant_threshold <= 1:
        raise ValueError("dominant_threshold must be between 0 and 1.")
    if consecutive_batches <= 0:
        raise ValueError("consecutive_batches must be greater than zero.")

    streak = 0
    first_batch: int | None = None
    previous_batch: int | None = None
    for record in records:
        if "batch_number" in record:
            batch_number = int(record["batch_number"])
        elif "batch" in record:
            batch_number = int(record["batch"])
        else:
            raise KeyError("Each record must contain 'batch_number' or 'batch'.")

        try:
            entropy = float(record[entropy_key])
            dominant_ratio = float(record[dominant_ratio_key])
        except KeyError as error:
            raise KeyError(
                f"Each record must contain {entropy_key!r} and "
                f"{dominant_ratio_key!r}."
            ) from error
        if not math.isfinite(entropy) or not math.isfinite(dominant_ratio):
            raise ValueError("Collapse metrics must be finite numbers.")

        qualifies = (
            entropy < entropy_threshold
            and dominant_ratio >= dominant_threshold
        )
        if qualifies:
            if previous_batch is None or batch_number != previous_batch + 1:
                streak = 0
                first_batch = batch_number
            elif streak == 0:
                first_batch = batch_number
            streak += 1
            if streak >= consecutive_batches:
                return first_batch
        else:
            streak = 0
            first_batch = None

        previous_batch = batch_number

    return None


def calculate_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float | list[int]]:
    """Calculate accuracy, entropy, prediction counts, and class concentration.

    Labels are used only for accuracy. All remaining metrics are derived from
    the model logits and do not depend on labels.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shaped [batch, classes], got {logits.shape}.")
    if logits.size(0) == 0:
        raise ValueError("Cannot calculate metrics for an empty batch.")
    if logits.size(1) < 2:
        raise ValueError("Expected logits for at least two classes.")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must be floating-point values.")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values.")
    if labels.ndim != 1 or labels.size(0) != logits.size(0):
        raise ValueError(
            f"Expected labels shaped [{logits.size(0)}], got {labels.shape}."
        )
    if labels.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("labels must contain integer class indices.")

    logits = logits.detach()
    labels = labels.detach().to(device=logits.device)
    num_classes = logits.size(1)
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError(f"labels must be in the range [0, {num_classes - 1}].")

    probabilities = F.softmax(logits, dim=1)
    marginal_probs = probabilities.mean(dim=0)
    conditional_entropy = -(
        probabilities * torch.log(probabilities.clamp_min(1e-8))
    ).sum(dim=1).mean()
    marginal_entropy = -(
        marginal_probs * torch.log(marginal_probs.clamp_min(1e-8))
    ).sum()
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == labels).float().mean()
    class_counts = torch.bincount(predictions, minlength=num_classes)
    dominant_class_ratio = class_counts.max().float() / predictions.numel()
    max_entropy = math.log(num_classes)

    return {
        "accuracy": accuracy.item(),
        "conditional_entropy": conditional_entropy.item(),
        "marginal_entropy": marginal_entropy.item(),
        "normalized_marginal_entropy": (
            marginal_entropy / max_entropy
        ).item(),
        "dominant_class_ratio": dominant_class_ratio.item(),
        "class_counts": class_counts.cpu().tolist(),
        "marginal_probabilities": marginal_probs.cpu().tolist(),
    }
