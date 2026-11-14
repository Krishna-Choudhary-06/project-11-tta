"""Run one persistent TENT model and optimizer over an ordered image stream."""

import argparse
import csv
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.stream import (
    DEFAULT_ROOT,
    EXPERIMENT_SEQUENCES,
    create_experiment_loader,
    create_stream_loader,
)
from models.resnet import create_model
from phase2_tent.adapt import adapt_batch, create_tent_optimizer
from utils.metrics import calculate_metrics, measure_time_to_collapse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "checkpoints" / "resnet18_cifar10.pth"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "phase3_results"
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def run_continuous_tent(
    model: nn.Module,
    stream_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output_path: str | Path,
    experiment: str,
    learning_rate: float,
    max_batches: int | None = None,
    experiment_metadata: dict[str, Any] | None = None,
    adaptation_loss_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    adaptation_fn: Callable[
        [nn.Module, torch.Tensor, torch.optim.Optimizer],
        tuple[dict[str, float], torch.Tensor],
    ] | None = None,
    objective_name: str = "tent_conditional_entropy",
) -> dict[str, Any]:
    """Adapt one model continuously and save one pre-update metrics row per batch.

    Labels are used only after the unlabeled adaptation step has computed
    logits and updated the model; they are never passed to ``adapt_batch``.
    """
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be greater than zero when provided.")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and greater than zero.")
    if adaptation_fn is not None and adaptation_loss_fn is not None:
        raise ValueError("Provide adaptation_fn or adaptation_loss_fn, not both.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_correct = 0
    total_samples = 0
    adaptation_loss_sum = 0.0
    conditional_grad_norm_sum = 0.0
    marginal_grad_norm_sum = 0.0
    diagnostic_batches = 0
    started_at = time.perf_counter()
    batches_processed = 0
    stream_start = 0
    collapse_records: list[dict[str, float | int]] = []
    rolling_collapse_records: list[dict[str, float | int]] = []
    rolling_predictions: deque[list[int]] = deque(maxlen=5)
    rolling_probability_sums: deque[tuple[int, list[float]]] = deque(maxlen=5)
    fieldnames = [
        "experiment",
        "objective",
        "lambda",
        "marginal_entropy_weight",
        "batch_number",
        "stream_start_index",
        "stream_end_index",
        "corruption",
        "severity",
        "batch_size",
        "adaptation_loss",
        "conditional_grad_norm",
        "marginal_grad_norm",
        "conditional_entropy",
        "accuracy_percent",
        "marginal_entropy",
        "normalized_marginal_entropy",
        "dominant_class_ratio",
        "rolling_window_batches",
        "rolling_window_samples",
        "rolling_marginal_entropy",
        "rolling_normalized_marginal_entropy",
        "rolling_dominant_class_ratio",
        "learning_rate",
        *(f"mean_probability_{name}" for name in CLASS_NAMES),
        *(f"predicted_{name}" for name in CLASS_NAMES),
    ]

    dataset = stream_loader.dataset
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for batch_number, (images, labels) in enumerate(stream_loader, start=1):
            if max_batches is not None and batch_number > max_batches:
                break

            stream_end = stream_start + labels.size(0)
            images = images.to(device)
            labels = labels.to(device)

            # adapt_batch returns logits from this forward pass, before step().
            # The optimizer and model persist across every loop iteration.
            adaptation_metrics: dict[str, float] = {}
            if adaptation_fn is not None:
                adaptation_metrics, logits = adaptation_fn(
                    model,
                    images,
                    optimizer,
                )
                adaptation_loss = adaptation_metrics["loss"]
            else:
                adaptation_loss, logits = adapt_batch(
                    model,
                    images,
                    optimizer,
                    loss_fn=adaptation_loss_fn,
                )
            metrics = calculate_metrics(logits, labels)
            correct = metrics["accuracy"] * labels.size(0)
            batch_size = labels.size(0)
            prediction_counts = metrics["class_counts"]
            marginal_probabilities = metrics["marginal_probabilities"]
            rolling_predictions.append(prediction_counts)
            rolling_probability_sums.append(
                (batch_size, marginal_probabilities)
            )
            rolling_samples = sum(
                sample_count for sample_count, _ in rolling_probability_sums
            )
            rolling_marginal = [
                sum(
                    sample_count * probabilities[class_index]
                    for sample_count, probabilities in rolling_probability_sums
                )
                / rolling_samples
                for class_index in range(len(CLASS_NAMES))
            ]
            rolling_entropy = -sum(
                probability * math.log(max(probability, 1e-8))
                for probability in rolling_marginal
            )
            rolling_normalized_entropy = rolling_entropy / math.log(
                len(CLASS_NAMES)
            )
            rolling_class_counts = [
                sum(counts[class_index] for counts in rolling_predictions)
                for class_index in range(len(CLASS_NAMES))
            ]
            rolling_dominant_ratio = max(rolling_class_counts) / rolling_samples

            corruption = (
                dataset.corruption_at(stream_start)
                if hasattr(dataset, "corruption_at")
                else "unknown"
            )
            severity = getattr(dataset, "severity", "")
            row: dict[str, Any] = {
                "experiment": experiment,
                "objective": objective_name,
                "lambda": adaptation_metrics.get(
                    "lambda",
                    (experiment_metadata or {}).get(
                        "marginal_entropy_weight", ""
                    ),
                ),
                "marginal_entropy_weight": (
                    (experiment_metadata or {}).get(
                        "marginal_entropy_weight", ""
                    )
                ),
                "batch_number": batch_number,
                "stream_start_index": stream_start,
                "stream_end_index": stream_end - 1,
                "corruption": corruption,
                "severity": severity,
                "batch_size": batch_size,
                "adaptation_loss": adaptation_loss,
                "conditional_grad_norm": adaptation_metrics.get(
                    "conditional_grad_norm", ""
                ),
                "marginal_grad_norm": adaptation_metrics.get(
                    "marginal_grad_norm", ""
                ),
                "conditional_entropy": metrics["conditional_entropy"],
                "accuracy_percent": 100.0 * metrics["accuracy"],
                "marginal_entropy": metrics["marginal_entropy"],
                "normalized_marginal_entropy": (
                    metrics["normalized_marginal_entropy"]
                ),
                "dominant_class_ratio": metrics["dominant_class_ratio"],
                "rolling_window_batches": len(rolling_probability_sums),
                "rolling_window_samples": rolling_samples,
                "rolling_marginal_entropy": rolling_entropy,
                "rolling_normalized_marginal_entropy": (
                    rolling_normalized_entropy
                ),
                "rolling_dominant_class_ratio": rolling_dominant_ratio,
                "learning_rate": learning_rate,
            }
            row.update(
                {
                    f"mean_probability_{name}": probability
                    for name, probability in zip(
                        CLASS_NAMES, marginal_probabilities
                    )
                }
            )
            row.update(
                {
                    f"predicted_{name}": count
                    for name, count in zip(CLASS_NAMES, prediction_counts)
                }
            )
            writer.writerow(row)
            csv_file.flush()
            collapse_records.append(
                {
                    "batch_number": batch_number,
                    "stream_end_index": stream_end - 1,
                    "normalized_marginal_entropy": metrics[
                        "normalized_marginal_entropy"
                    ],
                    "dominant_class_ratio": metrics["dominant_class_ratio"],
                    "rolling_normalized_marginal_entropy": (
                        rolling_normalized_entropy
                    ),
                    "rolling_dominant_class_ratio": rolling_dominant_ratio,
                }
            )
            if len(rolling_probability_sums) == rolling_probability_sums.maxlen:
                rolling_collapse_records.append(
                    {
                        "batch_number": batch_number,
                        "stream_end_index": stream_end - 1,
                        "normalized_marginal_entropy": (
                            rolling_normalized_entropy
                        ),
                        "dominant_class_ratio": rolling_dominant_ratio,
                    }
                )

            total_correct += correct
            total_samples += batch_size
            adaptation_loss_sum += adaptation_loss * batch_size
            if "conditional_grad_norm" in adaptation_metrics:
                conditional_grad_norm_sum += adaptation_metrics[
                    "conditional_grad_norm"
                ]
                marginal_grad_norm_sum += adaptation_metrics[
                    "marginal_grad_norm"
                ]
                diagnostic_batches += 1
            batches_processed += 1
            stream_start = stream_end

    if not batches_processed:
        raise ValueError("The stream loader produced no batches.")

    elapsed_seconds = time.perf_counter() - started_at
    collapse_start_batch = measure_time_to_collapse(collapse_records)
    rolling_collapse_start_batch = measure_time_to_collapse(
        rolling_collapse_records
    )

    def collapse_details(start_batch: int | None) -> dict[str, int | None]:
        if start_batch is None:
            return {
                "start_batch": None,
                "images_processed_at_start": None,
                "confirmation_batch": None,
                "images_processed_at_confirmation": None,
            }
        confirmation_batch = start_batch + 4
        start_record = collapse_records[start_batch - 1]
        confirmation_record = collapse_records[confirmation_batch - 1]
        return {
            "start_batch": start_batch,
            "images_processed_at_start": (
                int(start_record["stream_end_index"]) + 1
            ),
            "confirmation_batch": confirmation_batch,
            "images_processed_at_confirmation": (
                int(confirmation_record["stream_end_index"]) + 1
            ),
        }

    batch_collapse = collapse_details(collapse_start_batch)
    rolling_collapse = collapse_details(rolling_collapse_start_batch)
    summary = {
        "experiment": experiment,
        "objective": objective_name,
        "batches_processed": batches_processed,
        "images_processed": total_samples,
        "accuracy_percent": 100.0 * total_correct / total_samples,
        "mean_adaptation_loss": adaptation_loss_sum / total_samples,
        "mean_conditional_grad_norm": (
            conditional_grad_norm_sum / diagnostic_batches
            if diagnostic_batches
            else None
        ),
        "mean_marginal_grad_norm": (
            marginal_grad_norm_sum / diagnostic_batches
            if diagnostic_batches
            else None
        ),
        "lambda": (experiment_metadata or {}).get("marginal_entropy_weight"),
        "elapsed_seconds": elapsed_seconds,
        "learning_rate": learning_rate,
        "experiment_metadata": experiment_metadata or {},
        "collapse_start_batch": batch_collapse["start_batch"],
        "images_processed_at_collapse_start": (
            batch_collapse["images_processed_at_start"]
        ),
        "collapse_confirmation_batch": batch_collapse["confirmation_batch"],
        "images_processed_at_collapse_confirmation": (
            batch_collapse["images_processed_at_confirmation"]
        ),
        "rolling_collapse_start_batch": rolling_collapse["start_batch"],
        "images_processed_at_rolling_collapse_start": (
            rolling_collapse["images_processed_at_start"]
        ),
        "rolling_collapse_confirmation_batch": (
            rolling_collapse["confirmation_batch"]
        ),
        "images_processed_at_rolling_collapse_confirmation": (
            rolling_collapse["images_processed_at_confirmation"]
        ),
        "collapse_rule": {
            "normalized_marginal_entropy_below": 0.20,
            "dominant_class_ratio_at_least": 0.90,
            "consecutive_batches": 5,
            "rolling_window_batches": 5,
        },
        "csv_path": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously adapt TENT over an ordered CIFAR-10-C stream."
    )
    experiment_group = parser.add_mutually_exclusive_group()
    experiment_group.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENT_SEQUENCES),
        default="changing",
        help="Predefined static or changing corruption stream.",
    )
    experiment_group.add_argument(
        "--corruptions",
        nargs="+",
        help="Custom ordered corruption list, e.g. gaussian_noise motion_blur fog.",
    )
    parser.add_argument("--severity", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Optional limit for a short smoke run; omit for the full stream.",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Source checkpoint not found: {args.checkpoint}. Train the baseline first."
        )
    if args.corruptions:
        stream_name = "custom_" + "_".join(args.corruptions)
        stream_loader = create_stream_loader(
            root=args.root,
            corruptions=args.corruptions,
            severity=args.severity,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        stream_name = args.experiment
        stream_loader = create_experiment_loader(
            experiment=args.experiment,
            root=args.root,
            severity=args.severity,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(CLASS_NAMES)).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    # Configure once. Keep this optimizer and its momentum for the entire stream.
    optimizer = create_tent_optimizer(
        model,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )
    output_path = args.output or (
        DEFAULT_OUTPUT_DIR
        / f"{stream_name}_severity_{args.severity}_lr{args.learning_rate:g}.csv"
    )

    print(f"Experiment: {stream_name}")
    print(f"Corruption order: {', '.join(stream_loader.dataset.corruptions)}")
    print(f"Severity: {args.severity}")
    print(f"Images in stream: {len(stream_loader.dataset)}")
    print(f"Device: {device}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Source checkpoint loaded once: {args.checkpoint}")
    summary = run_continuous_tent(
        model=model,
        stream_loader=stream_loader,
        optimizer=optimizer,
        device=device,
        output_path=output_path,
        experiment=stream_name,
        learning_rate=args.learning_rate,
        max_batches=args.max_batches,
    )
    for name, value in summary.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
