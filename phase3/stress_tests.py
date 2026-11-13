"""Run controlled stress tests for continual TENT prediction collapse."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.stream import DEFAULT_ROOT, create_stream_loader
from models.resnet import create_model
from phase2_tent.adapt import create_tent_optimizer
from phase3.continual import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    run_continuous_tent,
)

LEARNING_RATES = (0.00025, 0.0005, 0.001, 0.002)
BATCH_SIZES = (8, 16, 32, 64)
REPEATED_NOISE_SEGMENTS = 10
IMBALANCED_CLASS = 0
IMBALANCED_CLASS_PROPORTION = 0.70
BIASED_CLASS = 0
CLASSIFIER_BIAS = 3.0
DEFAULT_STREAM_LENGTH = 30_000


class _SampledStream(Dataset[tuple[torch.Tensor, int]]):
    """Expose sampled source examples as a new sequential stream."""

    def __init__(self, dataset: Dataset, sampled_indices: np.ndarray) -> None:
        self.dataset = dataset
        self.sampled_indices = sampled_indices
        self.severity = getattr(dataset, "severity", None)

    def __len__(self) -> int:
        return len(self.sampled_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.dataset[int(self.sampled_indices[index])]

    def corruption_at(self, index: int) -> str:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return "gaussian_noise"


def _create_imbalanced_loader(
    root: Path,
    severity: int,
    batch_size: int,
    num_workers: int,
    stream_length: int,
    seed: int,
) -> tuple[DataLoader, dict[str, Any]]:
    """Sample a fixed-length Gaussian-noise stream with a 70% class-0 prior."""
    if stream_length <= 0:
        raise ValueError("stream_length must be greater than zero.")

    base_loader = create_stream_loader(
        root=root,
        corruptions=("gaussian_noise",),
        severity=severity,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dataset = base_loader.dataset
    labels = np.asarray(dataset.labels)
    class_indices = [np.flatnonzero(labels == index) for index in range(len(CLASS_NAMES))]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("The selected CIFAR-10-C severity block is missing a class.")

    target_counts = np.full(
        len(CLASS_NAMES),
        (stream_length * (1.0 - IMBALANCED_CLASS_PROPORTION))
        / (len(CLASS_NAMES) - 1),
        dtype=np.float64,
    )
    target_counts[IMBALANCED_CLASS] = (
        stream_length * IMBALANCED_CLASS_PROPORTION
    )
    counts = np.floor(target_counts).astype(np.int64)
    counts[-1] += stream_length - int(counts.sum())

    generator = np.random.default_rng(seed)
    sampled_indices = np.concatenate(
        [
            generator.choice(indices, size=int(count), replace=True)
            for indices, count in zip(class_indices, counts)
        ]
    )
    generator.shuffle(sampled_indices)
    loader = DataLoader(
        _SampledStream(dataset, sampled_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    metadata = {
        "stream_length": stream_length,
        "target_class": CLASS_NAMES[IMBALANCED_CLASS],
        "target_class_proportion": IMBALANCED_CLASS_PROPORTION,
        "class_counts": {
            CLASS_NAMES[index]: int(count)
            for index, count in enumerate(counts)
        },
        "sampling_seed": seed,
    }
    return loader, metadata


def _run_one(
    *,
    name: str,
    corruptions: tuple[str, ...],
    severity: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    root: Path,
    checkpoint: Path,
    output_dir: Path,
    num_workers: int,
    max_batches: int | None,
    device: torch.device,
    metadata: dict[str, Any] | None = None,
    bias_class: int | None = None,
    classifier_bias: float = CLASSIFIER_BIAS,
    imbalance_length: int = DEFAULT_STREAM_LENGTH,
    seed: int = 0,
) -> dict[str, Any]:
    if name == "class_imbalance":
        stream_loader, imbalance_metadata = _create_imbalanced_loader(
            root=root,
            severity=severity,
            batch_size=batch_size,
            num_workers=num_workers,
            stream_length=imbalance_length,
            seed=seed,
        )
        metadata = {**(metadata or {}), **imbalance_metadata}
    else:
        stream_loader = create_stream_loader(
            root=root,
            corruptions=corruptions,
            severity=severity,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    model = create_model(num_classes=len(CLASS_NAMES)).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    if bias_class is not None:
        if not 0 <= bias_class < len(CLASS_NAMES):
            raise ValueError(
                f"bias_class must be between 0 and {len(CLASS_NAMES) - 1}."
            )
        if not math.isfinite(classifier_bias):
            raise ValueError("classifier_bias must be finite.")
        with torch.no_grad():
            model.fc.bias.zero_()
            model.fc.bias[bias_class] = classifier_bias

    optimizer = create_tent_optimizer(
        model,
        learning_rate=learning_rate,
        momentum=momentum,
    )
    output_path = output_dir / (
        f"{name}_severity_{severity}_batch{batch_size}_lr{learning_rate:g}.csv"
    )
    summary = run_continuous_tent(
        model=model,
        stream_loader=stream_loader,
        optimizer=optimizer,
        device=device,
        output_path=output_path,
        experiment=name,
        learning_rate=learning_rate,
        max_batches=max_batches,
        experiment_metadata=metadata,
    )
    return summary


def update_master_results(output_dir: Path) -> Path:
    """Write a master table from complete experiment CSVs and summaries."""
    experiments = [
        ("Baseline", "Current LR", "static_severity_3_lr0.00025"),
        (
            "LR sweep",
            "0.0005",
            "learning_rate_0.0005_severity_3_batch32_lr0.0005",
        ),
        (
            "LR sweep",
            "0.001",
            "learning_rate_0.001_severity_3_batch32_lr0.001",
        ),
        (
            "LR sweep",
            "0.002",
            "learning_rate_0.002_severity_3_batch32_lr0.002",
        ),
        (
            "Biased model",
            "Airplane (classifier bias +3.0)",
            "biased_initialization_severity_3_batch32_lr0.00025",
        ),
        (
            "Imbalanced",
            "70% airplane",
            "class_imbalance_severity_3_batch32_lr0.00025",
        ),
        (
            "Repeated noise",
            "10 passes",
            "repeated_gaussian_noise_10x_severity_3_batch32_lr0.00025",
        ),
        *(
            (
                "Batch-size sweep",
                str(size),
                f"batch_size_{size}_severity_3_batch{size}_lr0.00025",
            )
            for size in BATCH_SIZES
        ),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "master_results.csv"
    rows = []
    for experiment, parameter, stem in experiments:
        csv_path = output_dir / f"{stem}.csv"
        summary_path = output_dir / f"{stem}.summary.json"
        if not csv_path.is_file() or not summary_path.is_file():
            continue
        with summary_path.open(encoding="utf-8") as file:
            summary = json.load(file)
        with csv_path.open(newline="", encoding="utf-8") as file:
            metric_rows = list(csv.DictReader(file))
        complete_windows = [
            row
            for row in metric_rows
            if int(row["rolling_window_batches"]) == 5
        ]
        if not complete_windows:
            continue
        rows.append(
            {
                "Experiment": experiment,
                "Parameter": parameter,
                "Accuracy (%)": f"{summary['accuracy_percent']:.2f}",
                "Rolling entropy min": (
                    f"{min(float(row['rolling_normalized_marginal_entropy']) for row in complete_windows):.4f}"
                ),
                "Dominant ratio max": (
                    f"{max(float(row['rolling_dominant_class_ratio']) for row in complete_windows):.4f}"
                ),
                "Collapse": (
                    "Yes"
                    if summary["rolling_collapse_start_batch"] is not None
                    else "No"
                ),
            }
        )

    fieldnames = (
        "Experiment",
        "Parameter",
        "Accuracy (%)",
        "Rolling entropy min",
        "Dominant ratio max",
        "Collapse",
    )
    with master_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return master_path


def run_stress_suite(
    stress: str,
    *,
    root: Path = DEFAULT_ROOT,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    severity: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.00025,
    momentum: float = 0.9,
    num_workers: int = 0,
    max_batches: int | None = None,
    imbalance_length: int = DEFAULT_STREAM_LENGTH,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run one or more stress conditions, reloading the source model each time."""
    valid_stresses = {
        "repeated-noise",
        "repeated-gaussian",
        "learning-rate-sweep",
        "class-imbalance",
        "biased-initialization",
        "batch-size-sweep",
        "all",
    }
    if stress not in valid_stresses:
        raise ValueError(
            f"Unknown stress test {stress!r}; choose from: "
            f"{', '.join(sorted(valid_stresses))}."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Source checkpoint not found: {checkpoint}. Train the baseline first."
        )
    if not 1 <= severity <= 5:
        raise ValueError("severity must be an integer from 1 to 5.")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative.")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs: list[dict[str, Any]] = []
    if stress in ("repeated-noise", "repeated-gaussian", "all"):
        configs.append(
            {
                "name": "repeated_gaussian_noise_10x",
                "corruptions": ("gaussian_noise",) * REPEATED_NOISE_SEGMENTS,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "metadata": {
                    "condition": "ten consecutive Gaussian-noise severity blocks",
                    "segments": REPEATED_NOISE_SEGMENTS,
                },
            }
        )
    if stress in ("learning-rate-sweep", "all"):
        configs.extend(
            {
                "name": f"learning_rate_{rate:g}",
                "corruptions": ("gaussian_noise",) * 3,
                "batch_size": batch_size,
                "learning_rate": rate,
                "metadata": {"condition": "learning-rate sweep"},
            }
            for rate in LEARNING_RATES
        )
    if stress in ("class-imbalance", "all"):
        configs.append(
            {
                "name": "class_imbalance",
                "corruptions": ("gaussian_noise",),
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "metadata": {"condition": "controlled class-prior imbalance"},
            }
        )
    if stress in ("biased-initialization", "all"):
        configs.append(
            {
                "name": "biased_initialization",
                "corruptions": ("gaussian_noise",) * 3,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "metadata": {
                    "condition": "source classifier output bias overwritten",
                    "biased_class": CLASS_NAMES[BIASED_CLASS],
                    "classifier_bias": CLASSIFIER_BIAS,
                },
                "bias_class": BIASED_CLASS,
            }
        )
    if stress in ("batch-size-sweep", "all"):
        configs.extend(
            {
                "name": f"batch_size_{size}",
                "corruptions": ("gaussian_noise",) * 3,
                "batch_size": size,
                "learning_rate": learning_rate,
                "metadata": {"condition": "batch-size sweep"},
            }
            for size in BATCH_SIZES
        )

    summaries = []
    for config in configs:
        summary = _run_one(
            **config,
            severity=severity,
            momentum=momentum,
            root=root,
            checkpoint=checkpoint,
            output_dir=output_dir,
            num_workers=num_workers,
            max_batches=max_batches,
            device=device,
            imbalance_length=imbalance_length,
            seed=seed,
        )
        summaries.append(summary)
        print(
            f"{summary['experiment']}: accuracy={summary['accuracy_percent']:.2f}%, "
            f"rolling collapse start={summary['rolling_collapse_start_batch']}, "
            f"CSV={summary['csv_path']}"
        )
    master_path = update_master_results(output_dir)
    print(f"Master results table: {master_path}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled continual-TENT prediction-collapse stress tests."
    )
    parser.add_argument(
        "--stress",
        required=True,
        choices=(
            "repeated-noise",
            "repeated-gaussian",
            "learning-rate-sweep",
            "class-imbalance",
            "biased-initialization",
            "batch-size-sweep",
            "all",
        ),
    )
    parser.add_argument("--severity", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imbalance-length", type=int, default=DEFAULT_STREAM_LENGTH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Optional per-run limit for smoke testing; omit for full experiments.",
    )
    args = parser.parse_args()
    run_stress_suite(
        args.stress,
        root=args.root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        severity=args.severity,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        imbalance_length=args.imbalance_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
