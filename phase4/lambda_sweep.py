"""Run a controlled lambda sweep for marginal-entropy-corrected TENT."""

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from datasets.stream import (
    DEFAULT_ROOT,
    EXPERIMENT_SEQUENCES,
    create_experiment_loader,
)
from models.resnet import create_model
from phase2_tent.adapt import create_tent_optimizer
from phase2_tent.corrected_tent import adapt_corrected_batch_with_logits
from phase3.continual import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    run_continuous_tent,
)

DEFAULT_LAMBDAS = (0.0, 0.1, 0.5, 1.0, 2.0)


def run_lambda_sweep(
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    experiment: str = "changing",
    severity: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.00025,
    momentum: float = 0.9,
    root: Path = DEFAULT_ROOT,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_workers: int = 0,
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    """Run each lambda from a fresh copy of the source checkpoint."""
    if not lambdas:
        raise ValueError("At least one lambda value is required.")
    if any(value < 0 or not math.isfinite(value) for value in lambdas):
        raise ValueError("Lambda values must be finite and nonnegative.")
    if len(set(lambdas)) != len(lambdas):
        raise ValueError("Lambda values must be unique.")
    if experiment not in EXPERIMENT_SEQUENCES:
        raise ValueError(
            f"Unknown experiment {experiment!r}; choose from "
            f"{', '.join(EXPERIMENT_SEQUENCES)}."
        )
    if not 1 <= severity <= 5:
        raise ValueError("severity must be an integer from 1 to 5.")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative.")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive.")
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in the range [0, 1).")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Source checkpoint not found: {checkpoint}. Train the baseline first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for marginal_entropy_weight in lambdas:
        loader = create_experiment_loader(
            experiment=experiment,
            root=root,
            severity=severity,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        model = create_model(num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        optimizer = create_tent_optimizer(
            model,
            learning_rate=learning_rate,
            momentum=momentum,
        )
        output_path = output_dir / (
            f"phase4_{experiment}_lambda{marginal_entropy_weight:g}"
            f"_severity_{severity}_batch{batch_size}_lr{learning_rate:g}.csv"
        )
        adaptation_fn = lambda current_model, images, current_optimizer: (
            adapt_corrected_batch_with_logits(
                current_model,
                images,
                current_optimizer,
                lambda_value=marginal_entropy_weight,
            )
        )
        summary = run_continuous_tent(
            model=model,
            stream_loader=loader,
            optimizer=optimizer,
            device=device,
            output_path=output_path,
            experiment=f"phase4_{experiment}_lambda{marginal_entropy_weight:g}",
            learning_rate=learning_rate,
            max_batches=max_batches,
            experiment_metadata={
                "marginal_entropy_weight": marginal_entropy_weight,
                "condition": "conditional entropy minus weighted batch marginal entropy",
                "corruptions": list(loader.dataset.corruptions),
                "severity": severity,
            },
            adaptation_fn=adaptation_fn,
            objective_name="conditional_entropy_minus_weighted_marginal_entropy",
        )
        results.append(summary)
        print(
            f"lambda={marginal_entropy_weight:g}: "
            f"accuracy={summary['accuracy_percent']:.2f}%, "
            f"rolling collapse start={summary['rolling_collapse_start_batch']}, "
            f"CSV={summary['csv_path']}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TENT with marginal-entropy-corrected adaptation."
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="Nonnegative marginal-entropy weights; include 0 for TENT baseline.",
    )
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENT_SEQUENCES),
        default="changing",
    )
    parser.add_argument("--severity", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Optional per-lambda limit for a smoke test.",
    )
    args = parser.parse_args()
    run_lambda_sweep(
        lambdas=tuple(args.lambdas),
        experiment=args.experiment,
        severity=args.severity,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        root=args.root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
