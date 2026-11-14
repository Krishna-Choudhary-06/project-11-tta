"""Run Phase 4 lambda/learning-rate experiments and compare their metrics."""

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from datasets.stream import DEFAULT_ROOT, create_experiment_loader
from models.resnet import create_model
from phase2_tent.adapt import create_tent_optimizer
from phase2_tent.corrected_tent import adapt_corrected_batch_with_logits
from phase3.continual import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    run_continuous_tent,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "phase4_results"
LAMBDA_VALUES = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
LEARNING_RATES = (0.00001, 0.00005, 0.0001, 0.00025, 0.0005)
DEFAULT_LEARNING_RATE = 0.00025
DEFAULT_BATCH_SIZE = 32
DEFAULT_MOMENTUM = 0.9
EXPERIMENTS = {
    "focused-grid": (
        (0.0, 0.0001),
        (0.1, 0.0001),
        (0.5, 0.0001),
        (0.1, 0.00005),
        (0.1, 0.00025),
        (0.5, 0.00025),
    ),
}


def _experiment_configs(mode: str) -> tuple[tuple[float, float], ...]:
    if mode == "lambda-sweep":
        return tuple((value, DEFAULT_LEARNING_RATE) for value in LAMBDA_VALUES)
    if mode == "focused-grid":
        return EXPERIMENTS[mode]
    if mode == "full-grid":
        return tuple(
            (value, rate)
            for value in LAMBDA_VALUES
            for rate in LEARNING_RATES
        )
    raise ValueError(f"Unknown experiment mode: {mode}")


def _summarize_csv(
    csv_path: Path,
    run_summary: dict[str, Any],
    lambda_value: float,
    learning_rate: float,
) -> dict[str, Any]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Experiment produced no batch records: {csv_path}")

    complete_windows = [
        row for row in rows if int(row["rolling_window_batches"]) == 5
    ]
    if not complete_windows:
        raise ValueError(
            f"Experiment needs at least five batches for rolling metrics: {csv_path}"
        )

    return {
        "experiment": run_summary["experiment"],
        "lambda": lambda_value,
        "learning_rate": learning_rate,
        "batch_size": int(rows[0]["batch_size"]),
        "accuracy_percent": float(run_summary["accuracy_percent"]),
        "mean_marginal_entropy": sum(
            float(row["marginal_entropy"]) for row in rows
        )
        / len(rows),
        "mean_conditional_grad_norm": sum(
            float(row["conditional_grad_norm"]) for row in rows
        )
        / len(rows),
        "mean_marginal_grad_norm": sum(
            float(row["marginal_grad_norm"]) for row in rows
        )
        / len(rows),
        "mean_rolling_marginal_entropy": sum(
            float(row["rolling_marginal_entropy"])
            for row in complete_windows
        )
        / len(complete_windows),
        "minimum_rolling_normalized_marginal_entropy": min(
            float(row["rolling_normalized_marginal_entropy"])
            for row in complete_windows
        ),
        "maximum_rolling_dominant_class_ratio": max(
            float(row["rolling_dominant_class_ratio"])
            for row in complete_windows
        ),
        "batch_collapse_start": run_summary["collapse_start_batch"],
        "rolling_collapse_start": run_summary["rolling_collapse_start_batch"],
        "images_processed": int(run_summary["images_processed"]),
        "csv_path": str(csv_path),
        "summary_path": str(csv_path.with_suffix(".summary.json")),
    }


def _load_rows(path: str) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in ("batch_number", "batch_size", "accuracy_percent"):
            row[field] = float(row[field])
        for name in CLASS_NAMES:
            row[f"predicted_{name}"] = int(row[f"predicted_{name}"])
    return rows


def _plot_comparisons(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    mode: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [(summary, _load_rows(summary["csv_path"])) for summary in summaries]

    entropy_path = output_dir / f"{mode}_marginal_entropy_comparison.png"
    fig, axis = plt.subplots(figsize=(11, 6))
    for summary, rows in records:
        label = f"lambda={summary['lambda']:g}, lr={summary['learning_rate']:g}"
        axis.plot(
            [row["batch_number"] for row in rows],
            [float(row["rolling_marginal_entropy"]) for row in rows],
            linewidth=1.2,
            label=label,
        )
    axis.set_xlabel("Batch number")
    axis.set_ylabel("Five-batch rolling marginal entropy (nats)")
    axis.set_title("Phase 4: marginal entropy by lambda and learning rate")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(entropy_path, dpi=150)
    plt.close(fig)

    accuracy_path = output_dir / f"{mode}_accuracy_comparison.png"
    fig, axis = plt.subplots(figsize=(11, 6))
    for summary, rows in records:
        label = f"lambda={summary['lambda']:g}, lr={summary['learning_rate']:g}"
        axis.plot(
            [row["batch_number"] for row in rows],
            [row["accuracy_percent"] for row in rows],
            linewidth=1.0,
            label=label,
        )
    axis.set_xlabel("Batch number")
    axis.set_ylabel("Batch accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Phase 4: accuracy by lambda and learning rate")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(accuracy_path, dpi=150)
    plt.close(fig)

    class_distribution_path = output_dir / f"{mode}_class_distribution.png"
    columns = 2
    rows_count = math.ceil(len(records) / columns)
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(15, 4.5 * rows_count),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for axis, (summary, batch_rows) in zip(axes.flat, records):
        batches = [int(row["batch_number"]) for row in batch_rows]
        class_shares = [
            [
                100.0
                * int(row[f"predicted_{name}"])
                / int(row["batch_size"])
                for name in CLASS_NAMES
            ]
            for row in batch_rows
        ]
        axis.stackplot(
            batches,
            *zip(*class_shares),
            labels=CLASS_NAMES,
            alpha=0.85,
        )
        axis.set_title(
            f"lambda={summary['lambda']:g}, lr={summary['learning_rate']:g}"
        )
        axis.set_ylim(0, 100)
        axis.grid(True, axis="y", alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("Batch number")
    for axis in axes[:, 0]:
        axis.set_ylabel("Predicted class (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1, 0.5),
        fontsize="small",
    )
    fig.suptitle("Phase 4: predicted class distribution by experiment")
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    fig.savefig(class_distribution_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    lambda_accuracy_path = output_dir / f"{mode}_lambda_vs_accuracy.png"
    fig, axis = plt.subplots(figsize=(9, 6))
    learning_rates = sorted({float(summary["learning_rate"]) for summary in summaries})
    for rate in learning_rates:
        rate_summaries = sorted(
            (
                summary
                for summary in summaries
                if float(summary["learning_rate"]) == rate
            ),
            key=lambda summary: float(summary["lambda"]),
        )
        axis.plot(
            [float(summary["lambda"]) for summary in rate_summaries],
            [float(summary["accuracy_percent"]) for summary in rate_summaries],
            marker="o",
            label=f"lr={rate:g}",
        )
    axis.set_xlabel("Marginal entropy weight (lambda)")
    axis.set_ylabel("Overall accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Phase 4: accuracy-diversity trade-off")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(lambda_accuracy_path, dpi=150)
    plt.close(fig)

    return {
        "marginal_entropy_plot": str(entropy_path),
        "accuracy_plot": str(accuracy_path),
        "class_distribution_plot": str(class_distribution_path),
        "lambda_vs_accuracy_plot": str(lambda_accuracy_path),
    }


def _write_summary_table(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    mode: str,
) -> Path:
    summary_path = output_dir / f"{mode}_summary.csv"
    fieldnames = [
        "experiment",
        "lambda",
        "learning_rate",
        "batch_size",
        "accuracy_percent",
        "mean_marginal_entropy",
        "mean_conditional_grad_norm",
        "mean_marginal_grad_norm",
        "mean_rolling_marginal_entropy",
        "minimum_rolling_normalized_marginal_entropy",
        "maximum_rolling_dominant_class_ratio",
        "batch_collapse_start",
        "rolling_collapse_start",
        "images_processed",
        "csv_path",
        "summary_path",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return summary_path


def run_experiments(
    mode: str = "focused-grid",
    experiment: str = "changing",
    severity: int = 3,
    batch_size: int = DEFAULT_BATCH_SIZE,
    momentum: float = DEFAULT_MOMENTUM,
    root: Path = DEFAULT_ROOT,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_workers: int = 0,
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    """Run each requested setting from a freshly loaded source checkpoint."""
    configurations = _experiment_configs(mode)
    if len(set(configurations)) != len(configurations):
        raise ValueError("Experiment configurations must not contain duplicates.")
    if not 1 <= severity <= 5:
        raise ValueError("severity must be an integer from 1 to 5.")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative.")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Source checkpoint not found: {checkpoint}. Train the baseline first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, (lambda_value, learning_rate) in enumerate(configurations, start=1):
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
            f"{mode}_{experiment}_e{index:02d}_lambda{lambda_value:g}"
            f"_lr{learning_rate:g}_severity{severity}_batch{batch_size}.csv"
        )
        adaptation_fn = lambda current_model, images, current_optimizer: (
            adapt_corrected_batch_with_logits(
                current_model,
                images,
                current_optimizer,
                lambda_value=lambda_value,
            )
        )
        print(
            f"Starting {mode} experiment {index}/{len(configurations)}: "
            f"lambda={lambda_value:g}, learning_rate={learning_rate:g}, "
            f"checkpoint={checkpoint}"
        )
        run_summary = run_continuous_tent(
            model=model,
            stream_loader=loader,
            optimizer=optimizer,
            device=device,
            output_path=output_path,
            experiment=(
                f"{mode}_{experiment}_lambda{lambda_value:g}_lr{learning_rate:g}"
            ),
            learning_rate=learning_rate,
            max_batches=max_batches,
            experiment_metadata={
                "marginal_entropy_weight": lambda_value,
                "learning_rate": learning_rate,
                "corruptions": list(loader.dataset.corruptions),
                "severity": severity,
                "batch_size": batch_size,
                "mode": mode,
            },
            adaptation_fn=adaptation_fn,
            objective_name="conditional_entropy_minus_weighted_marginal_entropy",
        )
        summaries.append(
            _summarize_csv(
                output_path,
                run_summary,
                lambda_value=lambda_value,
                learning_rate=learning_rate,
            )
        )

    summary_path = _write_summary_table(summaries, output_dir, mode)
    plot_paths = _plot_comparisons(summaries, output_dir, mode)
    for summary in summaries:
        print(
            f"lambda={summary['lambda']:g}, "
            f"lr={summary['learning_rate']:g}: "
            f"accuracy={summary['accuracy_percent']:.2f}%, "
            f"mean marginal entropy={summary['mean_marginal_entropy']:.4f}, "
            f"rolling collapse={summary['rolling_collapse_start']}"
        )
    print(f"Summary table: {summary_path}")
    for name, path in plot_paths.items():
        print(f"{name}: {path}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 lambda and learning-rate comparisons."
    )
    parser.add_argument(
        "--mode",
        choices=("lambda-sweep", "focused-grid", "full-grid"),
        default="focused-grid",
        help=(
            "Six requested lambda/LR pairs by default; lambda-sweep tests all "
            "six lambda values at LR 0.00025; full-grid tests every pair."
        ),
    )
    parser.add_argument(
        "--experiment",
        choices=("static", "changing"),
        default="changing",
    )
    parser.add_argument("--severity", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Optional per-experiment limit for a smoke test.",
    )
    args = parser.parse_args()
    run_experiments(
        mode=args.mode,
        experiment=args.experiment,
        severity=args.severity,
        batch_size=args.batch_size,
        momentum=args.momentum,
        root=args.root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
