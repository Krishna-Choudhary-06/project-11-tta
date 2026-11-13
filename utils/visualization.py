"""Plot continual TENT metrics and report sustained prediction collapse."""

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from utils.metrics import measure_time_to_collapse

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


def load_batch_records(csv_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate batch metric rows from a Phase 3 CSV."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Phase 3 metrics CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "batch_number",
            "accuracy_percent",
            "marginal_entropy",
            "rolling_marginal_entropy",
            "rolling_normalized_marginal_entropy",
            "rolling_dominant_class_ratio",
            "rolling_window_batches",
            "batch_size",
            *(f"predicted_{name}" for name in CLASS_NAMES),
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required metric columns: "
                f"{', '.join(sorted(missing))}."
            )
        records = list(reader)

    if not records:
        raise ValueError(f"No batch metric rows found in {path}.")

    for row in records:
        for field in (
            "batch_number",
            "accuracy_percent",
            "marginal_entropy",
            "rolling_marginal_entropy",
            "rolling_normalized_marginal_entropy",
            "rolling_dominant_class_ratio",
            "rolling_window_batches",
            "batch_size",
            *(f"predicted_{name}" for name in CLASS_NAMES),
        ):
            try:
                value = float(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {field!r} value in {path}: {row[field]!r}."
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {field!r} value in {path}.")
            row[field] = value
        for field in (
            "normalized_marginal_entropy",
            "dominant_class_ratio",
            "rolling_normalized_marginal_entropy",
            "rolling_dominant_class_ratio",
        ):
            if field in row and row[field] != "":
                row[field] = float(row[field])
            elif field == "normalized_marginal_entropy":
                row[field] = (
                    float(row["marginal_entropy"]) / math.log(len(CLASS_NAMES))
                )
            elif field == "dominant_class_ratio":
                predicted_total = sum(
                    float(row[f"predicted_{name}"]) for name in CLASS_NAMES
                )
                if predicted_total != int(row["batch_size"]):
                    raise ValueError(
                        f"Prediction counts in {path} do not sum to batch_size "
                        f"for batch {int(row['batch_number'])}."
                    )
                row[field] = max(
                    float(row[f"predicted_{name}"]) for name in CLASS_NAMES
                ) / predicted_total
            else:
                raise ValueError(
                    f"{path} lacks exact five-batch rolling metrics. "
                    "Rerun `python -m phase3.continual` to generate a CSV "
                    "with rolling metrics and per-class mean probabilities."
                )
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field!r} value in {path}.")
    return records


def plot_phase3_results(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    entropy_threshold: float = 0.20,
    dominant_threshold: float = 0.90,
    consecutive_batches: int = 5,
) -> dict[str, Any]:
    """Save entropy, accuracy, and predicted-class-distribution plots.

    Returns the plot paths and the first batch satisfying the sustained
    collapse rule, if any.
    """
    records = load_batch_records(csv_path)
    source_path = Path(csv_path)
    destination = Path(output_dir) if output_dir else source_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    batches = [int(row["batch_number"]) for row in records]

    batch_collapse_start = measure_time_to_collapse(
        records,
        entropy_threshold=entropy_threshold,
        dominant_threshold=dominant_threshold,
        consecutive_batches=consecutive_batches,
    )
    complete_rolling_records = [
        row for row in records if int(row["rolling_window_batches"]) == 5
    ]
    rolling_collapse_start = measure_time_to_collapse(
        complete_rolling_records,
        entropy_threshold=entropy_threshold,
        dominant_threshold=dominant_threshold,
        consecutive_batches=consecutive_batches,
        entropy_key="rolling_normalized_marginal_entropy",
        dominant_ratio_key="rolling_dominant_class_ratio",
    )
    batch_collapse_confirmation = (
        batch_collapse_start + consecutive_batches - 1
        if batch_collapse_start is not None
        else None
    )
    rolling_collapse_confirmation = (
        rolling_collapse_start + consecutive_batches - 1
        if rolling_collapse_start is not None
        else None
    )

    def images_processed(batch_number: int | None) -> int | None:
        if batch_number is None:
            return None
        return sum(
            int(row["batch_size"])
            for row in records
            if int(row["batch_number"]) <= batch_number
        )

    entropy_path = destination / f"{stem}_marginal_entropy.png"
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        batches,
        [float(row["marginal_entropy"]) for row in records],
        linewidth=1,
        alpha=0.55,
        label="Batch-wise marginal entropy",
    )
    axis.plot(
        batches,
        [float(row["rolling_marginal_entropy"]) for row in records],
        linewidth=1.5,
        label="Five-batch rolling marginal entropy",
    )
    axis.set_xlabel("Batch number")
    axis.set_ylabel("Marginal entropy (nats)")
    axis.set_title(f"{stem}: marginal entropy over time")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(entropy_path, dpi=150)
    plt.close(fig)

    accuracy_path = destination / f"{stem}_accuracy.png"
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        batches,
        [float(row["accuracy_percent"]) for row in records],
        linewidth=1,
        label="Batch accuracy",
    )
    axis.set_xlabel("Batch number")
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title(f"{stem}: accuracy over time")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(accuracy_path, dpi=150)
    plt.close(fig)

    distribution_path = destination / f"{stem}_class_distribution.png"
    class_shares = [
        [
            100.0 * float(row[f"predicted_{name}"]) / int(row["batch_size"])
            for name in CLASS_NAMES
        ]
        for row in records
    ]
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.stackplot(
        batches,
        *zip(*class_shares),
        labels=CLASS_NAMES,
        alpha=0.85,
    )
    axis.set_xlabel("Batch number")
    axis.set_ylabel("Predictions per class (% of batch)")
    axis.set_ylim(0, 100)
    axis.set_title(f"{stem}: predicted class distribution over time")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "csv_path": str(source_path),
        "marginal_entropy_plot": str(entropy_path),
        "accuracy_plot": str(accuracy_path),
        "class_distribution_plot": str(distribution_path),
        "batch_collapse_start_batch": batch_collapse_start,
        "batch_images_processed_at_start": images_processed(batch_collapse_start),
        "batch_collapse_confirmation_batch": batch_collapse_confirmation,
        "batch_images_processed_at_confirmation": images_processed(
            batch_collapse_confirmation
        ),
        "rolling_collapse_start_batch": rolling_collapse_start,
        "rolling_images_processed_at_start": images_processed(
            rolling_collapse_start
        ),
        "rolling_collapse_confirmation_batch": rolling_collapse_confirmation,
        "rolling_images_processed_at_confirmation": images_processed(
            rolling_collapse_confirmation
        ),
        "entropy_threshold": entropy_threshold,
        "dominant_threshold": dominant_threshold,
        "consecutive_batches": consecutive_batches,
        "rolling_window_batches": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Phase 3 metrics and calculate time-to-collapse."
    )
    parser.add_argument("csv_path", type=Path, help="Phase 3 batch metrics CSV.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--entropy-threshold", type=float, default=0.20)
    parser.add_argument("--dominant-threshold", type=float, default=0.90)
    parser.add_argument("--consecutive-batches", type=int, default=5)
    args = parser.parse_args()

    results = plot_phase3_results(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        entropy_threshold=args.entropy_threshold,
        dominant_threshold=args.dominant_threshold,
        consecutive_batches=args.consecutive_batches,
    )
    for name, value in results.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
