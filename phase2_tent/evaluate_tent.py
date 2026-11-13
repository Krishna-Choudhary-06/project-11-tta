"""Compare clean and corrupted source performance with online TENT."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.cifar_dataset import create_dataloaders
from datasets.corruptions import (
    CIFAR10C_ROOT,
    available_corruptions,
    create_cifar10c_loader,
)
from models.resnet import create_model
from phase1_baseline.evaluate import EvaluationResults, evaluate
from phase2_tent.adapt import adapt_batch, create_tent_optimizer

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "checkpoints"
    / "resnet18_cifar10.pth"
)


def evaluate_with_tent(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[EvaluationResults, float]:
    """Score each batch before adapting on its images, without labels."""
    model.train()
    class_names = list(
        getattr(
            data_loader.dataset,
            "classes",
            [
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
            ],
        )
    )
    num_classes = len(class_names)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    class_totals = torch.zeros(num_classes, dtype=torch.int64)
    class_correct = torch.zeros(num_classes, dtype=torch.int64)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_entropy = 0.0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Predictions are from this batch before its unlabeled adaptation step.
        entropy, logits = adapt_batch(model, images, optimizer)
        predictions = logits.argmax(dim=1)

        total_loss += criterion(logits, labels).item()
        total_correct += (predictions == labels).sum().item()
        batch_samples = labels.size(0)
        total_samples += batch_samples
        total_entropy += entropy * batch_samples

        labels_cpu = labels.cpu()
        predictions_cpu = predictions.cpu()
        class_totals += torch.bincount(labels_cpu, minlength=num_classes)
        class_correct += torch.bincount(
            labels_cpu[labels_cpu == predictions_cpu],
            minlength=num_classes,
        )
        confusion_matrix += torch.bincount(
            labels_cpu * num_classes + predictions_cpu,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    per_class_accuracy = [
        (correct / total * 100).item() if total else 0.0
        for correct, total in zip(class_correct, class_totals)
    ]
    results = EvaluationResults(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples * 100,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=confusion_matrix,
        class_names=class_names,
    )
    return results, total_entropy / total_samples


def _print_summary(title: str, results: EvaluationResults) -> None:
    print(f"\n{title}")
    print(f"Accuracy: {results.accuracy:.2f}%")
    print(f"Cross-entropy loss: {results.loss:.4f}")
    print("Class-wise accuracy:")
    for name, accuracy in zip(results.class_names, results.per_class_accuracy):
        print(f"  {name.capitalize()}: {accuracy:.2f}%")
    print("Confusion matrix (rows: actual, columns: predicted):")
    print(results.confusion_matrix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CIFAR-10 source performance and CIFAR-10-C TENT."
    )
    parser.add_argument(
        "--corruption",
        required=True,
        help="CIFAR-10-C corruption name, such as gaussian_noise.",
    )
    parser.add_argument("--severity", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. Train the baseline first."
        )
    corruptions = available_corruptions(CIFAR10C_ROOT)
    if args.corruption not in corruptions:
        raise ValueError(
            f"Unknown or unavailable corruption {args.corruption!r}. "
            f"Available corruptions: {corruptions or 'none found'}. "
            f"Download and extract CIFAR-10-C into {CIFAR10C_ROOT}; "
            "the folder must contain gaussian_noise.npy, labels.npy, "
            "and the other corruption arrays."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.checkpoint, map_location=device)

    _, clean_loader, num_classes = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    corrupted_loader = create_cifar10c_loader(
        corruption=args.corruption,
        severity=args.severity,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    source_model = create_model(num_classes=num_classes).to(device)
    source_model.load_state_dict(state_dict)
    _print_summary(
        "Clean CIFAR-10 — source model",
        evaluate(source_model, clean_loader, device),
    )
    _print_summary(
        f"CIFAR-10-C {args.corruption}, severity {args.severity} — source model",
        evaluate(source_model, corrupted_loader, device),
    )

    tent_model = create_model(num_classes=num_classes).to(device)
    tent_model.load_state_dict(state_dict)
    optimizer = create_tent_optimizer(tent_model)
    tent_results, mean_entropy = evaluate_with_tent(
        tent_model,
        corrupted_loader,
        optimizer,
        device,
    )
    _print_summary(
        f"CIFAR-10-C {args.corruption}, severity {args.severity} — TENT",
        tent_results,
    )
    print(f"Mean adaptation entropy: {mean_entropy:.4f}")


if __name__ == "__main__":
    main()
