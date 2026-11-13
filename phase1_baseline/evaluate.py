"""Evaluate a trained CIFAR-10 baseline model."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.cifar_dataset import create_dataloaders
from models.resnet import create_model

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "checkpoints"
    / "resnet18_cifar10.pth"
)


@dataclass
class EvaluationResults:
    loss: float
    accuracy: float
    per_class_accuracy: list[float]
    confusion_matrix: torch.Tensor
    class_names: list[str]


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> EvaluationResults:
    """Calculate average loss, accuracy, per-class accuracy, and confusion matrix."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
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
    confusion_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64
    )
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_totals = torch.zeros(num_classes, dtype=torch.int64)
    class_correct = torch.zeros(num_classes, dtype=torch.int64)

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)

            total_loss += criterion(logits, labels).item()
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

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
    return EvaluationResults(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples * 100,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=confusion_matrix,
        class_names=class_names,
    )


def print_evaluation(results: EvaluationResults) -> None:
    """Print a readable CIFAR-10 evaluation report."""
    print("\nBaseline Evaluation")
    print("\nDataset: CIFAR-10")
    print("Model: ResNet18")
    print(f"\nAccuracy: {results.accuracy:.2f}%")
    print(f"Loss: {results.loss:.4f}")
    print("\nClass-wise accuracy:")
    for name, accuracy in zip(results.class_names, results.per_class_accuracy):
        print(f"{name.capitalize()}: {accuracy:.2f}%")
    print("\nConfusion matrix (rows: actual, columns: predicted):")
    print(results.confusion_matrix)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Train the model first."
        )

    _, test_loader, num_classes = create_dataloaders()
    model = create_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Device: {device}")
    print_evaluation(evaluate(model, test_loader, device))


if __name__ == "__main__":
    main()
