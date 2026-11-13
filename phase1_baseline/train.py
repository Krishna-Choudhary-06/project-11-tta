"""Train the CIFAR-10 ResNet-18 baseline."""

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from datasets.cifar_dataset import create_dataloaders
from models.resnet import create_model
from phase1_baseline.evaluate import evaluate

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_WORKERS = 2
CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "checkpoints"
    / "resnet18_cifar10.pth"
)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return its average cross-entropy loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot train with an empty data loader.")

    return total_loss / total_samples


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    num_workers: int = NUM_WORKERS,
) -> nn.Module:
    """Train ResNet-18, evaluate after each epoch, and save its state dict."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, num_classes = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = create_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print(f"Training on: {device}")
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        test_results = evaluate(model, test_loader, device)
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train loss: {train_loss:.4f} | "
            f"Test loss: {test_results.loss:.4f} | "
            f"Test accuracy: {test_results.accuracy:.2f}%"
        )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved checkpoint to: {CHECKPOINT_PATH}")
    return model


if __name__ == "__main__":
    train_model()
