"""Download CIFAR-10 and create training and testing DataLoaders."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, int]:
    """Create CIFAR-10 train and test loaders, downloading data if needed."""
    data_dir = Path(__file__).resolve().parents[1] / "data" / "cifar10"
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, len(train_dataset.classes)


def main() -> None:
    train_loader, test_loader, num_classes = create_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print(f"\nImage batch shape: {list(images.shape)}")
    print(f"Label batch shape: {list(labels.shape)}")
    print(f"Labels: {labels.tolist()}")
    print(f"\nNumber of classes: {num_classes}")


if __name__ == "__main__":
    main()
