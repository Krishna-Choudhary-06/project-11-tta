"""Dataset utilities for the CIFAR-10-C corruption benchmark."""

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CIFAR10C_ROOT = Path(__file__).resolve().parents[1] / "data" / "cifar10_c"
CLASS_NAMES = [
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
]
NORMALIZE = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616),
)


def available_corruptions(root: Path = CIFAR10C_ROOT) -> list[str]:
    """List CIFAR-10-C corruption arrays found in the data directory."""
    data_dir = root / "CIFAR-10-C" if (root / "CIFAR-10-C").is_dir() else root
    if not data_dir.is_dir():
        return []
    return sorted(
        path.stem
        for path in data_dir.glob("*.npy")
        if path.name != "labels.npy"
    )


class CIFAR10C(Dataset[tuple[Tensor, int]]):
    """One corruption at one severity from the standard CIFAR-10-C arrays."""

    classes = CLASS_NAMES

    def __init__(
        self,
        corruption: str,
        severity: int = 1,
        root: Path = CIFAR10C_ROOT,
        transform: Callable | None = None,
    ) -> None:
        if severity not in range(1, 6):
            raise ValueError("severity must be an integer from 1 to 5.")

        data_dir = root / "CIFAR-10-C" if (root / "CIFAR-10-C").is_dir() else root
        corruption_path = data_dir / f"{corruption}.npy"
        labels_path = data_dir / "labels.npy"
        if not corruption_path.is_file() or not labels_path.is_file():
            available = available_corruptions(root)
            raise FileNotFoundError(
                f"Expected {corruption_path} and {labels_path}. "
                f"Available corruptions: {available or 'none found'}."
            )

        self.data = np.load(corruption_path, mmap_mode="r")
        all_labels = np.load(labels_path, mmap_mode="r")
        self.start = (severity - 1) * 10_000
        self.end = severity * 10_000
        if len(self.data) < self.end:
            raise ValueError(
                "CIFAR-10-C corruption arrays must contain 50,000 examples "
                "(10,000 for each severity)."
            )
        if len(all_labels) == 10_000:
            self.labels = all_labels
        elif len(all_labels) >= self.end:
            self.labels = all_labels[self.start : self.end]
        else:
            raise ValueError(
                "CIFAR-10-C labels must contain either 10,000 labels "
                "or 50,000 labels (10,000 for each severity)."
            )
        if self.data.ndim != 4 or self.data.shape[1:] != (32, 32, 3):
            raise ValueError(
                f"Expected images shaped [N, 32, 32, 3], got {self.data.shape}."
            )

        self.transform = transform or transforms.Compose(
            [transforms.ToTensor(), NORMALIZE]
        )

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        image = Image.fromarray(np.array(self.data[self.start + index]))
        return self.transform(image), int(self.labels[index])


def create_cifar10c_loader(
    corruption: str,
    severity: int = 1,
    batch_size: int = 32,
    num_workers: int = 2,
    root: Path = CIFAR10C_ROOT,
) -> DataLoader:
    """Create a deterministic data loader for a CIFAR-10-C subset."""
    dataset = CIFAR10C(corruption, severity, root=root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
