"""Ordered CIFAR-10-C streams for continual test-time adaptation."""

import argparse
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "cifar10_c"
IMAGES_PER_SEVERITY = 10_000
NUM_SEVERITIES = 5
IMAGE_SHAPE = (32, 32, 3)
MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
EXPERIMENT_SEQUENCES = {
    "static": ("gaussian_noise", "gaussian_noise", "gaussian_noise"),
    "changing": ("gaussian_noise", "motion_blur", "fog"),
}


class CorruptionStream(Dataset[tuple[Tensor, int]]):
    """Concatenate selected corruptions in order without copying them to RAM.

    Each corruption contributes the 10,000 test images from the requested
    severity block. Labels are shared across corruption segments, as in the
    CIFAR-10-C benchmark.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        corruptions: Sequence[str] = ("gaussian_noise",),
        severity: int = 3,
    ) -> None:
        if not 1 <= severity <= NUM_SEVERITIES:
            raise ValueError("severity must be an integer from 1 to 5.")
        if not corruptions:
            raise ValueError("corruptions must contain at least one corruption.")
        if any(
            not name or Path(name).name != name or name == "labels"
            for name in corruptions
        ):
            raise ValueError("Each corruption must be a filename stem, not a path.")

        root_path = Path(root)
        self.data_dir = (
            root_path / "CIFAR-10-C"
            if (root_path / "CIFAR-10-C").is_dir()
            else root_path
        )
        self.corruptions = tuple(corruptions)
        self.severity = severity
        self.start = (severity - 1) * IMAGES_PER_SEVERITY
        self.end = severity * IMAGES_PER_SEVERITY

        labels_path = self.data_dir / "labels.npy"
        if not labels_path.is_file():
            raise FileNotFoundError(f"CIFAR-10-C labels not found: {labels_path}")
        all_labels = np.load(labels_path, mmap_mode="r")
        if len(all_labels) == IMAGES_PER_SEVERITY:
            self.labels = all_labels
        elif len(all_labels) >= self.end:
            self.labels = all_labels[self.start : self.end]
        else:
            raise ValueError(
                f"Expected at least {IMAGES_PER_SEVERITY} labels in {labels_path}; "
                f"found {len(all_labels)}."
            )

        self.images: list[np.ndarray] = []
        self.segment_ends: list[int] = []
        total_images = 0
        for corruption in self.corruptions:
            data_path = self.data_dir / f"{corruption}.npy"
            if not data_path.is_file():
                raise FileNotFoundError(
                    f"CIFAR-10-C corruption not found: {data_path}"
                )
            data = np.load(data_path, mmap_mode="r")
            if data.ndim != 4 or data.shape[1:] != IMAGE_SHAPE:
                raise ValueError(
                    f"Expected {data_path} to have shape [N, 32, 32, 3], "
                    f"got {data.shape}."
                )
            if len(data) < self.end:
                raise ValueError(
                    f"Expected at least {self.end} images in {data_path}; "
                    f"found {len(data)}."
                )
            self.images.append(data)
            total_images += IMAGES_PER_SEVERITY
            self.segment_ends.append(total_images)

    def __len__(self) -> int:
        return len(self.corruptions) * IMAGES_PER_SEVERITY

    def corruption_at(self, index: int) -> str:
        """Return the corruption segment containing a stream index."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment = index // IMAGES_PER_SEVERITY
        return self.corruptions[segment]

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        segment = index // IMAGES_PER_SEVERITY
        image_index = self.start + index % IMAGES_PER_SEVERITY
        image = torch.from_numpy(
            np.array(self.images[segment][image_index], copy=True)
        ).permute(2, 0, 1).float().div_(255.0)
        image.sub_(MEAN).div_(STD)
        return image, int(self.labels[index % IMAGES_PER_SEVERITY])


class SegmentBatchSampler(Sampler[list[int]]):
    """Batch each corruption segment independently to keep shifts explicit."""

    def __init__(self, dataset: CorruptionStream, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        for segment in range(len(self.dataset.corruptions)):
            segment_start = segment * IMAGES_PER_SEVERITY
            segment_end = segment_start + IMAGES_PER_SEVERITY
            for batch_start in range(segment_start, segment_end, self.batch_size):
                yield list(
                    range(
                        batch_start,
                        min(batch_start + self.batch_size, segment_end),
                    )
                )

    def __len__(self) -> int:
        batches_per_segment = math.ceil(IMAGES_PER_SEVERITY / self.batch_size)
        return len(self.dataset.corruptions) * batches_per_segment


def _create_loader(
    dataset: CorruptionStream,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    return DataLoader(
        dataset,
        batch_sampler=SegmentBatchSampler(dataset, batch_size),
        num_workers=num_workers,
    )


def create_stream_loader(
    root: str | Path = DEFAULT_ROOT,
    corruptions: Sequence[str] = ("gaussian_noise",),
    severity: int = 3,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """Create an ordered loader with no batches crossing corruption changes."""
    dataset = CorruptionStream(root, corruptions, severity)
    return _create_loader(dataset, batch_size, num_workers)


def create_experiment_loader(
    experiment: str,
    root: str | Path = DEFAULT_ROOT,
    severity: int = 3,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """Create the predefined static or changing distribution-shift stream."""
    try:
        corruptions = EXPERIMENT_SEQUENCES[experiment]
    except KeyError as error:
        choices = ", ".join(EXPERIMENT_SEQUENCES)
        raise ValueError(
            f"Unknown experiment {experiment!r}; choose from: {choices}."
        ) from error
    return create_stream_loader(
        root=root,
        corruptions=corruptions,
        severity=severity,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an ordered CIFAR-10-C distribution-shift stream."
    )
    stream_choice = parser.add_mutually_exclusive_group()
    stream_choice.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENT_SEQUENCES),
        help="Predefined static or changing corruption sequence.",
    )
    stream_choice.add_argument(
        "--corruptions",
        nargs="+",
        help="Ordered CIFAR-10-C corruption names.",
    )
    parser.add_argument("--severity", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    corruptions = (
        EXPERIMENT_SEQUENCES[args.experiment]
        if args.experiment
        else tuple(args.corruptions or ("gaussian_noise",))
    )
    loader = create_stream_loader(
        root=args.root,
        corruptions=corruptions,
        severity=args.severity,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Corruption order: {', '.join(loader.dataset.corruptions)}")
    print(f"Severity: {args.severity}")
    print(f"Total images: {len(loader.dataset)}")
    images, labels = next(iter(loader))
    print(f"First batch image shape: {list(images.shape)}")
    print(f"First batch label shape: {list(labels.shape)}")
    last_index = min(args.batch_size, len(loader.dataset)) - 1
    print(
        "First-batch stream segment: "
        f"{loader.dataset.corruption_at(0)} -> "
        f"{loader.dataset.corruption_at(last_index)}"
    )


if __name__ == "__main__":
    main()
