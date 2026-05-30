"""Training scaffold for the Task 2 MNIST digit classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MNIST_DATA_DIR = TASK_ROOT / "data"


def download_mnist_dataset(data_dir: Path = DEFAULT_MNIST_DATA_DIR) -> Path:
    """Download torchvision MNIST into the Task 2 data directory."""
    import torchvision

    data_dir.mkdir(parents=True, exist_ok=True)
    torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    return data_dir / "MNIST"


class MNISTClassifier(nn.Module):
    """Small PyTorch classifier scaffold for 28x28 MNIST crops."""

    def __init__(self, input_size: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        return self.net(inputs)


def select_training_device(torch_module) -> str:
    if getattr(torch_module, "cuda", None) is not None and torch_module.cuda.is_available():
        return "cuda"

    if getattr(getattr(torch_module, "backends", None), "mps", None) is not None and torch_module.backends.mps.is_available():
        return "mps"

    return "cpu"


def train_mnist_classifier(dataset_dir: Path, output_path: Path) -> Path:
    from torch.utils.data import DataLoader, random_split
    import torchvision
    from torchvision.transforms import ToTensor

    device_str = select_training_device(torch)
    device = torch.device(device_str)

    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

    transform = ToTensor()
    train_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=True, download=True, transform=transform)
    if len(train_dataset) == 0:
        raise RuntimeError(f"MNIST training dataset contains no examples in {dataset_dir}")

    val_size = int(len(train_dataset) * 0.1)
    train_size = len(train_dataset) - val_size
    train_split, val_split = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_split, batch_size=128, shuffle=True, num_workers=2, pin_memory=(device_str != "cpu"))
    val_loader = DataLoader(val_split, batch_size=256, shuffle=False, num_workers=2, pin_memory=(device_str != "cpu"))

    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_accuracy = 0.0
    epochs = 5
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total if total > 0 else 0.0
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(output_path, state_dict=state)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task 2 MNIST digit classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_MNIST_DATA_DIR / "MNIST", help="Directory containing labeled MNIST board crops.")
    parser.add_argument("--output", type=Path, default=TASK_ROOT / "models" / "mnist_classifier.npz", help="Where to save the trained classifier.")
    parser.add_argument("--download-mnist", action="store_true", help="Download MNIST into tasks/task2-detector/data/MNIST before training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_mnist:
        dataset_path = download_mnist_dataset(DEFAULT_MNIST_DATA_DIR)
        print(f"Downloaded MNIST dataset to: {dataset_path}")
        return

    output_path = train_mnist_classifier(args.dataset_dir, args.output)
    print(f"Saved MNIST classifier to: {output_path}")


if __name__ == "__main__":
    main()
