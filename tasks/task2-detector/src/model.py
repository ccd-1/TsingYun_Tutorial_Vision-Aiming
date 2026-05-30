"""MNIST digit model scaffold for Task 2.

Detector code should call the inference function in this module. Training code
lives in train.py so detector.py stays focused on board detection, corner
geometry, and PnP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import torch
import torch.nn.functional as F

RgbPixel = tuple[int, int, int]
ImageLike = np.ndarray

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mnist_classifier.npz"

_MODEL_CACHE: dict[Path, object] = {}


class SimpleMNISTClassifier(torch.nn.Module):
    def __init__(self, input_channels: int = 1, hidden_units: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_channels * 28 * 28, hidden_units),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_units, num_classes),
        )

    def forward(self, inputs):
        return self.net(inputs)


def preprocess_mnist_crop(board_crop: ImageLike) -> np.ndarray:
    if board_crop is None:
        raise ValueError("board_crop must be provided")

    image = np.asarray(board_crop)
    if image.ndim == 3 and image.shape[2] >= 3:
        image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        image = image.copy()
    else:
        raise ValueError("Unsupported image shape for MNIST crop")

    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = np.clip(image, 0.0, 1.0)
    return image.reshape(1, 1, 28, 28)


def _build_default_model() -> torch.nn.Module:
    model = SimpleMNISTClassifier()
    model.eval()
    return model


def _load_model_from_npz(path: Path) -> torch.nn.Module:
    data = np.load(path, allow_pickle=True)
    model = SimpleMNISTClassifier()
    state_dict = model.state_dict()

    if "state_dict" in data.files:
        raw_state = data["state_dict"].item()
        for key, value in raw_state.items():
            state_dict[key] = torch.tensor(value)
        model.load_state_dict(state_dict)
        return model

    file_keys = set(data.files)
    if {"net.1.weight", "net.1.bias", "net.3.weight", "net.3.bias"}.issubset(file_keys):
        state_dict["net.1.weight"] = torch.tensor(data["net.1.weight"])
        state_dict["net.1.bias"] = torch.tensor(data["net.1.bias"])
        state_dict["net.3.weight"] = torch.tensor(data["net.3.weight"])
        state_dict["net.3.bias"] = torch.tensor(data["net.3.bias"])
        model.load_state_dict(state_dict)
        return model

    if {"weight0", "bias0", "weight1", "bias1"}.issubset(file_keys):
        state_dict["net.1.weight"] = torch.tensor(data["weight0"])
        state_dict["net.1.bias"] = torch.tensor(data["bias0"])
        state_dict["net.3.weight"] = torch.tensor(data["weight1"])
        state_dict["net.3.bias"] = torch.tensor(data["bias1"])
        model.load_state_dict(state_dict)
        return model

    raise ValueError(f"Unrecognized MNIST model format in {path}")


def load_mnist_model(model_path: Path = DEFAULT_MODEL_PATH) -> object:
    model_path = Path(model_path)
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    if not model_path.exists():
        model = _build_default_model()
        _MODEL_CACHE[model_path] = model
        return model

    if model_path.suffix.lower() in {".npz"}:
        try:
            model = _load_model_from_npz(model_path)
        except Exception:
            model = _build_default_model()
    elif model_path.suffix.lower() in {".pt", ".pth"}:
        try:
            model = SimpleMNISTClassifier()
            state = torch.load(model_path, map_location="cpu")
            if isinstance(state, dict) and not any(k.startswith("__") for k in state.keys()):
                model.load_state_dict(state)
            elif hasattr(state, "state_dict"):
                model.load_state_dict(state.state_dict())
            else:
                raise ValueError("Unsupported torch model format")
        except Exception:
            model = _build_default_model()
    else:
        model = _build_default_model()

    model.eval()
    _MODEL_CACHE[model_path] = model
    return model


def predict_mnist_digit(model: object, model_input: np.ndarray) -> tuple[int, float]:
    if model is None:
        raise ValueError("Model must be provided for MNIST prediction")
    if not isinstance(model_input, np.ndarray):
        raise ValueError("model_input must be a numpy array")

    tensor = torch.from_numpy(model_input.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output, dtype=torch.float32)

    if output.ndim == 1:
        output = output.unsqueeze(0)

    probabilities = F.softmax(output, dim=-1)
    probabilities = probabilities[0].cpu().numpy().astype(np.float32)
    digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    return digit, confidence


def classify_mnist_digit(board_crop: ImageLike, model_path: Path = DEFAULT_MODEL_PATH) -> tuple[int, float]:
    model_input = preprocess_mnist_crop(board_crop)
    model = load_mnist_model(model_path)
    return predict_mnist_digit(model, model_input)
