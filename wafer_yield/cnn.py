"""A small CNN for the 9 WM-811K pattern classes.

Input: 2 x 64 x 64 (wafer outline, fail mask). Three conv blocks, global average pooling,
one linear layer: about 100K weights, so it trains on a laptop. Patterns do not change
class when the wafer is rotated or flipped, so training uses those as augmentation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .maps import CLASSES, to_channels


class WaferCNN(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.body = nn.Sequential(block(2, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.body(x))

    def embed(self, x):
        """64 numbers per wafer from just before the classifier: wafers with similar
        defect shapes land close together, which powers the similar-wafer search."""
        return self.head[1](self.head[0](self.body(x)))


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random 90-degree rotation and flip per batch (cheap, and keeps die cells on the grid)."""
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if torch.rand(1) < 0.5 else x


def dihedral(x: torch.Tensor):
    """The 8 rotations and flips of a batch. A defect pattern keeps its class under all of them."""
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        yield r
        yield torch.flip(r, dims=(3,))


@torch.no_grad()
def probs_tta(model: "WaferCNN", x: torch.Tensor) -> torch.Tensor:
    """Test-time augmentation: average the softmax over the 8 rotations and flips."""
    return torch.stack([torch.softmax(model(v), dim=1) for v in dihedral(x)]).mean(0)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model: WaferCNN, wafers: list[np.ndarray], batch: int = 512, tta: bool = True) -> np.ndarray:
    """Class probabilities, one row per wafer (averaged over 8 rotations and flips when tta)."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(wafers), batch):
        x = torch.from_numpy(np.stack([to_channels(w) for w in wafers[i:i + batch]])).to(dev)
        p = probs_tta(model, x) if tta else torch.softmax(model(x), dim=1)
        out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), dtype=np.float32)


def load(path) -> WaferCNN:
    model = WaferCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()
