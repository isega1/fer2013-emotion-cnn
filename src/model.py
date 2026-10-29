"""Model definitions.

``EmotionCNN`` is reproduced from the original coursework code so that the
checkpoints trained in 2025 (``model_epoch_5.pth`` / ``model_epoch_10.pth``)
load without any key renaming. Parameter count: 10,927,751
(10,929,163 tensors in the state_dict; the 1,412 difference is BatchNorm
running statistics and `num_batches_tracked`, which are buffers, not weights).

The variants are the "6 models" family from the earlier report
(CNN-Light/Standard/Deep, MLP-Small/Medium/Large); they are registered here
so the comparison table can be produced from one code path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(cin: int, cout: int, pool: bool) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(cout),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return layers


class EmotionCNN(nn.Module):
    """AlexNet-style CNN for 48x48 single channel input (original model)."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(1, 64, pool=True),
            *_conv_block(64, 128, pool=True),
            *_conv_block(128, 256, pool=False),
            *_conv_block(256, 256, pool=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


class SmallCNN(nn.Module):
    """Three conv blocks with a single hidden layer (~0.3M parameters)."""

    def __init__(self, num_classes: int = 7, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(1, width, pool=True),
            *_conv_block(width, width * 2, pool=True),
            *_conv_block(width * 2, width * 4, pool=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(width * 4 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.features(x), 1))


class MLP(nn.Module):
    """Flattened 48x48 input, no convolution, for the no-prior baseline."""

    def __init__(self, num_classes: int = 7, hidden: int = 1024, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        cin = 48 * 48
        for _ in range(depth):
            layers += [nn.Linear(cin, hidden), nn.ReLU(inplace=True), nn.Dropout(0.3)]
            cin = hidden
        layers.append(nn.Linear(cin, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.flatten(x, 1))


MODELS = {
    "emotioncnn": lambda: EmotionCNN(),
    "cnn-light": lambda: SmallCNN(width=24),
    "cnn-small": lambda: SmallCNN(width=32),
    "mlp-small": lambda: MLP(hidden=512, depth=1),
    "mlp-medium": lambda: MLP(hidden=1024, depth=2),
    "mlp-large": lambda: MLP(hidden=2048, depth=2),
}


def build_model(name: str) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, choose from {sorted(MODELS)}")
    return MODELS[name]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
