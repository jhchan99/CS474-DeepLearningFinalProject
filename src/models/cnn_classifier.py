"""
1-D CNN classifier for water end-use classification.

Input:  (batch, 1, 128)  — single-channel flow-rate trace
Output: (batch, num_classes)  — raw logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """
    Stack of 1-D convolutional blocks followed by global average pooling and
    a linear classification head.

    Architecture per block:
      Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d(2)

    After all blocks the spatial dimension is pooled to a single vector with
    adaptive average pooling, then a dropout + linear head produces logits.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        channels: list[int] | None = None,
        kernel_size: int = 5,
        dropout: float = 0.3,
        use_metadata_head: bool = False,
        metadata_dim: int = 0,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            c_in = c_out

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.use_metadata_head = use_metadata_head and metadata_dim > 0
        head_in = c_in + (metadata_dim if self.use_metadata_head else 0)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(head_in, max(head_in // 2, 32)),
            nn.ReLU(inplace=True),
            nn.Linear(max(head_in // 2, 32), num_classes),
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)          # (B, C)
        if self.use_metadata_head and meta is not None:
            x = torch.cat([x, meta], dim=1)
        return self.head(x)


def build_cnn(num_classes: int, cfg, metadata_dim: int = 0) -> CNNClassifier:
    """Instantiate a CNNClassifier from a TrainConfig."""
    return CNNClassifier(
        num_classes=num_classes,
        channels=cfg.cnn_channels,
        kernel_size=cfg.cnn_kernel_size,
        dropout=cfg.cnn_dropout,
        use_metadata_head=getattr(cfg, "use_metadata_head", False),
        metadata_dim=metadata_dim,
    )
