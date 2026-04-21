"""
Multi-scale 1-D CNN classifier (InceptionTime-lite) for water end-use classification.

Input:  (batch, 1, 128)            flow trace
        (batch, metadata_dim)       optional scalar metadata (log1p(orig_len), sin/cos hour)

Output: (batch, num_classes)        raw logits

Architecture:
  1. Inception stem: parallel 1-D convolutions with kernel sizes {3, 7, 15, 31}
     concatenated along the channel dimension.
  2. Three stacked Conv / BN / ReLU / MaxPool blocks that progressively
     widen channels while halving the temporal dimension.
  3. Global average pooling over time.
  4. Optional concatenation of a scalar metadata vector.
  5. Two-layer MLP classification head with dropout.

The optional metadata branch lets the model see the true event duration
(``orig_len``) which was otherwise discarded when flow traces were resampled
to 128 steps — this is a known signal for bathtub/shower disambiguation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _InceptionStem(nn.Module):
    """Parallel 1-D convolutions with different receptive fields."""

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        kernel_sizes: tuple[int, ...] = (3, 7, 15, 31),
    ) -> None:
        super().__init__()
        branches: list[nn.Module] = []
        for k in kernel_sizes:
            branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.out_channels = branch_channels * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([b(x) for b in self.branches], dim=1)


class MultiScaleCNN(nn.Module):
    """Multi-scale 1-D CNN with optional scalar metadata head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        stem_branch_channels: int = 16,
        stem_kernels: tuple[int, ...] = (3, 7, 15, 31),
        channels: list[int] | None = None,
        dropout: float = 0.3,
        use_metadata_head: bool = False,
        metadata_dim: int = 0,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [96, 128, 160]

        self.stem = _InceptionStem(in_channels, stem_branch_channels, stem_kernels)

        blocks: list[nn.Module] = []
        c_in = self.stem.out_channels
        for c_out in channels:
            blocks += [
                nn.Conv1d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
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
        h = self.stem(x)
        h = self.features(h)
        h = self.pool(h).flatten(1)  # (B, C)
        if self.use_metadata_head and meta is not None:
            h = torch.cat([h, meta], dim=1)
        return self.head(h)


def build_multiscale_cnn(num_classes: int, cfg, metadata_dim: int = 0) -> MultiScaleCNN:
    """Instantiate a MultiScaleCNN from a TrainConfig."""
    channels = getattr(cfg, "multiscale_cnn_channels", None) or [96, 128, 160]
    stem_branch = getattr(cfg, "multiscale_cnn_stem_channels", 16)
    dropout = getattr(cfg, "multiscale_cnn_dropout", 0.3)
    kernels = tuple(getattr(cfg, "multiscale_cnn_kernels", (3, 7, 15, 31)))

    return MultiScaleCNN(
        num_classes=num_classes,
        stem_branch_channels=stem_branch,
        stem_kernels=kernels,
        channels=channels,
        dropout=dropout,
        use_metadata_head=getattr(cfg, "use_metadata_head", False),
        metadata_dim=metadata_dim,
    )
