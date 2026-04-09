"""
CNN + Bidirectional LSTM hybrid classifier for water end-use classification.

Input:  (batch, 1, 128)  — single-channel flow-rate trace
Output: (batch, num_classes)  — raw logits

Architecture:
  1. Convolutional front-end extracts local temporal features.
  2. BiLSTM reads the feature sequence to capture long-range dependencies.
  3. The concatenated final hidden states feed a linear classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        cnn_channels: list[int] | None = None,
        cnn_kernel: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64]

        # ── CNN front-end ────────────────────────────────────────────────────
        cnn_blocks: list[nn.Module] = []
        c_in = in_channels
        for c_out in cnn_channels:
            cnn_blocks += [
                nn.Conv1d(c_in, c_out, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            c_in = c_out
        self.cnn = nn.Sequential(*cnn_blocks)

        # ── Bidirectional LSTM ───────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=c_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        # ── Classification head ──────────────────────────────────────────────
        fc_in = lstm_hidden * 2  # bidirectional
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fc_in // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        x = self.cnn(x)              # (B, C, T')
        x = x.permute(0, 2, 1)      # (B, T', C) for LSTM batch_first
        _, (h, _) = self.lstm(x)
        # h: (num_layers*2, B, hidden) — take last layer fwd+bwd
        h_last = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2*hidden)
        return self.head(h_last)


def build_cnn_bilstm(num_classes: int, cfg) -> CNNBiLSTM:
    """Instantiate a CNNBiLSTM from a TrainConfig."""
    return CNNBiLSTM(
        num_classes=num_classes,
        cnn_channels=cfg.cnn_bilstm_channels,
        cnn_kernel=cfg.cnn_bilstm_kernel,
        lstm_hidden=cfg.cnn_bilstm_hidden,
        lstm_layers=cfg.cnn_bilstm_layers,
        dropout=cfg.cnn_bilstm_dropout,
    )
