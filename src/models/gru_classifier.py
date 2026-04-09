"""
GRU-based sequence classifier for water end-use classification.

Input:  (batch, 1, 128)  — single-channel flow-rate trace
Output: (batch, num_classes)  — raw logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """
    GRU sequence classifier.

    The input ``(B, 1, T)`` is transposed to ``(T, B, 1)`` before the GRU,
    then the last hidden state (or both directions for bidirectional) is passed
    through a two-layer MLP head.
    """

    def __init__(
        self,
        num_classes: int,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        fc_in = hidden_size * self.num_directions
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fc_in // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) -> (T, B, 1)
        x = x.permute(2, 0, 1)
        _, h = self.gru(x)
        # h: (num_layers * num_directions, B, hidden)
        if self.bidirectional:
            # concatenate the last forward and backward hidden states
            h_last = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2*hidden)
        else:
            h_last = h[-1]  # (B, hidden)
        return self.head(h_last)


def build_gru(num_classes: int, cfg) -> GRUClassifier:
    """Instantiate a GRUClassifier from a TrainConfig."""
    return GRUClassifier(
        num_classes=num_classes,
        hidden_size=cfg.gru_hidden,
        num_layers=cfg.gru_layers,
        dropout=cfg.gru_dropout,
        bidirectional=cfg.gru_bidirectional,
    )
