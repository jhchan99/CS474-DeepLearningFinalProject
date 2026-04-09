"""
PyTorch Dataset and DataLoader utilities for the water-event sequence data.

Loads events from ``events_manifest.csv`` and retrieves ``flow_128`` / ``label``
arrays from the corresponding NPZ shards on demand.  Shard files are opened once
and cached in memory so each file is read only once per process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.constants import PROCESSED_WATER_ROOT, SEQUENCES_DIR

Split = Literal["train", "val", "test"]


class WaterEventDataset(Dataset):
    """
    Index-based dataset over the manifest CSV and NPZ sequence shards.

    Each item is a tuple ``(flow_128, label_idx)`` where:
      - ``flow_128`` is a ``(1, 128)`` float32 tensor of L/min flow rate.
      - ``label_idx`` is a long scalar.

    Parameters
    ----------
    split:
        Which subset to load: ``"train"``, ``"val"``, or ``"test"``.
    manifest_path:
        Path to ``events_manifest.csv``.  Defaults to the standard location.
    sequences_dir:
        Directory containing NPZ shards.  Defaults to the standard location.
    """

    def __init__(
        self,
        split: Split,
        manifest_path: Path | None = None,
        sequences_dir: Path | None = None,
    ) -> None:
        manifest_path = manifest_path or (PROCESSED_WATER_ROOT / "events_manifest.csv")
        self._sequences_dir = sequences_dir or SEQUENCES_DIR

        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(
                f"No events found for split={split!r} in {manifest_path}. "
                "Run build_water_event_dataset.py to build the manifest first."
            )

        self._event_ids: list[str] = df["event_id"].tolist()
        self._label_idxs: np.ndarray = df["label_idx"].to_numpy(dtype=np.int64)
        self._shards: list[str] = df["shard"].tolist()

        # Build a compact position index: per shard, map event_id -> row index
        # populated lazily when the shard is first opened.
        self._shard_data: dict[str, dict[str, int]] = {}  # shard_name -> {event_id -> position}
        self._shard_arrays: dict[str, np.ndarray] = {}    # shard_name -> flow_128 array

        # Precompute unique labels and counts for sampler convenience
        self.label_idxs: np.ndarray = self._label_idxs
        self.num_classes: int = int(df["label_idx"].max()) + 1

    def __len__(self) -> int:
        return len(self._event_ids)

    def _load_shard(self, shard_name: str) -> None:
        path = self._sequences_dir / shard_name
        data = np.load(path, allow_pickle=True)
        ids = data["event_id"]
        id_to_pos = {str(ids[i]): i for i in range(len(ids))}
        self._shard_data[shard_name] = id_to_pos
        self._shard_arrays[shard_name] = data["flow_128"]  # (n, 128) float32

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_name = self._shards[idx]
        event_id = self._event_ids[idx]
        label = self._label_idxs[idx]

        if shard_name not in self._shard_data:
            self._load_shard(shard_name)

        pos = self._shard_data[shard_name][event_id]
        flow = self._shard_arrays[shard_name][pos]  # (128,) float32

        x = torch.from_numpy(flow).float().unsqueeze(0)  # (1, 128)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def class_weights(dataset: WaterEventDataset) -> torch.Tensor:
    """Inverse-frequency weights: ``n_total / (n_classes * count_per_class)``."""
    counts = np.bincount(dataset.label_idxs, minlength=dataset.num_classes).astype(np.float32)
    total = float(counts.sum())
    weights = total / (dataset.num_classes * np.where(counts > 0, counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(dataset: WaterEventDataset) -> WeightedRandomSampler:
    """Return a sampler that draws each class with equal probability."""
    cw = class_weights(dataset)
    sample_weights = cw[dataset.label_idxs]
    return WeightedRandomSampler(
        weights=sample_weights.numpy().tolist(),
        num_samples=len(dataset),
        replacement=True,
    )


def build_dataloaders(
    batch_size: int = 256,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
    manifest_path: Path | None = None,
    sequences_dir: Path | None = None,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_ds = WaterEventDataset("train", manifest_path, sequences_dir)
    val_ds = WaterEventDataset("val", manifest_path, sequences_dir)
    test_ds = WaterEventDataset("test", manifest_path, sequences_dir)

    sampler = make_weighted_sampler(train_ds) if use_weighted_sampler else None
    shuffle_train = sampler is None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def load_label_map(path: Path | None = None) -> dict[str, int]:
    path = path or (PROCESSED_WATER_ROOT / "label_encoding_benchmark.json")
    return json.loads(path.read_text(encoding="utf-8"))


def num_classes(path: Path | None = None) -> int:
    return len(load_label_map(path))
