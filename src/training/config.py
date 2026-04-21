"""
Training configuration loaded from a YAML file.

Usage::

    cfg = TrainConfig.from_yaml("configs/train_gru.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    # ── experiment identity ───────────────────────────────────────────────────
    run_name: str = "run"
    model: str = "gru"          # gru | cnn | cnn_bilstm

    # ── paths (resolved relative to project root) ─────────────────────────────
    manifest_path: str = "processed/water_events/events_manifest.csv"
    sequences_dir: str = "processed/water_events/sequences"
    label_map_path: str = "processed/water_events/label_encoding_benchmark.json"
    checkpoint_dir: str = "runs/{run_name}/checkpoints"
    log_dir: str = "runs/{run_name}/logs"

    # ── data loading ─────────────────────────────────────────────────────────
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    use_weighted_sampler: bool = False  # alternative to loss weighting

    # ── optimiser ────────────────────────────────────────────────────────────
    lr: float = 3e-4 
    weight_decay: float = 1e-4
    scheduler: str = "cosine"   # cosine | plateau | none
    # cosine params
    cosine_t_max: int = 50
    # plateau params
    plateau_patience: int = 5
    plateau_factor: float = 0.5

    # ── loss ─────────────────────────────────────────────────────────────────
    use_class_weights: bool = True  # inverse-frequency weights in cross-entropy
    loss_type: str = "ce"           # "ce" | "focal"
    focal_gamma: float = 2.0

    # ── augmentation (train split only) ──────────────────────────────────────
    augment: bool = False
    aug_noise_std: float = 0.0
    aug_amp_min: float = 1.0
    aug_amp_max: float = 1.0
    aug_time_warp: float = 0.0      # 0 = off; 0.1 = +/-10% random time-stretch

    # ── metadata head (scalar features concatenated before the FC head) ──────
    use_metadata_head: bool = False

    # ── training loop ─────────────────────────────────────────────────────────
    epochs: int = 50
    early_stop_patience: int = 10
    seed: int = 42

    # ── GRU-specific ─────────────────────────────────────────────────────────
    gru_hidden: int = 128
    gru_layers: int = 2
    gru_dropout: float = 0.3
    gru_bidirectional: bool = True

    # ── CNN-specific ──────────────────────────────────────────────────────────
    cnn_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_size: int = 5
    cnn_dropout: float = 0.3

    # ── CNN+BiLSTM-specific ───────────────────────────────────────────────────
    cnn_bilstm_channels: list[int] = field(default_factory=lambda: [32, 64])
    cnn_bilstm_kernel: int = 5
    cnn_bilstm_hidden: int = 128
    cnn_bilstm_layers: int = 2
    cnn_bilstm_dropout: float = 0.3

    # ── Multi-scale CNN-specific ──────────────────────────────────────────────
    multiscale_cnn_stem_channels: int = 16
    multiscale_cnn_kernels: list[int] = field(default_factory=lambda: [3, 7, 15, 31])
    multiscale_cnn_channels: list[int] = field(default_factory=lambda: [96, 128, 160])
    multiscale_cnn_dropout: float = 0.3

    # ── misc ─────────────────────────────────────────────────────────────────
    log_every_n_steps: int = 50
    save_best_only: bool = True

    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        obj = cls()
        for k, v in raw.items():
            if not hasattr(obj, k):
                raise ValueError(f"Unknown config key: {k!r}")
            setattr(obj, k, v)
        # Template path placeholders
        obj.checkpoint_dir = obj.checkpoint_dir.format(run_name=obj.run_name)
        obj.log_dir = obj.log_dir.format(run_name=obj.run_name)
        return obj

    def resolved_paths(self, project_root: Path) -> "TrainConfig":
        """Return a copy with all path strings resolved to absolute paths."""
        import copy
        cfg = copy.deepcopy(self)
        for attr in ("manifest_path", "sequences_dir", "label_map_path",
                     "checkpoint_dir", "log_dir"):
            val = getattr(cfg, attr)
            if not Path(val).is_absolute():
                setattr(cfg, attr, str(project_root / val))
        return cfg
