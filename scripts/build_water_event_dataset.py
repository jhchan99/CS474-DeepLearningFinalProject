#!/usr/bin/env python3
"""Build processed water-event tables, site splits, and sequence NPZ shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.water_event_pipeline import build_full_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42, help="RNG seed for site split")
    p.add_argument(
        "--max-events",
        type=int,
        default=None,
        metavar="N",
        help="Only process the first N benchmark events (for smoke tests)",
    )
    args = p.parse_args()
    build_full_pipeline(seed=args.seed, max_events=args.max_events)


if __name__ == "__main__":
    main()
