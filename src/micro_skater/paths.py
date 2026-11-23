# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Where the package data and the released checkpoints live (no absolute paths anywhere else)."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
# src/micro_skater -> repo root (an editable / PYTHONPATH checkout). Override with MICRO_SKATER_CHECKPOINTS.
REPO_ROOT = PACKAGE_DIR.parent.parent


def data_path(name: str) -> Path:
    """A file shipped in micro_skater/data (stance poses, spawn banks)."""
    return DATA_DIR / name


def checkpoint_path(name: str) -> Path:
    """A released checkpoint (checkpoints/ of this repo, or $MICRO_SKATER_CHECKPOINTS)."""
    root = os.environ.get("MICRO_SKATER_CHECKPOINTS")
    return (Path(root) if root else REPO_ROOT / "checkpoints") / name
