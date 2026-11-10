from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def isaaclab_source_dir(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "IsaacLab" / "source"


def ensure_isaaclab_source_on_path(root: Path | None = None) -> Path:
    source_dir = isaaclab_source_dir(root)
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Expected IsaacLab source directory at: {source_dir}. "
            "Copy your IsaacLab checkout into the repository root first."
        )
    source_str = str(source_dir)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)
    return source_dir

