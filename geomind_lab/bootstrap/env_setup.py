from __future__ import annotations

import os
from pathlib import Path

from .isaaclab_path import ensure_isaaclab_source_on_path, repo_root


def bootstrap_environment() -> Path:
    root = repo_root()
    ensure_isaaclab_source_on_path(root)
    os.environ.setdefault("GEOMIND_LAB_ROOT", str(root))
    os.environ.setdefault("GEOMIND_LAB_ASSETS_ROOT", str(root / "assets"))
    os.environ.setdefault("GEOMIND_LAB_OUTPUTS_ROOT", str(root / "outputs"))
    return root

