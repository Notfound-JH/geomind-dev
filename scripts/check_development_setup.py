#!/usr/bin/env python3
"""Check the editable checkout without starting Isaac Sim."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", action="store_true", help="Also check simulator and external asset prerequisites.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for name in ("numpy", "gymnasium", "pytest", "h5py"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Missing distribution: {name}; install -e '.[dev]'")
    for name in ("geomind", "geomind_lab", "geomind_client", "scripted_goal_policy"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            failures.append(f"Cannot resolve {name}; install this checkout in editable mode")
        elif not Path(spec.origin).resolve().is_relative_to(root):
            failures.append(f"{name} resolves outside this checkout: {spec.origin}")
        else:
            print(f"{name}: {spec.origin}")
    for relative in (
        "assets/annotations/annotated_tasks.json",
        "assets/scene_profiles/assets_catalog_with_metadata.json",
        "geomind/configs/legacy/layouts/standard.json",
        "geomind/configs/legacy/layouts/vla.json",
    ):
        path = root / relative
        if not path.is_file():
            failures.append(f"Missing checkout resource: {relative}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            failures.append(f"Invalid resource {relative}: {exc}")
    if args.simulation:
        if not (root / "IsaacLab/source").is_dir():
            failures.append("Missing IsaacLab/source; see migration/isaaclab/README.md")
        for name in ("isaacsim", "isaaclab", "torch"):
            try:
                dist = importlib.metadata.distribution(name)
                print(f"{name}: {dist.version}")
                direct_url = dist.read_text("direct_url.json")
                if direct_url:
                    print(f"{name} source: {json.loads(direct_url).get('url', '')}")
            except importlib.metadata.PackageNotFoundError:
                failures.append(f"Missing simulation distribution: {name}")
        assets = os.environ.get("GEOMIND_ASSETS_ROOT", "")
        if not assets or not Path(assets).is_dir():
            failures.append("Set GEOMIND_ASSETS_ROOT to the transferred external asset directory")
        print("GPU, asset completeness and runtime compatibility still require a real simulation smoke test.")
    for failure in failures:
        print(f"ERROR: {failure}")
    print(f"Development preflight: {len(failures)} error(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
