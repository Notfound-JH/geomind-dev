#!/usr/bin/env python3
"""Export randomized trajectories and full-camera videos for five simple GeoMind tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_RUNNER = REPO_ROOT / "scripts" / "collect_anchor_reference_trajectory_dataset.py"
LIGHTING_POOL_SIZE = 15
LIGHTING_RECENT_WINDOW = 5


@dataclass(frozen=True)
class TaskSpec:
    name: str
    protocol: str
    task_id: str
    layout_key: str
    scene_id: int
    task_index: int = 0


SIMPLE_TASKS = (
    TaskSpec("grasp_anchor_direction", "anchor_reference", "grasp_anchor_direction", "grasp_anchor_direction_scene65_base", 65),
    TaskSpec("grasp_anchor_distance", "anchor_reference", "grasp_anchor_distance", "grasp_anchor_distance_scene10_base", 10),
    TaskSpec("place_inside", "spatial_placement", "place_inside", "place_inside_scene65_skill", 65),
    TaskSpec("place_on_top", "spatial_placement", "place_on_top", "place_on_top_scene65_skill", 65),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None, help="Dataset output root.")
    parser.add_argument("--layouts", type=int, default=2, help="Randomized layouts per task (default: 2).")
    parser.add_argument("--rollouts-per-layout", type=int, default=2, help="Randomized trajectories per layout (default: 2).")
    parser.add_argument("--sample-hz", type=float, default=10.0, help="Export sampling frequency in Hz (default: 10).")
    parser.add_argument("--meta-seed", type=int, default=20260813, help="Base seed used to derive task/layout seeds.")
    parser.add_argument("--headless", action="store_true", default=True, help="Run Isaac Sim headless (default).")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Run with the Isaac Sim viewer.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a task failure.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting Isaac Sim.")
    return parser.parse_args()


def _task_seed(base_seed: int, task: TaskSpec) -> int:
    source = f"{base_seed}|{task.protocol}|{task.name}|{task.layout_key}|{task.scene_id}|{task.task_index}"
    return 1 + int.from_bytes(hashlib.blake2b(source.encode("utf-8"), digest_size=8).digest(), "big") % (2**31 - 2)


def _lighting_seed(base_seed: int, task: TaskSpec) -> int:
    source = f"lighting|{base_seed}|{task.protocol}|{task.name}|{task.layout_key}|{task.scene_id}|{task.task_index}"
    return 1 + int.from_bytes(hashlib.blake2b(source.encode("utf-8"), digest_size=8).digest(), "big") % (2**31 - 2)


def _lighting_pool_indices(base_seed: int, task: TaskSpec, layout_count: int) -> list[int]:
    rng = random.Random(_lighting_seed(base_seed, task))
    indices: list[int] = []
    for _ in range(int(layout_count)):
        recent = set(indices[-LIGHTING_RECENT_WINDOW:])
        candidates = [index for index in range(LIGHTING_POOL_SIZE) if index not in recent]
        if not candidates:
            raise RuntimeError("Unable to sample lighting pool index outside recent window")
        indices.append(rng.choice(candidates))
    return indices


def _extract_lighting_indices(command: list[str]) -> list[int]:
    try:
        raw_value = command[command.index("--lighting-pool-indices") + 1]
    except (ValueError, IndexError):
        return []
    return [int(item) for item in raw_value.split(",") if item]


def build_command(task: TaskSpec, args: argparse.Namespace, output_dir: Path) -> list[str]:
    lighting_indices = _lighting_pool_indices(args.meta_seed, task, args.layouts)
    command = [
        sys.executable,
        str(DATASET_RUNNER),
        "--protocol", task.protocol,
        "--task-id", task.task_id,
        "--layout-key", task.layout_key,
        "--scene-id", str(task.scene_id),
        "--task-index", str(task.task_index),
        "--layouts", str(args.layouts),
        "--rollouts-per-layout", str(args.rollouts_per_layout),
        "--sample-hz", str(args.sample_hz),
        "--meta-seed", str(_task_seed(args.meta_seed, task)),
        "--lighting-pool-indices", ",".join(str(index) for index in lighting_indices),
        "--out-dir", str(output_dir),
        "--compact-output",
        "--record-cameras",
        "--trajectory-augmentation-enable",
        "--verify-phase2",
        "--verify-phase3",
    ]
    if args.headless:
        command.append("--headless")
    if args.keep_going:
        command.append("--keep-going")
    return command


def main() -> int:
    args = parse_args()
    if args.layouts < 1 or args.rollouts_per_layout < 1:
        raise SystemExit("--layouts and --rollouts-per-layout must be positive.")
    if args.sample_hz <= 0.0:
        raise SystemExit("--sample-hz must be positive.")

    root = args.out_dir or (REPO_ROOT / "outputs" / f"simple_task_trajectory_dataset_{datetime.now():%Y%m%d_%H%M%S}")
    root = root.expanduser().resolve()
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    print(f"dataset_root={root}")
    print(f"task_count={len(SIMPLE_TASKS)} layouts_per_task={args.layouts} rollouts_per_layout={args.rollouts_per_layout}")
    print("cameras=all_rgb video_resolution=scene_cam")
    for index, task in enumerate(SIMPLE_TASKS, start=1):
        task_output = root / task.name
        command = build_command(task, args, task_output)
        print(f"[{index}/{len(SIMPLE_TASKS)}] {task.name}")
        print("  " + " ".join(command))
        if args.dry_run:
            manifest.append(
                {
                    "task": task.name,
                    "status": "dry_run",
                    "lighting_pool_indices": _extract_lighting_indices(command),
                    "command": command,
                }
            )
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        status = "success" if completed.returncode == 0 else "failed"
        manifest.append(
            {
                "task": task.name,
                "status": status,
                "returncode": completed.returncode,
                "lighting_pool_indices": _extract_lighting_indices(command),
                "command": command,
            }
        )
        if completed.returncode != 0 and not args.keep_going:
            break

    if not args.dry_run:
        (root / "collection_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"collection_manifest={root / 'collection_manifest.json'}")
    return 1 if any(item.get("status") == "failed" for item in manifest) else 0


if __name__ == "__main__":
    raise SystemExit(main())