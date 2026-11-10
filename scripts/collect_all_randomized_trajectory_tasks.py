#!/usr/bin/env python3
"""Collect randomized trajectories for all currently supported protocol tasks.

The child dataset runner creates L randomized layouts and K rollouts per layout.
This wrapper only supplies the task matrix and keeps every task in one dataset
root, with one directory per task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_RUNNER = REPO_ROOT / "scripts" / "collect_anchor_reference_trajectory_dataset.py"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    protocol: str
    task_id: str
    layout_key: str
    scene_id: int
    task_index: int = 0


DEFAULT_TASKS = (
    TaskSpec("grasp_anchor_direction", "anchor_reference", "grasp_anchor_direction", "grasp_anchor_direction_scene65_base", 65),
    TaskSpec("grasp_anchor_distance", "anchor_reference", "grasp_anchor_distance", "grasp_anchor_distance_scene10_base", 10),
    TaskSpec("trend_continuation", "geometric_inference", "trend_continuation", "geometric_inference_implicit_trend_line_3", 65),
    TaskSpec("perfect_midpoint", "geometric_inference", "perfect_midpoint", "geometric_inference_perfect_midpoint_line_2", 65),
    TaskSpec("isometric_extension", "geometric_inference", "isometric_extension", "geometric_inference_isometric_extension_line_2", 65),
    TaskSpec("point_symmetry", "geometric_inference", "point_symmetry", "geometric_inference_point_symmetry_line_2", 65),
    TaskSpec("square_completion", "geometric_inference", "square_completion", "geometric_inference_square_completion_corner_3", 65),
    TaskSpec("place_on_top", "spatial_placement", "place_on_top", "place_on_top_scene65_skill", 65),
    TaskSpec("place_inside", "spatial_placement", "place_inside", "place_inside_scene65_skill", 65),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect 2 randomized layouts x 3 trajectories for every supported randomized task.")
    parser.add_argument("--out-dir", type=Path, default=None, help="One large output root. Defaults to outputs/randomized_trajectory_dataset_<timestamp>.")
    parser.add_argument("--layouts", type=int, default=2, help="Randomized layouts per task.")
    parser.add_argument("--rollouts-per-layout", type=int, default=3, help="Trajectories per randomized layout.")
    parser.add_argument("--meta-seed", type=int, default=20260810, help="Base seed used to derive task/layout seeds.")
    parser.add_argument("--record-cameras", action="store_true", help="Also write camera RGB data and MP4 sidecars.")
    parser.add_argument("--headless", action="store_true", default=True, help="Run Isaac Sim headless (default).")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Run with the Isaac Sim viewer.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed task and return nonzero at the end.")
    parser.add_argument("--task-list", type=Path, default=None, help="Optional JSON task list replacing the built-in canonical matrix.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting Isaac Sim.")
    return parser.parse_args()


def load_tasks(path: Path | None) -> tuple[TaskSpec, ...]:
    if path is None:
        return DEFAULT_TASKS
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--task-list must contain a JSON list.")
    tasks: list[TaskSpec] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each task-list item must be an object.")
        tasks.append(
            TaskSpec(
                name=str(item["name"]),
                protocol=str(item["protocol"]),
                task_id=str(item.get("task_id", "")),
                layout_key=str(item["layout_key"]),
                scene_id=int(item["scene_id"]),
                task_index=int(item.get("task_index", 0)),
            )
        )
    if not tasks:
        raise ValueError("--task-list cannot be empty.")
    return tuple(tasks)


def task_meta_seed(base_seed: int, task: TaskSpec, task_index: int) -> int:
    payload = f"{int(base_seed)}|{int(task_index)}|{task.protocol}|{task.name}|{task.layout_key}|{task.scene_id}|{task.task_index}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return 1 + (int.from_bytes(digest, "big") % (2**31 - 2))


def build_command(task: TaskSpec, args: argparse.Namespace, task_out: Path, *, task_seed: int) -> list[str]:
    command = [
        sys.executable,
        str(DATASET_RUNNER),
        "--protocol", task.protocol,
        "--layouts", str(args.layouts),
        "--rollouts-per-layout", str(args.rollouts_per_layout),
        "--meta-seed", str(task_seed),
        "--out-dir", str(task_out),
        "--layout-key", task.layout_key,
        "--scene-id", str(task.scene_id),
        "--task-index", str(task.task_index),
        "--trajectory-augmentation-enable",
    ]
    if args.headless:
        command.append("--headless")
    if task.task_id and task.protocol != "spatial_placement":
        command.extend(["--task-id", task.task_id])
    if args.record_cameras:
        command.append("--record-cameras")
    if args.keep_going:
        command.append("--keep-going")
    return command


def main() -> int:
    args = parse_args()
    if args.layouts < 1 or args.rollouts_per_layout < 1:
        raise SystemExit("--layouts and --rollouts-per-layout must be positive.")
    tasks = load_tasks(args.task_list)
    root = args.out_dir or (REPO_ROOT / "outputs" / f"randomized_trajectory_dataset_{datetime.now():%Y%m%d_%H%M%S}")
    root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    print(f"dataset_root={root}")
    print(f"task_count={len(tasks)} layouts_per_task={args.layouts} rollouts_per_layout={args.rollouts_per_layout}")
    for index, task in enumerate(tasks, start=1):
        task_out = root / task.name
        seed = task_meta_seed(args.meta_seed, task, index - 1)
        command = build_command(task, args, task_out, task_seed=seed)
        print(f"[{index}/{len(tasks)}] {task.name}: {task.protocol} {task.layout_key} task_meta_seed={seed}")
        print("  " + " ".join(command))
        if args.dry_run:
            manifest.append({"task": task.name, "status": "dry_run", "task_meta_seed": seed, "command": command})
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        status = "success" if completed.returncode == 0 else "failed"
        manifest.append({"task": task.name, "status": status, "returncode": completed.returncode, "task_meta_seed": seed, "command": command})
        if completed.returncode != 0 and not args.keep_going:
            break

    (root / "collection_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = [item for item in manifest if item.get("status") == "failed"]
    print(f"collection_manifest={root / 'collection_manifest.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
