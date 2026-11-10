#!/usr/bin/env python3
"""Collect L random layouts x K rollouts using a supported protocol adapter.

Each subprocess creates exactly one environment with fixed layout seeds and
runs K resets inside it. The child protocol writes one H5 per episode.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.data.trajectory import validate_dataset_manifest
from geomind_lab.data.trajectory.validator import validate_phase5_dataset_manifest

PROTOCOL_RUNNERS = {
    "anchor_reference": REPO_ROOT / "scripts" / "debug_protocol_anchor_reference.sh",
    "geometric_inference": REPO_ROOT / "scripts" / "debug_protocol_geometric_inference.sh",
    "spatial_placement": REPO_ROOT / "scripts" / "debug_protocol_spatial_placement.sh",
}
_SEED_LIMIT = 2**31 - 1
_LIGHTING_POOL_SIZE = 15
_LIGHTING_RECENT_WINDOW = 5


def _json_dumps_stable(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _episode_h5_path(base_h5_path: Path, rollout_index: int) -> Path:
    if int(rollout_index) == 0:
        return base_h5_path
    if base_h5_path.stem.endswith("_0000"):
        return base_h5_path.with_name(f"{base_h5_path.stem[:-5]}_{int(rollout_index):04d}{base_h5_path.suffix}")
    return base_h5_path.with_name(f"{base_h5_path.stem}.episode_{int(rollout_index):04d}.h5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect independent trajectory rollouts across randomized layouts.")
    parser.add_argument("--layouts", type=int, default=10, help="Number of independently randomized layouts L.")
    parser.add_argument("--rollouts-per-layout", type=int, default=5, help="Episodes K collected from each layout.")
    parser.add_argument("--meta-seed", type=int, default=0, help="Seed used only to derive layout seed pairs.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Dataset output root.")
    parser.add_argument("--layout-key", default="grasp_anchor_direction_scene65_base")
    parser.add_argument("--scene-id", type=int, default=65)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-id", default="", help="Optional registered task id.")
    parser.add_argument("--variant", default="", help="Optional registered variant.")
    parser.add_argument("--object-mode", default="", help="Optional concrete/geometric mode.")
    parser.add_argument(
        "--protocol",
        choices=tuple(PROTOCOL_RUNNERS),
        default="anchor_reference",
        help="Protocol adapter used for the child rollout. Defaults to anchor_reference for compatibility.",
    )
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument(
        "--lighting-pool-indices",
        type=str,
        default="",
        help="Comma-separated lighting pool indices, one per layout. If unset, a deterministic default is sampled.",
    )
    parser.add_argument(
        "--lighting-seed",
        type=int,
        default=None,
        help="Optional seed for default lighting sampling when --lighting-pool-indices is unset.",
    )
    parser.add_argument("--record-cameras", action="store_true", help="Embed all 512x512 RGB cameras in H5 and save MP4s.")
    parser.add_argument("--compact-output", action="store_true", help="Use episode_XXXX.h5 and layout/videos output paths.")
    parser.add_argument(
        "--segment-source",
        choices=("stage_v1", "event_kinematics_v2"),
        default="stage_v1",
        help="Segmentation source for exported H5 labels.",
    )
    parser.add_argument(
        "--vary-trajectory-seed",
        action="store_true",
        help="Use different trajectory_seed for each rollout inside one layout. Phase5 augmentation enables this automatically.",
    )
    parser.add_argument(
        "--trajectory-augmentation-enable",
        action="store_true",
        help="Enable Phase 5 approach-only waypoint augmentation in the child protocol.",
    )
    parser.add_argument("--trajectory-augmentation-candidates", type=int, default=16)
    parser.add_argument("--trajectory-augmentation-segments", type=int, default=6)
    parser.add_argument("--trajectory-augmentation-reverse-probability", type=float, default=0.75)
    parser.add_argument("--trajectory-augmentation-reverse-ratio", type=float, default=0.30)
    parser.add_argument("--trajectory-augmentation-max-reverse", type=float, default=0.08)
    parser.add_argument("--trajectory-augmentation-max-segment-length", type=float, default=0.14)
    parser.add_argument("--trajectory-augmentation-min-clearance", type=float, default=0.08)
    parser.add_argument("--trajectory-augmentation-max-speed", type=float, default=0.30)
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument(
        "--layout-timeout-seconds",
        type=float,
        default=900.0,
        help="Maximum seconds allowed for one layout child process; 0 disables the timeout.",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue after failed layouts.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed layouts from manifest.json; incomplete or failed layouts are collected again.",
    )
    parser.add_argument(
        "--verify-phase2",
        action="store_true",
        help="After collection, run built-in Phase2 checks and write phase2_check_report.json.",
    )
    parser.add_argument(
        "--verify-phase3",
        action="store_true",
        help="After collection, validate manifest-linked H5 datasets and write phase3_check_report.json.",
    )
    parser.add_argument(
        "--verify-phase5",
        action="store_true",
        help="Verify K>=5 fixed-layout approach diversity and preserved grasp contact order from exported H5 files.",
    )
    parser.add_argument(
        "--verify-phase6",
        action="store_true",
        help="Fail on duplicate episode keys, missing H5 files, incomplete layouts, or invalid H5 datasets.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _layout_seeds(meta_seed: int, count: int) -> list[tuple[int, int]]:
    rng = random.Random(int(meta_seed))
    return [(rng.randrange(1, _SEED_LIMIT), rng.randrange(1, _SEED_LIMIT)) for _ in range(count)]


def _trajectory_seeds(meta_seed: int, layout_index: int, rollouts_per_layout: int, *, vary: bool) -> list[int]:
    # Phase2 default: keep one fixed trajectory seed per layout across K rollouts.
    local_seed = int(meta_seed) ^ (int(layout_index) * 0x9E3779B1) ^ 0x85EBCA6B
    rng = random.Random(local_seed)
    if not bool(vary):
        seed = rng.randrange(1, _SEED_LIMIT)
        return [seed for _ in range(int(rollouts_per_layout))]
    return [rng.randrange(1, _SEED_LIMIT) for _ in range(int(rollouts_per_layout))]


def _parse_lighting_pool_indices(raw_value: str, layout_count: int) -> list[int]:
    values = [item.strip() for item in str(raw_value or "").split(",") if item.strip()]
    if len(values) != int(layout_count):
        raise ValueError(
            f"--lighting-pool-indices must provide exactly {int(layout_count)} values; received {len(values)}"
        )
    indices: list[int] = []
    for value in values:
        index = int(value, 0)
        if index < 0 or index >= _LIGHTING_POOL_SIZE:
            raise ValueError(f"Lighting pool index out of range [0, {_LIGHTING_POOL_SIZE - 1}]: {index}")
        indices.append(index)
    return indices


def _sample_lighting_pool_indices(seed: int, layout_count: int) -> list[int]:
    rng = random.Random(int(seed))
    indices: list[int] = []
    for _ in range(int(layout_count)):
        recent = set(indices[-_LIGHTING_RECENT_WINDOW:])
        candidates = [index for index in range(_LIGHTING_POOL_SIZE) if index not in recent]
        if not candidates:
            raise RuntimeError("Unable to sample lighting pool index outside recent window")
        indices.append(rng.choice(candidates))
    return indices


def _resolve_lighting_pool_indices(args: argparse.Namespace) -> list[int]:
    if str(args.lighting_pool_indices or "").strip():
        return _parse_lighting_pool_indices(args.lighting_pool_indices, args.layouts)
    seed = int(args.lighting_seed if args.lighting_seed is not None else (int(args.meta_seed) ^ 0x4C474854))
    return _sample_lighting_pool_indices(seed, args.layouts)


def _should_vary_trajectory_seed(args: argparse.Namespace) -> bool:
    """Phase5 needs independent rollout seeds; Phase2 retains its fixed-seed control."""
    return bool(args.vary_trajectory_seed or args.trajectory_augmentation_enable)


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _manifest_payload(args: argparse.Namespace, layouts: list[dict[str, object]], *, run_status: str) -> dict[str, object]:
    return {
        "run_status": str(run_status),
        "generator_version": f"{args.protocol}_trajectory_runner_v1",
        "commit_hash": _git_commit_hash(),
        "config": vars(args),
        "layouts": layouts,
    }


def _layout_is_complete(layout: object, *, layout_index: int, jitter_seed: int, shift_seed: int, rollout_count: int) -> bool:
    if not isinstance(layout, dict):
        return False
    if (
        int(layout.get("layout_index", -1)) != int(layout_index)
        or int(layout.get("jitter_seed", -1)) != int(jitter_seed)
        or int(layout.get("shift_seed", -1)) != int(shift_seed)
        or int(layout.get("returncode", 1)) != 0
        or not bool(layout.get("collection_complete", False))
    ):
        return False
    episodes = layout.get("episodes", [])
    if not isinstance(episodes, list) or len(episodes) != int(rollout_count):
        return False
    return all(isinstance(episode, dict) and Path(str(episode.get("h5_path", ""))).is_file() for episode in episodes)


def _resume_config_matches(existing_payload: dict[str, object] | None, args: argparse.Namespace) -> bool:
    if not isinstance(existing_payload, dict):
        return False
    existing_config = existing_payload.get("config")
    if not isinstance(existing_config, dict):
        return False
    ignored = {"resume", "dry_run", "verify_phase2", "verify_phase3", "verify_phase5", "verify_phase6", "keep_going", "out_dir"}
    current_config = vars(args)
    return all(existing_config.get(key) == value for key, value in current_config.items() if key not in ignored)


def _episode_rows(layouts: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layout in layouts:
        if not isinstance(layout, dict):
            continue
        for episode in layout.get("episodes", []):
            if not isinstance(episode, dict):
                continue
            row = {
                "layout_index": layout.get("layout_index"),
                "layout_jitter_seed": layout.get("jitter_seed"),
                "layout_shift_seed": layout.get("shift_seed"),
                "lighting_pool_index": layout.get("lighting_pool_index"),
                "lighting_pool_name": layout.get("lighting_pool_name"),
                "layout_returncode": layout.get("returncode"),
                "collection_complete": layout.get("collection_complete"),
                **episode,
            }
            row["episode_key"] = str(row.get("episode_uid") or "").strip() or (
                f"layout_{row.get('layout_index')}_rollout_{row.get('rollout_index')}_seed_{row.get('trajectory_seed')}"
            )
            rows.append(row)
    return rows


def _write_episode_indexes(root: Path, layouts: list[dict[str, object]]) -> dict[str, object]:
    rows = _episode_rows(layouts)
    jsonl_path = root / "episodes.jsonl"
    csv_path = root / "episodes.csv"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    keys = [str(row["episode_key"]) for row in rows]
    return {
        "episode_count": len(rows),
        "unique_episode_key_count": len(set(keys)),
        "duplicate_episode_keys": sorted({key for key in keys if keys.count(key) > 1}),
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
    }


def _quality_report(layouts: list[dict[str, object]], index_report: dict[str, object], phase3_report: dict[str, object] | None) -> dict[str, object]:
    rows = _episode_rows(layouts)
    success_count = sum(bool(row.get("success", False)) for row in rows)
    h5_present_count = sum(Path(str(row.get("h5_path", ""))).is_file() for row in rows)
    report = {
        "episode_count": len(rows),
        "h5_present_count": h5_present_count,
        "success_count": success_count,
        "success_rate": None if not rows else success_count / len(rows),
        "unique_episode_keys": int(index_report["unique_episode_key_count"]),
        "duplicate_episode_keys": list(index_report["duplicate_episode_keys"]),
        "all_layouts_complete": all(bool(layout.get("collection_complete", False)) for layout in layouts),
        "phase3_h5_validation": None if phase3_report is None else dict(phase3_report.get("summary", {}) or {}),
    }
    report["passed"] = bool(
        report["episode_count"]
        and report["episode_count"] == report["h5_present_count"]
        and not report["duplicate_episode_keys"]
        and report["all_layouts_complete"]
        and (phase3_report is None or bool(phase3_report.get("passed", False)))
    )
    return report


def _episode_success_map(summary_payload: dict[str, object] | None) -> dict[int, bool]:
    if not isinstance(summary_payload, dict):
        return {}
    episodes = summary_payload.get("episodes", [])
    if not isinstance(episodes, list):
        return {}
    mapping: dict[int, bool] = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        try:
            idx = int(episode.get("episode_index", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0:
            continue
        mapping[idx] = bool(episode.get("success", False))
    return mapping


def _all_episode_h5_exist(base_h5_path: Path, rollout_count: int) -> bool:
    return all(_episode_h5_path(base_h5_path, rollout_index).is_file() for rollout_index in range(int(rollout_count)))


def _layout_status_payload(
    args: argparse.Namespace,
    *,
    layout_index: int,
    jitter_seed: int,
    shift_seed: int,
    rollout_seeds: list[int],
    layout_dir: Path,
    h5_path: Path,
    command: list[str],
    status: str,
    lighting_pool_index: int | None = None,
    lighting_pool_name: str | None = None,
    child_pid: int | None = None,
    returncode: int | None = None,
) -> dict[str, object]:
    return {
        "status": str(status),
        "layout_index": int(layout_index),
        "jitter_seed": int(jitter_seed),
        "shift_seed": int(shift_seed),
        "rollout_count": int(args.rollouts_per_layout),
        "rollout_seeds": [int(seed) for seed in rollout_seeds],
        "lighting_pool_index": None if lighting_pool_index is None else int(lighting_pool_index),
        "lighting_pool_name": None if lighting_pool_name is None else str(lighting_pool_name),
        "command": list(command),
        "child_pid": None if child_pid is None else int(child_pid),
        "returncode": None if returncode is None else int(returncode),
        "output_dir": str(layout_dir),
        "h5_base_path": str(h5_path),
        "expected_h5_paths": [str(_episode_h5_path(h5_path, rollout_index)) for rollout_index in range(int(args.rollouts_per_layout))],
    }


def _run_layout(
    args: argparse.Namespace,
    *,
    layout_index: int,
    jitter_seed: int,
    shift_seed: int,
    lighting_pool_index: int,
    root: Path,
) -> dict[str, object]:
    layout_dir = root / f"layout_{layout_index:04d}_j{jitter_seed}_s{shift_seed}"
    compact_output = str(os.environ.get("GEOMIND_COMPACT_OUTPUT", "")).strip().lower() in {"1", "true", "yes", "on"}
    h5_path = layout_dir / ("episode_0000.h5" if compact_output else "trajectory.h5")
    rollout_seeds = _trajectory_seeds(
        args.meta_seed,
        layout_index,
        args.rollouts_per_layout,
        vary=_should_vary_trajectory_seed(args),
    )
    lighting_pool_name = f"pool_{lighting_pool_index:02d}"
    env = dict(os.environ)
    env.update(
        {
            "OUTPUT_DIR": str(layout_dir),
            "SUMMARY_JSON_OUT": str(layout_dir / "summary.json"),
            "TRAJECTORY_JSON_OUT": str(layout_dir / "trajectory.json"),
            "TRAJECTORY_H5_OUT": str(h5_path),
            "CAMERA_VIDEO_DIR": str(layout_dir / "videos") if args.compact_output else "",
            "GEOMIND_COMPACT_OUTPUT": "1" if args.compact_output else "0",
            "TRAJECTORY_SAMPLE_HZ": str(args.sample_hz),
            "TRAJECTORY_RECORD_CAMERAS": "1" if args.record_cameras else "0",
            "HEADLESS_CAMERA_VIDEO": "1" if args.record_cameras else "0",
            "TRAJECTORY_SEGMENT_SOURCE": str(args.segment_source),
            "TRAJECTORY_SEEDS": ",".join(str(seed) for seed in rollout_seeds),
            "TRAJECTORY_SEED": str(rollout_seeds[0]),
            "TRAJECTORY_AUGMENTATION_ENABLE": "1" if args.trajectory_augmentation_enable else "0",
            "TRAJECTORY_AUGMENTATION_CANDIDATES": str(args.trajectory_augmentation_candidates),
            "TRAJECTORY_AUGMENTATION_SEGMENTS": str(args.trajectory_augmentation_segments),
            "TRAJECTORY_AUGMENTATION_REVERSE_PROBABILITY": str(args.trajectory_augmentation_reverse_probability),
            "TRAJECTORY_AUGMENTATION_REVERSE_RATIO": str(args.trajectory_augmentation_reverse_ratio),
            "TRAJECTORY_AUGMENTATION_MAX_REVERSE": str(args.trajectory_augmentation_max_reverse),
            "TRAJECTORY_AUGMENTATION_MAX_SEGMENT_LENGTH": str(args.trajectory_augmentation_max_segment_length),
            "TRAJECTORY_AUGMENTATION_MIN_CLEARANCE": str(args.trajectory_augmentation_min_clearance),
            "TRAJECTORY_AUGMENTATION_MAX_SPEED": str(args.trajectory_augmentation_max_speed),
            "GEOMIND_GENERATOR_VERSION": f"{args.protocol}_trajectory_runner_v1",
            "GEOMIND_GENERATOR_COMMIT": _git_commit_hash(),
            "LAYOUT_INDEX": str(layout_index),
            "AGENT_EPISODES": str(args.rollouts_per_layout),
            "LAYOUT_KEY": str(args.layout_key),
            "SCENE_ID": str(args.scene_id),
            "TASK_INDEX": str(args.task_index),
            "GEOMIND_LAYOUT_JITTER_ENABLE": "1",
            "GEOMIND_LAYOUT_JITTER_SEED": str(jitter_seed),
            "GEOMIND_LAYOUT_SHIFT_ENABLE": "1",
            "GEOMIND_LAYOUT_SHIFT_SEED": str(shift_seed),
            "GEOMIND_LIGHTING_POOL_ENABLE": "1",
            "GEOMIND_LIGHTING_POOL_INDEX": str(lighting_pool_index),
            "GEOMIND_LIGHTING_POOL_SEED": str(args.meta_seed),
            "GEOMIND_LIGHTING_POOL_NAME": lighting_pool_name,
        }
    )
    if args.headless:
        env["HEADLESS"] = "1"
    if args.task_id:
        env["TASK_ID"] = str(args.task_id)
    if args.variant:
        env["TASK_VARIANT"] = str(args.variant)
    if args.object_mode:
        env["OBJECT_MODE"] = str(args.object_mode)

    command = ["bash", str(PROTOCOL_RUNNERS[args.protocol])]
    if args.dry_run:
        return {
            "layout_index": layout_index,
            "jitter_seed": jitter_seed,
            "shift_seed": shift_seed,
            "rollout_count": int(args.rollouts_per_layout),
            "rollout_seeds": rollout_seeds,
            "command": command,
            "env": {
                key: env[key]
                for key in sorted(env)
                if key.startswith(("GEOMIND_LAYOUT", "TRAJECTORY_", "AGENT_", "LAYOUT_", "SCENE_", "TASK_"))
                or key.startswith("GEOMIND_LIGHTING_POOL")
            },
            "episodes": [
                {
                    "layout_index": int(layout_index),
                    "rollout_index": int(rollout_index),
                    "trajectory_seed": int(rollout_seed),
                    "h5_path": str(_episode_h5_path(h5_path, rollout_index)),
                }
                for rollout_index, rollout_seed in enumerate(rollout_seeds)
            ],
            "lighting_pool_index": lighting_pool_index,
            "lighting_pool_name": lighting_pool_name,
        }

    layout_dir.mkdir(parents=True, exist_ok=True)
    log_path = layout_dir / "console.log"
    status_path = layout_dir / "layout_status.json"
    _write_json_file(
        status_path,
        _layout_status_payload(
            args,
            layout_index=layout_index,
            jitter_seed=jitter_seed,
            shift_seed=shift_seed,
            rollout_seeds=rollout_seeds,
            layout_dir=layout_dir,
            h5_path=h5_path,
            command=command,
            status="starting",
            lighting_pool_index=lighting_pool_index,
            lighting_pool_name=lighting_pool_name,
        ),
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        _write_json_file(
            status_path,
            _layout_status_payload(
                args,
                layout_index=layout_index,
                jitter_seed=jitter_seed,
                shift_seed=shift_seed,
                rollout_seeds=rollout_seeds,
                layout_dir=layout_dir,
                h5_path=h5_path,
                command=command,
                status="running",
                lighting_pool_index=lighting_pool_index,
                lighting_pool_name=lighting_pool_name,
                child_pid=process.pid,
            ),
        )
        try:
            timeout = float(args.layout_timeout_seconds)
            child_returncode = process.wait(timeout=None if timeout <= 0.0 else timeout)
        except subprocess.TimeoutExpired:
            _write_json_file(
                status_path,
                _layout_status_payload(
                    args,
                    layout_index=layout_index,
                    jitter_seed=jitter_seed,
                    shift_seed=shift_seed,
                    rollout_seeds=rollout_seeds,
                    layout_dir=layout_dir,
                    h5_path=h5_path,
                    command=command,
                    status="timeout",
                    lighting_pool_index=lighting_pool_index,
                    lighting_pool_name=lighting_pool_name,
                    child_pid=process.pid,
                    returncode=-9,
                ),
            )
            process.kill()
            child_returncode = process.wait()

    summary_payload = _read_json(layout_dir / "summary.json")
    trajectory_payload = _read_json(layout_dir / "trajectory.json")
    success_map = _episode_success_map(summary_payload)
    trajectory_episodes = trajectory_payload.get("episodes", []) if isinstance(trajectory_payload, dict) else []
    parsed_episodes: list[dict[str, object]] = []
    if isinstance(trajectory_episodes, list) and trajectory_episodes:
        for episode in trajectory_episodes:
            if not isinstance(episode, dict):
                continue
            try:
                rollout_index = int(episode.get("rollout_index", episode.get("episode_index", -1)))
            except (TypeError, ValueError):
                continue
            if rollout_index < 0:
                continue
            h5_file = _episode_h5_path(h5_path, rollout_index)
            parsed_episodes.append(
                {
                    "layout_index": int(episode.get("layout_index", layout_index)),
                    "rollout_index": int(rollout_index),
                    "trajectory_seed": int(episode.get("trajectory_seed", rollout_seeds[min(rollout_index, len(rollout_seeds) - 1)])),
                    "episode_uid": str(episode.get("episode_uid", "")),
                    "h5_path": str(h5_file),
                    "h5_exists": bool(h5_file.is_file()),
                    "success": bool(success_map.get(rollout_index, False)),
                    "layout_randomization_record": episode.get("layout_randomization_record", {}),
                    "initial_runtime_object_poses": episode.get("initial_runtime_object_poses", {}),
                    "trajectory_augmentation_enabled": bool(episode.get("trajectory_augmentation_enabled", False)),
                    "augmentation_record": episode.get("augmentation_record", {}),
                }
            )

    if not parsed_episodes:
        parsed_episodes = [
            {
                "layout_index": int(layout_index),
                "rollout_index": int(rollout_index),
                "trajectory_seed": int(rollout_seed),
                "episode_uid": "",
                "h5_path": str(_episode_h5_path(h5_path, rollout_index)),
                "h5_exists": bool(_episode_h5_path(h5_path, rollout_index).is_file()),
                "success": False,
                "layout_randomization_record": {},
                "initial_runtime_object_poses": {},
                "trajectory_augmentation_enabled": False,
                "augmentation_record": {},
            }
            for rollout_index, rollout_seed in enumerate(rollout_seeds)
        ]

    all_h5_exist = _all_episode_h5_exist(h5_path, args.rollouts_per_layout)
    effective_returncode = child_returncode if child_returncode != 0 else (0 if all_h5_exist else 2)
    _write_json_file(
        status_path,
        _layout_status_payload(
            args,
            layout_index=layout_index,
            jitter_seed=jitter_seed,
            shift_seed=shift_seed,
            rollout_seeds=rollout_seeds,
            layout_dir=layout_dir,
            h5_path=h5_path,
            command=command,
            status="complete" if int(effective_returncode) == 0 else "failed",
            lighting_pool_index=lighting_pool_index,
            lighting_pool_name=lighting_pool_name,
            returncode=effective_returncode,
        ),
    )
    return {
        "layout_index": layout_index,
        "jitter_seed": jitter_seed,
        "shift_seed": shift_seed,
        "rollout_count": int(args.rollouts_per_layout),
        "rollout_seeds": rollout_seeds,
        "lighting_pool_index": lighting_pool_index,
        "lighting_pool_name": lighting_pool_name,
        "returncode": int(effective_returncode),
        "child_returncode": int(child_returncode),
        "collection_complete": bool(all_h5_exist),
        "output_dir": str(layout_dir),
        "h5_base_path": str(h5_path),
        "log_path": str(log_path),
        "episodes": parsed_episodes,
    }


def _phase2_check(layouts: list[dict[str, object]], *, rollouts_per_layout: int) -> dict[str, object]:
    checks: dict[str, object] = {}
    errors: list[str] = []

    if len(layouts) < 2 or int(rollouts_per_layout) < 2:
        errors.append("Phase2 推荐最小验收为 L>=2 且 K>=2。")

    layout_seed_pairs = {
        (int(layout.get("jitter_seed", -1)), int(layout.get("shift_seed", -1)))
        for layout in layouts
        if isinstance(layout, dict)
    }
    checks["distinct_layout_seed_pairs"] = len(layout_seed_pairs) >= min(2, len(layouts))
    if not checks["distinct_layout_seed_pairs"]:
        errors.append("不同 layout 的 jitter/shift seed 未区分。")

    cross_layout_pose_signatures: list[str] = []
    same_layout_pose_consistent = True
    h5_all_exist = True
    required_fields_ok = True
    required_fields = (
        "layout_index",
        "rollout_index",
        "trajectory_seed",
        "layout_randomization_record",
        "initial_runtime_object_poses",
    )

    for layout in layouts:
        if not isinstance(layout, dict):
            continue
        episodes = layout.get("episodes", [])
        if not isinstance(episodes, list) or not episodes:
            same_layout_pose_consistent = False
            continue
        local_pose_signatures: list[str] = []
        for episode in episodes:
            if not isinstance(episode, dict):
                required_fields_ok = False
                continue
            for key in required_fields:
                if key not in episode:
                    required_fields_ok = False
            if not bool(episode.get("h5_exists", False)):
                h5_all_exist = False
            poses = episode.get("initial_runtime_object_poses", {})
            local_pose_signatures.append(_json_dumps_stable(poses))
        if len(set(local_pose_signatures)) > 1:
            same_layout_pose_consistent = False
        cross_layout_pose_signatures.append(local_pose_signatures[0])

    checks["same_layout_initial_pose_consistent"] = same_layout_pose_consistent
    if not same_layout_pose_consistent:
        errors.append("同一 layout 内不同 rollout 的 initial_runtime_object_poses 不一致。")

    checks["cross_layout_initial_pose_differs"] = len(set(cross_layout_pose_signatures)) >= min(2, len(cross_layout_pose_signatures))
    if not checks["cross_layout_initial_pose_differs"]:
        errors.append("不同 layout 间未观察到初始位姿差异。")

    checks["all_h5_exist"] = h5_all_exist
    if not h5_all_exist:
        errors.append("存在缺失的 H5 输出文件。")

    checks["required_episode_fields_present"] = required_fields_ok
    if not required_fields_ok:
        errors.append("manifest episode 级字段缺失。")

    passed = all(bool(value) for value in checks.values()) and not errors
    return {
        "passed": bool(passed),
        "checks": checks,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    if args.layouts < 1 or args.rollouts_per_layout < 1:
        raise SystemExit("--layouts and --rollouts-per-layout must both be positive.")
    try:
        lighting_pool_indices = _resolve_lighting_pool_indices(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    root = args.out_dir or (REPO_ROOT / "outputs" / "trajectory_dataset" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    root = root.expanduser().resolve()
    seeds = _layout_seeds(args.meta_seed, args.layouts)
    print(f"collecting {args.layouts} layouts x {args.rollouts_per_layout} rollouts -> {root}", flush=True)

    manifest: list[dict[str, object]] = []
    existing_by_index: dict[int, dict[str, object]] = {}
    if args.resume:
        existing_payload = _read_json(root / "manifest.json")
        config_matches = _resume_config_matches(existing_payload, args)
        existing_layouts = [] if existing_payload is None else existing_payload.get("layouts", [])
        if config_matches and isinstance(existing_layouts, list):
            existing_by_index = {
                int(layout.get("layout_index", -1)): layout
                for layout in existing_layouts
                if isinstance(layout, dict) and int(layout.get("layout_index", -1)) >= 0
            }
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        _write_json_file(root / "manifest.json", _manifest_payload(args, manifest, run_status="running"))
    for index, (jitter_seed, shift_seed) in enumerate(seeds):
        print(f"[{index + 1}/{args.layouts}] layout jitter={jitter_seed} shift={shift_seed}", flush=True)
        existing = existing_by_index.get(index)
        if args.resume and _layout_is_complete(
            existing,
            layout_index=index,
            jitter_seed=jitter_seed,
            shift_seed=shift_seed,
            rollout_count=args.rollouts_per_layout,
        ):
            result = dict(existing)
            result["resumed"] = True
            print(f"[{index + 1}/{args.layouts}] resume: existing validated H5 files retained", flush=True)
        else:
            result = _run_layout(
                args,
                layout_index=index,
                jitter_seed=jitter_seed,
                shift_seed=shift_seed,
                lighting_pool_index=lighting_pool_indices[index],
                root=root,
            )
        manifest.append(result)
        if not args.dry_run:
            _write_json_file(root / "manifest.json", _manifest_payload(args, manifest, run_status="running"))
        if not args.dry_run and int(result.get("returncode", 1)) != 0 and not args.keep_going:
            break

    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        phase2_report = _phase2_check(manifest, rollouts_per_layout=int(args.rollouts_per_layout)) if args.verify_phase2 else None
        payload = _manifest_payload(
            args,
            manifest,
            run_status="complete" if all(int(item.get("returncode", 0)) == 0 for item in manifest) else "failed",
        )
        index_report = _write_episode_indexes(root, manifest)
        payload["episode_index"] = index_report
        if phase2_report is not None:
            payload["phase2_check"] = phase2_report
            (root / "phase2_check_report.json").write_text(
                json.dumps(phase2_report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        manifest_path = root / "manifest.json"
        _write_json_file(manifest_path, payload)
        if args.verify_phase3 or args.verify_phase6:
            phase3_report = validate_dataset_manifest(manifest_path)
            payload["phase3_check"] = {
                "passed": bool(phase3_report.get("passed", False)),
                "summary": dict(phase3_report.get("summary", {}) or {}),
            }
            (root / "phase3_check_report.json").write_text(
                json.dumps(phase3_report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _write_json_file(manifest_path, payload)
        if args.verify_phase5:
            phase5_report = validate_phase5_dataset_manifest(
                manifest_path,
                min_rollouts_per_layout=5,
            )
            payload["phase5_check"] = {"passed": bool(phase5_report.get("passed", False))}
            (root / "phase5_check_report.json").write_text(
                json.dumps(phase5_report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _write_json_file(manifest_path, payload)
        quality_report = _quality_report(
            manifest,
            index_report,
            phase3_report if (args.verify_phase3 or args.verify_phase6) else None,
        )
        payload["quality_report"] = quality_report
        (root / "quality_report.json").write_text(
            json.dumps(quality_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_json_file(manifest_path, payload)
        if phase2_report is not None:
            print(
                f"phase2_check: passed={phase2_report['passed']} details={root / 'phase2_check_report.json'}",
                flush=True,
            )
        if args.verify_phase3 or args.verify_phase6:
            phase3_pass = bool(payload.get("phase3_check", {}).get("passed", False))
            print(
                f"phase3_check: passed={phase3_pass} details={root / 'phase3_check_report.json'}",
                flush=True,
            )
        if args.verify_phase5:
            print(
                f"phase5_check: passed={payload['phase5_check']['passed']} details={root / 'phase5_check_report.json'}",
                flush=True,
            )
    run_succeeded = args.dry_run or all(int(item.get("returncode", 0)) == 0 for item in manifest)
    if not args.dry_run and args.verify_phase6:
        run_succeeded = run_succeeded and bool(payload.get("quality_report", {}).get("passed", False))
    return 0 if run_succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
