#!/usr/bin/env python3
"""Verification #1/#2/#4: base vs direct pairing inside IsaacLab.

Two phases.

``--mode dump`` boots IsaacLab once, and for every selected (task, level, trial) builds
the env for one ablation, resets it, and records the post-reset world pose of every rigid
object, the robot joint state, the goal metric inputs and the effective instruction. It
optionally saves the table_cam frame the policy would see.

``--mode compare`` diffs two dumps and reports, per trial, whether every object pose is
bitwise identical.

    # base side (reuses nothing from previous runs; it is a fresh deterministic reset)
    HEADLESS=1 python scripts/check_direct_pairing.py --mode dump --ablation base \
        --out docs/direct_ablation/pairing_base.json --frames-dir docs/direct_ablation/frames/base
    HEADLESS=1 python scripts/check_direct_pairing.py --mode dump --ablation direct \
        --out docs/direct_ablation/pairing_direct.json --frames-dir docs/direct_ablation/frames/direct
    python scripts/check_direct_pairing.py --mode compare \
        --base docs/direct_ablation/pairing_base.json --direct docs/direct_ablation/pairing_direct.json \
        --json docs/direct_ablation/pairing_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUCCESS_LEVELS = ("loose", "medium", "strict")
DEFAULT_BASE_SEED = 20260502
VLA_TASK_DIR = REPO_ROOT / "geomind" / "configs" / "vla_tasks"
from geomind.core.schema import normalize_task_config


def derived_seed(base_seed: int, task_index: int, level_index: int, trial_index: int) -> int:
    return int(base_seed) + int(task_index) * 100003 + int(level_index) * 1009 + int(trial_index)


def load_task_payloads() -> list[dict]:
    payloads = []
    for path in sorted(VLA_TASK_DIR.glob("*.json")):
        if path.name == "legacy_layouts.json":
            continue
        payload = normalize_task_config(json.loads(path.read_text(encoding="utf-8")))
        if payload.get("implementation_status") != "placeholder":
            payloads.append(payload)
    return payloads


def _scene_entity_names(scene) -> list[str]:
    names: set[str] = set()
    for attr in ("rigid_objects", "articulations", "extras", "deformable_objects", "sensors"):
        try:
            names.update(str(k) for k in dict(getattr(scene, attr, {}) or {}))
        except Exception:  # noqa: BLE001 - audit tool, best effort
            continue
    return sorted(names)


def _all_asset_poses(native_env) -> dict[str, list[float]]:
    """World pose of every scene entity, including static AssetBase extras.

    ``scene.rigid_objects`` only covers dynamic bodies; most direct-ablation tasks keep
    their anchors (and always the direct marker) as static assets, so restricting the
    pairing check to rigid objects would compare almost nothing.
    """
    from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.common import get_asset_pose_w

    poses: dict[str, list[float]] = {}
    for name in _scene_entity_names(native_env.scene):
        try:
            pos_w, quat_w = get_asset_pose_w(native_env, name)
        except Exception as exc:  # noqa: BLE001 - lights/sensors have no pose
            poses[name] = [f"unavailable:{type(exc).__name__}"]  # type: ignore[list-item]
            continue
        row = [float(v) for v in pos_w[0].detach().cpu().tolist()]
        row += [float(v) for v in quat_w[0].detach().cpu().tolist()]
        poses[name] = row
    return poses


def _save_frame(raw_obs: dict, out_path: Path) -> str:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        return f"frame_skipped:{type(exc).__name__}"
    native = ((raw_obs.get("state") or {}).get("native_obs") or {})
    policy = native.get("policy") if isinstance(native, dict) else None
    image = (policy or {}).get("table_cam") if isinstance(policy, dict) else None
    if image is None and isinstance(native, dict):
        image = native.get("table_cam")
    if image is None:
        return "frame_missing:table_cam"
    array = np.asarray(image)
    while array.ndim > 3:
        array = array[0]
    if array.ndim != 3:
        return f"frame_bad_shape:{array.shape}"
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        maximum = float(array.max()) if array.size else 1.0
        array = (array / maximum * 255.0).clip(0, 255).astype("uint8") if maximum > 1.5 else (
            array * 255.0
        ).clip(0, 255).astype("uint8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(out_path)
    return str(out_path)


def dump(args: argparse.Namespace) -> int:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("ENABLE_CAMERAS", "1")

    from geomind.core.loader import create_env
    from geomind.core.resolver import resolve_task_request
    from geomind.tasks import direct_ablation as direct

    payloads = load_task_payloads()
    selected = {t.strip() for t in str(args.tasks).split(",") if t.strip()}
    levels = [lvl.strip() for lvl in str(args.levels).split(",") if lvl.strip()]
    trials = [int(t) for t in str(args.trials).split(",") if str(t).strip()]

    out_path = Path(args.out)
    entries: dict[str, dict] = {}
    if args.resume and out_path.is_file():
        entries = json.loads(out_path.read_text(encoding="utf-8")).get("entries", {})

    for task_index, payload in enumerate(payloads):
        task_id = str(payload.get("task_id", ""))
        if selected and task_id not in selected:
            continue
        for level in levels:
            level_index = SUCCESS_LEVELS.index(level)
            for trial_index in trials:
                key = f"{task_id}|{level}|{trial_index}"
                if key in entries:
                    continue
                seed = derived_seed(int(args.base_seed), task_index, level_index, trial_index)
                resolved = resolve_task_request(task_id, max_episode_steps=int(args.max_steps), backend="isaaclab")
                env = create_env(
                    resolved.resolved_task_id,
                    max_episode_steps=int(args.max_steps),
                    backend="isaaclab",
                    backend_kwargs={
                        "seed": int(seed),
                        "enable_object_pose_perturb": False,
                        "object_pose_perturb_range": 0.0,
                        "object_pose_perturb_seed": int(seed),
                        "direct_ablation": {"ablation_id": str(args.ablation), "pair_id": key},
                    },
                )
                try:
                    raw_obs, info = env.reset()
                    delegate = getattr(env, "unwrapped", env)
                    delegate = getattr(delegate, "_runtime_env", delegate)
                    native = delegate._native_env  # noqa: SLF001 - audit tool
                    scene = native.scene
                    poses = {}
                    for name, asset in dict(getattr(scene, "rigid_objects", {}) or {}).items():
                        state = asset.data.root_state_w[0, :7].detach().cpu().tolist()
                        poses[str(name)] = [float(v) for v in state]
                    robot = scene["robot"]
                    entry = {
                        "task_id": task_id,
                        "task_index": task_index,
                        "level": level,
                        "trial_index": trial_index,
                        "seed": int(seed),
                        "ablation_id": str(args.ablation),
                        "rigid_object_poses": poses,
                        "all_asset_poses": _all_asset_poses(native),
                        "robot_joint_pos": [float(v) for v in robot.data.joint_pos[0].detach().cpu().tolist()],
                        "goal_distance": info.get("goal_distance"),
                        "success_radius": info.get("success_radius"),
                        "goal_kind": info.get("goal_kind"),
                        "protocol_resolver": info.get("protocol_resolver"),
                        "target_object_id": info.get("target_object_id"),
                        "anchor_object_id": info.get("anchor_object_id"),
                        "runtime_object_count": info.get("runtime_object_count"),
                        "instruction": raw_obs.get("instruction"),
                        "direct_ablation": delegate.direct_ablation_record,
                    }
                    if args.frames_dir:
                        entry["frame"] = _save_frame(
                            raw_obs, Path(args.frames_dir) / f"{task_id}__{level}__{trial_index}.png"
                        )
                    entries[key] = entry
                finally:
                    env.close()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(
                        {
                            "ablation_id": str(args.ablation),
                            "base_seed": int(args.base_seed),
                            "seed_derivation": "base_seed + task_index * 100003 + level_index * 1009 + trial_index",
                            "task_list_fingerprint": direct.task_list_fingerprint(
                                [str(p.get("task_id", "")) for p in payloads]
                            ),
                            "entries": entries,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    f"[dump] {key} seed={seed} rigid={len(entries[key]['rigid_object_poses'])} "
                    f"assets={len(entries[key]['all_asset_poses'])}"
                )
    print(f"[dump] wrote {out_path} entries={len(entries)}")
    return 0


def compare(args: argparse.Namespace) -> int:
    base_payload = json.loads(Path(args.base).read_text(encoding="utf-8"))
    direct_payload = json.loads(Path(args.direct).read_text(encoding="utf-8"))
    base_entries = base_payload.get("entries", {})
    direct_entries = direct_payload.get("entries", {})

    rows: list[dict] = []
    failures: list[str] = []
    if base_payload.get("task_list_fingerprint") != direct_payload.get("task_list_fingerprint"):
        failures.append("task_list_fingerprint mismatch between the two dumps")

    for key in sorted(set(base_entries) | set(direct_entries)):
        base_entry = base_entries.get(key)
        direct_entry = direct_entries.get(key)
        if base_entry is None or direct_entry is None:
            failures.append(f"{key}: missing on {'base' if base_entry is None else 'direct'} side")
            continue
        problems: list[str] = []
        if int(base_entry["seed"]) != int(direct_entry["seed"]):
            problems.append(f"seed {base_entry['seed']} != {direct_entry['seed']}")
        marker_name = "direct_marker"
        # Compare every scene entity we could read a pose for, not just rigid bodies:
        # most tasks keep their anchors static, so rigid_object_poses alone is far too weak.
        base_poses = dict(base_entry.get("all_asset_poses") or base_entry["rigid_object_poses"])
        direct_poses = dict(direct_entry.get("all_asset_poses") or direct_entry["rigid_object_poses"])
        shared = set(base_poses) & set(direct_poses)
        extra_direct = set(direct_poses) - set(base_poses)
        extra_base = set(base_poses) - set(direct_poses)
        if extra_base or (extra_direct - {marker_name}):
            problems.append(f"object set differs base_only={sorted(extra_base)} direct_only={sorted(extra_direct)}")
        max_delta = 0.0
        worst = ""
        unavailable: list[str] = []
        for name in sorted(shared):
            lhs = base_poses[name]
            rhs = direct_poses[name]
            if any(isinstance(v, str) for v in lhs) or any(isinstance(v, str) for v in rhs):
                if lhs != rhs:
                    problems.append(f"pose availability differs for {name}: {lhs} vs {rhs}")
                unavailable.append(name)
                continue
            if lhs != rhs:
                delta = max(abs(a - b) for a, b in zip(lhs, rhs))
                if delta > max_delta:
                    max_delta, worst = delta, name
        if max_delta > 0.0:
            problems.append(f"pose mismatch worst={worst} max_abs_delta={max_delta:.3e}")
        joint_delta = 0.0
        if base_entry["robot_joint_pos"] != direct_entry["robot_joint_pos"]:
            joint_delta = max(
                abs(a - b) for a, b in zip(base_entry["robot_joint_pos"], direct_entry["robot_joint_pos"])
            )
            problems.append(f"robot joint mismatch max_abs_delta={joint_delta:.3e}")
        if str(base_entry.get("target_object_id")) != str(direct_entry.get("target_object_id")):
            problems.append("target_object_id differs")
        if base_entry.get("success_radius") != direct_entry.get("success_radius"):
            problems.append("success_radius differs")
        base_hash = str((base_entry.get("direct_ablation") or {}).get("layout_hash", ""))
        direct_hash = str((direct_entry.get("direct_ablation") or {}).get("layout_hash", ""))
        if base_hash and direct_hash and base_hash != direct_hash:
            problems.append("layout_hash differs")
        if problems:
            failures.append(f"{key}: {'; '.join(problems)}")
        rows.append(
            {
                "pair_id": key,
                "seed": base_entry["seed"],
                "objects_compared": len(shared) - len(unavailable),
                "rigid_objects_compared": len(
                    set(base_entry["rigid_object_poses"]) & set(direct_entry["rigid_object_poses"])
                ),
                "pose_unavailable": sorted(unavailable),
                "max_pose_abs_delta": max_delta,
                "max_joint_abs_delta": joint_delta,
                "base_instruction": base_entry.get("instruction"),
                "direct_instruction": direct_entry.get("instruction"),
                "base_layout_hash": base_hash,
                "direct_layout_hash": direct_hash,
                "problems": problems,
            }
        )

    for row in rows:
        flag = "ok  " if not row["problems"] else "FAIL"
        print(
            f"{flag} {row['pair_id']:<62} seed={row['seed']} objs={row['objects_compared']} "
            f"max_delta={row['max_pose_abs_delta']:.3e}"
        )
    print(f"\npairs={len(rows)} failures={len(failures)}")
    for failure in failures:
        print(f"  FAIL {failure}")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "coverage": {
                        "checked": [
                            "post-reset world pose (pos+quat) of every rigid object, compared bitwise",
                            "robot joint positions after reset",
                            "seed equality per pair",
                            "target_object_id and success_radius equality",
                            "canonical layout hash equality",
                        ],
                        "not_checked": [
                            "trials outside the sampled (task, level, trial) grid given on the command line",
                            "poses at later timesteps (they depend on the policy, which is the thing under test)",
                            "static (non-rigid) props: they are AssetBase prims driven directly by cfg and are "
                            "covered by the layout hash instead",
                            "already-archived base result directories, which predate layout_hash and cannot be "
                            "re-derived without re-running them",
                        ],
                    },
                    "pairs": rows,
                    "failures": failures,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"json={out_path}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dump", "compare"), required=True)
    parser.add_argument("--ablation", choices=("base", "direct"), default="base")
    parser.add_argument("--out", default="")
    parser.add_argument("--base", default="")
    parser.add_argument("--direct", default="")
    parser.add_argument("--json", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--levels", default="loose")
    parser.add_argument("--trials", default="0")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--frames-dir", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "dump":
        if not args.out:
            raise SystemExit("--out is required for --mode dump")
        return dump(args)
    if not args.base or not args.direct:
        raise SystemExit("--base and --direct are required for --mode compare")
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
