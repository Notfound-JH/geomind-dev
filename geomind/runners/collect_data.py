from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .policy_runtime import _to_jsonable, run_episode, save_episode_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect trajectory data with oracle policy and small perturbations.")
    parser.add_argument("--task-id-env", required=True)
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-noise-scale", type=float, default=0.05, help="Trajectory perturbation scale.")
    parser.add_argument(
        "--object-perturb-range",
        type=float,
        default=0.02,
        help="Uniform XY perturbation range for target-object equivalent goal offset.",
    )
    parser.add_argument(
        "--native-object-pose-perturb-range",
        type=float,
        default=0.0,
        help="When backend=isaaclab, apply real target object XY pose perturbation per reset.",
    )
    parser.add_argument(
        "--isaac-render",
        dest="isaac_render",
        action="store_true",
        help="When backend=isaaclab, force GUI rendering (HEADLESS=0, ENABLE_CAMERAS=1).",
    )
    parser.add_argument(
        "--no-isaac-render",
        dest="isaac_render",
        action="store_false",
        help="When backend=isaaclab, force headless mode (HEADLESS=1, ENABLE_CAMERAS=1).",
    )
    parser.set_defaults(isaac_render=None)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    if str(args.backend).strip().lower() == "isaaclab" and args.isaac_render is not None:
        os.environ["HEADLESS"] = "0" if bool(args.isaac_render) else "1"
        os.environ.setdefault("ENABLE_CAMERAS", "1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir) if args.output_dir else Path("geomind") / "outputs" / "dataset" / (
        f"{timestamp}_{args.task_id_env}"
    )
    root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    episodes: list[dict[str, object]] = []
    for episode_index in range(max(1, int(args.episodes))):
        goal_perturb_xy = (
            float(rng.uniform(-float(args.object_perturb_range), float(args.object_perturb_range))),
            float(rng.uniform(-float(args.object_perturb_range), float(args.object_perturb_range))),
        )
        episode_start = time.perf_counter()
        episode = run_episode(
            task_id=str(args.task_id_env),
            policy_name="oracle",
            max_steps=int(args.max_steps),
            backend=str(args.backend),
            interactive=False,
            seed=int(args.seed) + episode_index,
            action_noise_scale=float(args.action_noise_scale),
            object_goal_perturb_xy=goal_perturb_xy,
            native_object_pose_perturb_range=float(args.native_object_pose_perturb_range),
        )
        elapsed_s = float(max(time.perf_counter() - episode_start, 0.0))
        time_steps = int(episode.summary.get("episode_steps", len(episode.trace)))
        render_steps = int(max(time_steps, 0))
        episode_dir = root / f"episode_{episode_index:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        save_episode_payload(output_dir=episode_dir, result=episode)
        (episode_dir / "summary.json").write_text(
            json.dumps(
                {
                    **dict(_to_jsonable(episode.summary)),
                    "episode_index": episode_index,
                    "policy_mode": "oracle",
                    "trajectory_duration_seconds": elapsed_s,
                    "trajectory_time_steps": time_steps,
                    "simulation_render_steps": render_steps,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        episodes.append(
            {
                "episode_index": episode_index,
                "path": str(episode_dir),
                "success": bool(episode.summary.get("success", False)),
                "episode_steps": int(episode.summary.get("episode_steps", 0)),
                "trajectory_duration_seconds": elapsed_s,
                "trajectory_time_steps": time_steps,
                "simulation_render_steps": render_steps,
                "reward_sum": float(episode.summary.get("reward_sum", 0.0)),
                "action_noise_scale": float(args.action_noise_scale),
                "object_goal_perturb_xy": [goal_perturb_xy[0], goal_perturb_xy[1]],
                "native_object_pose_perturb_range": float(args.native_object_pose_perturb_range),
            }
        )

    manifest = {
        "task_id_env": str(args.task_id_env),
        "backend": str(args.backend),
        "isaac_render": None if args.isaac_render is None else bool(args.isaac_render),
        "headless": os.environ.get("HEADLESS", ""),
        "enable_cameras": os.environ.get("ENABLE_CAMERAS", ""),
        "episodes": episodes,
        "default_policy": "oracle",
        "perturbation": {
            "action_noise_scale": float(args.action_noise_scale),
            "object_perturb_range": float(args.object_perturb_range),
            "native_object_pose_perturb_range": float(args.native_object_pose_perturb_range),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"dataset_dir={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
