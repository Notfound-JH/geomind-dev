from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .policy_runtime import EpisodeResult, SUPPORTED_POLICIES, run_episode, save_episode_payload


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _compact_native_obs(native_obs: Any) -> Any:
    if not isinstance(native_obs, dict):
        return native_obs
    compact: dict[str, Any] = {}
    policy = native_obs.get("policy")
    if isinstance(policy, dict):
        compact_policy: dict[str, Any] = {}
        for key, value in policy.items():
            if key.endswith("_cam") or "camera" in str(key).lower():
                shape = []
                cursor = value
                while isinstance(cursor, list) and cursor:
                    shape.append(len(cursor))
                    cursor = cursor[0]
                compact_policy[str(key)] = {"omitted": True, "shape": shape, "reason": "compact_trace"}
            else:
                compact_policy[str(key)] = value
        compact["policy"] = compact_policy
    if "subtask_terms" in native_obs:
        compact["subtask_terms"] = native_obs.get("subtask_terms")
    return compact


def _compact_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_trace: list[dict[str, Any]] = []
    for step in trace:
        if not isinstance(step, dict):
            continue
        item = dict(step)
        agent_obs = dict(item.get("agent_obs") or {})
        state = dict(agent_obs.get("state") or {})
        if "native_obs" in state:
            state["native_obs"] = _compact_native_obs(state.get("native_obs"))
        agent_obs["state"] = state
        item["agent_obs"] = agent_obs
        compact_trace.append(item)
    return compact_trace


def _camera_shapes_from_policy(policy: dict[str, Any]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for key, value in policy.items():
        if not (str(key).endswith("_cam") or "camera" in str(key).lower()):
            continue
        shape = []
        cursor = value
        while isinstance(cursor, list) and cursor:
            shape.append(len(cursor))
            cursor = cursor[0]
        shapes[str(key)] = shape
    return shapes


def _write_dataset_artifacts(*, output_dir: Path, summary: dict[str, Any], trace: list[dict[str, Any]]) -> None:
    trajectory = []
    for step in trace:
        trajectory.append(
            {
                "step": step.get("step"),
                "action": step.get("action", []),
                "reward": step.get("reward", 0.0),
                "terminated": bool(step.get("terminated", False)),
                "truncated": bool(step.get("truncated", False)),
                "meta": step.get("meta", {}),
            }
        )
    (output_dir / "trajectory.json").write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    first_obs = (trace[0].get("agent_obs") if trace else {}) or {}
    native_obs = ((first_obs.get("state") or {}).get("native_obs") or {})
    policy = native_obs.get("policy") if isinstance(native_obs, dict) else {}
    policy = policy if isinstance(policy, dict) else {}
    non_camera_policy = {
        str(key): value
        for key, value in policy.items()
        if not (str(key).endswith("_cam") or "camera" in str(key).lower())
    }
    observation_info = {
        "instruction": first_obs.get("instruction", ""),
        "episode_step": first_obs.get("episode_step", 0),
        "summary": summary,
        "initial_policy_observation": non_camera_policy,
        "camera_shapes": _camera_shapes_from_policy(policy),
        "first_step_meta": (trace[0].get("meta", {}) if trace else {}),
    }
    (output_dir / "observation_info.json").write_text(
        json.dumps(observation_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run env-policy interaction for a single GeoMind task.")
    parser.add_argument("--task-id-env", required=True, help="Task id used to construct the env.")
    parser.add_argument("--policy", default="oracle", choices=SUPPORTED_POLICIES)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--interactive", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--native-object-pose-perturb-range",
        type=float,
        default=0.0,
        help="Small deterministic XY perturb range, in meters, applied to the native target object on reset.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--ablation-id",
        default=os.environ.get("ABLATION_ID", "base"),
        choices=("base", "direct"),
        help="'direct' swaps in the direct-reference instruction/appearance override; layout and seed are untouched.",
    )
    parser.add_argument(
        "--pair-id",
        default=os.environ.get("PAIR_ID", ""),
        help="Opaque tag linking a base trial to its direct counterpart (task_id|level|trial_index).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("geomind") / "outputs" / "env_policy" / (
        f"{timestamp}_{args.task_id_env}_{args.policy}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_video_dir: str | None = None
    camera_video_fps = int(str(os.environ.get("GEOMIND_CAMERA_VIDEO_FPS", "60")).strip() or "60")
    camera_video_names: list[str] | None = None
    if _truthy_env("GEOMIND_SAVE_CAMERA_VIDEO", default=False) and str(args.backend).strip().lower() == "isaaclab":
        camera_video_dir = str(output_dir / "camera_videos")
        raw_cam = str(os.environ.get("GEOMIND_CAMERA_VIDEO_CAMERA", "all")).strip()
        parsed = [n.strip() for n in raw_cam.split(",") if n.strip()]
        wants_all = any(n.lower() in {"all", "*", "auto"} for n in parsed)
        if wants_all or not parsed:
            camera_video_names = None
        else:
            camera_video_names = parsed
    result = run_episode(
        task_id=str(args.task_id_env),
        policy_name=str(args.policy),
        max_steps=int(args.max_steps),
        backend=str(args.backend),
        interactive=bool(args.interactive),
        seed=int(args.seed),
        native_object_pose_perturb_range=float(args.native_object_pose_perturb_range),
        camera_video_dir=camera_video_dir,
        camera_video_fps=int(camera_video_fps),
        camera_video_names=camera_video_names,
        ablation_id=str(args.ablation_id),
        pair_id=str(args.pair_id),
    )
    video_status: dict[str, Any] = {"saved": False, "reason": "disabled"}
    in_summary = result.summary.get("camera_video") if isinstance(result.summary, dict) else None
    if isinstance(in_summary, dict) and in_summary.get("saved"):
        video_status = dict(in_summary)
    elif _truthy_env("GEOMIND_SAVE_CAMERA_VIDEO", default=False):
        from geomind.tools.export_trace_camera_video import export_camera_video_from_trace

        camera_names = [
            name.strip()
            for name in str(os.environ.get("GEOMIND_CAMERA_VIDEO_CAMERA", "wrist_cam")).split(",")
            if name.strip()
        ]
        if not camera_names:
            camera_names = ["wrist_cam"]
        fps = int(str(os.environ.get("GEOMIND_CAMERA_VIDEO_FPS", "60")).strip() or "60")
        if (
            str(args.backend).strip().lower() == "isaaclab"
            and _truthy_env("GEOMIND_CAMERA_VIDEO_SYNC_CONTROL", default=False)
            and not _truthy_env("GEOMIND_CAMERA_VIDEO_USE_MANUAL_FPS", default=False)
        ):
            chz = result.summary.get("control_step_hz") if isinstance(result.summary, dict) else None
            if isinstance(chz, (int, float)) and float(chz) > 0.0:
                fps = max(int(round(float(chz))), 1)
        video_dir = output_dir / "camera_videos"
        video_status = {"saved": False, "videos": []}
        for camera_name in camera_names:
            video_path = video_dir / f"{camera_name}.mp4"
            status = export_camera_video_from_trace(
                trace=result.trace,
                camera_name=camera_name,
                output_path=video_path,
                fps=fps,
            )
            video_status["videos"].append(status)
        video_status["saved"] = any(bool(item.get("saved")) for item in video_status["videos"])

    trace_mode = str(os.environ.get("GEOMIND_TRACE_MODE", "full")).strip().lower()
    result_to_save = result
    if trace_mode in {"compact", "lite", "small"}:
        result_to_save = EpisodeResult(summary=result.summary, trace=_compact_trace(result.trace))

    save_episode_payload(output_dir=output_dir, result=result_to_save)
    if _truthy_env("GEOMIND_DATASET_ARTIFACTS", default=False):
        _write_dataset_artifacts(output_dir=output_dir, summary=result.summary, trace=result_to_save.trace)
    if _truthy_env("GEOMIND_SAVE_CAMERA_VIDEO", default=False):
        (output_dir / "camera_video_status.json").write_text(
            json.dumps(video_status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"trace_len={len(result.trace)}")
    if video_status.get("saved"):
        for item in video_status.get("videos", []):
            if item.get("saved"):
                print(f"camera_video={item.get('output_video')}")
    print(f"result_dir={output_dir}")
    # Force-exit to terminate any daemon threads left by a timed-out env.close()
    # (Isaac Sim RTX/DLSS shutdown can hang indefinitely; all results are already
    # written above so it is safe to bypass Python atexit hooks here).
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
