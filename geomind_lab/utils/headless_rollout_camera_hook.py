"""Headless rollout 下写相机 MP4 的共用逻辑（OUTPUT_DIR、HEADLESS_CAMERA_VIDEO、子目录命名）。"""

from __future__ import annotations

import os
from pathlib import Path


def truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def camera_rollout_video_enabled(args_cli) -> bool:
    if bool(getattr(args_cli, "camera_rollout_video", False)):
        return True
    return truthy_env("HEADLESS_CAMERA_VIDEO")


def rollout_camera_slug(task_label: str | None, env) -> str:
    def _slug(text: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))

    if task_label:
        return _slug(task_label)
    layout_key = str(getattr(env.cfg, "task_layout_key", "layout"))
    task_index = int(getattr(env.cfg, "task_index", 0))
    scene_id = int(getattr(env.cfg, "scene_id", 0))
    return _slug(f"scene{scene_id}_{layout_key}_task{task_index}")


def resolve_rollout_camera_video_dir(args_cli, task_label: str | None, env) -> Path | None:
    if not bool(getattr(args_cli, "headless", False)):
        return None
    if not camera_rollout_video_enabled(args_cli):
        return None
    out_root = os.environ.get("OUTPUT_DIR", "").strip()
    if not out_root:
        print(
            "headless_camera_video: skipped (OUTPUT_DIR unset; export OUTPUT_DIR or unset HEADLESS_CAMERA_VIDEO)",
            flush=True,
        )
        return None
    compact_video_root = os.environ.get("CAMERA_VIDEO_DIR", "").strip()
    root_path = Path(compact_video_root or out_root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    if compact_video_root:
        return root_path
    return root_path / "camera_videos" / rollout_camera_slug(task_label, env)
