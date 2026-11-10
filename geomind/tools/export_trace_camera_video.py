from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_uint8_image(value: Any) -> np.ndarray | None:
    try:
        image = np.asarray(value)
    except Exception:
        return None
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        return None
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


def _camera_payload(step: dict[str, Any], camera_name: str) -> Any:
    agent_obs = step.get("agent_obs") or {}
    native_obs = ((agent_obs.get("state") or {}).get("native_obs") or {})
    policy_obs = native_obs.get("policy") if isinstance(native_obs, dict) else None
    if isinstance(policy_obs, dict) and camera_name in policy_obs:
        return policy_obs.get(camera_name)
    cameras = agent_obs.get("cameras") or {}
    if isinstance(cameras, dict) and camera_name in cameras:
        return cameras.get(camera_name)
    return None


def _save_gif(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(output_path, frames, fps=max(int(fps), 1))
        return
    except Exception:
        pass

    from PIL import Image

    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(1000 / max(int(fps), 1))
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _save_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.mimwrite(output_path, frames, fps=max(int(fps), 1), macro_block_size=1)


def _save_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    if output_path.suffix.lower() == ".mp4":
        _save_mp4(frames, output_path=output_path, fps=fps)
        return
    _save_gif(frames, output_path=output_path, fps=fps)


def export_camera_video_from_trace(
    *,
    trace: list[dict[str, Any]],
    camera_name: str,
    output_path: Path,
    fps: int = 5,
    step_duration_s: float | None = None,
) -> dict[str, Any]:
    frames: list[np.ndarray] = []
    for step in trace:
        if not isinstance(step, dict):
            continue
        frame = _as_uint8_image(_camera_payload(step, str(camera_name)))
        if frame is not None:
            frames.append(frame)

    frame_repeat = 1
    expanded_frames = frames
    fps_value = max(int(fps), 1)
    if step_duration_s is not None:
        try:
            step_duration_value = float(step_duration_s)
        except Exception:
            step_duration_value = 0.0
        if step_duration_value > 0.0:
            frame_repeat = max(int(round(step_duration_value * float(fps_value))), 1)
            if frame_repeat > 1:
                expanded_frames = []
                for frame in frames:
                    for _ in range(frame_repeat):
                        expanded_frames.append(frame)

    status = {
        "saved": False,
        "camera": str(camera_name),
        "frame_count": len(frames),
        "encoded_frame_count": len(expanded_frames),
        "frame_repeat": int(frame_repeat),
        "output_video": str(output_path),
    }
    if not frames:
        status["reason"] = "missing_camera_frames"
        return status

    _save_video(expanded_frames, output_path=output_path, fps=fps_value)
    status["saved"] = True
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a camera GIF from a GeoMind trace.json.")
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--camera", default="wrist_cam")
    parser.add_argument("--output-video", default="")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--step-duration-s", type=float, default=0.0)
    args = parser.parse_args()

    trace_path = Path(args.trace_json)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(trace, list):
        raise ValueError(f"Trace payload must be a list, got: {type(trace)!r}")

    output_path = (
        Path(args.output_video)
        if args.output_video
        else trace_path.parent / f"{str(args.camera).strip() or 'camera'}.gif"
    )
    status = export_camera_video_from_trace(
        trace=trace,
        camera_name=str(args.camera),
        output_path=output_path,
        fps=int(args.fps),
        step_duration_s=(float(args.step_duration_s) if float(args.step_duration_s) > 0.0 else None),
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status.get("saved") else 1


if __name__ == "__main__":
    raise SystemExit(main())
