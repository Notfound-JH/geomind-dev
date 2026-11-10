"""在 headless 运行时将场景中各 IsaacLab Camera 传感器的 RGB 存成视频。"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def infer_headless_from_env() -> bool:
    """根据环境变量推断是否为无界面运行（与 IsaacLab 文档中的 HEADLESS / LIVESTREAM 语义一致）。

    注意：若在代码里用 ``AppLauncher(headless=True)`` 启动但未设置 ``HEADLESS=1``，此处可能为
    False。此时请显式传入 ``headless=True``。
    """
    try:
        headless_flag = int(os.environ.get("HEADLESS", "0"))
    except ValueError:
        headless_flag = 0
    if headless_flag != 0:
        return True
    try:
        livestream = int(os.environ.get("LIVESTREAM", "0"))
    except ValueError:
        livestream = 0
    return livestream in (1, 2)


def _env_candidates(env: Any) -> list[Any]:
    candidates = [
        env,
        getattr(env, "unwrapped", None),
        getattr(env, "_runtime_env", None),
        getattr(env, "_native_env", None),
    ]
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None:
        candidates.extend(
            [
                getattr(unwrapped, "_runtime_env", None),
                getattr(unwrapped, "_native_env", None),
            ]
        )
    runtime_env = getattr(unwrapped, "_runtime_env", None) if unwrapped is not None else None
    if runtime_env is not None:
        candidates.append(getattr(runtime_env, "_native_env", None))
    return [candidate for candidate in candidates if candidate is not None]


def _resolve_camera_env(env: Any) -> Any:
    for candidate in _env_candidates(env):
        sensors = getattr(getattr(candidate, "scene", None), "sensors", None)
        if sensors:
            return candidate
    return env


def infer_physics_step_hz(env: Any) -> float:
    for candidate in _env_candidates(env):
        physics_dt = getattr(candidate, "physics_dt", None)
        try:
            physics_dt_value = float(physics_dt)
            if physics_dt_value > 0.0:
                return 1.0 / physics_dt_value
        except Exception:
            pass
        cfg = getattr(candidate, "cfg", None)
        sim = getattr(cfg, "sim", None)
        dt = getattr(sim, "dt", None)
        try:
            dt_value = float(dt)
            if dt_value > 0.0:
                return 1.0 / dt_value
        except Exception:
            pass
    return 0.0


def infer_action_repeat(env: Any) -> int:
    for candidate in _env_candidates(env):
        cfg = getattr(candidate, "cfg", None)
        decimation = getattr(cfg, "decimation", None)
        try:
            value = int(decimation)
            if value > 0:
                return value
        except Exception:
            pass
    return 1


def resolve_camera_sampling_config(env: Any, target_sample_hz: float) -> dict[str, float | int | bool]:
    target_hz = max(float(target_sample_hz), 1.0)
    camera_env = _resolve_camera_env(env)
    physics_hz = infer_physics_step_hz(camera_env)
    capture_on_sim_step = bool(physics_hz > 0.0 and hasattr(camera_env, "sim") and hasattr(camera_env.sim, "step"))
    sim_step_stride = max(int(round(float(physics_hz) / target_hz)), 1) if capture_on_sim_step else 1
    actual_hz = float(physics_hz) / float(sim_step_stride) if capture_on_sim_step else 0.0
    return {
        "target_sample_hz": float(target_hz),
        "actual_sample_hz": float(actual_hz),
        "physics_step_hz": float(physics_hz),
        "action_repeat": int(infer_action_repeat(camera_env)),
        "sim_step_stride": int(sim_step_stride),
        "capture_on_sim_step": bool(capture_on_sim_step),
    }


def create_rollout_camera_recorder(
    env: ManagerBasedRLEnv,
    *,
    max_frames: int | None = None,
    sample_hz: float = 60.0,
    match_camera_resolution: str | None = "scene_cam",
    stream_video_dir: Path | str | None = None,
    stream_video_fps: float = 30.0,
) -> "RolloutCameraRecorder":
    sampling = resolve_camera_sampling_config(env, target_sample_hz=float(sample_hz))
    recorder = RolloutCameraRecorder(
        env,
        max_frames=max_frames,
        capture_on_sim_step=bool(sampling["capture_on_sim_step"]),
        sim_step_stride=int(sampling["sim_step_stride"]),
        output_resolution=_camera_resolution(env, match_camera_resolution),
        stream_video_dir=stream_video_dir,
        stream_video_fps=float(stream_video_fps),
    )
    recorder.sampling_config = sampling
    return recorder


def _camera_resolution(env: Any, camera_name: str | None) -> tuple[int, int] | None:
    if not camera_name:
        return None
    sensors = getattr(getattr(_resolve_camera_env(env), "scene", None), "sensors", {}) or {}
    sensor = sensors.get(str(camera_name))
    cfg = getattr(sensor, "cfg", None)
    height = getattr(cfg, "height", None)
    width = getattr(cfg, "width", None)
    try:
        height, width = int(height), int(width)
    except (TypeError, ValueError):
        return None
    return (height, width) if height > 0 and width > 0 else None


def _resize_rgb(image: torch.Tensor, resolution: tuple[int, int] | None) -> torch.Tensor:
    if resolution is None or tuple(image.shape[:2]) == tuple(resolution):
        return image
    import torch.nn.functional as functional

    height, width = resolution
    nchw = image.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    resized = functional.interpolate(nchw, size=(height, width), mode="bilinear", align_corners=False)[0]
    return resized.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()


def _sanitize_camera_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def camera_video_filename(name: str) -> str:
    """Return the stable MP4 filename used for a camera sensor name."""
    return f"{_sanitize_camera_filename(name)}.mp4"


def _iter_camera_sensors(env: ManagerBasedRLEnv) -> list[tuple[str, object]]:
    sensors = getattr(getattr(env, "scene", None), "sensors", {}) or {}
    try:
        from isaaclab.sensors import Camera  # type: ignore
    except Exception:
        Camera = None  # type: ignore[assignment]

    camera_items: list[tuple[str, object]] = []
    for name, sensor in sensors.items():
        if Camera is not None and isinstance(sensor, Camera):
            camera_items.append((str(name), sensor))
            continue
        data = getattr(sensor, "data", None)
        output = getattr(data, "output", None)
        if isinstance(output, dict) and "rgb" in output:
            camera_items.append((str(name), sensor))
    camera_items.sort(key=lambda item: item[0])
    return camera_items


def list_rgb_camera_names(env: ManagerBasedRLEnv) -> list[str]:
    """Return the sorted set of expected RGB camera sensor names."""
    return [name for name, _sensor in _iter_camera_sensors(_resolve_camera_env(env))]


def _rgb_env0_to_uint8_hwc(rgb: torch.Tensor) -> torch.Tensor | None:
    if rgb is None or rgb.numel() == 0:
        return None
    image = rgb[0].detach().cpu()
    if image.ndim != 3 or image.shape[-1] < 3:
        return None
    image = image[..., :3]
    if image.dtype != torch.uint8:
        if image.is_floating_point():
            max_val = float(image.max().item())
            if max_val <= 1.0 + 1e-5:
                image = (image * 255.0).clamp(0.0, 255.0).to(torch.uint8)
            else:
                image = image.clamp(0.0, 255.0).to(torch.uint8)
        else:
            image = image.clamp(0, 255).to(torch.uint8)
    return image.contiguous()


def capture_all_camera_rgb(
    env: ManagerBasedRLEnv,
    *,
    width: int = 512,
    height: int = 512,
) -> dict[str, torch.Tensor]:
    """Capture every RGB camera for env 0 and resize it to a fixed HWC shape."""
    import torch.nn.functional as functional

    frames: dict[str, torch.Tensor] = {}
    camera_env = _resolve_camera_env(env)
    for name, sensor in _iter_camera_sensors(camera_env):
        sensor.update(dt=0.0, force_recompute=True)
        image = _rgb_env0_to_uint8_hwc(getattr(getattr(sensor, "data", None), "output", {}).get("rgb"))
        if image is None:
            continue
        if tuple(image.shape[:2]) != (int(height), int(width)):
            nchw = image.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
            image = functional.interpolate(nchw, size=(int(height), int(width)), mode="bilinear", align_corners=False)[0]
            image = image.clamp(0, 255).to(dtype=torch.uint8).permute(1, 2, 0).contiguous()
        frames[str(name)] = image.detach().cpu()
    return frames


def _python_float_fps(fps: float) -> float:
    """PyAV/torchvision 需要原生 ``float``；``numpy.float64`` 等会触发 ``to_avrational`` 报错。"""
    item = getattr(fps, "item", None)
    if callable(item):
        return float(item())
    return float(fps)


def _write_mp4_via_ffmpeg(path: Path, clip_u8: torch.Tensor, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("PATH 中未找到 ffmpeg，无法编码 MP4。")
    _t, height, width, _c = clip_u8.shape
    height = int(height)
    width = int(width)
    raw = clip_u8.numpy().tobytes()
    fps_str = str(_python_float_fps(fps))
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        fps_str,
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(path),
    ]
    proc = subprocess.run(cmd, input=raw, capture_output=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg 编码失败 (exit {proc.returncode}): {err[:800]}")


def _write_mp4_clip(path: Path, clip: torch.Tensor, fps: float) -> None:
    """Write ``clip`` (T, H, W, C) uint8 RGB to MP4. Try torchvision (PyAV); on failure use ffmpeg stdin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError(f"clip must be (T,H,W,3) uint8, got shape {tuple(clip.shape)}")
    clip_u8 = clip.detach().cpu().contiguous()
    if clip_u8.dtype != torch.uint8:
        clip_u8 = clip_u8.to(torch.uint8)

    fps_py = _python_float_fps(fps)
    tv_error: BaseException | None = None
    try:
        from torchvision.io import write_video

        write_video(str(path), clip_u8, fps=fps_py, video_codec="libx264", options={"crf": "18"})
        return
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        tv_error = exc

    if shutil.which("ffmpeg") is not None:
        _write_mp4_via_ffmpeg(path, clip_u8, fps_py)
        return

    if tv_error is not None:
        raise tv_error
    raise ImportError(
        "无法写出 MP4：torchvision 不可用或失败，且 PATH 中找不到 ffmpeg。"
        "请安装 ffmpeg（系统包）或修复 PyAV/torchvision。"
    ) from None


def record_rgb_frame_lists(
    env: ManagerBasedRLEnv,
    num_frames: int,
    *,
    warmup_steps: int = 32,
    make_actions: Callable[[ManagerBasedRLEnv], torch.Tensor] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """采集各 Camera 的 RGB 帧列表（env 0）。先 ``warmup_steps`` 次零动作预热渲染，避免片头大量黑帧/空缓冲。

    每帧顺序：update → 读 RGB →（非末帧）step。读不到时用上一帧有效图填充，保证各相机长度一致。
    """
    if int(num_frames) <= 0:
        return {}

    camera_names = [name for name, _sensor in _iter_camera_sensors(env)]
    if not camera_names:
        return {}

    zero_actions = torch.zeros(env.action_space.shape, device=env.device)
    actions_fn = make_actions if make_actions is not None else (lambda e: zero_actions)
    last_valid: dict[str, torch.Tensor] = {}

    for _ in range(max(0, int(warmup_steps))):
        for name in camera_names:
            sensor = env.scene.sensors[name]
            sensor.update(dt=0.0, force_recompute=True)
            hwc = _rgb_env0_to_uint8_hwc(sensor.data.output.get("rgb"))
            if hwc is not None:
                last_valid[name] = hwc.clone()
        env.step(zero_actions)

    camera_names = [name for name in camera_names if name in last_valid]
    if not camera_names:
        return {}

    frame_buffers: dict[str, list[torch.Tensor]] = {name: [] for name in camera_names}

    for frame_index in range(int(num_frames)):
        for name in camera_names:
            sensor = env.scene.sensors[name]
            sensor.update(dt=0.0, force_recompute=True)
            hwc = _rgb_env0_to_uint8_hwc(sensor.data.output.get("rgb"))
            if hwc is not None:
                last_valid[name] = hwc.clone()
            else:
                hwc = last_valid[name]
            frame_buffers[name].append(hwc.clone())

        if frame_index < int(num_frames) - 1:
            env.step(actions_fn(env))

    return frame_buffers


def merge_rgb_frame_lists(
    pre: dict[str, list[torch.Tensor]],
    post: dict[str, list[torch.Tensor]],
) -> dict[str, list[torch.Tensor]]:
    """按时间顺序拼接两段同相机帧列表。"""
    keys = sorted(set(pre) | set(post))
    merged: dict[str, list[torch.Tensor]] = {}
    for key in keys:
        left = list(pre.get(key, []))
        right = list(post.get(key, []))
        if left or right:
            merged[key] = left + right
    return merged


def save_rgb_frame_lists_to_mp4s(
    frame_lists: dict[str, list[torch.Tensor]],
    output_dir: Path | str,
    *,
    fps: float = 30.0,
    camera_names: tuple[str, ...] | list[str] | set[str] | None = None,
) -> list[Path]:
    """将 ``record_rgb_frame_lists`` / ``merge_rgb_frame_lists`` 的结果写成各相机 MP4。"""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    fps_py = _python_float_fps(fps)
    allowed_names = None if camera_names is None else {str(name) for name in camera_names}
    for name in sorted(frame_lists.keys()):
        if allowed_names is not None and str(name) not in allowed_names:
            continue
        frames = frame_lists[name]
        if not frames:
            continue
        clip = torch.stack(frames, dim=0)
        if clip.ndim != 4 or clip.shape[-1] != 3:
            continue
        path = out / camera_video_filename(name)
        _write_mp4_clip(path, clip, fps_py)
        saved.append(path)
    return saved


def save_all_camera_videos_headless(
    env: ManagerBasedRLEnv,
    output_dir: Path | str,
    *,
    headless: bool | None = None,
    num_frames: int,
    fps: float = 30.0,
    make_actions: Callable[[ManagerBasedRLEnv], torch.Tensor] | None = None,
    warmup_steps: int = 32,
) -> list[Path]:
    """在 headless 模式下为每个 Camera 传感器写入一段 MP4（仅使用 env 0 的 RGB）。

    Args:
        env: 已 reset、相机已启用的 ``ManagerBasedRLEnv``。
        output_dir: 输出目录；不存在时会创建。
        headless: 是否视为 headless。``None`` 时用 :func:`infer_headless_from_env`；为 ``False`` 时直接
            返回空列表（不写文件）。
        num_frames: 采集帧数；每帧在 step 之前抓取当前画面，共执行 ``num_frames - 1`` 次 ``step``。
        fps: 视频帧率。
        make_actions: 每步动作；默认全零动作。
        warmup_steps: 写入前零动作预热步数，减轻片头无效帧。

    Returns:
        已写入的 ``.mp4`` 路径列表；非 headless 时为空列表。
    """
    if headless is None:
        headless = infer_headless_from_env()
    if not headless:
        return []

    if int(num_frames) <= 0:
        return []

    lists = record_rgb_frame_lists(
        env,
        int(num_frames),
        warmup_steps=int(warmup_steps),
        make_actions=make_actions,
    )
    return save_rgb_frame_lists_to_mp4s(lists, output_dir, fps=float(fps))


class RolloutCameraRecorder(AbstractContextManager["RolloutCameraRecorder"]):
    """Record RGB frames continuously across a real rollout by patching ``env.step``."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        *,
        max_frames: int | None = None,
        capture_on_sim_step: bool = False,
        sim_step_stride: int = 1,
        output_resolution: tuple[int, int] | None = None,
        stream_video_dir: Path | str | None = None,
        stream_video_fps: float = 30.0,
    ):
        self.env = env
        self.camera_env = _resolve_camera_env(env)
        self.max_frames = None if max_frames is None or int(max_frames) <= 0 else int(max_frames)
        self.capture_on_sim_step = bool(capture_on_sim_step)
        self.sim_step_stride = max(int(sim_step_stride), 1)
        self.output_resolution = output_resolution
        self.stream_video_dir = None if stream_video_dir is None else Path(stream_video_dir).expanduser().resolve()
        self.stream_video_fps = float(stream_video_fps)
        self._stream_processes: dict[str, subprocess.Popen[bytes]] = {}
        self._stream_paths: dict[str, Path] = {}
        self._finalized_stream_paths: list[Path] = []
        self.sampling_config: dict[str, float | int | bool] = {}
        self.frame_buffers: dict[str, list[torch.Tensor]] = {}
        self.last_valid: dict[str, torch.Tensor] = {}
        self._original_step = None
        self._original_sim_step = None
        self._installed = False
        self._frame_count = 0
        self._sim_step_counter = 0

    @property
    def frame_count(self) -> int:
        return int(self._frame_count)

    def _at_capacity(self) -> bool:
        return self.max_frames is not None and self._frame_count >= self.max_frames

    def capture_now(self) -> None:
        if self._at_capacity():
            return
        camera_items = _iter_camera_sensors(self.camera_env)
        if not camera_items:
            return

        captured_any = False
        for name, sensor in camera_items:
            sensor.update(dt=0.0, force_recompute=True)
            hwc = _rgb_env0_to_uint8_hwc(getattr(getattr(sensor, "data", None), "output", {}).get("rgb"))
            if hwc is not None:
                self.last_valid[name] = hwc.clone()
            else:
                hwc = self.last_valid.get(name)
            if hwc is None:
                continue
            frame = _resize_rgb(hwc, self.output_resolution).clone()
            if self.stream_video_dir is None:
                self.frame_buffers.setdefault(name, []).append(frame)
            else:
                self._write_stream_frame(name, frame)
            captured_any = True

        if captured_any:
            self._frame_count += 1

    def install(self) -> "RolloutCameraRecorder":
        if self._installed:
            return self

        if self.capture_on_sim_step and hasattr(self.camera_env, "sim") and hasattr(self.camera_env.sim, "step"):
            original_sim_step = self.camera_env.sim.step

            def patched_sim_step(*args, **kwargs):
                result = original_sim_step(*args, **kwargs)
                self._sim_step_counter += 1
                if self._sim_step_counter % self.sim_step_stride == 0:
                    self.capture_now()
                return result

            self._original_sim_step = original_sim_step
            self.camera_env.sim.step = patched_sim_step  # type: ignore[method-assign]
        else:
            original_step = self.env.step

            def patched_step(*args, **kwargs):
                result = original_step(*args, **kwargs)
                self.capture_now()
                return result

            self._original_step = original_step
            self.env.step = patched_step  # type: ignore[method-assign]
        self._installed = True
        return self

    def _write_stream_frame(self, name: str, frame: torch.Tensor) -> None:
        process = self._stream_processes.get(name)
        if process is None:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                self.stream_video_dir = None
                self.frame_buffers.setdefault(name, []).append(frame)
                return
            self.stream_video_dir.mkdir(parents=True, exist_ok=True)
            path = self.stream_video_dir / camera_video_filename(name)
            height, width = int(frame.shape[0]), int(frame.shape[1])
            process = subprocess.Popen(
                [
                    ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                    "-vcodec", "rawvideo", "-s", f"{width}x{height}",
                    "-pix_fmt", "rgb24", "-r", str(self.stream_video_fps), "-i", "-",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path),
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._stream_processes[name] = process
            self._stream_paths[name] = path
        if process.stdin is None:
            return
        try:
            process.stdin.write(frame.numpy().tobytes())
        except (BrokenPipeError, OSError):
            process.stdin.close()
            process.wait()
            self._stream_processes.pop(name, None)
            self._stream_paths.pop(name, None)

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._original_step is not None:
            self.env.step = self._original_step  # type: ignore[method-assign]
        if self._original_sim_step is not None and hasattr(self.camera_env, "sim"):
            self.camera_env.sim.step = self._original_sim_step  # type: ignore[method-assign]
        self._original_step = None
        self._original_sim_step = None
        self._installed = False

    def close(self) -> None:
        """Stop stream encoders and release retained frame buffers."""
        self.uninstall()
        if self._stream_processes:
            self._close_streams()
        self.frame_buffers.clear()

    def _close_streams(self) -> list[Path]:
        paths: list[Path] = []
        for name, process in list(self._stream_processes.items()):
            if process.stdin is not None:
                process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg camera export failed for {name}: {stderr[:800]}")
            path = self._stream_paths.get(name)
            if path is not None:
                paths.append(path)
        self._finalized_stream_paths.extend(paths)
        self._stream_processes.clear()
        self._stream_paths.clear()
        return sorted(paths)

    def save_mp4s(
        self,
        output_dir: Path | str,
        *,
        fps: float = 30.0,
        camera_names: tuple[str, ...] | list[str] | set[str] | None = None,
    ) -> list[Path]:
        if self.stream_video_dir is not None:
            paths = self._close_streams() if self._stream_processes else list(self._finalized_stream_paths)
            allowed_names = None if camera_names is None else {str(name) for name in camera_names}
            return [path for path in paths if path.stem in allowed_names] if allowed_names is not None else paths
        return save_rgb_frame_lists_to_mp4s(self.frame_buffers, output_dir, fps=fps, camera_names=camera_names)

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.uninstall()
        return None
