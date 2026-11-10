#!/usr/bin/env python3
"""批量对比布局随机化效果：对同一个任务用 N 对随机种子各跑一次，导出开局相机画面。

用法示例::

    # 对 structure1 跑 10 对种子 + 1 次基线（随机化全关）
    python scripts/compare_layout_randomization.py structure1

    # 只跑 4 对、加大抖动幅度、固定元种子便于复现
    python scripts/compare_layout_randomization.py structure1 \
        --pairs 4 --meta-seed 20260726 --jitter-xy 0.05 --shift-max 0.15

    # 转发额外参数给底层 debug 脚本
    python scripts/compare_layout_randomization.py place3 --extra --variant skill

产物::

    logs/layout_randomization_compare/<task>/<timestamp>/
        frames/00_j<seed>_s<seed>__<camera>.png   <- 用来肉眼对比的开局截图
        runs/run_00/console.log                    <- 每次运行的完整输出
        runs/run_00/camera_videos/...              <- 原始 mp4
        summary.json                               <- 种子、策略、物体坐标汇总

注意：本脚本会真实拉起 Isaac Sim（headless），每次运行都要重新启动仿真，耗时较长。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEBUG_TASK_SH = REPO_ROOT / "scripts" / "debug_task.sh"

_RANDOMIZER_LINE = re.compile(r"^\[layout-randomizer\]\s*(.*)$")
_SEED_MAX = 2**31 - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用多对随机种子重复运行同一任务并导出开局相机画面，用于验证布局随机化。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("task", help="任务别名或 task_id，例如 structure1 / perfect_midpoint / place3")
    parser.add_argument("--pairs", type=int, default=10, help="随机种子对的数量")
    parser.add_argument("--meta-seed", type=int, default=None, help="生成种子对所用的元种子，缺省则每次不同")
    parser.add_argument("--no-baseline", action="store_true", help="不额外跑一次随机化全关的基线")

    parser.add_argument("--no-jitter", action="store_true", help="关闭小偏移（仅做整体平移）")
    parser.add_argument("--no-shift", action="store_true", help="关闭整体平移（仅做小偏移）")
    parser.add_argument("--jitter-xy", type=float, default=None, help="小偏移幅度，缺省用 randomizer 默认值 0.02")
    parser.add_argument("--shift-max", type=float, default=None, help="整体平移半径上限，缺省用默认值 0.15")
    parser.add_argument("--min-clearance", type=float, default=None, help="物体间最小间距，缺省 0.025")
    parser.add_argument("--bounds", default=None, help='安全区 "xmin,xmax,ymin,ymax"')

    parser.add_argument("--episodes", type=int, default=1, help="每次运行的 episode 数（越小越快）")
    parser.add_argument(
        "--video-frames",
        type=int,
        default=0,
        help=(
            "相机片段最大帧数，0 表示录全程且不追加该参数。"
            "注意 debug_task.sh 的参数转发会拆散带值选项，非 0 时请改用底层 wrapper 直接运行。"
        ),
    )
    parser.add_argument("--frame-index", type=int, default=0, help="从视频中抽取第几帧作为开局画面")
    parser.add_argument("--timeout", type=int, default=1800, help="单次运行超时秒数")
    parser.add_argument("--out-dir", default=None, help="输出根目录，缺省自动生成")
    parser.add_argument("--keep-going", action="store_true", help="某次运行失败时继续跑剩余种子")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令与种子，不真正运行")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "其余参数原样转发给底层 debug 脚本（必须放在最后）。"
            "只有 wrapper 自己认识的选项（如 --variant / --layout-key / --scene-id）才能安全转发。"
        ),
    )
    return parser.parse_args()


def build_run_env(args: argparse.Namespace, output_dir: Path, jitter_seed: int | None, shift_seed: int | None) -> dict[str, str]:
    env = dict(os.environ)
    env["HEADLESS"] = "1"
    env["HEADLESS_CAMERA_VIDEO"] = "1"
    env["OUTPUT_DIR"] = str(output_dir)
    env["AGENT_EPISODES"] = str(args.episodes)
    env["GEOMIND_LAYOUT_VERBOSE"] = "1"

    jitter_on = jitter_seed is not None
    shift_on = shift_seed is not None
    env["GEOMIND_LAYOUT_JITTER_ENABLE"] = "1" if jitter_on else "0"
    env["GEOMIND_LAYOUT_SHIFT_ENABLE"] = "1" if shift_on else "0"
    if jitter_on:
        env["GEOMIND_LAYOUT_JITTER_SEED"] = str(jitter_seed)
    else:
        env.pop("GEOMIND_LAYOUT_JITTER_SEED", None)
    if shift_on:
        env["GEOMIND_LAYOUT_SHIFT_SEED"] = str(shift_seed)
    else:
        env.pop("GEOMIND_LAYOUT_SHIFT_SEED", None)

    for key, value in (
        ("GEOMIND_LAYOUT_JITTER_XY", args.jitter_xy),
        ("GEOMIND_LAYOUT_SHIFT_MAX", args.shift_max),
        ("GEOMIND_LAYOUT_MIN_CLEARANCE", args.min_clearance),
    ):
        if value is not None:
            env[key] = repr(float(value))
    if args.bounds:
        env["GEOMIND_LAYOUT_BOUNDS"] = str(args.bounds)
    return env


def build_command(args: argparse.Namespace) -> list[str]:
    command = ["bash", str(DEBUG_TASK_SH), str(args.task)]
    forwarded: list[str] = []
    if args.video_frames > 0:
        forwarded += ["--camera-rollout-video-frames", str(args.video_frames)]
    extra = [token for token in args.extra if token != "--"]
    forwarded += extra
    if forwarded:
        command += ["--"] + forwarded
    return command


def extract_frame(video: Path, destination: Path, frame_index: int) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{max(int(frame_index), 0)})",
        "-vsync",
        "vfr",
        "-frames:v",
        "1",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return destination.is_file()


def collect_randomizer_lines(log_path: Path) -> list[str]:
    if not log_path.is_file():
        return []
    lines: list[str] = []
    for raw in log_path.read_text(errors="replace").splitlines():
        matched = _RANDOMIZER_LINE.match(raw.strip())
        if matched:
            lines.append(matched.group(1))
    return lines


def run_once(
    args: argparse.Namespace,
    command: list[str],
    label: str,
    run_dir: Path,
    jitter_seed: int | None,
    shift_seed: int | None,
    frames_dir: Path,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console.log"
    env = build_run_env(args, run_dir, jitter_seed, shift_seed)

    started = time.time()
    with log_path.open("w") as log_file:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
    elapsed = time.time() - started

    videos = sorted((run_dir / "camera_videos").rglob("*.mp4"))
    frames: list[str] = []
    for video in videos:
        camera = video.stem
        destination = frames_dir / f"{label}__{camera}.png"
        if extract_frame(video, destination, args.frame_index):
            frames.append(str(destination.relative_to(frames_dir.parent)))

    return {
        "label": label,
        "jitter_seed": jitter_seed,
        "shift_seed": shift_seed,
        "returncode": completed.returncode,
        "elapsed_sec": round(elapsed, 1),
        "randomizer": collect_randomizer_lines(log_path),
        "videos": [str(video.relative_to(run_dir)) for video in videos],
        "frames": frames,
        "log": str(log_path),
    }


def main() -> int:
    args = parse_args()

    if not DEBUG_TASK_SH.is_file():
        print(f"找不到 {DEBUG_TASK_SH}", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print("警告：未找到 ffmpeg，将只保留 mp4 而不导出 png。", file=sys.stderr)
    if args.no_jitter and args.no_shift:
        print("--no-jitter 和 --no-shift 不能同时使用，否则每次运行都相同。", file=sys.stderr)
        return 2
    if args.pairs < 1:
        print("--pairs 至少为 1。", file=sys.stderr)
        return 2

    meta_seed = args.meta_seed if args.meta_seed is not None else random.randrange(_SEED_MAX)
    rng = random.Random(meta_seed)
    seed_pairs = [
        (
            None if args.no_jitter else rng.randrange(1, _SEED_MAX),
            None if args.no_shift else rng.randrange(1, _SEED_MAX),
        )
        for _ in range(args.pairs)
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else REPO_ROOT / "logs" / "layout_randomization_compare" / str(args.task) / timestamp
    )
    frames_dir = out_dir / "frames"
    runs_dir = out_dir / "runs"
    command = build_command(args)

    print("=" * 72)
    print("GeoMind 布局随机化对比")
    print("=" * 72)
    print(f"task        : {args.task}")
    print(f"command     : {' '.join(command)}")
    print(f"meta_seed   : {meta_seed}")
    print(f"out_dir     : {out_dir}")
    print(f"baseline    : {'no' if args.no_baseline else 'yes'}")
    print("seed pairs  :")
    for index, (jitter_seed, shift_seed) in enumerate(seed_pairs):
        print(f"  [{index:02d}] jitter={jitter_seed} shift={shift_seed}")
    print("=" * 72)

    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, int | None, int | None]] = []
    if not args.no_baseline:
        plan.append(("base_off", None, None))
    plan += [
        (f"{index:02d}_j{jitter_seed}_s{shift_seed}", jitter_seed, shift_seed)
        for index, (jitter_seed, shift_seed) in enumerate(seed_pairs)
    ]

    results: list[dict] = []
    for position, (label, jitter_seed, shift_seed) in enumerate(plan, start=1):
        run_dir = runs_dir / label
        print(f"\n[{position}/{len(plan)}] {label} -> {run_dir}", flush=True)
        try:
            result = run_once(args, command, label, run_dir, jitter_seed, shift_seed, frames_dir)
        except subprocess.TimeoutExpired:
            result = {"label": label, "jitter_seed": jitter_seed, "shift_seed": shift_seed, "returncode": -1, "error": "timeout"}
        results.append(result)

        status = "ok" if result.get("returncode") == 0 else f"FAILED(rc={result.get('returncode')})"
        print(f"    status : {status}  ({result.get('elapsed_sec', '?')}s)")
        for line in result.get("randomizer", []):
            print(f"    rand   : {line}")
        if result.get("frames"):
            for frame in result["frames"]:
                print(f"    frame  : {frame}")
        elif result.get("returncode") == 0:
            print("    frame  : 未导出（该场景可能没有 Camera sensor，见 console.log）")

        if result.get("returncode") not in (0, None):
            log_path = Path(result.get("log", ""))
            if log_path.is_file():
                tail = log_path.read_text(errors="replace").splitlines()[-15:]
                print("    --- console.log 末尾 ---")
                for line in tail:
                    print(f"    | {line}")

        if result.get("returncode") != 0 and not args.keep_going:
            print("\n运行失败，已中止。加 --keep-going 可跳过失败项继续。", file=sys.stderr)
            break

    summary = {
        "task": args.task,
        "command": command,
        "meta_seed": meta_seed,
        "frame_index": args.frame_index,
        "jitter_xy": args.jitter_xy,
        "shift_max": args.shift_max,
        "min_clearance": args.min_clearance,
        "bounds": args.bounds,
        "runs": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    total_frames = sum(len(result.get("frames", [])) for result in results)
    print("\n" + "=" * 72)
    print(f"完成：{len(results)} 次运行，导出 {total_frames} 张开局截图")
    print(f"截图目录：{frames_dir}")
    print(f"汇总文件：{summary_path}")
    print("=" * 72)
    return 0 if all(result.get("returncode") == 0 for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
