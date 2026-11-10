#!/usr/bin/env python3
"""
交互式人工审核：遍历一次 eval run 下的视频，逐条输入 1（成功）或 0（失败），
写入 JSON 状态文件并可导出 CSV / 打印按任务汇总的成功率。

目录约定（与常见 eval 输出一致）：
  <runs_root>/runs/<task_id>/<level>/seed_<int>/episode_<idx>/camera_videos/<camera>.mp4
复核时会读取同一条 episode 下的 observation_info.json 中的 instruction 作为指令目标。
默认：result.json 中自动化 success=true 的 trial 会写入状态为 auto_rule_success（预标成功），但仍排进人工队列观看确认；加 --no-queue-auto-rule-success 可恢复旧行为（预标成功即不再进队）。
每个 task_id 下待人工超过 12 条则随机抽 12 条，抽样池写入状态文件，resume 时沿用同一池；已手标确认的不再出现（可用参数关闭、重抽或改上限）。

用法示例：
  python scripts/human_review_run_videos.py /path/to/run_eval_internvla
  # 默认 --player auto：有桌面 + 已装 mpv 才自动弹窗播片；无头 Linux 自动行式（适合 VS Code 点开 mp4）
  # 默认按 <runs_root>/summary.json 的自动化 success_rate 升序（成功率为 0 的 task+level 优先）
  python scripts/human_review_run_videos.py /path/to/run_eval_internvla --order path
  # 无头 / 仅 SSH：用行式，在 VS Code 里点路径打开视频
  python scripts/human_review_run_videos.py /path/to/run_eval_internvla --player line
  python scripts/human_review_run_videos.py /path/to/run_eval_internvla --task SomeTask_v0
  # 仅一个 --task 且未写 --state 时：状态写入 <runs>/human_check/<task_id>_human_check.json（与总表 human_review_state.json 分离）
  # 指定 --task 且未写 --max-human-per-task 时默认 0：该任务内未标的轨迹全部入队
  # 强制清空该任务复核记录后重标：加 --force-human-check
  # 不想复核「规则预标成功」的条目、恢复旧跳过行为：加 --no-queue-auto-rule-success
  python scripts/human_review_run_videos.py /path/to/run_eval_internvla --export-csv human_labels.csv
  python scripts/human_review_run_videos.py /path/to/run_eval_internvla --summarize-only /path/to/human_review_state.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import re
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import termios
import textwrap
import tty
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal


EPISODE_RE = re.compile(r"episode_(\d+)$")
SEED_RE = re.compile(r"seed_(\d+)$")


@dataclass(frozen=True)
class Trial:
    """一次 trial：task / level / seed / episode，对应一个 episode 目录下的一段视频。"""

    task: str
    level: str
    seed: str
    episode: str
    video_path: Path

    @property
    def trial_id(self) -> str:
        return f"{self.task}|{self.level}|{self.seed}|{self.episode}"

    @property
    def episode_dir(self) -> Path:
        """camera_videos 的上一级，即 episode_xxxx 目录。"""
        return self.video_path.parent.parent

    @classmethod
    def from_video_path(cls, video: Path) -> Trial | None:
        # .../task/level/seed_XXX/episode_YYYY/camera_videos/foo.mp4
        parts = video.parts
        try:
            cam_idx = parts.index("camera_videos")
        except ValueError:
            return None
        if cam_idx < 4:
            return None
        episode_dir = parts[cam_idx - 1]
        seed_dir = parts[cam_idx - 2]
        level = parts[cam_idx - 3]
        task = parts[cam_idx - 4]
        em = EPISODE_RE.match(episode_dir)
        sm = SEED_RE.match(seed_dir)
        if not em or not sm:
            return None
        return cls(
            task=task,
            level=level,
            seed=sm.group(1),
            episode=em.group(1),
            video_path=video.resolve(),
        )


def load_task_level_auto_success_rates(summary_path: Path) -> dict[tuple[str, str], float] | None:
    """读取 eval 的 summary.json 里 by_task_level，得到 (task_id, success_level) -> 自动化 success_rate。"""
    if not summary_path.is_file():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    out: dict[tuple[str, str], float] = {}
    for row in data.get("by_task_level", []):
        tid = row.get("task_id")
        sl = row.get("success_level")
        sr = row.get("success_rate")
        if isinstance(tid, str) and isinstance(sl, str) and isinstance(sr, (int, float)):
            out[(tid, sl)] = float(sr)
    return out or None


def sort_trials_auto_rate_first(
    trials: list[Trial],
    rates: dict[tuple[str, str], float],
    *,
    path_tiebreak: bool = True,
) -> None:
    """按自动化成功率升序：success_rate 为 0 的 task+level 最先，其次更高；无统计的组排在 1.0 之后。"""

    def rate_for(t: Trial) -> float:
        r = rates.get((t.task, t.level))
        if r is not None:
            return r
        # 无对应行：排在已知「全失败」之后、避免误插到最前
        return 1.0

    def key(t: Trial) -> tuple:
        r = rate_for(t)
        tie = (str(t.video_path),) if path_tiebreak else ()
        return (r, t.task, t.level, t.seed, int(t.episode)) + tie

    trials.sort(key=key)


def apply_trial_order(
    trials: list[Trial],
    order: str,
    runs_root: Path,
    summary_json: Path | None,
) -> None:
    summary_path = (summary_json or (runs_root / "summary.json")).resolve()
    rates = load_task_level_auto_success_rates(summary_path)

    if order == "auto_rate":
        if rates is None:
            print(
                f"未找到或无法解析 {summary_path} 中的 by_task_level，回退为 path 排序。",
                file=sys.stderr,
            )
            trials.sort(key=lambda t: str(t.video_path))
        else:
            sort_trials_auto_rate_first(trials, rates)
            print(
                f"已按 {summary_path.name} 的自动化 success_rate 升序排列（成功率为 0 的 task+level 优先）。",
            )
        return
    if order == "task":
        trials.sort(key=lambda t: (t.task, t.level, t.seed, int(t.episode), str(t.video_path)))
    elif order == "random":
        rng = random.Random(42)
        rng.shuffle(trials)
    else:
        trials.sort(key=lambda t: str(t.video_path))


def discover_trials(runs_root: Path, camera_name: str) -> list[Trial]:
    runs_dir = runs_root / "runs"
    if not runs_dir.is_dir():
        raise SystemExit(f"未找到 runs 目录: {runs_dir}")
    pattern = f"**/camera_videos/{camera_name}"
    videos = sorted(runs_dir.glob(pattern))
    by_episode: dict[tuple[str, str, str, str], Trial] = {}
    for v in videos:
        t = Trial.from_video_path(v)
        if t is None:
            continue
        key = (t.task, t.level, t.seed, t.episode)
        # 若同一 episode 下重复命中多个文件，保留第一个（已按路径排序）
        by_episode.setdefault(key, t)
    return list(by_episode.values())


def filter_trials_by_task(
    trials: list[Trial],
    *,
    task_ids: list[str] | None,
    task_regex: str | None,
) -> list[Trial]:
    """按 task_id 筛选：--task 为完全匹配（可多个并集）；--task-regex 为正则并集。"""
    parts = [x.strip() for x in (task_ids or []) if isinstance(x, str) and x.strip()]
    idset = set(parts)
    rex = re.compile(task_regex.strip()) if (task_regex and str(task_regex).strip()) else None
    if not idset and not rex:
        return trials
    out: list[Trial] = []
    for t in trials:
        if idset and t.task in idset:
            out.append(t)
        elif rex is not None and rex.search(t.task):
            out.append(t)
    return out


def load_episode_instruction(episode_dir: Path) -> str:
    """从 episode 目录下的 observation_info.json 读取 VLA 自然语言指令。"""
    info = episode_dir / "observation_info.json"
    if not info.is_file():
        return ""
    try:
        data = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    ins = data.get("instruction")
    if ins is None:
        return ""
    return str(ins).strip()


def format_instruction_block(text: str, *, width: int) -> str:
    if not text:
        return "（未找到 observation_info.json 或 instruction 为空）"
    lines = textwrap.wrap(
        text,
        width=max(40, width),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines)


def load_state(path: Path) -> dict:
    if not path.is_file():
        return {"version": 1, "labels": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "labels" not in data:
        data["labels"] = {}
    return data


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def labels_int_from_state(state: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in state.get("labels", {}).items():
        if isinstance(v, dict) and "human_success" in v:
            out[k] = int(v["human_success"])
        elif isinstance(v, int):
            out[k] = int(v)
    return out


def is_trial_pending_human_review(
    tid: str, state: dict, *, queue_auto_rule_success: bool
) -> bool:
    """是否仍须进入人工复核队列（未标、或仅规则预标且要求排队复核）。"""
    ent = state.get("labels", {}).get(tid)
    if ent is None:
        return True
    if isinstance(ent, int):
        return False
    if isinstance(ent, dict):
        if "human_success" not in ent:
            return True
        if queue_auto_rule_success and ent.get("source") == "auto_rule_success":
            return True
        return False
    return True


def load_eval_rule_success(episode_dir: Path) -> bool | None:
    """读取 episode/result.json 中自动化成功标志；无文件或解析失败返回 None（不自动跳过）。"""
    rj = episode_dir / "result.json"
    if not rj.is_file():
        return None
    try:
        data = json.loads(rj.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "success" in data:
        return bool(data["success"])
    ev = data.get("evaluation")
    if isinstance(ev, dict) and "success" in ev:
        return bool(ev["success"])
    return None


def apply_auto_rule_success_labels(
    trials: list[Trial],
    state: dict,
    state_path: Path,
    *,
    enabled: bool,
) -> int:
    """对尚无 labels 条目的 trial，若 result.json 判定 success 则写入 human_success=1（source=auto_rule_success 预标）。"""
    if not enabled:
        return 0
    state.setdefault("labels", {})
    added = 0
    for t in trials:
        tid = t.trial_id
        if tid in state["labels"]:
            continue
        ok = load_eval_rule_success(t.episode_dir)
        if ok is not True:
            continue
        state["labels"][tid] = {
            "human_success": 1,
            "updated_at": _now_iso(),
            "source": "auto_rule_success",
        }
        added += 1
    if added:
        save_state(state_path, state)
    return added


def resolve_human_review_queue(
    ordered_trials: list[Trial],
    state: dict,
    state_path: Path,
    *,
    max_per_task: int,
    seed: int,
    reset_pools: bool,
    queue_auto_rule_success: bool,
) -> tuple[list[Trial], int, int]:
    """
    构建本回合人工复核队列（支持 resume）。

    - 默认：未标、或仅有 auto_rule_success 预标的 trial 仍入队，待人眼确认。
    - queue_auto_rule_success=False 时：与旧行为一致，已写入 auto_rule_success 的不再入队。
    - max_per_task <= 0：全部待人工入队，不写抽样池。
    - max_per_task > 0：每个 task 首次超过上限时在 state.meta.human_sample_pools 中固定 trial_id 列表；
      之后继续跑只在该池里尚未标注的子集上复核，不会重新随机换一批（除非池已标完、改种子、或 --fresh-human-sample）。
    """
    trial_by_id = {t.trial_id: t for t in ordered_trials}
    need = [
        t
        for t in ordered_trials
        if is_trial_pending_human_review(t.trial_id, state, queue_auto_rule_success=queue_auto_rule_success)
    ]
    n_need = len(need)
    if not need:
        return [], n_need, 0
    if max_per_task <= 0:
        return need, n_need, 0

    meta = state.setdefault("meta", {})
    if reset_pools:
        meta.pop("human_sample_pools", None)

    pools: dict = meta.get("human_sample_pools")
    if not isinstance(pools, dict):
        pools = {}
        meta["human_sample_pools"] = pools

    if meta.get("human_sample_seed") != seed:
        pools.clear()
    meta["human_sample_seed"] = seed

    rng = random.Random(seed)
    picked_ids: set[str] = set()
    changed = False

    by_task: dict[str, list[Trial]] = defaultdict(list)
    for t in need:
        by_task[t.task].append(t)

    for task, grp in by_task.items():
        if len(grp) <= max_per_task:
            ids = [x.trial_id for x in grp]
            picked_ids.update(ids)
            if pools.get(task) != ids:
                pools[task] = ids
                changed = True
            continue

        old_pool = pools.get(task)
        if isinstance(old_pool, list) and old_pool:
            survivors = [
                tid
                for tid in old_pool
                if tid in trial_by_id
                and is_trial_pending_human_review(tid, state, queue_auto_rule_success=queue_auto_rule_success)
            ]
            if survivors:
                picked_ids.update(survivors)
                pruned = [tid for tid in old_pool if tid in trial_by_id]
                if pruned != old_pool:
                    pools[task] = pruned
                    changed = True
                continue

        k_draw = min(len(grp), max_per_task)
        new_ids = [x.trial_id for x in rng.sample(grp, k_draw)]
        pools[task] = new_ids
        picked_ids.update(new_ids)
        changed = True

    if changed:
        save_state(state_path, state)

    queue = [
        t
        for t in ordered_trials
        if t.trial_id in picked_ids
        and is_trial_pending_human_review(t.trial_id, state, queue_auto_rule_success=queue_auto_rule_success)
    ]
    dropped = max(0, n_need - len(queue))
    return queue, n_need, dropped


def summarize(labels: dict[str, int], trials: Iterable[Trial]) -> None:
    trial_by_id = {t.trial_id: t for t in trials}
    by_task: dict[str, list[int]] = defaultdict(list)
    by_task_level: dict[tuple[str, str], list[int]] = defaultdict(list)
    missing = 0
    for tid, v in labels.items():
        if v not in (0, 1):
            continue
        tr = trial_by_id.get(tid)
        if tr is None:
            missing += 1
            continue
        by_task[tr.task].append(v)
        by_task_level[(tr.task, tr.level)].append(v)

    def rate(vals: list[int]) -> tuple[int, int, float]:
        n = len(vals)
        s = sum(vals)
        return s, n, (s / n) if n else 0.0

    print("\n=== 人工标注汇总 ===")
    o_s, o_n, o_r = rate([v for v in labels.values() if v in (0, 1)])
    print(f"总体: {o_s}/{o_n} = {o_r:.4f}")
    print("\n按任务:")
    for task in sorted(by_task.keys()):
        s, n, r = rate(by_task[task])
        print(f"  {task}: {s}/{n} = {r:.4f}")
    print("\n按任务 + level:")
    for (task, level) in sorted(by_task_level.keys()):
        s, n, r = rate(by_task_level[(task, level)])
        print(f"  {task} [{level}]: {s}/{n} = {r:.4f}")
    if missing:
        print(f"\n（有 {missing} 条标注的 trial_id 在当前扫描结果中未找到，已忽略汇总）")


def export_csv(path: Path, trials: list[Trial], labels: dict[str, int]) -> None:
    rows = []
    for t in trials:
        lab = labels.get(t.trial_id)
        rows.append(
            {
                "trial_id": t.trial_id,
                "task": t.task,
                "level": t.level,
                "seed": t.seed,
                "episode": t.episode,
                "video_path": str(t.video_path),
                "human_success": "" if lab is None else lab,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["trial_id", "task", "level", "seed", "episode", "video_path", "human_success"],
        )
        w.writeheader()
        w.writerows(rows)


def try_open_video(path: Path) -> None:
    for cmd in (["xdg-open", str(path)], ["open", str(path)]):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue


def _likely_graphic_session() -> bool:
    """Linux 下无 DISPLAY/WAYLAND 时视为无本地图形，mpv 弹窗不可用（常见于无头机、纯 SSH）。"""
    if platform.system() == "Linux":
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _wait_path_exists(path: Path, timeout_s: float = 8.0, interval_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(interval_s)
    return False


def _mpv_ipc_send(sock_path: Path, command: list) -> None:
    payload = json.dumps({"command": command}, ensure_ascii=False) + "\n"
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(str(sock_path))
        client.sendall(payload.encode("utf-8"))
        try:
            client.settimeout(0.5)
            client.recv(65536)
        except OSError:
            pass
    finally:
        client.close()


def _mpv_loadfile(sock_path: Path, video: Path) -> None:
    _mpv_ipc_send(sock_path, ["loadfile", str(video.resolve()), "replace"])


def _mpv_quit(sock_path: Path | None) -> None:
    if sock_path is None or not sock_path.exists():
        return
    try:
        _mpv_ipc_send(sock_path, ["quit"])
    except OSError:
        pass


def _clear_terminal() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _read_review_key() -> str:
    """终端原始模式读取单键；↑/↓ 返回 'up'/'down'。"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch in ("1", "0", "u", "U", "s", "S", "q", "Q", "?", "\r", "\n"):
            return ch.lower() if ch.isalpha() else ch
        if ch == "\x1b":
            if select.select([sys.stdin], [], [], 0.08)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":
                        return "up"
                    if ch3 == "B":
                        return "down"
                elif ch2 == "O":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":
                        return "up"
                    if ch3 == "B":
                        return "down"
            return "esc"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _persist_labels(state: dict, labels: dict[str, int], state_path: Path) -> None:
    state["labels"] = {k: {"human_success": int(v), "updated_at": _now_iso()} for k, v in labels.items()}
    save_state(state_path, state)


def run_interactive(
    trials: list[Trial],
    state_path: Path,
    open_player: bool,
    eval_root: Path,
    *,
    show_instruction: bool,
    instruction_width: int,
    ui: Literal["mpv", "line"],
    queue_auto_rule_success: bool,
) -> None:
    state = load_state(state_path)
    labels: dict[str, int] = {}
    for k, v in state.get("labels", {}).items():
        if isinstance(v, dict) and "human_success" in v:
            labels[k] = int(v["human_success"])
        elif isinstance(v, int):
            labels[k] = v
    history: list[str] = []
    instruction_cache: dict[str, str] = {}

    def instruction_for(t: Trial) -> str:
        key = str(t.episode_dir.resolve())
        if key not in instruction_cache:
            instruction_cache[key] = load_episode_instruction(t.episode_dir)
        return instruction_cache[key]

    def rule_autofill_banner(t: Trial) -> str | None:
        ent = state.get("labels", {}).get(t.trial_id)
        if (
            queue_auto_rule_success
            and isinstance(ent, dict)
            and ent.get("source") == "auto_rule_success"
            and "human_success" in ent
        ):
            return (
                f"【规则预标】result.json 为成功，已记 human_success={int(ent['human_success'])}；"
                "请观看后按 1/0 确认或改正。"
            )
        return None

    def render_trial_screen(t: Trial, idx: int, n: int) -> None:
        _clear_terminal()
        done_here = len(trials) - n
        print(f"共 {len(trials)} 条 trial，已标 {done_here}，待标 {n}  |  队列中 {idx + 1}/{n}")
        print("-" * 72)
        print(t.trial_id)
        print(f"视频: {t.video_path}")
        hint = rule_autofill_banner(t)
        if hint:
            print(f"\n{hint}\n")
        if show_instruction:
            block = format_instruction_block(instruction_for(t), width=instruction_width)
            print("\n指令 / 目标:")
            print(block)
        print()
        if ui == "mpv":
            print("↑ 上一条视频   ↓ 下一条视频   |   1 成功   0 失败   s 跳过   u 撤销   q 退出   ? 帮助")
        else:
            print("输入: 1 成功   0 失败   s 跳过   u 撤销   q 退出   ? 帮助")
        print(f"\n状态文件: {state_path}")

    pending = [
        t
        for t in trials
        if is_trial_pending_human_review(t.trial_id, state, queue_auto_rule_success=queue_auto_rule_success)
    ]
    done_n = len(trials) - len(pending)
    mpv_proc: subprocess.Popen | None = None
    mpv_sock: Path | None = None

    try:
        if not pending:
            print("没有待标注的 trial。")
            return

        if ui == "mpv":
            if not shutil.which("mpv"):
                raise SystemExit("未找到 mpv。请安装: sudo apt install mpv   或使用 --player line")
            mpv_sock = Path(tempfile.gettempdir()) / f"geomind_hr_{os.getpid()}.sock"
            try:
                if mpv_sock.exists():
                    mpv_sock.unlink()
            except OSError:
                pass
            mpv_proc = subprocess.Popen(
                [
                    "mpv",
                    "--really-quiet",
                    "--idle=yes",
                    "--keep-open=yes",
                    "--force-window=yes",
                    f"--input-ipc-server={mpv_sock}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not _wait_path_exists(mpv_sock):
                print("mpv IPC 套接字未在超时内创建，请检查 mpv 是否正常。", file=sys.stderr)
                raise SystemExit(1)

            print(
                f"共 {len(trials)} 条 trial，已标注 {done_n}，待标注 {len(pending)}。\n"
                "已启动 mpv：在**本终端**用 ↑/↓ 切换待标视频，mpv 会自动换片；按 1/0 写入当前条结果。\n"
                f"状态文件: {state_path}\n",
                flush=True,
            )
            idx = 0
            while pending and 0 <= idx < len(pending):
                t = pending[idx]
                render_trial_screen(t, idx, len(pending))
                try:
                    assert mpv_sock is not None
                    _mpv_loadfile(mpv_sock, t.video_path)
                except OSError as exc:
                    print(f"\nmpv 加载失败: {exc}", file=sys.stderr)

                while True:
                    k = _read_review_key()
                    if k in ("\r", "\n"):
                        continue
                    if k == "?":
                        print(
                            "\n↑/↓ 在待标注列表中切换（mpv 自动换片）\n"
                            "1 / 0  标注当前条成功或失败并前进\n"
                            "s 跳过当前条   u 撤销上一标注   q 保存退出\n"
                        )
                        continue
                    if k == "q":
                        _persist_labels(state, labels, state_path)
                        print("\n已保存并退出。")
                        return
                    if k == "up":
                        idx = max(0, idx - 1)
                        break
                    if k == "down":
                        idx = min(len(pending) - 1, idx + 1)
                        break
                    if k == "u":
                        if not history:
                            print("\n没有可撤销的记录。")
                            continue
                        last_id = history.pop()
                        labels.pop(last_id, None)
                        _persist_labels(state, labels, state_path)
                        pending = [
                            x
                            for x in trials
                            if is_trial_pending_human_review(
                                x.trial_id, state, queue_auto_rule_success=queue_auto_rule_success
                            )
                        ]
                        if not pending:
                            print("\n已全部标完（撤销后无待标）。")
                            return
                        for j, tt in enumerate(pending):
                            if tt.trial_id == last_id:
                                idx = j
                                break
                        else:
                            idx = min(idx, len(pending) - 1)
                        print(f"\n已撤销: {last_id}")
                        break
                    if k == "s":
                        idx += 1
                        if idx >= len(pending):
                            _persist_labels(state, labels, state_path)
                            print("\n已跳过到队列末尾，状态已保存。")
                            return
                        break
                    if k in ("1", "0"):
                        val = int(k)
                        labels[t.trial_id] = val
                        history.append(t.trial_id)
                        state.setdefault("meta", {})
                        state["meta"]["eval_root"] = str(eval_root)
                        slot = idx
                        _persist_labels(state, labels, state_path)
                        pending = [
                            x
                            for x in trials
                            if is_trial_pending_human_review(
                                x.trial_id, state, queue_auto_rule_success=queue_auto_rule_success
                            )
                        ]
                        if not pending:
                            print(f"\n已记录 {t.trial_id} -> {val}  （全部完成）")
                            return
                        idx = min(slot, len(pending) - 1)
                        break
            _persist_labels(state, labels, state_path)
            print(f"\n状态已保存: {state_path}")
            return

        print(
            f"共 {len(trials)} 条 trial，已标注 {done_n}，待标注 {len(pending)}。\n"
            "输入: 1=成功  0=失败  u=撤销上一条  s=跳过本条  q=保存并退出\n"
            "     ? 显示帮助\n"
            f"状态文件: {state_path}\n"
        )
        idx = 0
        while idx < len(pending):
            t = pending[idx]
            print("-" * 72)
            print(f"[{idx + 1}/{len(pending)}] {t.trial_id}")
            print(f"视频: {t.video_path}")
            hint = rule_autofill_banner(t)
            if hint:
                print(hint)
            if show_instruction:
                block = format_instruction_block(instruction_for(t), width=instruction_width)
                print("指令 / 目标:")
                print(block)
            if open_player:
                try_open_video(t.video_path)

            raw = input("标注 (1/0/u/s/q/?): ").strip().lower()
            if raw in ("?", "h", "help"):
                print("1 成功 | 0 失败 | u 撤销 | s 跳过 | q 退出")
                if show_instruction:
                    print("（每条会显示 observation_info.json 中的 instruction）")
                print()
                continue
            if raw == "q":
                break
            if raw == "u":
                if not history:
                    print("没有可撤销的记录。\n")
                    continue
                last_id = history.pop()
                labels.pop(last_id, None)
                _persist_labels(state, labels, state_path)
                print(f"已撤销: {last_id}\n")
                pending = [
                    x
                    for x in trials
                    if is_trial_pending_human_review(
                        x.trial_id, state, queue_auto_rule_success=queue_auto_rule_success
                    )
                ]
                idx = 0
                for j, tt in enumerate(pending):
                    if tt.trial_id == last_id:
                        idx = j
                        break
                continue
            if raw == "s":
                idx += 1
                continue
            if raw not in ("0", "1"):
                print("无效输入，请重试。\n")
                continue
            val = int(raw)
            labels[t.trial_id] = val
            history.append(t.trial_id)
            state.setdefault("meta", {})
            state["meta"]["eval_root"] = str(eval_root)
            _persist_labels(state, labels, state_path)
            print(f"已记录 {t.trial_id} -> {val}\n")
            idx += 1

        _persist_labels(state, labels, state_path)
        print(f"状态已保存: {state_path}")
    finally:
        if mpv_proc is not None:
            _mpv_quit(mpv_sock)
            try:
                mpv_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                mpv_proc.kill()


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="人工审核 eval run 视频并统计成功率")
    p.add_argument(
        "runs_root",
        nargs="?",
        type=Path,
        help="eval 输出根目录（其下应有 runs/）",
    )
    p.add_argument("--camera", default="table_cam.mp4", help="camera_videos 下的文件名，默认 table_cam.mp4")
    p.add_argument(
        "--state",
        type=Path,
        default=None,
        help="JSON 状态路径；默认 <runs_root>/human_review_state.json；仅一个 --task 时默认 human_check/<task_id>_human_check.json",
    )
    p.add_argument(
        "--force-human-check",
        action="store_true",
        help="若状态文件在 <runs_root>/human_check/ 下，启动前删除该文件以强制重新复核（危险：丢失该任务专存记录）",
    )
    p.add_argument("--open-player", action="store_true", help="行式模式（--player line）下每条用 xdg-open/open 打开视频")
    p.add_argument(
        "--player",
        choices=("auto", "mpv", "line"),
        default="auto",
        help="复核界面：auto 有 mpv 且有图形会话则用 mpv+↑↓；无头 Linux 自动行式。mpv/line 见说明",
    )
    p.add_argument(
        "--no-instruction",
        action="store_true",
        help="不读取 episode 下 observation_info.json，复核时不打印指令",
    )
    p.add_argument(
        "--instruction-width",
        type=int,
        default=72,
        metavar="N",
        help="终端上指令自动换行宽度（默认 72）",
    )
    p.add_argument(
        "--max-human-per-task",
        type=int,
        default=None,
        metavar="N",
        help="每 task 入队上限（抽样用）。未指定时：整 run 默认 12；若用了 --task/--task-regex 则默认 0（该筛选下未标全入队）。0=不限制",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="按 task 抽样随机种子；与状态文件中记录不一致时会清空抽样池并重抽",
    )
    p.add_argument(
        "--no-auto-rule-success",
        action="store_true",
        help="不把 result.json 中 success=true 的 trial 写入 auto_rule_success 预标（全部待人手标）",
    )
    p.add_argument(
        "--no-queue-auto-rule-success",
        action="store_true",
        help="已 auto_rule_success 预标成功的 trial 不再进人工队列（旧行为：不观看）；默认会进队待人眼确认",
    )
    p.add_argument(
        "--fresh-human-sample",
        action="store_true",
        help="丢弃已保存的每 task 人工抽样池并重新抽样（默认 resume 沿用旧池，已标过的不再出现）",
    )
    p.add_argument(
        "--order",
        choices=("auto_rate", "path", "task", "random"),
        default="auto_rate",
        help="遍历顺序：auto_rate 按 summary.json 自动化成功率升序（0 优先，默认）| path | task | random",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="与 --order auto_rate 配合，指定 summary.json（默认 <runs_root>/summary.json）",
    )
    p.add_argument(
        "--task",
        action="append",
        default=None,
        metavar="TASK_ID",
        help="只处理 task_id 完全一致的 trial；可重复为并集。仅一条 --task 且无 --state 时用 human_check/<task>_human_check.json",
    )
    p.add_argument(
        "--task-regex",
        default=None,
        metavar="PATTERN",
        help="只处理 task_id 匹配该正则的 trial；与 --task 并集。未同时指定 --max-human-per-task 时默认不抽样",
    )
    p.add_argument("--export-csv", type=Path, default=None, help="导出含 human_success 列的 CSV（未标的为空）")
    p.add_argument(
        "--summarize-only",
        type=Path,
        default=None,
        help="仅读取该 state JSON 并与 runs_root 扫描结果汇总打印（不进入交互）",
    )
    args = p.parse_args(argv)

    task_filter_on = bool(args.task) or bool(args.task_regex and str(args.task_regex).strip())
    if args.max_human_per_task is None:
        args.max_human_per_task = 0 if task_filter_on else 12
        if task_filter_on:
            print(
                "与 --task/--task-regex 联用且未指定 --max-human-per-task：使用 0（不抽样），"
                "筛选范围内尚未标注的 trial 全部进入队列；已标过的（含此前抽样已验证）不再出现。",
                file=sys.stderr,
            )

    def _apply_task_filter(trials_in: list[Trial]) -> list[Trial]:
        spec = bool(args.task) or bool(args.task_regex and str(args.task_regex).strip())
        tlist = filter_trials_by_task(
            trials_in,
            task_ids=args.task,
            task_regex=args.task_regex,
        )
        if spec and not tlist:
            print("没有 trial 匹配 --task / --task-regex。", file=sys.stderr)
            raise SystemExit(1)
        if spec:
            print(f"按任务筛选: {len(trials_in)} -> {len(tlist)} 条 trial。", file=sys.stderr)
        return tlist

    if args.summarize_only is not None:
        sp = args.summarize_only
        if not sp.is_file():
            print(f"找不到 state 文件: {sp}", file=sys.stderr)
            return 1
        if args.runs_root is None:
            print("与 --summarize-only 一起使用时仍需提供 runs_root 以扫描 trial 列表。", file=sys.stderr)
            return 1
        runs_rr = args.runs_root.resolve()
        trials = discover_trials(runs_rr, args.camera)
        apply_trial_order(trials, args.order, runs_rr, args.summary_json)
        trials = _apply_task_filter(trials)
        st = load_state(sp)
        raw_labels = st.get("labels", {})
        labels: dict[str, int] = {}
        for k, v in raw_labels.items():
            if isinstance(v, dict):
                labels[k] = int(v["human_success"])
            else:
                labels[k] = int(v)
        summarize(labels, trials)
        return 0

    if args.runs_root is None:
        p.print_help()
        return 1

    runs_root = args.runs_root.resolve()

    def _single_task_id_for_dedicated_state() -> str | None:
        if args.task_regex and str(args.task_regex).strip():
            return None
        if not args.task or len(args.task) != 1:
            return None
        t = args.task[0].strip()
        return t or None

    if args.state is not None:
        state_path = Path(args.state).resolve()
    else:
        st_only = _single_task_id_for_dedicated_state()
        if st_only is not None:
            state_path = (runs_root / "human_check" / f"{st_only}_human_check.json").resolve()
            print(f"单任务复核状态文件: {state_path}", file=sys.stderr)
        else:
            state_path = (runs_root / "human_review_state.json").resolve()

    trials = discover_trials(runs_root, args.camera)
    apply_trial_order(trials, args.order, runs_root, args.summary_json)
    trials = _apply_task_filter(trials)

    hc_dir = (runs_root / "human_check").resolve()
    if args.force_human_check and state_path.is_file():
        if state_path.parent == hc_dir:
            state_path.unlink()
            print(f"已 --force-human-check，已删除: {state_path}", file=sys.stderr)
        else:
            print(
                f"--force-human-check 仅删除位于 {hc_dir} 下的状态文件，当前路径未删除: {state_path}",
                file=sys.stderr,
            )

    if args.export_csv:
        st = load_state(state_path)
        raw_labels = st.get("labels", {})
        labels: dict[str, int] = {}
        for k, v in raw_labels.items():
            if isinstance(v, dict):
                labels[k] = int(v["human_success"])
            else:
                labels[k] = int(v)
        export_csv(args.export_csv.resolve(), trials, labels)
        print(f"已导出: {args.export_csv.resolve()}")
        summarize(labels, trials)
        return 0

    if not sys.stdin.isatty():
        print("标准输入不是终端，无法进行交互式标注。", file=sys.stderr)
        return 1

    has_mpv = bool(shutil.which("mpv"))
    has_gui = _likely_graphic_session()
    if args.player == "mpv" and not has_mpv:
        print("指定了 --player mpv 但未找到 mpv 可执行文件。", file=sys.stderr)
        return 1
    if args.player == "mpv" and not has_gui:
        print(
            "警告：当前无 DISPLAY/WAYLAND_DISPLAY，mpv 多半无法出窗口；"
            "无头复核请改用 --player line，在 VS Code 资源管理器中打开打印的 mp4 路径。",
            file=sys.stderr,
        )
    if args.player == "auto":
        use_mpv = has_mpv and has_gui
        ui: Literal["mpv", "line"] = "mpv" if use_mpv else "line"
        if not has_mpv:
            print(
                "未检测到 mpv：使用行式复核（本机有桌面且已装 mpv 时可自动弹窗播片）。",
                file=sys.stderr,
            )
        elif not has_gui:
            print(
                "未检测到图形会话（无 DISPLAY/WAYLAND_DISPLAY）：使用行式复核。\n"
                "数据在远端时，可在 VS Code 远程工作区里展开路径，点开 mp4 观看后再回终端按 1/0。",
                file=sys.stderr,
            )
    elif args.player == "mpv":
        ui = "mpv"
    else:
        ui = "line"

    state = load_state(state_path)
    n_auto = apply_auto_rule_success_labels(
        trials,
        state,
        state_path,
        enabled=not args.no_auto_rule_success,
    )
    if n_auto:
        state = load_state(state_path)
    labels = labels_int_from_state(state)
    queue_auto = not args.no_queue_auto_rule_success
    human_trials, n_need, dropped = resolve_human_review_queue(
        trials,
        state,
        state_path,
        max_per_task=args.max_human_per_task,
        seed=args.sample_seed,
        reset_pools=args.fresh_human_sample,
        queue_auto_rule_success=queue_auto,
    )
    cap = args.max_human_per_task
    if args.no_auto_rule_success:
        auto_msg = "已关闭 result.json 规则预标"
    elif queue_auto:
        auto_msg = f"规则预标成功: {n_auto} 条（仍排入人工队列观看确认）"
    else:
        auto_msg = f"规则预标成功: {n_auto} 条（--no-queue-auto-rule-success：预标条目不进队、不观看）"
    msg_parts = [
        auto_msg,
        f"仍待人工确认: {n_need} 条",
        f"本回合进入队列: {len(human_trials)} 条",
    ]
    if cap > 0 and dropped > 0:
        msg_parts.append(f"按 task 每任务至多 {cap} 条抽样，未入队 {dropped} 条（种子 {args.sample_seed}）")
    elif cap > 0 and n_need > 0:
        msg_parts.append(f"各 task 待人工条数均未超过 {cap}，全部进入队列")
    if cap > 0:
        if queue_auto:
            msg_parts.append(
                "resume：每 task 抽样池保存在状态文件；已手标确认的不再入队；"
                "改 --sample-seed 或加 --fresh-human-sample 会清空池并重抽。"
            )
        else:
            msg_parts.append(
                "resume：每 task 抽样池保存在状态文件，已标/仅规则预标且不入队的 trial 不再出现；"
                "改 --sample-seed 或加 --fresh-human-sample 会清空池并重抽。"
            )
    elif cap == 0 and n_need > 0:
        if queue_auto:
            msg_parts.append(
                "当前为不抽样模式：筛选范围内待确认的 trial 全部入队（含规则预标成功项，待人眼按 1/0 确认）。"
            )
        else:
            msg_parts.append(
                "当前为不抽样模式：筛选范围内未标的 trial 全部入队（已有人工/规则预标且不入队条目不重复复核）。"
            )
    print("\n".join(msg_parts), flush=True)

    if not human_trials:
        st0 = load_state(state_path)
        still_pending = sum(
            1
            for t in trials
            if is_trial_pending_human_review(t.trial_id, st0, queue_auto_rule_success=queue_auto)
        )
        if still_pending == 0:
            print("当前筛选范围内已无待人工确认条目。", flush=True)
        else:
            print("人工队列为空（多为每 task 抽样上限未抽中；或检查筛选条件）。", flush=True)
        summarize(labels, trials)
        return 0

    run_interactive(
        human_trials,
        state_path,
        args.open_player,
        runs_root,
        show_instruction=not args.no_instruction,
        instruction_width=args.instruction_width,
        ui=ui,
        queue_auto_rule_success=queue_auto,
    )
    st = load_state(state_path)
    labels = labels_int_from_state(st)
    summarize(labels, trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
