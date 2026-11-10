from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from geomind.core.registry import registry
from geomind.core.resolver import resolve_task_request
from geomind.runners.policy_runtime import SUPPORTED_POLICIES, run_episode, save_episode_payload, try_save_trajectory_video


def _load_preset_task_ids(name: str) -> list[str]:
    preset_path = Path("geomind") / "configs" / "presets" / f"{name}.json"
    if not preset_path.is_file():
        return []
    payload = json.loads(preset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    task_ids = payload.get("task_ids", [])
    if not isinstance(task_ids, list):
        return []
    return [str(item).strip() for item in task_ids if str(item).strip()]


def _resolve_task_ids(group: str, explicit_task_ids: list[str]) -> list[str]:
    if explicit_task_ids:
        return [resolve_task_request(task_id).resolved_task_id for task_id in explicit_task_ids]

    preset_task_ids = _load_preset_task_ids(str(group).strip().lower())
    if preset_task_ids:
        return [resolve_task_request(task_id).resolved_task_id for task_id in preset_task_ids]

    normalized = str(group).strip().lower()
    if normalized in {"base", "aug"}:
        candidate_ids = registry.select_by_split(normalized)
    else:
        candidate_ids = registry.list_task_ids(include_placeholders=False)
    return [resolve_task_request(task_id).resolved_task_id for task_id in candidate_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GeoMind task suites with env-policy runtime.")
    parser.add_argument("--group", default="base", help="Task group: base | aug | all")
    parser.add_argument("--task-ids", nargs="*", default=[], help="Explicit task ids")
    parser.add_argument("--policy", default="oracle", choices=SUPPORTED_POLICIES)
    parser.add_argument("--interactive", action="store_true", default=False)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--no-save-video", action="store_true", default=False)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    registry.register_gym_envs()
    task_ids = _resolve_task_ids(args.group, list(args.task_ids))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir) if args.output_dir else Path("geomind") / "outputs" / "suite" / timestamp
    root.mkdir(parents=True, exist_ok=True)

    save_video = bool(args.save_video and not args.no_save_video)
    per_task_results = []
    for task_id in task_ids:
        task_output = root / task_id
        task_output.mkdir(parents=True, exist_ok=True)
        episode_results: list[dict[str, object]] = []
        for episode_index in range(max(1, int(args.episodes_per_task))):
            episode = run_episode(
                task_id=task_id,
                policy_name=str(args.policy),
                max_steps=int(args.max_steps),
                backend=str(args.backend),
                interactive=bool(args.interactive),
                seed=int(args.seed) + episode_index,
            )
            episode_dir = task_output / f"episode_{episode_index:03d}"
            save_episode_payload(output_dir=episode_dir, result=episode)
            video_status = {"saved": False, "reason": "disabled"}
            if save_video:
                video_status = try_save_trajectory_video(
                    trace=episode.trace,
                    output_path=episode_dir / "trajectory.gif",
                    fps=10,
                )
            episode_results.append(
                {
                    "episode_index": episode_index,
                    "summary": episode.summary,
                    "video": video_status,
                }
            )
        success_count = sum(1 for item in episode_results if bool(dict(item["summary"]).get("success", False)))
        reach_count = sum(1 for item in episode_results if bool(dict(item["summary"]).get("reach_success", False)))
        task_summary = {
            "task_id": task_id,
            "policy": str(args.policy),
            "interactive": bool(args.interactive),
            "episodes": episode_results,
            "episodes_total": len(episode_results),
            "success_count": success_count,
            "success_rate": float(success_count) / float(len(episode_results)) if episode_results else 0.0,
            "reach_count": reach_count,
            "reach_rate": float(reach_count) / float(len(episode_results)) if episode_results else 0.0,
        }
        (task_output / "task_summary.json").write_text(
            json.dumps(task_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        per_task_results.append(task_summary)

    success_count = sum(int(item.get("success_count", 0)) for item in per_task_results)
    reach_count = sum(int(item.get("reach_count", 0)) for item in per_task_results)
    total_episodes = sum(int(item.get("episodes_total", 0)) for item in per_task_results)
    summary = {
        "group": args.group,
        "policy": str(args.policy),
        "interactive": bool(args.interactive),
        "episodes_per_task": int(args.episodes_per_task),
        "save_video": bool(save_video),
        "task_count": len(task_ids),
        "episode_count": total_episodes,
        "success_count": success_count,
        "success_rate": (float(success_count) / float(total_episodes)) if total_episodes else 0.0,
        "reach_count": reach_count,
        "reach_rate": (float(reach_count) / float(total_episodes)) if total_episodes else 0.0,
        "max_steps": int(args.max_steps),
        "backend": str(args.backend),
        "tasks": per_task_results,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"suite_dir={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
