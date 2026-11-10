#!/usr/bin/env python3
"""Write the resolved GeoMind task name and natural-language goal to a text file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import (  # noqa: E402
    resolve_task_annotation,
    resolve_task_request,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, help="Debug task id from debug_task_registry.json")
    parser.add_argument("--output", required=True, type=Path, help="Destination task_info.txt path")
    args = parser.parse_args()

    request = resolve_task_request(task_id=args.task_id)
    annotation = resolve_task_annotation(request.layout_key, request.task_index)
    task_name = str(annotation.get("task_id") or request.task_id).strip()
    goal = str((annotation.get("protocol") or {}).get("query_text") or "").strip()
    if not goal:
        goal = str(annotation.get("query_text") or annotation.get("instruction") or "").strip()

    payload = "\n".join(
        (
            f"Task name: {task_name}",
            f"Task id: {request.task_id}",
            f"Variant: {request.variant}",
            f"Scene: {request.scene} (scene_id={request.scene_id})",
            f"Layout: {request.layout_key} (task_index={request.task_index})",
            f"Goal: {goal or '(no annotation goal text)'}",
            "",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())