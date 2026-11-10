#!/usr/bin/env python3
"""打印 GeoMind 任务注册表中的全部自然语言指令（与 build_task_metadata 一致）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.mimic.utils.task_metadata_registry import (  # noqa: E402
    build_task_metadata,
    declared_task_ids,
    supported_task_versions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="列出所有已注册 task_id 在各 task_version 下的指令文本。")
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 数组，每项含 task_id, task_version, instruction",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="按 instruction 去重：只打印每种指令出现一次（附带 task_id:version 列表）",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for task_id in sorted(declared_task_ids()):
        for task_version in supported_task_versions(task_id):
            meta = build_task_metadata(task_id, task_version=task_version)
            instruction = str(meta.get("instruction", "") or "").strip()
            rows.append(
                {
                    "task_id": task_id,
                    "task_version": task_version,
                    "instruction": instruction,
                }
            )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if args.dedupe:
        by_text: dict[str, list[str]] = {}
        for row in rows:
            key = row["instruction"]
            tag = f"{row['task_id']}:{row['task_version']}"
            by_text.setdefault(key, []).append(tag)
        for instruction in sorted(by_text.keys(), key=lambda s: (len(s), s)):
            tags = ", ".join(sorted(by_text[instruction]))
            print(instruction)
            print(f"  -> {tags}")
            print()
        return

    for row in rows:
        print(f"{row['task_id']}\t{row['task_version']}\t{row['instruction']}")


if __name__ == "__main__":
    main()
