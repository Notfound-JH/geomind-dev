#!/usr/bin/env python3
"""Rewrite annotated_tasks.json to only core layout keys (task profiles + test extras)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geomind_lab.tasks.manager_based.manipulation.geomind.task_profiles import (  # noqa: E402
    ANNOTATION_LAYOUT_KEYS_TEST_EXTRA,
    registered_annotation_layout_keys,
)


def main() -> int:
    src = REPO / "assets" / "annotations" / "annotated_tasks.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print("Expected top-level object", file=sys.stderr)
        return 1

    keep = set(registered_annotation_layout_keys()) | set(ANNOTATION_LAYOUT_KEYS_TEST_EXTRA)
    filtered = {k: v for k, v in payload.items() if k in keep}
    removed = sorted(set(payload) - set(filtered))

    out_path = src
    bak = src.with_suffix(".json.bak")
    bak.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"kept {len(filtered)} layouts, removed {len(removed)} (backup: {bak})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
