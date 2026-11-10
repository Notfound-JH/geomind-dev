from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate selected legacy annotation layouts into geomind local assets.")
    parser.add_argument(
        "--source",
        default="assets/annotations/annotated_tasks.json",
        help="Source annotated_tasks.json path (repo-relative).",
    )
    parser.add_argument("--layout-keys", nargs="+", required=True)
    parser.add_argument("--output", default="geomind/configs/legacy/layouts/standard.json")
    args = parser.parse_args()

    source_path = Path(args.source)
    payload = _read_json(source_path)
    kept = {key: payload[key] for key in args.layout_keys if key in payload}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"migrated_layouts={len(kept)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
