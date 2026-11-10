from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize run_suite output directory.")
    parser.add_argument("--suite-dir", required=True)
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    summary_path = suite_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    task_results = []
    for result_path in suite_dir.glob("*/result.json"):
        task_results.append(json.loads(result_path.read_text(encoding="utf-8")))
    success = sum(1 for item in task_results if bool(item.get("success", False)))
    output = {
        "task_count": len(task_results),
        "success_count": success,
        "success_rate": (float(success) / float(len(task_results))) if task_results else 0.0,
        "tasks": task_results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
