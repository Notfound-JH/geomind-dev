from __future__ import annotations

import argparse

from geomind_lab.bootstrap import bootstrap_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect GeoMind demonstrations.")
    parser.add_argument("--task", type=str, help="Registered IsaacLab task id.")
    parser.add_argument("--teleop-device", type=str, default="keyboard")
    return parser


def main() -> None:
    root = bootstrap_environment()
    args = build_parser().parse_args()
    print(f"GeoMind Lab collect CLI is ready at {root}. Implement task launch for: {args.task}")


if __name__ == "__main__":
    main()

