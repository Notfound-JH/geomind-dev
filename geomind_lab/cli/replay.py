from __future__ import annotations

import argparse

from geomind_lab.bootstrap import bootstrap_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay GeoMind demonstrations.")
    parser.add_argument("--task", type=str, help="Registered IsaacLab task id.")
    parser.add_argument("--dataset", type=str, help="Dataset file path.")
    return parser


def main() -> None:
    root = bootstrap_environment()
    args = build_parser().parse_args()
    print(f"GeoMind Lab replay CLI is ready at {root}. Implement replay for dataset: {args.dataset}")


if __name__ == "__main__":
    main()

