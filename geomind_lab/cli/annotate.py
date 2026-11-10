from __future__ import annotations

import argparse

from geomind_lab.bootstrap import bootstrap_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate GeoMind demonstrations.")
    parser.add_argument("--input", type=str, help="Input episode or dataset path.")
    return parser


def main() -> None:
    root = bootstrap_environment()
    args = build_parser().parse_args()
    print(f"GeoMind Lab annotate CLI is ready at {root}. Implement annotation for: {args.input}")


if __name__ == "__main__":
    main()

