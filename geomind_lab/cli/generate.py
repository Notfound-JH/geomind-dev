from __future__ import annotations

import argparse

from geomind_lab.bootstrap import bootstrap_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GeoMind mimic datasets.")
    parser.add_argument("--task", type=str, help="Registered IsaacLab task id.")
    parser.add_argument("--source-dataset", type=str, help="Input dataset path.")
    return parser


def main() -> None:
    root = bootstrap_environment()
    args = build_parser().parse_args()
    print(
        "GeoMind Lab generate CLI is ready at "
        f"{root}. Implement mimic generation for source dataset: {args.source_dataset}"
    )


if __name__ == "__main__":
    main()

