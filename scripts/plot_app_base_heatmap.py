#!/usr/bin/env python3
"""Wrapper: GeoMind-Base family heatmap only. See scripts/plot_appendix_dashboard.py."""
from __future__ import annotations

import pathlib
import subprocess
import sys


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    cmd = [sys.executable, str(here / "plot_appendix_dashboard.py"), "--only", "heatmap", *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
