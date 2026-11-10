#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = REPO_ROOT / "logs" / "scene_check_images"
DEFAULT_OUTPUT_PATH = DEFAULT_IMAGE_ROOT / "scene_check_images_table.md"
SCENE_DIR_PATTERN = re.compile(r"^scene_(\d+)$")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan scene_check_images and export a Markdown table with scene id, "
            "ego_view_camera image, and table_cam image."
        )
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Directory containing scene_xxx folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output Markdown file path.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=320,
        help="Rendered width for each image inside the Markdown table.",
    )
    return parser


def sort_key(scene_dir: Path) -> tuple[int, str]:
    match = SCENE_DIR_PATTERN.match(scene_dir.name)
    if match:
        return (int(match.group(1)), scene_dir.name)
    return (10**9, scene_dir.name)


def resolve_image(scene_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
        candidate = scene_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def to_cell(image_path: Path | None, output_path: Path, width: int) -> str:
    if image_path is None:
        return "-"

    relative_path = os.path.relpath(image_path, output_path.parent).replace(os.sep, "/")
    return f'<img src="{relative_path}" width="{int(width)}" />'


def collect_rows(image_root: Path, output_path: Path, image_width: int) -> list[str]:
    rows: list[str] = []
    scene_dirs = sorted(
        (path for path in image_root.iterdir() if path.is_dir() and SCENE_DIR_PATTERN.match(path.name)),
        key=sort_key,
    )

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        ego_image = resolve_image(scene_dir, "ego_view_camera")
        table_image = resolve_image(scene_dir, "table_cam")
        rows.append(
            f"| {scene_id} | {to_cell(ego_image, output_path, image_width)} | "
            f"{to_cell(table_image, output_path, image_width)} |"
        )
    return rows


def build_markdown(image_root: Path, output_path: Path, image_width: int) -> str:
    rows = collect_rows(image_root, output_path, image_width)
    header = [
        "# Scene Check Images",
        "",
        f"Source: `{image_root}`",
        "",
        "| scene_id | ego_view_camera | table_cam |",
        "| --- | --- | --- |",
    ]
    return "\n".join(header + rows) + "\n"


def main() -> int:
    args = build_argparser().parse_args()
    image_root = args.image_root.resolve()
    output_path = args.output.resolve()

    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(image_root, output_path, args.image_width)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote Markdown table to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
