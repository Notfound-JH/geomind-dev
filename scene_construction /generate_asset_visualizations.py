#!/usr/bin/env python3
"""Generate overview images for non-stage asset folders."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_ASSETS_ROOT = Path("/home/elm/projects/geomind_assets")
SUPPORTED_ASSET_EXTENSIONS = {
    ".obj",
    ".stl",
    ".urdf",
    ".usd",
    ".usda",
    ".usdc",
}
PREVIEW_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
SKIP_DIRECTORY_NAMES = {".thumbs", "asset_images"}
HEADER_HEIGHT = 96
CARD_WIDTH = 276
CARD_HEIGHT = 332
PREVIEW_BOX = (244, 220)
CANVAS_PADDING = 28
CARD_GAP = 18
BACKGROUND_COLOR = "#f3f0ea"
HEADER_TEXT_COLOR = "#1f1b16"
CARD_BACKGROUND_COLOR = "#fffdf9"
CARD_BORDER_COLOR = "#d8d0c4"
ACCENT_COLORS = (
    "#d7e7f5",
    "#ead7f5",
    "#dcefd9",
    "#f5e2d4",
    "#f6efcf",
    "#d9ece6",
)


@dataclass(frozen=True)
class AssetEntry:
    asset_path: Path
    preview_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan an asset root, skip all stage-related folders, and build one "
            "overview image per asset directory."
        )
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=DEFAULT_ASSETS_ROOT,
        help=f"Asset root to scan. Default: {DEFAULT_ASSETS_ROOT}",
    )
    parser.add_argument(
        "--output-dir-name",
        default="asset_images",
        help="Name of the output directory to create under each asset folder.",
    )
    parser.add_argument(
        "--output-file-name",
        default="asset_overview.png",
        help="Filename for the generated overview image.",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=6,
        help="Upper bound for the number of cards per row.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the folders that would be processed.",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def should_skip_directory(directory: Path, assets_root: Path, output_dir_name: str) -> bool:
    if directory == assets_root:
        return False

    relative_parts = directory.relative_to(assets_root).parts
    lowered_parts = [part.lower() for part in relative_parts]

    if any(part.startswith(".") for part in relative_parts):
        return True
    if any("stage" in part for part in lowered_parts):
        return True
    if directory.name in SKIP_DIRECTORY_NAMES:
        return True
    if directory.name == output_dir_name:
        return True
    return False


def is_supported_asset_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_ASSET_EXTENSIONS


def discover_asset_directories(
    assets_root: Path, output_dir_name: str
) -> list[tuple[Path, list[Path]]]:
    asset_directories: list[tuple[Path, list[Path]]] = []

    for current_root, dirnames, filenames in os.walk(assets_root):
        current_path = Path(current_root)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not should_skip_directory(current_path / dirname, assets_root, output_dir_name)
        ]

        if should_skip_directory(current_path, assets_root, output_dir_name):
            continue

        asset_files = sorted(
            current_path / filename
            for filename in filenames
            if is_supported_asset_file(current_path / filename)
        )
        if asset_files:
            asset_directories.append((current_path, asset_files))

    return sorted(asset_directories, key=lambda item: str(item[0].relative_to(assets_root)))


def find_preview(asset_path: Path) -> Path | None:
    parent = asset_path.parent
    candidate_names = (
        f"{asset_path.name}.png",
        f"{asset_path.name}.jpg",
        f"{asset_path.name}.jpeg",
        f"{asset_path.stem}.png",
        f"{asset_path.stem}.jpg",
        f"{asset_path.stem}.jpeg",
        f"{asset_path.stem}.webp",
    )

    thumb_root = parent / ".thumbs"
    preferred_dirs = [thumb_root / "256x256", thumb_root, parent]
    for base_dir in preferred_dirs:
        if not base_dir.exists():
            continue
        for candidate_name in candidate_names:
            candidate = base_dir / candidate_name
            if candidate.is_file() and candidate.suffix.lower() in PREVIEW_EXTENSIONS:
                return candidate

    if thumb_root.exists():
        for candidate_name in candidate_names:
            matches = sorted(thumb_root.rglob(candidate_name))
            if matches:
                return matches[0]

    return None


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.replace("-", " ").replace("_", " ").split()
    if not words:
        return [text]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines[:2]


def create_placeholder_preview(asset_path: Path) -> Image.Image:
    digest = hashlib.md5(asset_path.stem.encode("utf-8"), usedforsecurity=False).digest()
    preview = Image.new("RGB", PREVIEW_BOX, color=ACCENT_COLORS[digest[0] % len(ACCENT_COLORS)])
    draw = ImageDraw.Draw(preview)
    title_font = load_font(28)
    meta_font = load_font(18)

    title = asset_path.stem.replace("_", " ")
    extension = asset_path.suffix.lstrip(".").upper()
    title_box = draw.textbbox((0, 0), title[:28], font=title_font)
    title_width = title_box[2] - title_box[0]
    title_height = title_box[3] - title_box[1]
    draw.text(
        ((PREVIEW_BOX[0] - title_width) / 2, PREVIEW_BOX[1] * 0.34 - title_height / 2),
        title[:28],
        fill="#20303d",
        font=title_font,
    )

    ext_box = draw.textbbox((0, 0), extension, font=meta_font)
    ext_width = ext_box[2] - ext_box[0]
    ext_height = ext_box[3] - ext_box[1]
    pill_w = ext_width + 26
    pill_h = ext_height + 14
    pill_x = (PREVIEW_BOX[0] - pill_w) / 2
    pill_y = PREVIEW_BOX[1] * 0.62
    draw.rounded_rectangle(
        (pill_x, pill_y, pill_x + pill_w, pill_y + pill_h),
        radius=12,
        fill="#fffdf9",
        outline="#98acb9",
        width=2,
    )
    draw.text(
        (pill_x + (pill_w - ext_width) / 2, pill_y + (pill_h - ext_height) / 2 - 1),
        extension,
        fill="#20303d",
        font=meta_font,
    )
    return preview


def load_preview_image(asset_path: Path, preview_path: Path | None) -> Image.Image:
    if preview_path is None:
        return create_placeholder_preview(asset_path)

    with Image.open(preview_path) as image:
        return ImageOps.pad(
            image.convert("RGB"),
            PREVIEW_BOX,
            method=Image.Resampling.LANCZOS,
            color="#ffffff",
            centering=(0.5, 0.5),
        )


def create_asset_card(asset_path: Path, preview_path: Path | None) -> Image.Image:
    card = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), color=CARD_BACKGROUND_COLOR)
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle(
        (0, 0, CARD_WIDTH - 1, CARD_HEIGHT - 1),
        radius=18,
        outline=CARD_BORDER_COLOR,
        width=2,
        fill=CARD_BACKGROUND_COLOR,
    )

    preview = load_preview_image(asset_path, preview_path)
    preview_x = (CARD_WIDTH - PREVIEW_BOX[0]) // 2
    preview_y = 16
    card.paste(preview, (preview_x, preview_y))

    title_font = load_font(18)
    meta_font = load_font(14)
    label_draw = ImageDraw.Draw(card)
    label_lines = wrap_text(label_draw, asset_path.stem, title_font, CARD_WIDTH - 28)

    text_y = preview_y + PREVIEW_BOX[1] + 14
    for line in label_lines:
        line_box = label_draw.textbbox((0, 0), line, font=title_font)
        line_width = line_box[2] - line_box[0]
        label_draw.text(
            ((CARD_WIDTH - line_width) / 2, text_y),
            line,
            fill=HEADER_TEXT_COLOR,
            font=title_font,
        )
        text_y += 22

    meta = asset_path.suffix.lstrip(".").upper()
    meta_box = label_draw.textbbox((0, 0), meta, font=meta_font)
    meta_width = meta_box[2] - meta_box[0]
    label_draw.text(
        ((CARD_WIDTH - meta_width) / 2, CARD_HEIGHT - 36),
        meta,
        fill="#655d53",
        font=meta_font,
    )
    return card


def build_overview_image(
    asset_dir: Path, assets_root: Path, entries: Iterable[AssetEntry], max_columns: int
) -> Image.Image:
    entry_list = list(entries)
    asset_count = len(entry_list)
    columns = min(max_columns, max(1, math.ceil(math.sqrt(asset_count))))
    rows = math.ceil(asset_count / columns)

    canvas_width = CANVAS_PADDING * 2 + columns * CARD_WIDTH + (columns - 1) * CARD_GAP
    canvas_height = (
        HEADER_HEIGHT
        + CANVAS_PADDING
        + rows * CARD_HEIGHT
        + (rows - 1) * CARD_GAP
        + CANVAS_PADDING
    )

    overview = Image.new("RGB", (canvas_width, canvas_height), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(overview)
    title_font = load_font(30)
    subtitle_font = load_font(18)
    folder_label = str(asset_dir.relative_to(assets_root))
    subtitle = f"{asset_count} assets"

    draw.text((CANVAS_PADDING, 22), folder_label, fill=HEADER_TEXT_COLOR, font=title_font)
    draw.text((CANVAS_PADDING, 58), subtitle, fill="#655d53", font=subtitle_font)

    for index, entry in enumerate(entry_list):
        row = index // columns
        column = index % columns
        x = CANVAS_PADDING + column * (CARD_WIDTH + CARD_GAP)
        y = HEADER_HEIGHT + CANVAS_PADDING + row * (CARD_HEIGHT + CARD_GAP)
        card = create_asset_card(entry.asset_path, entry.preview_path)
        overview.paste(card, (x, y))

    return overview


def build_entries(asset_files: Iterable[Path]) -> list[AssetEntry]:
    return [AssetEntry(asset_path=asset_file, preview_path=find_preview(asset_file)) for asset_file in asset_files]


def process_directory(
    asset_dir: Path,
    asset_files: list[Path],
    assets_root: Path,
    output_dir_name: str,
    output_file_name: str,
    max_columns: int,
    dry_run: bool,
) -> Path:
    output_dir = asset_dir / output_dir_name
    output_path = output_dir / output_file_name

    if dry_run:
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    overview = build_overview_image(asset_dir, assets_root, build_entries(asset_files), max_columns)
    overview.save(output_path)
    return output_path


def main() -> int:
    args = parse_args()
    assets_root = args.assets_root.expanduser().resolve()

    if not assets_root.exists():
        raise FileNotFoundError(f"Asset root does not exist: {assets_root}")

    asset_directories = discover_asset_directories(assets_root, args.output_dir_name)
    if not asset_directories:
        print(f"No eligible asset directories found under {assets_root}")
        return 0

    print(f"Scanning asset root: {assets_root}")
    print(f"Found {len(asset_directories)} non-stage asset directories.")

    for asset_dir, asset_files in asset_directories:
        output_path = process_directory(
            asset_dir=asset_dir,
            asset_files=asset_files,
            assets_root=assets_root,
            output_dir_name=args.output_dir_name,
            output_file_name=args.output_file_name,
            max_columns=args.max_columns,
            dry_run=args.dry_run,
        )
        relative_dir = asset_dir.relative_to(assets_root)
        print(f"- {relative_dir}: {len(asset_files)} assets -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
