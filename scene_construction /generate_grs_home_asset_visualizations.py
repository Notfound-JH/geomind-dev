#!/usr/bin/env python3
"""Render GRScenes home assets into preview grids under geomind_assets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_MODELS_ROOT = Path(
    "/home/elm/projects/InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes/models"
)
DEFAULT_DEST_ROOT = Path("/home/elm/projects/geomind_assets")
DEFAULT_OUTPUT_NAMES = {
    "layout": "GRS_home_assets_layouts",
    "object": "GRS_home_assets_objects",
}
SUPPORTED_INSTANCE_FILES = ("instance.usd", "instance.usda")

HEADER_HEIGHT = 96
CARD_WIDTH = 280
CARD_HEIGHT = 350
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
FILL_BACKGROUND_RGB = (243, 240, 234)


@dataclass(frozen=True)
class CategoryRecord:
    kind: str
    output_name: str
    relative_dir: Path
    asset_paths: tuple[Path, ...]


@dataclass(frozen=True)
class TileEntry:
    title: str
    subtitle: str
    preview_path: Path | None
    accent_seed: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render GRScenes home assets into per-category previews and overview sheets "
            "under a mirrored destination root."
        )
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help=f"GRScenes models root. Default: {DEFAULT_MODELS_ROOT}",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help=f"Destination root to hold the generated image folders. Default: {DEFAULT_DEST_ROOT}",
    )
    parser.add_argument(
        "--layout-output-name",
        default=DEFAULT_OUTPUT_NAMES["layout"],
        help=f"Destination folder name for layout categories. Default: {DEFAULT_OUTPUT_NAMES['layout']}",
    )
    parser.add_argument(
        "--object-output-name",
        default=DEFAULT_OUTPUT_NAMES["object"],
        help=f"Destination folder name for object categories. Default: {DEFAULT_OUTPUT_NAMES['object']}",
    )
    parser.add_argument(
        "--kind",
        choices=("all", "layout", "object"),
        default="all",
        help="Restrict generation to one source branch.",
    )
    parser.add_argument(
        "--max-assets-per-category",
        type=int,
        default=16,
        help="Maximum number of rendered assets to sample per category.",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=4,
        help="Upper bound for tiles per row in overview sheets.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Render resolution per asset preview.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=6,
        help="Number of SimulationApp updates after scene changes before capture.",
    )
    parser.add_argument(
        "--limit-categories",
        type=int,
        default=0,
        help="Optional limit for testing; 0 means no limit.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing preview PNGs when present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be generated without rendering or writing output files.",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int = 2,
) -> list[str]:
    words = text.replace("-", " ").replace("_", " ").replace("/", " / ").split()
    if not words:
        return [text]

    lines: list[str] = []
    current_words = [words[0]]
    overflow = False

    for word in words[1:]:
        candidate_words = [*current_words, word]
        candidate = " ".join(candidate_words)
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current_words = candidate_words
            continue

        lines.append(" ".join(current_words))
        current_words = [word]
        if len(lines) == max_lines - 1:
            overflow = True
            break

    if len(lines) < max_lines:
        if overflow:
            remaining_words = current_words
            consumed = sum(len(line.replace("/", " / ").split()) for line in lines) + len(current_words)
            remaining_words.extend(words[consumed:])
            lines.append(" ".join(remaining_words))
        else:
            lines.append(" ".join(current_words))

    trimmed: list[str] = []
    for index, line in enumerate(lines[:max_lines]):
        normalized = line.replace(" / ", "/").strip()
        if index == max_lines - 1 and overflow:
            normalized = normalized.rstrip()
            if normalized and not normalized.endswith("…"):
                normalized = normalized + "…"
        trimmed.append(normalized)
    return trimmed or [text]


def discover_categories(
    models_root: Path,
    output_names: dict[str, str],
    selected_kind: str,
    limit_categories: int,
) -> list[CategoryRecord]:
    kinds = ("layout", "object") if selected_kind == "all" else (selected_kind,)
    discovered: list[CategoryRecord] = []

    for kind in kinds:
        kind_root = models_root / kind
        if not kind_root.exists():
            print(f"Skipping missing source branch: {kind_root}")
            continue

        category_to_assets: dict[Path, list[Path]] = {}
        seen_asset_dirs: set[Path] = set()
        for instance_name in SUPPORTED_INSTANCE_FILES:
            for asset_path in kind_root.rglob(instance_name):
                asset_dir = asset_path.parent
                if asset_dir in seen_asset_dirs:
                    continue
                seen_asset_dirs.add(asset_dir)
                category_dir = asset_dir.parent
                relative_dir = category_dir.relative_to(kind_root)
                category_to_assets.setdefault(relative_dir, []).append(asset_path)

        category_items = sorted(category_to_assets.items(), key=lambda item: str(item[0]))
        if limit_categories > 0:
            category_items = category_items[:limit_categories]

        for relative_dir, assets in category_items:
            discovered.append(
                CategoryRecord(
                    kind=kind,
                    output_name=output_names[kind],
                    relative_dir=relative_dir,
                    asset_paths=tuple(sorted(assets)),
                )
            )

    return discovered


def sample_evenly(paths: tuple[Path, ...], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[len(paths) // 2]]

    indices = np.linspace(0, len(paths) - 1, num=limit, dtype=int)
    sampled: list[Path] = []
    seen: set[Path] = set()
    for index in indices.tolist():
        candidate = paths[index]
        if candidate in seen:
            continue
        sampled.append(candidate)
        seen.add(candidate)
    return sampled


def create_placeholder_preview(seed_text: str, subtitle: str) -> Image.Image:
    color = ACCENT_COLORS[sum(seed_text.encode("utf-8")) % len(ACCENT_COLORS)]
    preview = Image.new("RGB", PREVIEW_BOX, color=color)
    draw = ImageDraw.Draw(preview)
    title_font = load_font(24)
    meta_font = load_font(16)

    title = seed_text.replace("_", " ").replace("/", " ")
    title_box = draw.textbbox((0, 0), title[:28], font=title_font)
    title_width = title_box[2] - title_box[0]
    title_height = title_box[3] - title_box[1]
    draw.text(
        ((PREVIEW_BOX[0] - title_width) / 2, PREVIEW_BOX[1] * 0.34 - title_height / 2),
        title[:28],
        fill="#20303d",
        font=title_font,
    )

    subtitle_box = draw.textbbox((0, 0), subtitle[:28], font=meta_font)
    subtitle_width = subtitle_box[2] - subtitle_box[0]
    draw.text(
        ((PREVIEW_BOX[0] - subtitle_width) / 2, PREVIEW_BOX[1] * 0.62),
        subtitle[:28],
        fill="#20303d",
        font=meta_font,
    )
    return preview


def build_border_mask(rgb: np.ndarray, threshold: int = 18) -> np.ndarray:
    height, width = rgb.shape[:2]
    near_black = (
        (rgb[..., 0] <= threshold)
        & (rgb[..., 1] <= threshold)
        & (rgb[..., 2] <= threshold)
    )

    mask = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        if near_black[0, x]:
            queue.append((0, x))
        if near_black[height - 1, x]:
            queue.append((height - 1, x))
    for y in range(height):
        if near_black[y, 0]:
            queue.append((y, 0))
        if near_black[y, width - 1]:
            queue.append((y, width - 1))

    while queue:
        y, x = queue.popleft()
        if mask[y, x] or not near_black[y, x]:
            continue
        mask[y, x] = True
        if y > 0:
            queue.append((y - 1, x))
        if y + 1 < height:
            queue.append((y + 1, x))
        if x > 0:
            queue.append((y, x - 1))
        if x + 1 < width:
            queue.append((y, x + 1))

    return mask


def postprocess_render(rgba: np.ndarray) -> Image.Image:
    if rgba.shape[-1] == 4:
        rgb = rgba[..., :3].copy()
        alpha = rgba[..., 3]
        use_alpha = np.any(alpha < 8) and np.any(alpha > 8)
        background_mask = alpha < 8 if use_alpha else build_border_mask(rgb)
    else:
        rgb = rgba[..., :3].copy()
        background_mask = build_border_mask(rgb)

    rgb[background_mask] = np.array(FILL_BACKGROUND_RGB, dtype=np.uint8)
    foreground_mask = ~background_mask

    if np.any(foreground_mask):
        ys, xs = np.where(foreground_mask)
        margin = max(10, int(0.06 * max(rgb.shape[0], rgb.shape[1])))
        y_min = max(0, int(ys.min()) - margin)
        y_max = min(rgb.shape[0], int(ys.max()) + margin + 1)
        x_min = max(0, int(xs.min()) - margin)
        x_max = min(rgb.shape[1], int(xs.max()) + margin + 1)
        rgb = rgb[y_min:y_max, x_min:x_max]

    return Image.fromarray(rgb, mode="RGB")


def load_preview_image(path: Path | None, title: str, subtitle: str) -> Image.Image:
    if path is None or not path.exists():
        return create_placeholder_preview(title, subtitle)
    with Image.open(path) as image:
        return ImageOps.pad(
            image.convert("RGB"),
            PREVIEW_BOX,
            method=Image.Resampling.LANCZOS,
            color=FILL_BACKGROUND_RGB,
            centering=(0.5, 0.5),
        )


def create_tile_card(entry: TileEntry) -> Image.Image:
    card = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), color=CARD_BACKGROUND_COLOR)
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle(
        (0, 0, CARD_WIDTH - 1, CARD_HEIGHT - 1),
        radius=18,
        outline=CARD_BORDER_COLOR,
        width=2,
        fill=CARD_BACKGROUND_COLOR,
    )

    preview = load_preview_image(entry.preview_path, entry.accent_seed, entry.subtitle)
    preview_x = (CARD_WIDTH - PREVIEW_BOX[0]) // 2
    preview_y = 16
    card.paste(preview, (preview_x, preview_y))

    title_font = load_font(18)
    subtitle_font = load_font(14)
    text_draw = ImageDraw.Draw(card)
    title_lines = wrap_text(text_draw, entry.title, title_font, CARD_WIDTH - 28)
    subtitle_lines = wrap_text(text_draw, entry.subtitle, subtitle_font, CARD_WIDTH - 32)

    text_y = preview_y + PREVIEW_BOX[1] + 12
    for line in title_lines:
        box = text_draw.textbbox((0, 0), line, font=title_font)
        width = box[2] - box[0]
        text_draw.text(
            ((CARD_WIDTH - width) / 2, text_y),
            line,
            fill=HEADER_TEXT_COLOR,
            font=title_font,
        )
        text_y += 22

    text_y += 4
    for line in subtitle_lines:
        box = text_draw.textbbox((0, 0), line, font=subtitle_font)
        width = box[2] - box[0]
        text_draw.text(
            ((CARD_WIDTH - width) / 2, text_y),
            line,
            fill="#655d53",
            font=subtitle_font,
        )
        text_y += 18

    return card


def build_overview_image(
    title: str,
    subtitle: str,
    entries: Iterable[TileEntry],
    max_columns: int,
) -> Image.Image:
    tile_entries = list(entries)
    if not tile_entries:
        tile_entries = [TileEntry(title="empty", subtitle="0 assets", preview_path=None, accent_seed=title)]

    columns = min(max_columns, max(1, math.ceil(math.sqrt(len(tile_entries)))))
    rows = math.ceil(len(tile_entries) / columns)

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

    draw.text((CANVAS_PADDING, 22), title, fill=HEADER_TEXT_COLOR, font=title_font)
    draw.text((CANVAS_PADDING, 58), subtitle, fill="#655d53", font=subtitle_font)

    for index, entry in enumerate(tile_entries):
        row = index // columns
        column = index % columns
        x = CANVAS_PADDING + column * (CARD_WIDTH + CARD_GAP)
        y = HEADER_HEIGHT + CANVAS_PADDING + row * (CARD_HEIGHT + CARD_GAP)
        overview.paste(create_tile_card(entry), (x, y))

    return overview


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class IsaacAssetRenderer:
    def __init__(self, resolution: int, warmup_frames: int) -> None:
        argv_backup = sys.argv[:]
        sys.argv = [sys.argv[0]]
        from isaacsim import SimulationApp

        self._simulation_app = SimulationApp({"headless": True})
        self._simulation_app.update()
        sys.argv = argv_backup

        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdLux

        self._rep = rep
        self._omni_usd = omni.usd
        self._Gf = Gf
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._UsdLux = UsdLux
        self._resolution = resolution
        self._warmup_frames = warmup_frames

        self._stage = None
        self._render_product = None
        self._rgb_annotator = None
        self._camera_path = "/World/PreviewCamera"
        self._reset_stage()

    def close(self) -> None:
        if self._rgb_annotator is not None and self._render_product is not None:
            try:
                self._rgb_annotator.detach([self._render_product.path])
            except Exception:
                pass
        self._simulation_app.close()

    def render_to_path(self, asset_path: Path, output_path: Path) -> None:
        self._clear_asset_prims()
        asset_prim = self._stage.DefinePrim("/World/Asset", "Xform")
        asset_prim.GetReferences().AddReference(str(asset_path))
        self._set_translation(asset_prim, (0.0, 0.0, 0.0))

        self._pump_updates(self._warmup_frames)
        aligned_range = self._compute_aligned_range(asset_prim)
        if aligned_range.IsEmpty():
            raise RuntimeError(f"Empty world bound for {asset_path}")

        midpoint = aligned_range.GetMidpoint()
        minimum = aligned_range.GetMin()
        offset = (-float(midpoint[0]), -float(midpoint[1]), -float(minimum[2]))
        self._set_translation(asset_prim, offset)

        self._pump_updates(self._warmup_frames)
        aligned_range = self._compute_aligned_range(asset_prim)
        if aligned_range.IsEmpty():
            raise RuntimeError(f"Empty recentered world bound for {asset_path}")

        size = aligned_range.GetSize()
        center = aligned_range.GetMidpoint()
        max_extent = max(float(size[0]), float(size[1]), float(size[2]), 0.25)
        diagonal = max(float(np.linalg.norm([float(size[0]), float(size[1]), float(size[2])])), 0.5)
        distance = max(max_extent * 2.6, diagonal * 1.6, 1.2)

        camera_direction = np.array([-1.6, -1.2, 1.0], dtype=float)
        camera_direction /= np.linalg.norm(camera_direction)
        camera_position = np.array(
            [
                float(center[0]),
                float(center[1]),
                float(center[2]) + max(0.08 * max_extent, 0.02),
            ],
            dtype=float,
        ) + camera_direction * distance
        target = np.array(
            [
                float(center[0]),
                float(center[1]),
                float(center[2]) + max(0.05 * float(size[2]), 0.02),
            ],
            dtype=float,
        )

        self._set_camera_pose(camera_position, target)
        self._pump_updates(self._warmup_frames + 2)

        rgba = self._rgb_annotator.get_data()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            raise RuntimeError(f"No RGB data returned for {asset_path}")

        image = postprocess_render(np.array(rgba))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    def _reset_stage(self) -> None:
        if self._rgb_annotator is not None and self._render_product is not None:
            try:
                self._rgb_annotator.detach([self._render_product.path])
            except Exception:
                pass

        self._omni_usd.get_context().new_stage()
        self._pump_updates(4)
        self._stage = self._omni_usd.get_context().get_stage()
        self._stage.DefinePrim("/World", "Xform")
        self._create_lights()
        self._create_camera()
        self._pump_updates(4)

    def _create_camera(self) -> None:
        camera = self._UsdGeom.Camera.Define(self._stage, self._camera_path)
        camera.CreateFocalLengthAttr(20.0)
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateVerticalApertureAttr(24.0)
        camera.CreateClippingRangeAttr((0.01, 100000.0))
        self._set_camera_pose(np.array([3.0, 3.0, 2.0]), np.array([0.0, 0.0, 0.0]))

        self._render_product = self._rep.create.render_product(
            self._camera_path,
            (self._resolution, self._resolution),
        )
        self._rgb_annotator = self._rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annotator.attach([self._render_product.path])

    def _create_lights(self) -> None:
        dome = self._UsdLux.DomeLight.Define(self._stage, "/World/DomeLight")
        dome.CreateIntensityAttr(1800.0)
        dome.CreateColorAttr((1.0, 1.0, 1.0))

        key = self._UsdLux.DistantLight.Define(self._stage, "/World/KeyLight")
        key.CreateIntensityAttr(4500.0)
        self._set_look_at(key.GetPrim(), (8.0, -8.0, 10.0), (0.0, 0.0, 0.0))

        fill = self._UsdLux.DistantLight.Define(self._stage, "/World/FillLight")
        fill.CreateIntensityAttr(2400.0)
        self._set_look_at(fill.GetPrim(), (-10.0, 6.0, 8.0), (0.0, 0.0, 0.0))

    def _clear_asset_prims(self) -> None:
        for prim_path in ("/World/Asset",):
            if self._stage.GetPrimAtPath(prim_path).IsValid():
                self._stage.RemovePrim(prim_path)
        self._pump_updates(2)

    def _compute_aligned_range(self, prim) -> object:
        bbox_cache = self._UsdGeom.BBoxCache(
            self._Usd.TimeCode.Default(),
            [self._UsdGeom.Tokens.default_],
            useExtentsHint=True,
        )
        return bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()

    def _set_translation(self, prim, translation: tuple[float, float, float]) -> None:
        xformable = self._UsdGeom.Xformable(prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                translate_op = op
                break
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        translate_op.Set(self._Gf.Vec3d(*translation))

    def _set_camera_pose(self, position: np.ndarray, target: np.ndarray) -> None:
        camera_prim = self._stage.GetPrimAtPath(self._camera_path)
        self._set_look_at(camera_prim, tuple(position.tolist()), tuple(target.tolist()))

    def _set_look_at(
        self,
        prim,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> None:
        xformable = self._UsdGeom.Xformable(prim)
        translate_op = None
        orient_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                translate_op = op
            elif op.GetOpName() == "xformOp:orient":
                orient_op = op
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if orient_op is None:
            orient_op = xformable.AddOrientOp()

        look_at = self._Gf.Matrix4d().SetLookAt(
            self._Gf.Vec3d(*position),
            self._Gf.Vec3d(*target),
            self._Gf.Vec3d(0.0, 0.0, 1.0),
        )
        rotation = look_at.ExtractRotationQuat()
        translate_op.Set(self._Gf.Vec3d(*position))
        orient_op.Set(
            self._Gf.Quatf(
                float(rotation.GetReal()),
                self._Gf.Vec3f(*[float(value) for value in rotation.GetImaginary()]),
            )
        )

    def _pump_updates(self, count: int) -> None:
        for _ in range(max(1, count)):
            self._simulation_app.update()


def build_category_manifest(
    record: CategoryRecord,
    sampled_assets: list[Path],
    preview_paths: list[Path],
    output_dir: Path,
) -> dict:
    return {
        "kind": record.kind,
        "source_root": str(output_dir),
        "category": str(record.relative_dir),
        "total_assets": len(record.asset_paths),
        "sampled_assets": [
            {
                "asset_id": sampled.parent.name,
                "source_path": str(sampled),
                "preview_path": str(preview),
            }
            for sampled, preview in zip(sampled_assets, preview_paths)
        ],
    }


def process_category(
    renderer: IsaacAssetRenderer,
    record: CategoryRecord,
    dest_root: Path,
    max_assets_per_category: int,
    max_columns: int,
    skip_existing: bool,
) -> tuple[Path, Path, list[Path]]:
    output_category_root = dest_root / record.output_name / record.relative_dir
    asset_images_dir = output_category_root / "asset_images"
    previews_dir = asset_images_dir / "previews"
    asset_images_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    sampled_assets = sample_evenly(record.asset_paths, max_assets_per_category)
    preview_paths: list[Path] = []

    renderer._reset_stage()
    for asset_path in sampled_assets:
        preview_path = previews_dir / f"{asset_path.parent.name}.png"
        if not (skip_existing and preview_path.exists()):
            renderer.render_to_path(asset_path, preview_path)
        preview_paths.append(preview_path)

    category_title = str(record.relative_dir)
    category_subtitle = f"sampled {len(sampled_assets)} / {len(record.asset_paths)} assets"
    category_entries = [
        TileEntry(
            title=asset_path.parent.name,
            subtitle=asset_path.parent.parent.name,
            preview_path=preview_path,
            accent_seed=asset_path.parent.name,
        )
        for asset_path, preview_path in zip(sampled_assets, preview_paths)
    ]

    overview = build_overview_image(
        title=category_title,
        subtitle=category_subtitle,
        entries=category_entries,
        max_columns=max_columns,
    )
    overview_path = asset_images_dir / "asset_overview.png"
    overview.save(overview_path)

    manifest = build_category_manifest(record, sampled_assets, preview_paths, output_category_root)
    write_json(asset_images_dir / "manifest.json", manifest)
    return output_category_root, overview_path, preview_paths


def build_root_outputs(
    dest_root: Path,
    output_name: str,
    category_records: list[dict],
    max_columns: int,
) -> None:
    output_root = dest_root / output_name
    asset_images_dir = output_root / "asset_images"
    asset_images_dir.mkdir(parents=True, exist_ok=True)

    total_assets = sum(int(record["total_assets"]) for record in category_records)
    entries = [
        TileEntry(
            title=record["category"],
            subtitle=f"sampled {record['sampled_assets']} / {record['total_assets']}",
            preview_path=Path(record["cover_preview"]) if record["cover_preview"] else None,
            accent_seed=record["category"],
        )
        for record in category_records
    ]
    overview = build_overview_image(
        title=output_name,
        subtitle=f"{len(category_records)} categories, {total_assets} assets",
        entries=entries,
        max_columns=max_columns,
    )
    overview.save(asset_images_dir / "asset_overview.png")
    write_json(
        asset_images_dir / "manifest.json",
        {
            "output_name": output_name,
            "category_count": len(category_records),
            "total_assets": total_assets,
            "categories": category_records,
        },
    )


def main() -> int:
    args = parse_args()
    models_root = args.models_root.expanduser().resolve()
    dest_root = args.dest_root.expanduser().resolve()

    if not models_root.exists():
        raise FileNotFoundError(f"Models root does not exist: {models_root}")

    output_names = {
        "layout": args.layout_output_name,
        "object": args.object_output_name,
    }
    category_records = discover_categories(
        models_root=models_root,
        output_names=output_names,
        selected_kind=args.kind,
        limit_categories=args.limit_categories,
    )
    if not category_records:
        print(f"No categories discovered under {models_root}")
        return 0

    print(f"Scanning source root: {models_root}")
    print(f"Discovered {len(category_records)} categories.")

    if args.dry_run:
        for record in category_records:
            output_dir = dest_root / record.output_name / record.relative_dir / "asset_images"
            sampled_count = len(sample_evenly(record.asset_paths, args.max_assets_per_category))
            print(
                f"- {record.kind}:{record.relative_dir} -> {output_dir} "
                f"(sample {sampled_count}/{len(record.asset_paths)})"
            )
        return 0

    root_records: dict[str, list[dict]] = {name: [] for name in output_names.values()}
    renderer = IsaacAssetRenderer(resolution=args.resolution, warmup_frames=args.warmup_frames)

    try:
        for index, record in enumerate(category_records, start=1):
            output_root, overview_path, preview_paths = process_category(
                renderer=renderer,
                record=record,
                dest_root=dest_root,
                max_assets_per_category=args.max_assets_per_category,
                max_columns=args.max_columns,
                skip_existing=args.skip_existing,
            )
            sampled_count = min(args.max_assets_per_category, len(record.asset_paths))
            root_records[record.output_name].append(
                {
                    "kind": record.kind,
                    "category": str(record.relative_dir),
                    "output_dir": str(output_root),
                    "overview_path": str(overview_path),
                    "cover_preview": str(preview_paths[0]) if preview_paths else "",
                    "total_assets": len(record.asset_paths),
                    "sampled_assets": sampled_count,
                }
            )
            print(
                f"[{index}/{len(category_records)}] {record.kind}:{record.relative_dir} "
                f"-> {overview_path}"
            )
    finally:
        renderer.close()

    for output_name, records in root_records.items():
        if records:
            build_root_outputs(
                dest_root=dest_root,
                output_name=output_name,
                category_records=records,
                max_columns=args.max_columns,
            )

    print(f"Finished writing previews under {dest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
