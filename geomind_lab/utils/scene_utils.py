from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_PRIM_DEF_RE = re.compile(r'^\s*(def|over|class)\s+(\w+)\s+"([^"]+)"')
_SCALAR_ATTR_RE = re.compile(r"^\s*(?:double|float)\s+([A-Za-z0-9_:]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
_XFORM_ATTR_KEYS = {
    "xformOp:translate": "translate",
    "xformOp:orient": "orient",
    "xformOp:scale": "scale",
}
DEFAULT_RUNTIME_WALL_HEIGHT = 2.5


def resolve_scene_profiles_root() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "scene_profiles"


@dataclass
class PrimNode:
    name: str
    prim_type: str
    path: str
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list["PrimNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.prim_type,
            "path": self.path,
            "attrs": self.attrs,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class WallSpec:
    name: str
    path: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    size: float

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return tuple(max(abs(self.size * axis), 1e-6) for axis in self.scale)


def _parse_tuple(text: str) -> tuple[float, ...] | None:
    match = re.search(r"\(([^)]*)\)", text)
    if not match:
        return None
    values: list[float] = []
    for raw in match.group(1).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            return None
    return tuple(values)


def _parse_attr_line(line: str) -> tuple[str, Any] | None:
    for key, alias in _XFORM_ATTR_KEYS.items():
        if key in line:
            values = _parse_tuple(line)
            if values is None:
                return None
            return alias, values
    if "xformOpOrder" in line:
        match = re.search(r"xformOpOrder\s*=\s*\[([^\]]*)\]", line)
        if match:
            items = [item.strip().strip('"') for item in match.group(1).split(",") if item.strip()]
            return "xformOpOrder", items
    scalar_match = _SCALAR_ATTR_RE.match(line)
    if scalar_match and scalar_match.group(1) == "size":
        return "size", float(scalar_match.group(2))
    return None


def _coerce_tuple(value: Any, length: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)) and len(value) >= length:
        return tuple(float(item) for item in value[:length])
    return default


def _iter_prim_nodes(nodes: list[PrimNode]) -> list[PrimNode]:
    walked: list[PrimNode] = []

    def walk(node: PrimNode) -> None:
        walked.append(node)
        for child in node.children:
            walk(child)

    for node in nodes:
        walk(node)
    return walked


def parse_usda_scene(scene_path: str | Path) -> list[PrimNode]:
    path = Path(scene_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scene USD file not found: {path}")

    root = PrimNode(name="__ROOT__", prim_type="Root", path="/")
    stack: list[PrimNode] = [root]
    pending_node: PrimNode | None = None

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        prim_match = _PRIM_DEF_RE.match(line)
        if prim_match:
            prim_type = prim_match.group(2)
            name = prim_match.group(3)
            parent_path = stack[-1].path.rstrip("/")
            full_path = f"{parent_path}/{name}" if parent_path else f"/{name}"
            node = PrimNode(name=name, prim_type=prim_type, path=full_path)

            if "{" in line:
                stack[-1].children.append(node)
                stack.append(node)
            else:
                pending_node = node
            continue

        if "{" in line and pending_node is not None:
            stack[-1].children.append(pending_node)
            stack.append(pending_node)
            pending_node = None

        attr = _parse_attr_line(line)
        if attr and stack:
            key, value = attr
            stack[-1].attrs[key] = value

        if "}" in line:
            close_count = line.count("}")
            for _ in range(close_count):
                if len(stack) > 1:
                    stack.pop()

    return root.children


def extract_xform_tree(nodes: list[PrimNode]) -> list[dict[str, Any]]:
    def filter_node(node: PrimNode) -> dict[str, Any] | None:
        children = [child for child in (filter_node(c) for c in node.children) if child is not None]
        if node.prim_type == "Xform":
            return {
                "name": node.name,
                "type": node.prim_type,
                "path": node.path,
                "attrs": node.attrs,
                "children": children,
            }
        if children:
            return {
                "name": node.name,
                "type": node.prim_type,
                "path": node.path,
                "attrs": node.attrs,
                "children": children,
            }
        return None

    result: list[dict[str, Any]] = []
    for node in nodes:
        filtered = filter_node(node)
        if filtered is not None:
            result.append(filtered)

    root_empty_xforms = [
        item for item in result if item.get("type") == "Xform" and not item.get("children")
    ]
    if not root_empty_xforms:
        return result

    remaining = [item for item in result if item not in root_empty_xforms]
    object_node = next((item for item in remaining if item.get("name") == "object"), None)
    if object_node is None:
        return result

    object_children = object_node.setdefault("children", [])
    object_children.extend(root_empty_xforms)
    return remaining


def has_wall_prims(scene: str | Path | list[PrimNode]) -> bool:
    return bool(extract_wall_specs(scene))


def extract_wall_specs(scene: str | Path | list[PrimNode]) -> tuple[WallSpec, ...]:
    nodes = parse_usda_scene(scene) if isinstance(scene, (str, Path)) else scene
    walls: list[WallSpec] = []
    for node in _iter_prim_nodes(nodes):
        if node.prim_type != "Cube" or "wall" not in node.name.lower():
            continue
        attrs = node.attrs
        walls.append(
            WallSpec(
                name=node.name,
                path=node.path,
                position=_coerce_tuple(attrs.get("translate"), 3, (0.0, 0.0, 0.0)),
                orientation=_coerce_tuple(attrs.get("orient"), 4, (1.0, 0.0, 0.0, 0.0)),
                scale=_coerce_tuple(attrs.get("scale"), 3, (1.0, 1.0, 1.0)),
                size=float(attrs.get("size", 2.0)),
            )
        )
    return tuple(sorted(walls, key=lambda wall: wall.path))


def resolve_wall_runtime_dimensions(
    wall_spec: WallSpec,
    *,
    wall_height: float = DEFAULT_RUNTIME_WALL_HEIGHT,
) -> tuple[float, float, float]:
    thickness, width, _ = wall_spec.dimensions
    return (thickness, width, float(wall_height))


def resolve_wall_runtime_position(
    wall_spec: WallSpec,
    *,
    wall_height: float = DEFAULT_RUNTIME_WALL_HEIGHT,
    ground_z: float = 0.0,
) -> tuple[float, float, float]:
    return (wall_spec.position[0], wall_spec.position[1], float(ground_z) + float(wall_height) * 0.5)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Parse a USDA scene and extract Xform prim structure.")
    parser.add_argument("scene_path", type=str, help="Path to the .usda scene file.")
    parser.add_argument("--xform-only", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    nodes = parse_usda_scene(args.scene_path)
    data: Any = extract_xform_tree(nodes) if args.xform_only else [node.to_dict() for node in nodes]

    if args.pretty:
        print(json.dumps(data, indent=2, ensure_ascii=True))
    else:
        print(json.dumps(data, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    _main()
