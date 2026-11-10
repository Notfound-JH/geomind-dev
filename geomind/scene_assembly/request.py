"""Small, explicit scene-assembly language; this does not execute robot actions.

Examples::

    put 4 blue cans in grid of 2* *2 in scene 60.
    place 5 blue_can in t_shape in scene 60
    在场景60中，将4个蓝色罐摆成2×2网格。

Structured requests use canonical asset names. Natural-language requests also
accept the aliases below. Unrecognised text is rejected instead of being ignored.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from numbers import Real


class RequestError(ValueError):
    """The request is ambiguous, unsupported, or outside the assembly contract."""


ASSET_IDS = (
    "plate_1", "plate_2", "blue_can", "red_can", "cube_can", "low_can",
    "glass_can", "green_tin", "rubiks_cube", "red_box", "brown_box",
    "yellow_bottle", "white_bottle", "glass_cup", "banana", "apple", "lime",
    "pomegranate", "avocado", "bowl_1", "bowl_2", "basket_1", "basket_2",
    "basket_3", "bin", "cup_1", "cup_2", "pot_1", "pot_2", "basket_4",
    "box_1", "box_2", "box_3", "blue_tin", "tin",
)
LAYOUT_TYPES = ("grid", "line", "ring", "random", "t_shape")


def _alias_key(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", unicodedata.normalize("NFKC", value).strip().lower())


_ALIASES = {_alias_key(asset): asset for asset in ASSET_IDS}
_EXTRA_ALIASES = {
    "plate_1": ("plate 1", "plates 1", "1号盘", "盘子1", "盘子一"),
    "plate_2": ("plate 2", "plates 2", "2号盘", "盘子2", "盘子二"),
    "blue_can": ("blue cans", "blue coffee can", "blue coffee cans", "蓝色罐", "蓝罐", "蓝色咖啡罐"),
    "red_can": ("red cans", "tomato soup can", "tomato soup cans", "红色罐", "红罐", "番茄汤罐"),
    "cube_can": ("cube cans", "potted meat can", "potted meat cans", "方罐", "方形罐", "肉罐头"),
    "low_can": ("low cans", "tuna can", "tuna cans", "矮罐", "金枪鱼罐头"),
    "glass_can": ("glass cans", "glass jar", "glass jars", "玻璃罐"),
    "green_tin": ("green tins", "绿色罐", "绿罐"),
    "rubiks_cube": ("rubiks cubes", "rubik's cube", "rubik's cubes", "rubik cube", "rubik cubes", "魔方"),
    "red_box": ("red boxes", "cracker box", "cracker boxes", "红色盒", "红盒", "饼干盒"),
    "brown_box": ("brown boxes", "pudding box", "pudding boxes", "棕色盒", "布丁盒"),
    "yellow_bottle": ("yellow bottles", "mustard bottle", "mustard bottles", "黄色瓶", "黄瓶", "芥末瓶"),
    "white_bottle": ("white bottles", "bleach bottle", "bleach bottles", "白色瓶", "白瓶", "漂白剂瓶"),
    "glass_cup": ("glass cups", "玻璃杯"),
    "banana": ("bananas", "香蕉"),
    "apple": ("apples", "苹果"),
    "lime": ("limes", "青柠", "青柠檬"),
    "pomegranate": ("pomegranates", "石榴"),
    "avocado": ("avocados", "牛油果", "鳄梨"),
    "bowl_1": ("bowls 1", "1号碗", "碗1"),
    "bowl_2": ("bowls 2", "2号碗", "碗2"),
    "basket_1": ("baskets 1", "1号篮", "篮子1"),
    "basket_2": ("baskets 2", "2号篮", "篮子2"),
    "basket_3": ("baskets 3", "3号篮", "篮子3"),
    "basket_4": ("baskets 4", "4号篮", "篮子4"),
    "bin": ("bins", "垃圾桶", "收纳桶"),
    "cup_1": ("cups 1", "1号杯", "杯子1"),
    "cup_2": ("cups 2", "2号杯", "杯子2"),
    "pot_1": ("pots 1", "1号锅", "锅1"),
    "pot_2": ("pots 2", "2号锅", "锅2"),
    "box_1": ("boxes 1", "1号盒", "盒子1"),
    "box_2": ("boxes 2", "2号盒", "盒子2"),
    "box_3": ("boxes 3", "3号盒", "盒子3"),
    "blue_tin": ("blue tins", "蓝色铁罐"),
    "tin": ("tins", "铁罐"),
}
for _asset, _names in _EXTRA_ALIASES.items():
    for _name in _names:
        _ALIASES[_alias_key(_name)] = _asset


def normalize_asset(name: str) -> str:
    """Resolve a supported name without guessing ambiguous generic asset names."""
    if not isinstance(name, str) or _alias_key(name) not in _ALIASES:
        raise RequestError(f"Unknown asset {name!r}; use a supported canonical asset ID.")
    return _ALIASES[_alias_key(name)]


def _integer(value: object, field: str, low: int, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        limit = f"{low}..{high}" if high is not None else f">= {low}"
        raise RequestError(f"{field} must be an integer {limit} (not a boolean).")
    return value


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise RequestError(f"{field} must be a finite positive number.")
    return float(value)


def _bounded_number(value: object, field: str, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= high:
        raise RequestError(f"{field} must be a finite number in 0..{high:g}.")
    return float(value)


def _mapping(value: object, field: str, allowed: set[str], required: set[str]) -> Mapping:
    if not isinstance(value, Mapping):
        raise RequestError(f"{field} must be an object.")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise RequestError(f"Unknown {field} fields: {sorted(map(str, unknown))}.")
    if missing:
        raise RequestError(f"Missing {field} fields: {sorted(missing)}.")
    return value


def validate_request(data: dict) -> dict:
    """Return a fresh canonical request, or raise :class:`RequestError`.

    ``schema_version`` and ``seed`` default to 1 and 0. A grid needs rows and
    cols together; when both are omitted, only a perfect-square count is inferred.
    ``spacing_m`` is optional on all layouts. Line also accepts ``axis`` (x/y),
    ring accepts ``radius_m``. An explicit ``randomization`` object enables
    randomization by default and receives bounded defaults; absent objects stay
    absent for backwards-compatible canonical requests. Geometry-dependent
    checks happen in layouts.py.
    """
    data = _mapping(data, "request", {"schema_version", "scene_id", "object", "layout", "seed", "randomization"}, {"scene_id", "object", "layout"})
    version = _integer(data.get("schema_version", 1), "schema_version", 1, 1)
    scene_id = _integer(data["scene_id"], "scene_id", 1, 102)
    obj = _mapping(data["object"], "object", {"asset", "count"}, {"asset", "count"})
    if not isinstance(obj["asset"], str) or obj["asset"] not in ASSET_IDS:
        raise RequestError("object.asset must be a supported canonical asset ID.")
    count = _integer(obj["count"], "object.count", 1, 64)
    supplied = _mapping(data["layout"], "layout", {"type", "rows", "cols", "spacing_m", "axis", "radius_m"}, {"type"})
    kind = supplied["type"]
    if not isinstance(kind, str) or kind not in LAYOUT_TYPES:
        raise RequestError(f"layout.type must be one of {LAYOUT_TYPES}.")
    allowed = {"type", "spacing_m"} | {"grid": {"rows", "cols"}, "line": {"axis"}, "ring": {"radius_m"}}.get(kind, set())
    if set(supplied) - allowed:
        raise RequestError(f"Fields {sorted(set(supplied) - allowed)} are invalid for {kind}.")
    layout = {"type": kind}
    if "spacing_m" in supplied:
        layout["spacing_m"] = _positive_number(supplied["spacing_m"], "layout.spacing_m")
    if kind == "grid":
        if ("rows" in supplied) != ("cols" in supplied):
            raise RequestError("grid requires both rows and cols.")
        if "rows" in supplied:
            rows = _integer(supplied["rows"], "layout.rows", 1, 64)
            cols = _integer(supplied["cols"], "layout.cols", 1, 64)
        else:
            rows = cols = math.isqrt(count)
            if rows * cols != count:
                raise RequestError("A non-square grid needs explicit rows and cols.")
        if rows * cols != count:
            raise RequestError(f"Grid {rows}x{cols} needs {rows * cols} objects; requested {count}.")
        layout.update(rows=rows, cols=cols)
    elif kind == "line":
        axis = supplied.get("axis", "y")
        if axis not in ("x", "y"):
            raise RequestError("line axis must be 'x' or 'y'.")
        layout["axis"] = axis
    elif kind == "ring" and "radius_m" in supplied:
        layout["radius_m"] = _positive_number(supplied["radius_m"], "layout.radius_m")
    elif kind == "t_shape" and count != 5:
        raise RequestError("t_shape requires exactly 5 objects.")
    seed = _integer(data.get("seed", 0), "seed", 0)
    result = {"schema_version": version, "scene_id": scene_id, "object": {"asset": obj["asset"], "count": count}, "layout": layout, "seed": seed}
    if "randomization" in data:
        supplied_randomization = _mapping(data["randomization"], "randomization", {"enabled", "jitter_xy_m", "shift_max_m", "yaw_deg"}, set())
        enabled = supplied_randomization.get("enabled", True)
        if type(enabled) is not bool:
            raise RequestError("randomization.enabled must be a boolean.")
        result["randomization"] = {
            "enabled": enabled,
            "jitter_xy_m": _bounded_number(supplied_randomization.get("jitter_xy_m", 0.02), "randomization.jitter_xy_m", 0.05),
            "shift_max_m": _bounded_number(supplied_randomization.get("shift_max_m", 0.15), "randomization.shift_max_m", 0.3),
            "yaw_deg": _bounded_number(supplied_randomization.get("yaw_deg", 7.5), "randomization.yaw_deg", 180.0),
        }
    return result


# Repeated/Markdown-escaped asterisks occur in the original user example. This
# compatibility is local to a full dimension expression, not global unescaping.
_SHAPE = r"(?P<rows>\d+)\s*(?:x|×|(?:\\?\*\s*)+)\s*(?P<cols>\d+)"


def _parse_layout(text: str) -> dict:
    phrase = text.strip().lower()
    for pattern in (rf"grid(?:\s+of)?\s+{_SHAPE}", rf"{_SHAPE}\s+grid", rf"{_SHAPE}\s*(?:网格|矩阵)", rf"(?:网格|矩阵)\s*{_SHAPE}"):
        match = re.fullmatch(pattern, phrase)
        if match:
            return {"type": "grid", "rows": int(match["rows"]), "cols": int(match["cols"])}
    match = re.fullmatch(r"(?P<rows>\d+)\s*行\s*(?P<cols>\d+)\s*列(?:网格|矩阵)?", phrase)
    if match:
        return {"type": "grid", "rows": int(match["rows"]), "cols": int(match["cols"])}
    if phrase in ("grid", "网格", "矩阵"):
        return {"type": "grid"}
    if phrase in ("line", "a line", "直线", "一排", "一行"):
        return {"type": "line"}
    match = re.fullmatch(r"line\s+along\s+([xy])(?:\s*axis)?", phrase)
    if match:
        return {"type": "line", "axis": match[1]}
    if phrase in ("ring", "a ring", "circle", "a circle", "圆环", "环形", "圆形"):
        return {"type": "ring"}
    if phrase in ("random", "random layout", "scatter", "随机", "随机摆放", "随机分布"):
        return {"type": "random"}
    if phrase in ("t_shape", "t shape", "t-shape", "t-shaped", "t字形", "t形"):
        return {"type": "t_shape"}
    raise RequestError(f"Unsupported layout expression {text!r}.")


def parse_request(text: str) -> dict:
    """Parse one complete assembly instruction in a deliberately bounded grammar."""
    if not isinstance(text, str) or not text.strip():
        raise RequestError("An assembly instruction must be a nonempty string.")
    phrase = unicodedata.normalize("NFKC", text).strip().lower()
    phrase = re.sub(r"[.!。!]$", "", phrase).strip()
    english = re.fullmatch(
        r"(?:put|place|arrange)\s+(?P<count>\d+)\s+(?P<asset>.+?)\s+(?:in|as|into)\s+(?P<layout>.+?)\s+in\s+scene\s*(?P<scene>\d+)(?:\s+with\s+(?:(?P<randomization>randomization)(?:\s+and\s+seed\s+(?P<random_seed>\d+))?|seed\s+(?P<seed>\d+)))?",
        phrase,
    )
    patterns = (
        r"在\s*场景\s*(?P<scene>\d+)\s*(?:中|里)?\s*[,，]?\s*(?:将|把)?\s*(?P<count>\d+)\s*(?:个|只|件)?\s*(?P<asset>.+?)\s*(?:摆成|排成|排列成|放成)\s*(?P<layout>.+)",
        r"在\s*场景\s*(?P<scene>\d+)\s*(?:中|里)?\s*[,，]?\s*(?:放置|摆放)\s*(?P<count>\d+)\s*(?:个|只|件)?\s*(?P<asset>.+?)\s*[,，]?\s*(?:按|以)\s*(?P<layout>.+?)(?:排列|摆放)?",
        r"(?:将|把)\s*(?P<count>\d+)\s*(?:个|只|件)?\s*(?P<asset>.+?)\s*(?:按|以)\s*(?P<layout>.+?)\s*(?:摆放|排列)在\s*场景\s*(?P<scene>\d+)\s*(?:中|里)?",
    )
    match = english
    if match is None:
        for pattern in patterns:
            match = re.fullmatch(pattern, phrase)
            if match:
                break
    if match is None:
        raise RequestError("Use 'put <count> <asset> in <layout> in scene <1..102>' or '在场景60中,将4个蓝色罐摆成2x2网格'.")
    groups = match.groupdict()
    request = {
        "scene_id": int(groups["scene"]),
        "object": {"asset": normalize_asset(groups["asset"].strip()), "count": int(groups["count"])},
        "layout": _parse_layout(groups["layout"]),
        "seed": int(groups.get("seed") or groups.get("random_seed") or 0),
    }
    if groups.get("randomization"):
        request["randomization"] = {}
    return validate_request(request)
