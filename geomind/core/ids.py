from __future__ import annotations

import re

from .types import ParsedTaskId

_STYLE_A_PATTERN = re.compile(
    r"^(?P<family>[A-Za-z]+)"
    r"(?P<split>Base|Aug)"
    r"(?P<variant>Base|Sem|Visual|Execution|Vla)"
    r"_(?P<task>[A-Za-z0-9]+)_v(?P<ver>\d+)$"
)

_COMPACT_PATTERN = re.compile(
    r"^(?P<task>[A-Z][A-Za-z0-9]*?)(?P<variant>Sem|Visual|Execution)?"
    r"(?P<vla>Vla)?_v(?P<ver>\d+)$"
)


def _snake_to_title(text: str) -> str:
    return "".join(chunk.capitalize() for chunk in text.split("_") if chunk)


def is_public_task_id(task_id: str) -> bool:
    text = str(task_id).strip()
    return bool(_STYLE_A_PATTERN.fullmatch(text) or _COMPACT_PATTERN.fullmatch(text))


def parse_task_id(task_id: str) -> ParsedTaskId:
    text = str(task_id).strip()
    match = _STYLE_A_PATTERN.match(text)
    if match:
        return ParsedTaskId(
            raw=text,
            family=match.group("family").lower(),
            split=match.group("split").lower(),
            variant=match.group("variant").lower(),
            task_name=match.group("task"),
            version=int(match.group("ver")),
        )
    match = _COMPACT_PATTERN.match(text)
    if match:
        return ParsedTaskId(
            raw=text,
            family="unknown",
            split="base",
            variant="vla" if match.group("vla") else (match.group("variant") or "base").lower(),
            task_name=match.group("task"),
            version=int(match.group("ver")),
        )
    # Fallback for existing snake-style ids, e.g. "track_single".
    fallback_task = _snake_to_title(text) if "_" in text else text
    return ParsedTaskId(
        raw=text,
        family="unknown",
        split="base",
        variant="base",
        task_name=fallback_task,
        version=0,
    )


def make_task_id(*, family: str = "", split: str = "base", variant: str = "base", task_name: str, version: int) -> str:
    """Build compatible runtime keys; the task definition stores its bare name.

    Keep accepting family/split so existing scaffold callers remain compatible.
    Variant suffixes keep semantic, visual and VLA tasks distinct.
    """
    suffixes = {
        "base": "",
        "sem": "Sem",
        "semantic": "Sem",
        "visual": "Visual",
        "execution": "Execution",
        "exec": "Execution",
        "vla": "Vla",
    }
    normalized = str(variant).strip().lower()
    if normalized not in suffixes:
        raise ValueError(f"Unsupported task variant: {variant!r}")
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", task_name):
        raise ValueError("task_name must be a CamelCase name without a category prefix.")
    return f"{task_name}{suffixes[normalized]}_v{int(version)}"
