from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
import os
import random
import re
import secrets
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geomind_lab.mimic.utils.config_semantics_adapter import normalize_annotation_entry, normalize_config_semantics
from geomind_lab.mimic.utils.task_metadata_registry import build_task_metadata, resolve_task_id
from geomind_lab.tasks.manager_based.manipulation.geomind.layout_randomizer import randomize_runtime_object_layout
from geomind_lab.tasks.manager_based.manipulation.geomind.task_profiles import (
    get_task_profile,
    normalize_scene_value,
    resolve_task_backend,
    resolve_profile_task_id,
    resolve_profile_variant,
    resolve_profile_variant_profile,
    scene_name_for_id,
)
from geomind_lab.utils import asset_utils

if TYPE_CHECKING:
    from .geomind_scene_cfg import SceneProfile


DEFAULT_OBJECT_MODE = "geometric"
SUPPORTED_OBJECT_MODES = ("concrete", "geometric")


@dataclass(frozen=True)
class GoalSpec:
    object_id: str
    xy: tuple[float, float]
    tolerance_r: float = 0.06


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    instruction: str
    scene_id: int
    layout_key: str
    goals: tuple[GoalSpec, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedTaskRequest:
    task_id: str
    variant: str
    scene_id: int
    scene: str
    recommended_scene_id: int
    recommended_scene: str
    scene_overridden: bool
    layout_key: str
    task_index: int
    object_mode: str
    backend: str = ""
    raw_task_id: str = ""


@dataclass(frozen=True)
class LoadedTask:
    request: ResolvedTaskRequest
    task_spec: dict[str, Any]
    task_label: str


@dataclass(frozen=True)
class LightweightSceneProfile:
    scene_id: int
    object_names: tuple[str, ...]
    xforms: dict[str, dict[str, tuple[float, ...]]]
    basket_name: str | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def resolve_annotated_tasks_path() -> Path:
    env_override = os.environ.get("GEOMIND_LAB_TASK_SPECS_PATH") or os.environ.get("GEOMIND_ANNOTATED_TASKS_PATH")
    if env_override:
        return Path(env_override)

    candidates = (
        repo_root() / "assets" / "annotations" / "annotated_tasks.json",
        repo_root() / "assets" / "tasks" / "annotated_tasks.json",
        repo_root() / "assets" / "tasks" / "annotated_tasks_1.json",
        repo_root() / "annotated_tasks_1.json",
        Path("/home/elm/projects/geomind/annotated_tasks_1.json"),
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def load_task_payload(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load_annotated_tasks_cached(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.is_file():
        return {}
    try:
        payload = load_task_payload(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_annotated_tasks(path: str | Path | None = None) -> dict[str, Any]:
    """Load merged task layout index: default annotated_tasks.json plus geomind layout overlays.

    Overlays (later keys win on collision) are merged so IsaacLab ``load_task_spec`` sees
    VLA layout keys from ``geomind/configs/legacy/layouts/vla.json`` without duplicating
    the full ``annotated_tasks.json`` into the repo.

    Set ``GEOMIND_DISABLE_LAYOUT_OVERLAYS=1`` to load only the primary JSON (for debugging).
    """
    resolved = Path(path).resolve() if path is not None else resolve_annotated_tasks_path().resolve()
    merged = dict(_load_annotated_tasks_cached(str(resolved)))
    disable = str(os.environ.get("GEOMIND_DISABLE_LAYOUT_OVERLAYS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if disable:
        return merged
    default_path = resolve_annotated_tasks_path().resolve()
    if resolved != default_path:
        return merged
    root = repo_root()
    for relative in (
        Path("geomind") / "configs" / "legacy" / "layouts" / "standard.json",
        Path("geomind") / "configs" / "legacy" / "layouts" / "vla.json",
    ):
        overlay_path = root / relative
        if not overlay_path.is_file():
            continue
        try:
            extra = load_task_payload(overlay_path)
        except Exception:
            continue
        if isinstance(extra, dict):
            merged.update(extra)
    # Physical initialization is owned by independent v3 layout files. Keep
    # legacy task protocols/GT in the compatibility annotation index only.
    for layout_path in sorted((root / "geomind" / "configs" / "layouts").glob("*.json")):
        layout_definition = load_task_payload(layout_path)
        if not isinstance(layout_definition, dict) or layout_definition.get("format") != "LegacyInitialization":
            continue
        key = layout_definition.get("source_layout")
        if key not in merged:
            raise ValueError(f"Missing legacy source layout {key!r} for {layout_path}")
        physical_layout = deepcopy(merged[key])
        physical_layout["scene_structure"] = deepcopy(layout_definition["scene_structure"])
        initialization = layout_definition["initialization"]
        if set(initialization) - {"runtime_objects", "background_overrides", "object_mode"}:
            raise ValueError(f"Layout initialization contains task semantics: {layout_path}")
        for task in physical_layout.get("tasks", []):
            annotation = task.setdefault("annotation", {})
            adaptation = annotation.setdefault("scene_adaptation", {})
            if "runtime_objects" in initialization:
                old_objects = {obj["name"]: obj for obj in adaptation.get("runtime_objects", [])}
                new_objects = deepcopy(initialization["runtime_objects"])
                if old_objects and set(old_objects) != {obj["name"] for obj in new_objects}:
                    raise ValueError(f"Legacy protocol object identities would change: {layout_path}")
                for obj in new_objects:
                    old_role = old_objects.get(obj["name"], {}).get("role")
                    if old_role is not None:
                        obj["role"] = old_role
                adaptation["runtime_objects"] = new_objects
            for field in ("background_overrides", "object_mode"):
                if field in initialization:
                    adaptation[field] = deepcopy(initialization[field])
        merged[key] = physical_layout
    return merged


def normalize_object_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    alias_map = {
        "": "",
        "object": "concrete",
        "objects": "concrete",
        "concrete": "concrete",
        "specific": "concrete",
        "specific_objects": "concrete",
        "real": "concrete",
        "realistic": "concrete",
        "shape": "geometric",
        "shapes": "geometric",
        "geometry": "geometric",
        "geometric": "geometric",
        "primitive": "geometric",
        "primitives": "geometric",
    }
    resolved = alias_map.get(normalized)
    if resolved is None:
        valid = ", ".join(SUPPORTED_OBJECT_MODES)
        raise ValueError(f"Unsupported object_mode {value!r}. Expected one of: {valid}")
    return resolved


def resolve_layout(layout_key: str, path: str | Path | None = None) -> dict[str, Any]:
    payload = load_annotated_tasks(path)
    layout = payload.get(str(layout_key), {})
    return dict(layout) if isinstance(layout, dict) else {}


def resolve_task_payload(layout_key: str, task_index: int, path: str | Path | None = None) -> dict[str, Any]:
    tasks = resolve_layout(layout_key, path).get("tasks", [])
    if not isinstance(tasks, list) or task_index < 0 or task_index >= len(tasks):
        return {}
    task_payload = tasks[task_index]
    return dict(task_payload) if isinstance(task_payload, dict) else {}


def resolve_task_annotation(layout_key: str, task_index: int, path: str | Path | None = None) -> dict[str, Any]:
    annotation = resolve_task_payload(layout_key, task_index, path).get("annotation", {})
    return dict(annotation) if isinstance(annotation, dict) else {}


def resolve_task_label(layout_key: str, task_index: int, path: str | Path | None = None) -> str:
    annotation = resolve_task_annotation(layout_key, task_index, path)
    instance_semantics = annotation.get("instance_semantics", {})
    if isinstance(instance_semantics, dict):
        instance_id = instance_semantics.get("instance_id")
        if isinstance(instance_id, str) and instance_id.strip():
            return str(instance_id)
    task_id = annotation.get("task_id")
    if isinstance(task_id, str) and task_id.strip():
        return str(task_id)
    return f"{layout_key}__task_{task_index}"


def _annotation_protocol(annotation: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(annotation, Mapping):
        return {}
    protocol = annotation.get("protocol", {})
    return dict(protocol) if isinstance(protocol, Mapping) else {}


def _annotation_task_id_candidates(annotation: Mapping[str, Any] | None) -> set[str]:
    candidates: set[str] = set()
    if not isinstance(annotation, Mapping):
        return candidates

    raw_task_id = str(annotation.get("task_id", "")).strip()
    legacy_task_id = str(annotation.get("legacy_task_id", "")).strip()
    if raw_task_id:
        candidates.add(raw_task_id)
    if legacy_task_id:
        candidates.add(legacy_task_id)

    protocol = _annotation_protocol(annotation)
    resolved_task_id = resolve_task_id(
        raw_task_id=raw_task_id or legacy_task_id,
        family=protocol.get("family"),
        variant=protocol.get("variant"),
        protocol=protocol,
    )
    if resolved_task_id:
        candidates.add(str(resolved_task_id))
    return {candidate for candidate in candidates if candidate}


def resolve_task_locator(task_id: str, path: str | Path | None = None) -> dict[str, Any]:
    requested_task_id = str(task_id).strip()
    if not requested_task_id:
        raise ValueError("task_id must be a non-empty string.")

    requested_task_id = resolve_task_id(requested_task_id) or requested_task_id
    payload = load_annotated_tasks(path)
    matches: list[dict[str, Any]] = []
    for layout_key, layout in payload.items():
        if not isinstance(layout, dict):
            continue
        scene_id = int(layout.get("scene_id", 1))
        tasks = layout.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        for task_index, task_payload in enumerate(tasks):
            if not isinstance(task_payload, dict):
                continue
            annotation = task_payload.get("annotation", {})
            if requested_task_id not in _annotation_task_id_candidates(annotation if isinstance(annotation, dict) else {}):
                continue
            matches.append(
                {
                    "task_id": requested_task_id,
                    "layout_key": str(layout_key),
                    "task_index": int(task_index),
                    "scene_id": scene_id,
                    "instruction": str(task_payload.get("instruction", "")),
                }
            )

    if not matches:
        raise KeyError(f"Unknown task_id: {requested_task_id!r}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous task_id: {requested_task_id!r} matched {len(matches)} tasks.")
    return matches[0]


def resolve_task_locators(
    task_id: str,
    *,
    scene: Any = None,
    layout_key: str | None = None,
    task_index: int | None = None,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    requested_task_id = str(task_id).strip()
    if not requested_task_id:
        return []

    requested_task_id = resolve_task_id(requested_task_id) or requested_task_id
    requested_scene_id = normalize_scene_value(scene)
    requested_layout_key = str(layout_key or "").strip()
    requested_task_index = None if task_index is None else int(task_index)

    payload = load_annotated_tasks(path)
    matches: list[dict[str, Any]] = []
    for candidate_layout_key, layout in payload.items():
        if not isinstance(layout, dict):
            continue
        candidate_scene_id = int(layout.get("scene_id", 1))
        if requested_scene_id is not None and candidate_scene_id != requested_scene_id:
            continue
        if requested_layout_key and str(candidate_layout_key) != requested_layout_key:
            continue
        tasks = layout.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        for candidate_task_index, task_payload in enumerate(tasks):
            if requested_task_index is not None and candidate_task_index != requested_task_index:
                continue
            if not isinstance(task_payload, dict):
                continue
            annotation = task_payload.get("annotation", {})
            annotation_payload = annotation if isinstance(annotation, dict) else {}
            if requested_task_id not in _annotation_task_id_candidates(annotation_payload):
                continue
            matches.append(
                {
                    "task_id": requested_task_id,
                    "layout_key": str(candidate_layout_key),
                    "task_index": int(candidate_task_index),
                    "scene_id": candidate_scene_id,
                    "instruction": str(task_payload.get("instruction", "")),
                }
            )
    return matches


def _annotation_default_object_mode(annotation: Mapping[str, Any] | None) -> str:
    if not isinstance(annotation, Mapping):
        return ""
    scene_adaptation = annotation.get("scene_adaptation", {})
    if isinstance(scene_adaptation, Mapping):
        resolved = normalize_object_mode(scene_adaptation.get("object_mode"))
        if resolved:
            return resolved
    protocol = _annotation_protocol(annotation)
    resolved = normalize_object_mode(protocol.get("object_mode"))
    if resolved:
        return resolved
    return ""


def resolve_task_request(
    *,
    task_id: Any = None,
    variant: Any = None,
    scene: Any = None,
    scene_id: Any = None,
    layout_key: Any = None,
    task_index: Any = None,
    object_mode: Any = None,
    path: str | Path | None = None,
) -> ResolvedTaskRequest:
    requested_layout_key = str(layout_key or "").strip()
    requested_task_index = None if task_index in (None, "") else int(task_index)
    requested_scene = scene if scene not in (None, "") else scene_id
    requested_scene_id = normalize_scene_value(requested_scene)
    requested_object_mode = normalize_object_mode(object_mode)

    requested_task_id = str(task_id or "").strip()
    resolved_task_id, implicit_variant = resolve_profile_task_id(requested_task_id)
    raw_task_id = requested_task_id
    if not resolved_task_id:
        resolved_task_id = requested_task_id
    requested_variant = str(variant or implicit_variant or "").strip()
    profile = get_task_profile(resolved_task_id)
    resolved_variant = resolve_profile_variant(resolved_task_id, requested_variant)
    variant_profile = resolve_profile_variant_profile(resolved_task_id, resolved_variant)
    if (
        variant_profile is None
        and requested_variant
        and not requested_layout_key
        and profile is not None
        and getattr(profile, "variants", None)
    ):
        known = sorted(profile.variants)
        raise ValueError(
            f"Unknown variant {requested_variant!r} for task_id {resolved_task_id!r}. "
            f"Known variants: {', '.join(known)}. "
            "Pass --layout-key explicitly to target a layout that has no registered variant."
        )
    backend = resolve_task_backend(resolved_task_id)

    resolved_layout_key = requested_layout_key
    resolved_task_index = 0 if requested_task_index is None else int(requested_task_index)
    recommended_scene_id: int | None = None

    if requested_layout_key:
        layout = resolve_layout(requested_layout_key, path)
        if layout:
            recommended_scene_id = int(layout.get("scene_id", 1))
        elif variant_profile is not None:
            recommended_scene_id = int(variant_profile.scene_id)
        elif requested_scene_id is not None:
            recommended_scene_id = int(requested_scene_id)
        else:
            recommended_scene_id = 1
        if requested_task_index is None and variant_profile is not None and variant_profile.layout_key == requested_layout_key:
            resolved_task_index = int(variant_profile.task_index)
    elif variant_profile is not None:
        resolved_layout_key = str(variant_profile.layout_key)
        resolved_task_index = int(variant_profile.task_index if requested_task_index is None else requested_task_index)
        recommended_scene_id = int(variant_profile.scene_id)
    elif resolved_task_id:
        matches = resolve_task_locators(
            resolved_task_id,
            scene=requested_scene_id,
            layout_key=requested_layout_key or None,
            task_index=requested_task_index,
            path=path,
        )
        if len(matches) > 1:
            preview = ", ".join(
                f"{candidate['layout_key']}:{candidate['task_index']}@scene{candidate['scene_id']}"
                for candidate in matches[:5]
            )
            raise ValueError(
                f"Ambiguous task_id: {resolved_task_id!r}. Matching instances: {preview}. "
                "Refine with variant, scene, layout_key, or task_index."
            )
        if matches:
            match = matches[0]
            resolved_layout_key = str(match["layout_key"])
            resolved_task_index = int(match["task_index"] if requested_task_index is None else requested_task_index)
            recommended_scene_id = int(match["scene_id"])
    if not resolved_layout_key:
        raise ValueError("Unable to resolve layout_key from the given task selectors.")

    resolved_annotation = resolve_task_annotation(resolved_layout_key, resolved_task_index, path)
    resolved_protocol = _annotation_protocol(resolved_annotation)
    if not resolved_task_id:
        resolved_task_id = (
            resolve_task_id(
                raw_task_id=resolved_annotation.get("task_id"),
                family=resolved_protocol.get("family"),
                variant=resolved_protocol.get("variant"),
                protocol=resolved_protocol,
            )
            or str(resolved_annotation.get("task_id") or "")
        )
        profile = get_task_profile(resolved_task_id)
        backend = resolve_task_backend(resolved_task_id)
    if profile is not None and not resolved_variant:
        resolved_variant = str(profile.default_variant)
    elif not resolved_variant:
        resolved_variant = str(resolved_protocol.get("task_version") or resolved_protocol.get("variant") or "")

    resolved_scene_id = int(
        requested_scene_id
        if requested_scene_id is not None
        else (recommended_scene_id if recommended_scene_id is not None else 1)
    )
    resolved_recommended_scene_id = int(recommended_scene_id if recommended_scene_id is not None else resolved_scene_id)
    annotation_object_mode = _annotation_default_object_mode(resolved_annotation)
    resolved_object_mode = (
        requested_object_mode
        or annotation_object_mode
        or (profile.default_object_mode if profile is not None else "")
        or DEFAULT_OBJECT_MODE
    )

    return ResolvedTaskRequest(
        task_id=str(resolved_task_id),
        variant=str(resolved_variant),
        scene_id=resolved_scene_id,
        scene=scene_name_for_id(resolved_scene_id),
        recommended_scene_id=resolved_recommended_scene_id,
        recommended_scene=scene_name_for_id(resolved_recommended_scene_id),
        scene_overridden=resolved_scene_id != resolved_recommended_scene_id,
        layout_key=resolved_layout_key,
        task_index=resolved_task_index,
        object_mode=resolved_object_mode,
        backend=str(backend),
        raw_task_id=raw_task_id or str(resolved_annotation.get("task_id") or resolved_task_id),
    )


def _sort_by_suffix(names: list[str]) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        match = re.search(r"_(\d+)$", name)
        suffix = int(match.group(1)) if match else 0
        return (suffix, name)

    return sorted(names, key=key)


def _build_object_name_mapping(source_names: list[str], target_names: list[str]) -> dict[str, str]:
    if not source_names or not target_names:
        return {}
    ordered_src = _sort_by_suffix(source_names)
    ordered_tgt = _sort_by_suffix(target_names)
    return {src: tgt for src, tgt in zip(ordered_src, ordered_tgt)}


def _collect_protocol_object_ids(payload: Any, used_names: set[str]) -> None:
    if isinstance(payload, dict):
        object_id = payload.get("object_id")
        if isinstance(object_id, str) and object_id:
            used_names.add(object_id)
        for value in payload.values():
            _collect_protocol_object_ids(value, used_names)
        return
    if isinstance(payload, list):
        for value in payload:
            _collect_protocol_object_ids(value, used_names)


def _remap_protocol_object_ids(payload: Any, mapping: dict[str, str]) -> Any:
    if isinstance(payload, dict):
        remapped: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "object_id" and isinstance(value, str):
                remapped[key] = mapping.get(value, value)
            else:
                remapped[key] = _remap_protocol_object_ids(value, mapping)
        return remapped
    if isinstance(payload, list):
        return [_remap_protocol_object_ids(value, mapping) for value in payload]
    return payload


def remap_task_objects(task_spec: dict[str, Any], scene_objects: list[str]) -> dict[str, Any]:
    scene_set = set(scene_objects)
    used_names: set[str] = set()

    object_positions = task_spec.get("object_positions", {})
    used_names.update(object_positions.keys())

    goal_sequence = task_spec.get("goal_sequence", [])
    for step in goal_sequence:
        for goal in step.get("goals", []):
            object_id = goal.get("object_id")
            if object_id:
                used_names.add(object_id)
    _collect_protocol_object_ids(task_spec.get("protocol"), used_names)

    if not used_names or used_names.issubset(scene_set):
        return task_spec

    mapping = _build_object_name_mapping(list(used_names), scene_objects)

    remapped_positions: dict[str, Any] = {}
    for name, payload in object_positions.items():
        remapped_positions[mapping.get(name, name)] = payload
    task_spec["object_positions"] = remapped_positions

    for step in goal_sequence:
        for goal in step.get("goals", []):
            object_id = goal.get("object_id")
            if object_id in mapping:
                goal["object_id"] = mapping[object_id]

    protocol = task_spec.get("protocol")
    if isinstance(protocol, dict):
        task_spec["protocol"] = _remap_protocol_object_ids(protocol, mapping)
    return task_spec


def _extract_protocol_spec(annotation: dict[str, Any], instruction: str) -> dict[str, Any]:
    protocol = annotation.get("protocol")
    if not isinstance(protocol, dict) or not protocol:
        return {}

    resolved = deepcopy(protocol)
    query_text = resolved.get("query_text")
    if not isinstance(query_text, str) or not query_text.strip():
        resolved["query_text"] = str(instruction)
    return resolved


def _extract_scene_adaptation(annotation: dict[str, Any]) -> dict[str, Any]:
    scene_adaptation = annotation.get("scene_adaptation", {})
    return deepcopy(scene_adaptation) if isinstance(scene_adaptation, dict) else {}


def _build_runtime_object_positions(runtime_objects: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    object_positions: dict[str, dict[str, list[float]]] = {}
    for runtime_object in runtime_objects:
        if not isinstance(runtime_object, dict):
            continue
        object_name = str(runtime_object.get("name", "")).strip()
        pose = runtime_object.get("pose", {})
        xy = pose.get("xy", []) if isinstance(pose, dict) else []
        if not object_name or not isinstance(xy, list) or len(xy) < 2:
            continue
        object_positions[object_name] = {"position": [float(xy[0]), float(xy[1])]}
    return object_positions


_POST_ACTION_RELOCATION_MODULE = "post_action_relocation"
_TRACK_REINDEX_TRACK_NAMES = {
    "state_reindexing",
    "reorder_with_distractors",
    "historical_identity_tracking",
    "object_identity_tracking",
}
_TRACK_REINDEX_CUBOID_SCALE = 0.6


def _should_scale_track_reindex_cuboids(protocol: Mapping[str, Any] | None) -> bool:
    if not isinstance(protocol, Mapping):
        return False
    module_name = str(protocol.get("module_name", "")).strip().lower()
    track_name = str(protocol.get("track_name", "")).strip().lower()
    return module_name == _POST_ACTION_RELOCATION_MODULE and track_name in _TRACK_REINDEX_TRACK_NAMES


def _scale_runtime_cuboid_sizes(runtime_objects: list[dict[str, Any]], *, scale: float) -> None:
    safe_scale = float(scale)
    if safe_scale <= 0.0:
        return
    for runtime_object in runtime_objects:
        if not isinstance(runtime_object, dict):
            continue
        if str(runtime_object.get("kind", "")).strip() != "primitive_cuboid":
            continue
        size = runtime_object.get("size")
        if not isinstance(size, list) or len(size) < 3:
            continue
        try:
            sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        except (TypeError, ValueError):
            continue
        runtime_object["size"] = [
            round(max(sx * safe_scale, 1.0e-4), 6),
            round(max(sy * safe_scale, 1.0e-4), 6),
            round(max(sz * safe_scale, 1.0e-4), 6),
        ]


def resolve_task_spec_object_names(task_spec: dict[str, Any] | None, scene_profile: "SceneProfile" | None = None) -> list[str]:
    if isinstance(task_spec, dict):
        runtime_objects = task_spec.get("runtime_objects", [])
        if isinstance(runtime_objects, list) and runtime_objects:
            runtime_names = [
                str(runtime_object.get("name", "")).strip()
                for runtime_object in runtime_objects
                if isinstance(runtime_object, dict) and str(runtime_object.get("name", "")).strip()
            ]
            if runtime_names:
                return runtime_names

        object_positions = task_spec.get("object_positions", {})
        if isinstance(object_positions, dict) and object_positions:
            return [str(name) for name in object_positions.keys()]

    if scene_profile is not None and getattr(scene_profile, "object_names", None):
        return [str(name) for name in scene_profile.object_names]

    return []


def resolve_task_protocol(task_spec: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(task_spec, dict):
        return {}
    raw_protocol = task_spec.get("protocol", {})
    return dict(raw_protocol) if isinstance(raw_protocol, dict) else {}


def resolve_task_instruction(task_spec: dict[str, Any] | None, task_protocol: dict[str, Any] | None = None) -> str:
    protocol = resolve_task_protocol(task_spec) if task_protocol is None else (
        dict(task_protocol) if isinstance(task_protocol, dict) else {}
    )
    query_text = protocol.get("query_text")
    if isinstance(query_text, str) and query_text.strip():
        return str(query_text)
    if not isinstance(task_spec, dict):
        return ""
    return str(task_spec.get("instruction", ""))


def _build_lightweight_scene_profile(
    layout_key: str,
    scene_id: int,
    *,
    path: str | Path | None = None,
) -> LightweightSceneProfile:
    layout = resolve_layout(layout_key, path)
    scene_structure = layout.get("scene_structure", {}) if isinstance(layout, dict) else {}
    object_positions = scene_structure.get("object_positions", {}) if isinstance(scene_structure, dict) else {}
    resolved_positions = object_positions if isinstance(object_positions, dict) else {}

    xforms: dict[str, dict[str, tuple[float, ...]]] = {}
    for object_name, payload in resolved_positions.items():
        if not isinstance(payload, Mapping):
            continue
        position = payload.get("position", ())
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) < 2:
            continue
        try:
            xforms[str(object_name)] = {
                "pos": (float(position[0]), float(position[1]), 0.75),
                "rot": (1.0, 0.0, 0.0, 0.0),
                "scale": (1.0, 1.0, 1.0),
            }
        except (TypeError, ValueError):
            continue

    return LightweightSceneProfile(
        scene_id=int(scene_id),
        object_names=tuple(xforms.keys()),
        xforms=xforms,
        basket_name=None,
    )


def _runtime_object_scale(runtime_object: Mapping[str, Any] | None) -> tuple[float, float, float]:
    if not isinstance(runtime_object, Mapping):
        return (1.0, 1.0, 1.0)
    scale = runtime_object.get("scale", ())
    if isinstance(scale, Sequence) and not isinstance(scale, (str, bytes)) and len(scale) >= 3:
        try:
            return (float(scale[0]), float(scale[1]), float(scale[2]))
        except (TypeError, ValueError):
            return (1.0, 1.0, 1.0)
    return (1.0, 1.0, 1.0)


def _runtime_object_dimensions(runtime_object: Mapping[str, Any] | None) -> tuple[float, float, float] | None:
    if not isinstance(runtime_object, Mapping):
        return None
    kind = str(runtime_object.get("kind", "")).strip()
    if kind == "primitive_cuboid":
        size = runtime_object.get("size", ())
        if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) >= 3:
            try:
                return (abs(float(size[0])), abs(float(size[1])), abs(float(size[2])))
            except (TypeError, ValueError):
                return None
    if kind == "primitive_sphere":
        try:
            radius = abs(float(runtime_object.get("radius", 0.0)))
        except (TypeError, ValueError):
            return None
        diameter = 2.0 * radius
        return (diameter, diameter, diameter)
    if kind == "primitive_cylinder":
        try:
            radius = abs(float(runtime_object.get("radius", 0.0)))
            height = abs(float(runtime_object.get("height", 0.0)))
        except (TypeError, ValueError):
            return None
        diameter = 2.0 * radius
        return (diameter, diameter, height)
    if kind == "primitive_cone":
        try:
            radius = abs(float(runtime_object.get("radius", 0.0)))
            height = abs(float(runtime_object.get("height", 0.0)))
        except (TypeError, ValueError):
            return None
        diameter = 2.0 * radius
        return (diameter, diameter, height)
    if kind == "usd":
        asset_category = str(runtime_object.get("asset_category", "")).strip()
        asset_key = str(runtime_object.get("asset_key", "")).strip()
        asset_info = asset_utils.get_asset_info_by_name(asset_category, asset_key) if asset_category and asset_key else None
        if not isinstance(asset_info, Mapping):
            return None
        dims = asset_info.get("dimensions", {})
        if not isinstance(dims, Mapping):
            return None
        scale_x, scale_y, scale_z = _runtime_object_scale(runtime_object)
        try:
            return (
                abs(float(dims.get("x", 0.0)) * scale_x),
                abs(float(dims.get("y", 0.0)) * scale_y),
                abs(float(dims.get("z", 0.0)) * scale_z),
            )
        except (TypeError, ValueError):
            return None
    return None


def _runtime_object_color_label(runtime_object: Mapping[str, Any] | None) -> str:
    if not isinstance(runtime_object, Mapping):
        return ""
    color_source = runtime_object.get("visual_color")
    if not isinstance(color_source, Sequence) or isinstance(color_source, (str, bytes)) or len(color_source) < 3:
        material_override = runtime_object.get("material_override", {})
        if isinstance(material_override, Mapping):
            color_source = material_override.get("diffuse_color")
    if not isinstance(color_source, Sequence) or isinstance(color_source, (str, bytes)) or len(color_source) < 3:
        return ""
    try:
        red = float(color_source[0])
        green = float(color_source[1])
        blue = float(color_source[2])
    except (TypeError, ValueError):
        return ""
    if blue >= max(red, green) and blue >= 0.45:
        return "blue"
    if red >= max(green, blue) and red >= 0.45:
        if green >= 0.32 and blue < 0.25:
            return "orange"
        return "red"
    if green >= max(red, blue) and green >= 0.45:
        return "green"
    if abs(red - green) <= 0.1 and abs(green - blue) <= 0.1:
        return "gray"
    if red >= 0.3 and green >= 0.2 and blue <= 0.25:
        return "brown"
    return ""


def _runtime_object_role(runtime_object: Mapping[str, Any] | None) -> str:
    if not isinstance(runtime_object, Mapping):
        return ""
    return str(runtime_object.get("role", "")).strip().lower()


def _object_category_from_shape(shape: str, *, object_mode: str) -> str:
    if object_mode == "concrete":
        return {
            "cuboid": "box",
            "cube": "box",
            "sphere": "cup",
            "cylinder": "bottle",
            "ring": "plate",
            "plate": "plate",
            "bowl": "bowl",
        }.get(shape, "object")
    return {
        "box": "cube",
        "cup": "cylinder",
        "bottle": "cylinder",
        "plate": "ring",
        "bowl": "cylinder",
        "cuboid": "cube",
        "sphere": "sphere",
        "cylinder": "cylinder",
        "ring": "ring",
    }.get(shape, "cube")


def _runtime_object_category(runtime_object: Mapping[str, Any], *, object_mode: str) -> str:
    kind = str(runtime_object.get("kind", "")).strip()
    role = _runtime_object_role(runtime_object)
    asset_key = str(runtime_object.get("asset_key", "")).strip().lower()
    if object_mode == "concrete":
        if "bowl" in asset_key or "bowl" in role:
            return "bowl"
        if "plate" in asset_key or "plate" in role:
            return "plate"
        if kind == "primitive_sphere":
            return "cup"
        if kind == "primitive_cylinder":
            return "bottle"
        return "box"
    if "plate" in asset_key:
        return "ring"
    if kind == "primitive_sphere":
        return "sphere"
    if kind == "primitive_cylinder" or kind == "primitive_cone" or "bowl" in asset_key:
        return "cylinder"
    return "cube"


def _should_preserve_structural_runtime_object(runtime_object: Mapping[str, Any]) -> bool:
    role = _runtime_object_role(runtime_object)
    return bool(role.startswith("container_") or role.endswith("_wall") or role.endswith("_floor"))


def _transform_runtime_object_for_mode(runtime_object: Mapping[str, Any], *, object_mode: str) -> dict[str, Any]:
    resolved = deepcopy(dict(runtime_object))
    resolved["object_category"] = _runtime_object_category(resolved, object_mode=object_mode)
    if _should_preserve_structural_runtime_object(resolved):
        return resolved

    kind = str(resolved.get("kind", "")).strip()
    asset_key = str(resolved.get("asset_key", "")).strip().lower()
    name = str(resolved.get("name", "")).strip()
    role = _runtime_object_role(resolved)
    dimensions = _runtime_object_dimensions(resolved)

    # Visual-layout distractor meshes (cup / bottle / bowl USD) must not be collapsed into
    # primitive cylinders in geometric object_mode, or distinct clutter types are lost.
    if name.startswith("viz_clutter_") and role == "distractor":
        return resolved
    if role == "plate_trinket":
        return resolved

    if object_mode == "geometric" and kind == "usd" and ("plate" in asset_key or "bowl" in asset_key):
        radius = 0.06
        height = 0.02
        if dimensions is not None:
            radius = max(float(dimensions[0]), float(dimensions[1])) * 0.5
            height = max(float(dimensions[2]), 0.012)
        if "plate" in asset_key:
            height = max(min(height, 0.018), 0.01)
        resolved.pop("asset_category", None)
        resolved.pop("asset_key", None)
        resolved.pop("scale", None)
        resolved.pop("material_override", None)
        resolved["kind"] = "primitive_cylinder"
        resolved["radius"] = round(float(radius), 6)
        resolved["height"] = round(float(height), 6)
        return resolved

    if object_mode == "concrete" and kind == "primitive_sphere":
        diameter = 0.06
        if dimensions is not None:
            diameter = max(float(dimensions[0]), 0.05)
        resolved["kind"] = "primitive_cylinder"
        resolved["radius"] = round(float(max(diameter * 0.33, 0.018)), 6)
        resolved["height"] = round(float(max(diameter * 0.9, 0.05)), 6)
        return resolved

    return resolved


def _apply_object_mode_to_protocol(task_spec: dict[str, Any], *, object_mode: str) -> None:
    protocol = task_spec.get("protocol", {})
    if not isinstance(protocol, dict):
        return

    runtime_objects = task_spec.get("runtime_objects", [])
    runtime_by_name = {
        str(runtime_object.get("name", "")).strip(): runtime_object
        for runtime_object in runtime_objects
        if isinstance(runtime_object, dict) and str(runtime_object.get("name", "")).strip()
    }

    object_semantics = protocol.get("object_semantics", {})
    resolved_semantics: dict[str, dict[str, str]] = {}
    if isinstance(object_semantics, Mapping):
        for object_name, payload in object_semantics.items():
            if not isinstance(payload, Mapping):
                continue
            semantics = {str(key): str(value) for key, value in payload.items() if str(key).strip() and str(value).strip()}
            runtime_object = runtime_by_name.get(str(object_name).strip(), {})
            category = _runtime_object_category(runtime_object, object_mode=object_mode) if runtime_object else ""
            if category:
                semantics["object_category"] = category
            if "color" not in semantics:
                color_label = _runtime_object_color_label(runtime_object)
                if color_label:
                    semantics["color"] = color_label
            resolved_semantics[str(object_name)] = semantics
    else:
        for object_name, runtime_object in runtime_by_name.items():
            semantics: dict[str, str] = {}
            color_label = _runtime_object_color_label(runtime_object)
            if color_label:
                semantics["color"] = color_label
            semantics["object_category"] = _runtime_object_category(runtime_object, object_mode=object_mode)
            resolved_semantics[object_name] = semantics
    if resolved_semantics:
        protocol["object_semantics"] = resolved_semantics

    gt_spec = protocol.get("gt_spec", {})
    if isinstance(gt_spec, dict):
        target_filters = gt_spec.get("target_filters", {})
        if isinstance(target_filters, dict):
            shape_value = str(target_filters.get("shape", "")).strip().lower()
            if shape_value:
                target_filters["object_category"] = _object_category_from_shape(shape_value, object_mode=object_mode)
            gt_spec["target_filters"] = target_filters

    protocol["object_mode"] = object_mode
    task_spec["protocol"] = protocol


def _rewrite_instruction_for_object_mode(text: str, *, object_mode: str) -> str:
    if not text:
        return text
    if object_mode == "concrete":
        replacements = (
            ("木纹圆柱体", "木纹瓶子"),
            ("圆柱体", "瓶子"),
            ("小球", "杯子"),
            ("球体", "杯子"),
            ("方块", "盒子"),
            ("木块", "木盒"),
            ("蓝盘", "蓝色盘子"),
            ("红盘", "红色盘子"),
        )
    else:
        replacements = (
            ("木纹瓶子", "木纹圆柱体"),
            ("盘子", "圆环"),
            ("瓶子", "圆柱体"),
            ("杯子", "球体"),
            ("盒子", "方块"),
            ("木盒", "木块"),
        )
    resolved = str(text)
    for old, new in replacements:
        resolved = resolved.replace(old, new)
    return resolved


def apply_object_mode(task_spec: dict[str, Any], *, object_mode: str | None) -> dict[str, Any]:
    resolved_object_mode = normalize_object_mode(object_mode)
    if not resolved_object_mode:
        return task_spec

    task_spec["object_mode"] = resolved_object_mode
    runtime_objects = task_spec.get("runtime_objects", [])
    if isinstance(runtime_objects, list) and runtime_objects:
        transformed_runtime_objects = [
            _transform_runtime_object_for_mode(runtime_object, object_mode=resolved_object_mode)
            if isinstance(runtime_object, dict)
            else runtime_object
            for runtime_object in runtime_objects
        ]
        task_spec["runtime_objects"] = transformed_runtime_objects
        task_spec["object_positions"] = _build_runtime_object_positions(
            [runtime_object for runtime_object in transformed_runtime_objects if isinstance(runtime_object, dict)]
        )
        scene_adaptation = task_spec.get("scene_adaptation", {})
        if isinstance(scene_adaptation, dict):
            scene_adaptation["runtime_objects"] = deepcopy(transformed_runtime_objects)
            scene_adaptation["object_mode"] = resolved_object_mode
            task_spec["scene_adaptation"] = scene_adaptation

    _apply_object_mode_to_protocol(task_spec, object_mode=resolved_object_mode)
    task_spec["instruction"] = _rewrite_instruction_for_object_mode(str(task_spec.get("instruction", "")), object_mode=resolved_object_mode)
    protocol = task_spec.get("protocol", {})
    if isinstance(protocol, dict):
        protocol["query_text"] = _rewrite_instruction_for_object_mode(
            str(protocol.get("query_text", task_spec.get("instruction", ""))),
            object_mode=resolved_object_mode,
        )
        task_spec["protocol"] = protocol

    metadata = task_spec.get("task_metadata", {})
    if isinstance(metadata, dict):
        metadata["object_mode"] = resolved_object_mode
        task_spec["task_metadata"] = metadata
    return task_spec


def build_default_task_spec(scene_profile: "SceneProfile") -> dict[str, Any]:
    object_names = list(scene_profile.object_names)
    ordered = _sort_by_suffix(object_names)
    if not ordered:
        return {
            "task_name": "empty_scene",
            "instruction": "No object is present in this scene.",
            "goal_sequence": [],
            "object_positions": {},
        }

    default_positions = {
        name: {"position": list(scene_profile.xforms[name]["pos"][:2])} for name in object_names
    }
    leftmost = ordered[0]
    basket_xy = [0.0, 0.0]
    if scene_profile.basket_name is not None:
        basket_xy = list(scene_profile.xforms[scene_profile.basket_name]["pos"][:2])
    return {
        "task_name": "place_in_basket",
        "instruction": "Put the leftmost object inside the container.",
        "goal_sequence": [
            {
                "step": 1,
                "goals": [
                    {
                        "object_id": leftmost,
                        "xy": basket_xy,
                        "tolerance": {"pos_r": 0.08},
                    }
                ],
            }
        ],
        "object_positions": default_positions,
    }


# viz_clutter_*: static mesh collision only (no rigid dynamics). Plate anchor uses kinematic USD in annotations.
_DIRECTIONAL_VISUAL_CLUTTER_PHYSICS: dict[str, Any] = {
    "rigid_body": {"rigid_body_enabled": False},
    "collision": {
        "type": "static_mesh",
        "contact_offset": 0.0015,
        "rest_offset": 0.0,
    },
    "material": {
        "static_friction": 1.2,
        "dynamic_friction": 1.0,
        "restitution": 0.0,
        "friction_combine_mode": "max",
    },
}

# plate_trinket_*: small dynamic rigid primitives on the plate (collide with plate/cube/each other).
_DIRECTIONAL_VISUAL_TRINKET_PHYSICS: dict[str, Any] = {
    "rigid_body": {
        "rigid_body_enabled": True,
        "kinematic_enabled": False,
        "linear_damping": 2.5,
        "angular_damping": 5.0,
        "max_linear_velocity": 2.0,
        "max_angular_velocity": 20.0,
        "solver_position_iteration_count": 16,
        "solver_velocity_iteration_count": 4,
        "sleep_threshold": 0.002,
        "stabilization_threshold": 0.001,
    },
    "collision": {
        "collision_enabled": True,
        "contact_offset": 0.0015,
        "rest_offset": 0.0,
    },
    "material": {
        "static_friction": 1.2,
        "dynamic_friction": 1.0,
        "restitution": 0.0,
        "friction_combine_mode": "max",
    },
}

_PLACE_DIRECTIONAL_SCENE65_VISUAL_LAYOUT = "place_directional_scene65_visual"
# Random clutter semantic kinds (no standalone large cylinder clutter; tiny cylinders may appear as plate trinkets).
_VISUAL_CLUTTER_KINDS: tuple[str, ...] = ("can", "bottle", "bowl")
# ``place_directional_scene65_visual`` gt_spec.target_zone in world XY (anchor = blue_plate center).
_DIRECTIONAL_VISUAL_ZONE_OFFSET_XY: tuple[float, float] = (0.18, 0.0)
_DIRECTIONAL_VISUAL_ZONE_HALF_EXTENTS: tuple[float, float] = (0.05, 0.06)
_VIZ_CLUTTER_RADIAL_MIN_M: float = 0.12
_VIZ_CLUTTER_RADIAL_MAX_M: float = 0.18
_VIZ_CLUTTER_CUBE_CLEARANCE_M: float = 0.11
# Extra XY gap between blue_plate footprint and clutter footprint (static mesh overlap avoidance in XY).
_VIZ_CLUTTER_PLATE_GAP_M: float = 0.018
# plate_trinket_*: XY clearance between trinket footprints and inset from the blue plate rim (disk model).
_PLATE_TRINKET_PAIR_GAP_M: float = 0.006
_PLATE_TRINKET_PLATE_INSET_M: float = 0.008
_PLATE_TRINKET_CYCLE: tuple[str, ...] = (
    "primitive_cuboid",
    "primitive_sphere",
    "primitive_cylinder",
    "primitive_cone",
)
_PLATE_TRINKET_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.22, 0.22),
    (0.2, 0.78, 0.32),
    (0.25, 0.45, 0.95),
    (0.95, 0.82, 0.18),
    (0.78, 0.35, 0.92),
    (0.12, 0.88, 0.88),
)


def _extract_runtime_pose_xy(runtime_objects: Sequence[Any], object_name: str) -> tuple[float, float] | None:
    for runtime_object in runtime_objects:
        if not isinstance(runtime_object, dict):
            continue
        if str(runtime_object.get("name", "")).strip() != object_name:
            continue
        pose = runtime_object.get("pose", {})
        if not isinstance(pose, dict):
            return None
        xy = pose.get("xy", [])
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            return None
        try:
            return (float(xy[0]), float(xy[1]))
        except (TypeError, ValueError):
            return None
    return None


def _target_zone_center_xy(anchor_xy: tuple[float, float]) -> tuple[float, float]:
    return (
        float(anchor_xy[0]) + _DIRECTIONAL_VISUAL_ZONE_OFFSET_XY[0],
        float(anchor_xy[1]) + _DIRECTIONAL_VISUAL_ZONE_OFFSET_XY[1],
    )


def _point_in_directional_target_zone_xy(
    px: float,
    py: float,
    *,
    anchor_xy: tuple[float, float],
) -> bool:
    zcx, zcy = _target_zone_center_xy(anchor_xy)
    hx, hy = _DIRECTIONAL_VISUAL_ZONE_HALF_EXTENTS
    return abs(px - zcx) <= hx + 1e-9 and abs(py - zcy) <= hy + 1e-9


def _directional_visual_blue_plate_xy_radius_m(runtime_objects: Sequence[Any]) -> float:
    """Approximate XY clearance from anchor to blue_plate rim (half of max horizontal USD extent)."""
    for runtime_object in runtime_objects:
        if not isinstance(runtime_object, dict):
            continue
        if str(runtime_object.get("name", "")).strip() != "blue_plate":
            continue
        dims = _runtime_object_dimensions(runtime_object)
        if dims is None:
            break
        return 0.5 * max(float(dims[0]), float(dims[1])) + 1e-6
    info = asset_utils.get_asset_info_by_name("catch_object", "plate_2")
    if isinstance(info, dict):
        d = info.get("dimensions", {})
        if isinstance(d, dict):
            try:
                dx = abs(float(d.get("x", 0.0))) * 0.01
                dy = abs(float(d.get("y", 0.0))) * 0.01
                return 0.5 * max(dx, dy) + 1e-6
            except (TypeError, ValueError):
                pass
    return 0.13


def _runtime_object_xy_footprint_radius_m(runtime_object: Mapping[str, Any]) -> float:
    dims = _runtime_object_dimensions(runtime_object)
    if dims is None:
        return 0.08
    return 0.5 * max(float(dims[0]), float(dims[1])) + 1e-6


def _sample_viz_clutter_xy(
    rng: random.Random,
    *,
    anchor_xy: tuple[float, float],
    cube_xy: tuple[float, float],
    plate_xy_radius_m: float,
    clutter_xy_radius_m: float,
    max_attempts: int = 220,
) -> tuple[float, float]:
    """Sample XY outside legal rect and away from cube; radially outside plate + clutter footprints."""
    ax, ay = anchor_xy
    cx, cy = cube_xy
    r_geom = float(plate_xy_radius_m) + float(clutter_xy_radius_m) + float(_VIZ_CLUTTER_PLATE_GAP_M)
    r_min = max(_VIZ_CLUTTER_RADIAL_MIN_M, r_geom)
    r_max = max(_VIZ_CLUTTER_RADIAL_MAX_M, r_min + 0.028)
    r_min_sq = r_min * r_min
    r_max_sq = r_max * r_max
    for _ in range(max_attempts):
        theta = rng.uniform(0.0, 2.0 * math.pi)
        rad = math.sqrt(rng.uniform(0.0, 1.0) * (r_max_sq - r_min_sq) + r_min_sq)
        px = ax + rad * math.cos(theta)
        py = ay + rad * math.sin(theta)
        if _point_in_directional_target_zone_xy(px, py, anchor_xy=anchor_xy):
            continue
        if math.hypot(px - cx, py - cy) < _VIZ_CLUTTER_CUBE_CLEARANCE_M:
            continue
        return (px, py)
    # Rare fallback: offset along +X beyond plate + clutter.
    return (ax + r_min + 0.01, ay)


def _pose_on_xy(xy: tuple[float, float], rng: random.Random) -> dict[str, Any]:
    return {
        "xy": [float(xy[0]), float(xy[1])],
        "yaw": rng.uniform(-0.55, 0.55),
    }


def _sample_plate_trinket_xy(
    rng: random.Random,
    *,
    anchor_xy: tuple[float, float],
    plate_r: float,
    trinket_xy_r: float,
    placed: list[tuple[float, float, float]],
    max_attempts: int = 140,
) -> tuple[float, float] | None:
    """Sample plate center offset in disk inside the plate rim, outside gt target rect, away from other trinkets."""
    ax, ay = anchor_xy
    margin = float(_PLATE_TRINKET_PLATE_INSET_M)
    gap = float(_PLATE_TRINKET_PAIR_GAP_M)
    r_self = float(trinket_xy_r)
    max_rr = float(plate_r) - r_self - margin
    if max_rr <= 1e-5:
        return None
    max_rr_sq = max_rr * max_rr
    for _ in range(max_attempts):
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = math.sqrt(rng.uniform(0.0, 1.0) * max_rr_sq)
        tx = ax + rad * math.cos(ang)
        ty = ay + rad * math.sin(ang)
        if _point_in_directional_target_zone_xy(tx, ty, anchor_xy=anchor_xy):
            continue
        ok = True
        for ox, oy, o_r in placed:
            if math.hypot(tx - ox, ty - oy) < r_self + float(o_r) + gap:
                ok = False
                break
        if ok:
            return (tx, ty)
    return None


def _fallback_sample_plate_trinket_xy(
    *,
    anchor_xy: tuple[float, float],
    plate_r: float,
    trinket_xy_r: float,
    placed: list[tuple[float, float, float]],
    index_hint: int,
) -> tuple[float, float] | None:
    ax, ay = anchor_xy
    margin = float(_PLATE_TRINKET_PLATE_INSET_M)
    gap = float(_PLATE_TRINKET_PAIR_GAP_M)
    r_self = float(trinket_xy_r)
    max_rr = float(plate_r) - r_self - margin
    if max_rr <= 1e-5:
        return None
    golden = 2.39996322972865332
    for step in range(72):
        frac = (step + 0.5) / 72.0
        rr = max_rr * (0.1 + 0.9 * frac * frac)
        ang = golden * float(step) + float(index_hint) * 1.83
        tx = ax + rr * math.cos(ang)
        ty = ay + rr * math.sin(ang)
        if _point_in_directional_target_zone_xy(tx, ty, anchor_xy=anchor_xy):
            continue
        if all(math.hypot(tx - ox, ty - oy) >= r_self + float(o_r) + gap for ox, oy, o_r in placed):
            return (tx, ty)
    return None


def _emergency_plate_trinket_xy(
    *,
    anchor_xy: tuple[float, float],
    plate_r: float,
    trinket_xy_r: float,
    placed: list[tuple[float, float, float]],
    slot_index: int,
) -> tuple[float, float]:
    """Dense ring search when random and golden-angle fallbacks fail (keeps trinkets inside the plate disk when possible)."""
    ax, ay = anchor_xy
    margin = float(_PLATE_TRINKET_PLATE_INSET_M)
    gap = float(_PLATE_TRINKET_PAIR_GAP_M)
    r_self = float(trinket_xy_r)
    max_rr = max(1e-4, float(plate_r) - r_self - margin)
    n_ring = 18
    for ring in range(5):
        rr = max_rr * (0.12 + 0.88 * (ring + 0.5) / 5.0)
        for k in range(n_ring):
            ang = (k + slot_index * 0.41) * (2.0 * math.pi / n_ring)
            tx = ax + rr * math.cos(ang)
            ty = ay + rr * math.sin(ang)
            if _point_in_directional_target_zone_xy(tx, ty, anchor_xy=anchor_xy):
                continue
            if all(math.hypot(tx - ox, ty - oy) >= r_self + float(o_r) + gap for ox, oy, o_r in placed):
                return (tx, ty)
    return (ax + 0.015 * float(1 + slot_index % 5), ay + 0.012 * float(slot_index % 3))


def _build_directional_visual_clutter_fallback_usd(
    *,
    name: str,
    pose: dict[str, Any],
    rng: random.Random,
    physics: dict[str, Any],
) -> dict[str, Any]:
    for candidates, scale in (
        (("yellow_bottle", "white_bottle"), [0.55, 0.55, 0.55]),
        (("bowl_1", "bowl_2"), [0.01, 0.01, 0.01]),
    ):
        key = _pick_catch_object_key(rng, candidates)
        if key is None:
            continue
        return {
            "name": name,
            "kind": "usd",
            "asset_category": "catch_object",
            "asset_key": key,
            "role": "distractor",
            "scale": list(scale),
            "physics": deepcopy(physics),
            "pose": pose,
        }
    return {
        "name": name,
        "kind": "usd",
        "asset_category": "catch_object",
        "asset_key": "yellow_bottle",
        "role": "distractor",
        "scale": [0.55, 0.55, 0.55],
        "physics": deepcopy(physics),
        "pose": pose,
    }


def _plate_trinket_object_without_pose(*, index: int, rng: random.Random) -> dict[str, Any]:
    """Small colorful rigid primitives on the blue plate (decoration only); caller sets ``pose`` after XY sampling."""
    _g = 4.0
    physics = deepcopy(_DIRECTIONAL_VISUAL_TRINKET_PHYSICS)
    shape = _PLATE_TRINKET_CYCLE[index % len(_PLATE_TRINKET_CYCLE)]
    color = list(_PLATE_TRINKET_COLORS[index % len(_PLATE_TRINKET_COLORS)])
    for channel in range(3):
        jitter = rng.uniform(-0.06, 0.06)
        color[channel] = max(0.05, min(0.99, float(color[channel]) + jitter))
    rough = round(rng.uniform(0.35, 0.55), 3)
    name = f"plate_trinket_{index + 1}"
    if shape == "primitive_cuboid":
        s = 0.0075 * _g
        size = [s, s, s]
        return {
            "name": name,
            "kind": "primitive_cuboid",
            "role": "plate_trinket",
            "size": size,
            "visual_color": color,
            "visual_material": {"fallback_diffuse_color": color, "fallback_roughness": rough},
            "physics": physics,
        }
    if shape == "primitive_sphere":
        return {
            "name": name,
            "kind": "primitive_sphere",
            "role": "plate_trinket",
            "radius": 0.0048 * _g,
            "visual_color": color,
            "visual_material": {"fallback_diffuse_color": color, "fallback_roughness": rough},
            "physics": physics,
        }
    if shape == "primitive_cylinder":
        return {
            "name": name,
            "kind": "primitive_cylinder",
            "role": "plate_trinket",
            "radius": 0.0038 * _g,
            "height": 0.0085 * _g,
            "visual_color": color,
            "visual_material": {"fallback_diffuse_color": color, "fallback_roughness": rough},
            "physics": physics,
        }
    return {
        "name": name,
        "kind": "primitive_cone",
        "role": "plate_trinket",
        "radius": 0.0042 * _g,
        "height": 0.009 * _g,
        "visual_color": color,
        "visual_material": {"fallback_diffuse_color": color, "fallback_roughness": rough},
        "physics": physics,
    }


def _inject_place_directional_visual_plate_trinkets(
    runtime_objects: list[Any],
    *,
    rng: random.Random,
) -> None:
    anchor_xy = _extract_runtime_pose_xy(runtime_objects, "blue_plate")
    if anchor_xy is None:
        anchor_xy = (0.39, 0.02)
    plate_r = _directional_visual_blue_plate_xy_radius_m(runtime_objects)
    placed: list[tuple[float, float, float]] = []
    count = rng.randint(4, 6)
    for i in range(count):
        body = _plate_trinket_object_without_pose(index=i, rng=rng)
        r_xy = _runtime_object_xy_footprint_radius_m(body)
        xy = _sample_plate_trinket_xy(
            rng,
            anchor_xy=anchor_xy,
            plate_r=plate_r,
            trinket_xy_r=r_xy,
            placed=placed,
        )
        if xy is None:
            xy = _fallback_sample_plate_trinket_xy(
                anchor_xy=anchor_xy,
                plate_r=plate_r,
                trinket_xy_r=r_xy,
                placed=placed,
                index_hint=i,
            )
        if xy is None:
            xy = _emergency_plate_trinket_xy(
                anchor_xy=anchor_xy,
                plate_r=plate_r,
                trinket_xy_r=r_xy,
                placed=placed,
                slot_index=i,
            )
        tx, ty = float(xy[0]), float(xy[1])
        body["pose"] = {"xy": [tx, ty], "yaw": rng.uniform(-0.5, 0.5)}
        placed.append((tx, ty, r_xy))
        runtime_objects.append(body)


def _resolve_visual_clutter_rng() -> random.Random:
    raw = str(os.environ.get("GEOMIND_VISUAL_CLUTTER_SEED", "")).strip()
    if raw:
        try:
            seed = int(raw, 0)
        except ValueError:
            seed = (hash(raw) & 0xFFFFFFFFFFFFFFFF) ^ 0x9E3779B97F4A7C15
        return random.Random(seed)
    return random.Random(secrets.randbits(64))


def _catch_object_key_present(asset_key: str) -> bool:
    return asset_utils.get_asset_info_by_name("catch_object", asset_key) is not None


def _pick_catch_object_key(rng: random.Random, candidates: tuple[str, ...]) -> str | None:
    present = [key for key in candidates if _catch_object_key_present(key)]
    if not present:
        return None
    return rng.choice(present)


def _build_directional_visual_clutter_object(
    *,
    name: str,
    clutter_kind: str,
    pose: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    physics = deepcopy(_DIRECTIONAL_VISUAL_CLUTTER_PHYSICS)
    if clutter_kind == "can":
        key = _pick_catch_object_key(rng, ("blue_can", "red_can"))
        if key is None:
            return _build_directional_visual_clutter_fallback_usd(name=name, pose=pose, rng=rng, physics=physics)
        scale = [0.45, 0.45, 0.45] if key == "blue_can" else [0.52, 0.52, 0.52]
        return {
            "name": name,
            "kind": "usd",
            "asset_category": "catch_object",
            "asset_key": key,
            "role": "distractor",
            "scale": scale,
            "physics": physics,
            "pose": pose,
        }
    if clutter_kind == "bottle":
        key = _pick_catch_object_key(rng, ("yellow_bottle", "white_bottle"))
        if key is None:
            return _build_directional_visual_clutter_fallback_usd(name=name, pose=pose, rng=rng, physics=physics)
        return {
            "name": name,
            "kind": "usd",
            "asset_category": "catch_object",
            "asset_key": key,
            "role": "distractor",
            "scale": [0.55, 0.55, 0.55],
            "physics": physics,
            "pose": pose,
        }
    if clutter_kind == "bowl":
        key = _pick_catch_object_key(rng, ("bowl_1", "bowl_2"))
        if key is None:
            return _build_directional_visual_clutter_fallback_usd(name=name, pose=pose, rng=rng, physics=physics)
        return {
            "name": name,
            "kind": "usd",
            "asset_category": "catch_object",
            "asset_key": key,
            "role": "distractor",
            "scale": [0.01, 0.01, 0.01],
            "physics": physics,
            "pose": pose,
        }
    return _build_directional_visual_clutter_fallback_usd(name=name, pose=pose, rng=rng, physics=physics)


def _randomize_place_directional_scene65_visual_clutter(
    runtime_objects: list[Any],
    *,
    rng: random.Random | None = None,
) -> None:
    """In-place: ``viz_clutter_*`` → random can (blue/red YCB) / bottle / bowl.

    XY radius is at least ``max(12 cm, plate_radius + clutter_radius + gap)`` so static clutter clears the blue_plate mesh in XY.

    ``GEOMIND_VISUAL_CLUTTER_KIND`` may be ``can`` / ``bottle`` / ``bowl`` (``cup`` is accepted as alias for ``can``).
    """
    active = rng if rng is not None else _resolve_visual_clutter_rng()
    forced_raw = str(os.environ.get("GEOMIND_VISUAL_CLUTTER_KIND", "")).strip().lower()
    if forced_raw == "cup":
        forced_raw = "can"
    if forced_raw == "cylinder":
        forced_raw = ""
    force_variant = forced_raw if forced_raw in _VISUAL_CLUTTER_KINDS else ""

    anchor_xy = _extract_runtime_pose_xy(runtime_objects, "blue_plate")
    cube_xy = _extract_runtime_pose_xy(runtime_objects, "movable_cube")
    if anchor_xy is None:
        anchor_xy = (0.39, 0.02)
    if cube_xy is None:
        cube_xy = (0.24, -0.11)

    plate_r = _directional_visual_blue_plate_xy_radius_m(runtime_objects)

    for index, obj in enumerate(runtime_objects):
        if not isinstance(obj, dict):
            continue
        if _runtime_object_role(obj) != "distractor":
            continue
        name = str(obj.get("name", "")).strip()
        if not name.startswith("viz_clutter_"):
            continue
        variant = force_variant if force_variant else active.choice(_VISUAL_CLUTTER_KINDS)
        temp_pose: dict[str, Any] = {"xy": [float(anchor_xy[0]), float(anchor_xy[1])], "yaw": 0.0}
        built = _build_directional_visual_clutter_object(
            name=name,
            clutter_kind=variant,
            pose=temp_pose,
            rng=active,
        )
        clutter_r = _runtime_object_xy_footprint_radius_m(built)
        xy = _sample_viz_clutter_xy(
            active,
            anchor_xy=anchor_xy,
            cube_xy=cube_xy,
            plate_xy_radius_m=plate_r,
            clutter_xy_radius_m=clutter_r,
        )
        built["pose"] = _pose_on_xy(xy, active)
        runtime_objects[index] = built


def _build_task_metadata(
    *,
    layout_key: str,
    task_index: int,
    layout: dict[str, Any] | None,
    task_data: dict[str, Any] | None,
    spec: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(layout, dict) and isinstance(task_data, dict):
        semantic_view = normalize_annotation_entry(
            layout_key=layout_key,
            task_index=task_index,
            layout=layout,
            task_payload=task_data,
        )
    else:
        semantic_view = normalize_config_semantics(
            raw_task_id=spec.get("task_name"),
            layout_key=layout_key,
            task_index=task_index,
            instruction=spec.get("instruction"),
        )
    task_metadata = semantic_view.get("task_metadata", {})
    return dict(task_metadata) if isinstance(task_metadata, dict) else {}


def load_task_spec(
    layout_key: str,
    task_index: int,
    scene_profile: "SceneProfile",
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    defaults = build_default_task_spec(scene_profile)
    payload = load_annotated_tasks(path)
    layout = payload.get(layout_key)
    if not isinstance(layout, dict):
        defaults["task_metadata"] = _build_task_metadata(
            layout_key=layout_key,
            task_index=task_index,
            layout=None,
            task_data=None,
            spec=defaults,
        )
        return defaults

    tasks = layout.get("tasks", [])
    if not isinstance(tasks, list) or task_index < 0 or task_index >= len(tasks):
        defaults["task_metadata"] = _build_task_metadata(
            layout_key=layout_key,
            task_index=task_index,
            layout=layout,
            task_data=None,
            spec=defaults,
        )
        return defaults

    task_data = tasks[task_index]
    if not isinstance(task_data, dict):
        defaults["task_metadata"] = _build_task_metadata(
            layout_key=layout_key,
            task_index=task_index,
            layout=layout,
            task_data=None,
            spec=defaults,
        )
        return defaults

    annotation = task_data.get("annotation", {})
    if not isinstance(annotation, dict):
        annotation = {}
    scene_structure = layout.get("scene_structure", {})
    if not isinstance(scene_structure, dict):
        scene_structure = {}
    object_positions = scene_structure.get("object_positions", defaults["object_positions"])
    if not isinstance(object_positions, dict):
        object_positions = defaults["object_positions"]

    spec = {
        "task_name": str(task_data.get("task_category", f"task_{task_index + 1}")),
        "instruction": str(task_data.get("instruction", defaults["instruction"])),
        "goal_sequence": annotation.get("goal_sequence", defaults["goal_sequence"]),
        "object_positions": object_positions,
    }
    protocol = _extract_protocol_spec(annotation, spec["instruction"])
    if protocol:
        spec["protocol"] = protocol
        spec["instruction"] = resolve_task_instruction(spec, protocol)
    scene_adaptation = _extract_scene_adaptation(annotation)
    runtime_objects = scene_adaptation.get("runtime_objects", [])
    preview_proxy_goal = scene_adaptation.get("preview_proxy_goal", {})
    if isinstance(preview_proxy_goal, dict) and preview_proxy_goal:
        spec["preview_proxy_goal"] = deepcopy(preview_proxy_goal)
    has_runtime_object_injection = isinstance(runtime_objects, list) and bool(runtime_objects)
    if has_runtime_object_injection:
        spec["scene_adaptation"] = scene_adaptation
        resolved_runtime_objects = deepcopy(runtime_objects)
        if _should_scale_track_reindex_cuboids(protocol):
            _scale_runtime_cuboid_sizes(resolved_runtime_objects, scale=_TRACK_REINDEX_CUBOID_SCALE)
        if str(layout_key) == _PLACE_DIRECTIONAL_SCENE65_VISUAL_LAYOUT:
            _visual_rng = _resolve_visual_clutter_rng()
            _randomize_place_directional_scene65_visual_clutter(resolved_runtime_objects, rng=_visual_rng)
            _inject_place_directional_visual_plate_trinkets(resolved_runtime_objects, rng=_visual_rng)
        applied_randomization = randomize_runtime_object_layout(
            resolved_runtime_objects,
            protocol=protocol,
            layout_key=str(layout_key),
        )
        spec["runtime_objects"] = resolved_runtime_objects
        if applied_randomization is not None:
            spec["applied_layout_randomization"] = applied_randomization
        if isinstance(scene_adaptation, dict):
            scene_adaptation["runtime_objects"] = resolved_runtime_objects
            if applied_randomization is not None:
                scene_adaptation["applied_layout_randomization"] = applied_randomization
        spec["layout_object_positions"] = deepcopy(object_positions)
        spec["object_positions"] = _build_runtime_object_positions(resolved_runtime_objects)

    if not has_runtime_object_injection:
        spec = remap_task_objects(spec, list(scene_profile.object_names))
        for object_name in scene_profile.object_names:
            spec.setdefault("object_positions", {}).setdefault(
                object_name, {"position": list(scene_profile.xforms[object_name]["pos"][:2])}
            )
    spec["task_metadata"] = _build_task_metadata(
        layout_key=layout_key,
        task_index=task_index,
        layout=layout,
        task_data=task_data,
        spec=spec,
    )
    return spec


def load_task(
    *,
    task_id: Any = None,
    variant: Any = None,
    scene: Any = None,
    scene_id: Any = None,
    layout_key: Any = None,
    task_index: Any = None,
    object_mode: Any = None,
    scene_profile: "SceneProfile" | None = None,
    path: str | Path | None = None,
    task_contract: dict[str, Any] | None = None,
) -> LoadedTask:
    request = resolve_task_request(
        task_id=task_id,
        variant=variant,
        scene=scene,
        scene_id=scene_id,
        layout_key=layout_key,
        task_index=task_index,
        object_mode=object_mode,
        path=path,
    )
    resolved_scene_profile = scene_profile
    if resolved_scene_profile is None:
        try:
            from .geomind_scene_cfg import get_scene_profile
        except (ImportError, ModuleNotFoundError):
            resolved_scene_profile = _build_lightweight_scene_profile(
                request.layout_key,
                request.scene_id,
                path=path,
            )
        else:
            resolved_scene_profile = get_scene_profile(request.scene_id)

    task_spec = load_task_spec(
        request.layout_key,
        request.task_index,
        resolved_scene_profile,
        path=path,
    )
    task_spec = apply_object_mode(task_spec, object_mode=request.object_mode)
    from .task_evolution import apply_task_contract
    task_spec = apply_task_contract(task_spec, request.task_id, task_contract)

    task_metadata = dict(task_spec.get("task_metadata", {}) or {})
    if request.task_id:
        task_metadata["task_id"] = request.task_id
    task_metadata["variant"] = request.variant
    task_metadata["scene"] = request.scene
    task_metadata["scene_name"] = request.scene
    task_metadata["scene_id"] = int(request.scene_id)
    task_metadata["recommended_scene"] = request.recommended_scene
    task_metadata["recommended_scene_name"] = request.recommended_scene
    task_metadata["recommended_scene_id"] = int(request.recommended_scene_id)
    task_metadata["scene_overridden"] = bool(request.scene_overridden)
    task_metadata["layout_key"] = str(request.layout_key)
    task_metadata["task_index"] = int(request.task_index)
    task_metadata["object_mode"] = request.object_mode
    if request.backend:
        task_metadata["backend"] = request.backend
    task_spec["task_metadata"] = task_metadata
    task_spec["resolved_task_request"] = {
        "task_id": request.task_id,
        "variant": request.variant,
        "scene": request.scene,
        "scene_name": request.scene,
        "scene_id": int(request.scene_id),
        "recommended_scene": request.recommended_scene,
        "recommended_scene_name": request.recommended_scene,
        "recommended_scene_id": int(request.recommended_scene_id),
        "scene_overridden": bool(request.scene_overridden),
        "layout_key": request.layout_key,
        "task_index": int(request.task_index),
        "object_mode": request.object_mode,
        "backend": request.backend,
        "raw_task_id": request.raw_task_id,
    }

    return LoadedTask(
        request=request,
        task_spec=task_spec,
        task_label=resolve_task_label(request.layout_key, request.task_index, path),
    )
