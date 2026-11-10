"""Direct-reference ablation overrides for the GeoMind VLA task suite.

The ablation keeps the scene, object poses, intervention scripts and the seed of a
task byte-identical to its base version; the only differences are

* the instruction text (both ``TaskSpec.instruction`` and ``protocol.query_text``), and
* either the GT target's colour (``recolor``) or an added visual-only marker disc
  placed at the GT goal point (``marker``).

Every mutation goes through :func:`apply_direct_override`, which asserts that no key
outside an explicit whitelist changed. That assertion is what makes base/direct
pairing provable rather than merely intended.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "direct_ablation" / "direct_tasks.json"

BASE_ABLATION_ID = "base"
DIRECT_ABLATION_ID = "direct"

MARKER_NAME = "direct_marker"

#: Paths inside the task spec the override is allowed to touch. Anything else is a bug.
_ALLOWED_MUTATIONS = (
    "instruction",
    "protocol.query_text",
    "runtime_objects[*].visual_color",
    "runtime_objects[*].visual_material",
    f"runtime_objects[+] name={MARKER_NAME}",
    f"object_positions[+] {MARKER_NAME}",
    "scene_adaptation.runtime_objects (mirror of runtime_objects)",
)

_PRIMITIVE_KINDS = {"primitive_cuboid", "primitive_sphere", "primitive_cylinder", "primitive_cone"}


class DirectOverrideError(RuntimeError):
    """Raised when a direct variant cannot be built deterministically."""


def _load_config() -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "tasks" not in payload:
        raise DirectOverrideError(f"Malformed direct ablation config: {CONFIG_PATH}")
    return payload


_CONFIG: dict[str, Any] | None = None


def config() -> dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config()
    return _CONFIG


def task_entry(task_id: str) -> dict[str, Any]:
    entry = config()["tasks"].get(str(task_id))
    if entry is None:
        raise DirectOverrideError(
            f"No direct-ablation entry for task_id={task_id!r}. "
            f"Add it to {CONFIG_PATH} before running the direct variant."
        )
    return dict(entry)


def is_direct(ablation_id: str | None) -> bool:
    return str(ablation_id or BASE_ABLATION_ID).strip().lower() == DIRECT_ABLATION_ID


def is_greenish(color: Any) -> bool:
    """Colour-selection rule, hard-coded so the choice is reproducible from the record."""
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return False
    try:
        red, green, blue = float(color[0]), float(color[1]), float(color[2])
    except (TypeError, ValueError):
        return False
    return green > 0.45 and green > red * 1.4 and green > blue * 1.4


def _pick_color(*, other_colors: list[Any]) -> tuple[list[float], str]:
    colors = config()["colors"]
    if any(is_greenish(color) for color in other_colors):
        return [float(v) for v in colors["magenta"]], "green_already_used_in_scene"
    return [float(v) for v in colors["green"]], "green_unused_in_scene"


def _runtime_objects(task_spec: dict[str, Any]) -> list[dict[str, Any]]:
    objects = task_spec.get("runtime_objects", [])
    if not isinstance(objects, list) or not objects:
        raise DirectOverrideError("task_spec has no runtime_objects; direct ablation cannot be applied.")
    return objects


def _protocol(task_spec: dict[str, Any]) -> dict[str, Any]:
    protocol = task_spec.get("protocol", {})
    return protocol if isinstance(protocol, dict) else {}


def resolve_direct_target(task_spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve the GT binding of a task without depending on ``execution_binding``.

    Returns ``{"target_object_id": str, "goal_xy": [x, y] | None, "goal_xy_source": str}``.
    Historical ``result.json`` files often carry an empty ``execution_binding.target_object_id``
    because only the protocol-oracle branch fills it in; the authoritative source is the layout's
    ``protocol.gt_spec`` / ``protocol.target_object`` / ``goal_sequence``.
    """
    protocol = _protocol(task_spec)
    gt_spec = protocol.get("gt_spec", {})
    gt_spec = gt_spec if isinstance(gt_spec, dict) else {}

    target_object_id = str(gt_spec.get("target_object_id", "") or "").strip()
    source = "gt_spec.target_object_id"
    if protocol.get("runtime_target_binding") == "protocol_gt":
        from geomind_lab.tasks.manager_based.manipulation.geomind.task_evolution import resolve_contract_target_id
        target_object_id = resolve_contract_target_id(task_spec)
        source = "task_contract.predicted_identity"
    if not target_object_id:
        target_object = protocol.get("target_object", {})
        if isinstance(target_object, dict):
            target_object_id = str(target_object.get("object_id", "") or "").strip()
            source = "protocol.target_object.object_id"

    goal_xy: list[float] | None = None
    goal_xy_source = ""
    goal_sequence = task_spec.get("goal_sequence", [])
    if isinstance(goal_sequence, list) and goal_sequence:
        goals = (goal_sequence[0] or {}).get("goals", []) if isinstance(goal_sequence[0], dict) else []
        if isinstance(goals, list) and goals and isinstance(goals[0], dict):
            xy = goals[0].get("xy", [])
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                goal_xy = [float(xy[0]), float(xy[1])]
                goal_xy_source = "goal_sequence[0].goals[0].xy"

    if goal_xy is None:
        zone = gt_spec.get("target_zone", {})
        zone = zone if isinstance(zone, dict) else {}
        offset = zone.get("center_offset_from_anchor", ())
        anchor_id = str(gt_spec.get("anchor_object_id", "") or "").strip()
        if not anchor_id:
            anchor_object = protocol.get("anchor_object", {})
            if isinstance(anchor_object, dict):
                anchor_id = str(anchor_object.get("object_id", "") or "").strip()
        if anchor_id and isinstance(offset, (list, tuple)) and len(offset) >= 2:
            anchor_xy = _object_xy(task_spec, anchor_id)
            if anchor_xy is not None:
                # Mirrors isaaclab_native_delegate._placement_goal_distance: the judged zone
                # centre is anchor world XY + center_offset_from_anchor.
                goal_xy = [anchor_xy[0] + float(offset[0]), anchor_xy[1] + float(offset[1])]
                goal_xy_source = f"gt_spec.target_zone.center_offset_from_anchor(anchor={anchor_id})"

    if not target_object_id:
        raise DirectOverrideError("Could not resolve a GT target object id from the task spec.")
    return {
        "target_object_id": target_object_id,
        "target_object_id_source": source,
        "goal_xy": goal_xy,
        "goal_xy_source": goal_xy_source,
    }


def _object_xy(task_spec: dict[str, Any], object_id: str) -> list[float] | None:
    for runtime_object in _runtime_objects(task_spec):
        if str(runtime_object.get("name", "")).strip() != str(object_id):
            continue
        pose = runtime_object.get("pose", {})
        xy = pose.get("xy", []) if isinstance(pose, dict) else []
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            return [float(xy[0]), float(xy[1])]
    positions = task_spec.get("object_positions", {})
    if isinstance(positions, dict):
        entry = positions.get(str(object_id), {})
        position = entry.get("position", []) if isinstance(entry, dict) else []
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            return [float(position[0]), float(position[1])]
    return None


def _canonical_layout(task_spec: dict[str, Any]) -> dict[str, Any]:
    """Everything that must be identical between base and direct."""
    objects = []
    for runtime_object in task_spec.get("runtime_objects", []) or []:
        if not isinstance(runtime_object, dict):
            continue
        if str(runtime_object.get("name", "")).strip() == MARKER_NAME:
            continue
        stripped = {
            key: value
            for key, value in runtime_object.items()
            if key not in {"visual_color", "visual_material", "material_override"}
        }
        objects.append(stripped)
    protocol = {key: value for key, value in _protocol(task_spec).items() if key != "query_text"}
    return {
        "runtime_objects": objects,
        "protocol": protocol,
        "goal_sequence": task_spec.get("goal_sequence", []),
    }


def layout_hash(task_spec: dict[str, Any]) -> str:
    payload = json.dumps(_canonical_layout(task_spec), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_list_fingerprint(task_ids: list[str]) -> str:
    payload = "\n".join(str(task_id) for task_id in task_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_only_allowed_mutations(before: dict[str, Any], after: dict[str, Any]) -> None:
    if layout_hash(before) != layout_hash(after):
        raise DirectOverrideError(
            "Direct override changed pairing-relevant layout state. Allowed mutations are: "
            + ", ".join(_ALLOWED_MUTATIONS)
        )
    before_names = [str(o.get("name", "")) for o in before.get("runtime_objects", []) if isinstance(o, dict)]
    after_names = [str(o.get("name", "")) for o in after.get("runtime_objects", []) if isinstance(o, dict)]
    added = [name for name in after_names if name not in before_names]
    removed = [name for name in before_names if name not in after_names]
    if removed or any(name != MARKER_NAME for name in added):
        raise DirectOverrideError(f"Direct override changed the object set: added={added} removed={removed}")


def _build_marker(*, xy: list[float], color: list[float]) -> dict[str, Any]:
    marker_cfg = config()["marker"]
    return {
        "name": str(marker_cfg.get("name", MARKER_NAME)),
        "kind": str(marker_cfg.get("kind", "primitive_cylinder")),
        "role": str(marker_cfg.get("role", "direct_marker")),
        "radius": float(marker_cfg.get("radius", 0.04)),
        "height": float(marker_cfg.get("height", 0.002)),
        "pose": {"xy": [float(xy[0]), float(xy[1])], "yaw": 0.0},
        "visual_color": [float(value) for value in color],
        # Visual only: no rigid body and no collider, so it can neither be grasped nor
        # perturb the physics of the placed object. Routes through
        # geomind_env_cfg._runtime_should_spawn_static_asset -> AssetBaseCfg.
        "physics": {
            "rigid_body": {"rigid_body_enabled": False},
            "collision": {"collision_enabled": False},
        },
    }


def apply_direct_override(
    *,
    task_spec: dict[str, Any],
    task_id: str,
    base_instruction: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(direct_task_spec, record)``.

    ``task_spec`` is not mutated. ``record`` is persisted into ``result.json`` under
    ``direct_ablation`` so that pairing can be audited after the fact.
    """
    entry = task_entry(task_id)
    mechanism = str(entry.get("mechanism", "none")).strip().lower()
    original_query_text = str(_protocol(task_spec).get("query_text", "") or "")
    original_instruction = str(base_instruction or task_spec.get("instruction", "") or original_query_text)
    base_hash = layout_hash(task_spec)

    record: dict[str, Any] = {
        "ablation_id": DIRECT_ABLATION_ID,
        "mechanism": mechanism,
        "gradient_tier": str(entry.get("gradient_tier", "")),
        "original_instruction": original_instruction,
        "original_query_text": original_query_text,
        "effective_instruction": original_instruction,
        "instruction_sources_patched": [],
        "recolored_object_id": "",
        "applied_color": None,
        "color_choice_reason": "",
        "cleared_visual_material": False,
        "marker": None,
        "layout_hash": base_hash,
        "config_path": str(CONFIG_PATH),
        "notes": str(entry.get("notes", "")),
    }

    if mechanism == "none":
        record["effective_instruction"] = original_instruction
        return deepcopy(task_spec), record

    instruction = str(entry.get("instruction", "") or "").strip()
    if not instruction:
        raise DirectOverrideError(f"Direct entry for {task_id!r} has mechanism={mechanism!r} but no instruction.")

    direct_spec = deepcopy(task_spec)
    binding = resolve_direct_target(direct_spec)
    objects = _runtime_objects(direct_spec)

    if mechanism == "recolor":
        expected = str(entry.get("expected_target_object_id", "") or "").strip()
        if expected and expected != binding["target_object_id"]:
            raise DirectOverrideError(
                f"{task_id}: resolved GT target {binding['target_object_id']!r} does not match the "
                f"configured expectation {expected!r}."
            )
        target = next((o for o in objects if str(o.get("name", "")).strip() == binding["target_object_id"]), None)
        if target is None:
            raise DirectOverrideError(f"{task_id}: GT target {binding['target_object_id']!r} is not a runtime object.")
        kind = str(target.get("kind", "")).strip()
        if kind not in _PRIMITIVE_KINDS:
            raise DirectOverrideError(
                f"{task_id}: GT target {binding['target_object_id']!r} is kind={kind!r}; only primitive shapes "
                "can be recoloured through visual_color. USD assets would need material_override."
            )
        others = [o.get("visual_color") for o in objects if str(o.get("name", "")).strip() != binding["target_object_id"]]
        material = target.get("visual_material")
        material = material if isinstance(material, dict) else {}
        is_mdl = str(material.get("type", "")).strip().lower() == "mdl"
        others_green = any(is_greenish(other) for other in others)
        if is_greenish(target.get("visual_color")) and not others_green and not is_mdl:
            # The target is already the unique green object. Touching it at all would make this
            # an appearance change; leaving it alone keeps the task strictly instruction-only.
            color = [float(value) for value in target.get("visual_color")]
            reason = "already_green_unique_no_change"
            cleared = False
        else:
            color, reason = _pick_color(other_colors=others)
            # visual_material with type=mdl short-circuits _runtime_visual_material and would hide
            # the new diffuse colour entirely, so it must go. Non-MDL materials are kept (they only
            # carry roughness/fallback) and their fallback colour is updated in place.
            cleared = False
            if is_mdl:
                target.pop("visual_material", None)
                cleared = True
            elif material:
                material["fallback_diffuse_color"] = list(color)
            target["visual_color"] = list(color)
        record.update(
            {
                "recolored_object_id": binding["target_object_id"],
                "applied_color": list(color),
                "color_choice_reason": reason,
                "cleared_visual_material": cleared,
            }
        )
    elif mechanism == "marker":
        goal_xy = binding.get("goal_xy")
        if goal_xy is None:
            raise DirectOverrideError(f"{task_id}: could not resolve a GT goal point for the marker.")
        expected_source = str(entry.get("marker_source", "") or "")
        if expected_source == "goal_sequence" and not str(binding["goal_xy_source"]).startswith("goal_sequence"):
            raise DirectOverrideError(f"{task_id}: expected marker_source=goal_sequence, got {binding['goal_xy_source']!r}")
        if expected_source == "target_zone" and not str(binding["goal_xy_source"]).startswith("gt_spec.target_zone"):
            raise DirectOverrideError(f"{task_id}: expected marker_source=target_zone, got {binding['goal_xy_source']!r}")
        color, reason = _pick_color(other_colors=[o.get("visual_color") for o in objects])
        marker = _build_marker(xy=goal_xy, color=color)
        objects.append(marker)
        positions = direct_spec.setdefault("object_positions", {})
        if isinstance(positions, dict):
            positions[MARKER_NAME] = {"position": [float(goal_xy[0]), float(goal_xy[1])]}
        record.update(
            {
                "applied_color": list(color),
                "color_choice_reason": reason,
                "marker": {
                    "name": MARKER_NAME,
                    "xy": [float(goal_xy[0]), float(goal_xy[1])],
                    "radius": float(marker["radius"]),
                    "height": float(marker["height"]),
                    "color": list(color),
                    "source": binding["goal_xy_source"],
                    "collision_enabled": False,
                    "rigid_body_enabled": False,
                },
            }
        )
    else:
        raise DirectOverrideError(f"{task_id}: unknown mechanism {mechanism!r}")

    expected_color_name = str(entry.get("expected_color", "") or "").strip()
    if expected_color_name:
        expected_color = [float(v) for v in config()["colors"][expected_color_name]]
        no_change = str(record["color_choice_reason"]).endswith("no_change")
        if no_change:
            if expected_color_name != "green" or not is_greenish(record["applied_color"]):
                raise DirectOverrideError(
                    f"{task_id}: target kept its original colour {record['applied_color']} but the config "
                    f"expects {expected_color_name}."
                )
        elif record["applied_color"] != expected_color:
            raise DirectOverrideError(
                f"{task_id}: colour selection produced {record['applied_color']} but the config expects "
                f"{expected_color_name}={expected_color}. Update the config or the scene."
            )

    # Both instruction sources: TaskSpec.instruction is patched in the delegate, protocol.query_text here.
    direct_spec["instruction"] = instruction
    protocol = direct_spec.get("protocol")
    if isinstance(protocol, dict):
        protocol["query_text"] = instruction
        record["instruction_sources_patched"].append("protocol.query_text")
    record["instruction_sources_patched"].append("task_spec.instruction")

    scene_adaptation = direct_spec.get("scene_adaptation")
    if isinstance(scene_adaptation, dict):
        scene_adaptation["runtime_objects"] = objects

    _assert_only_allowed_mutations(task_spec, direct_spec)
    record["effective_instruction"] = instruction
    record["layout_hash"] = layout_hash(direct_spec)
    if record["layout_hash"] != base_hash:
        raise DirectOverrideError(f"{task_id}: layout hash changed ({base_hash} -> {record['layout_hash']}).")
    return direct_spec, record


def base_record(*, task_spec: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Record written for base runs so base/direct result files line up field for field."""
    return {
        "ablation_id": BASE_ABLATION_ID,
        "mechanism": "none",
        "gradient_tier": "base",
        "original_instruction": str(instruction),
        "original_query_text": str(_protocol(task_spec).get("query_text", "") or ""),
        "effective_instruction": str(instruction),
        "instruction_sources_patched": [],
        "recolored_object_id": "",
        "applied_color": None,
        "color_choice_reason": "",
        "cleared_visual_material": False,
        "marker": None,
        "layout_hash": layout_hash(task_spec),
        "config_path": str(CONFIG_PATH),
        "notes": "",
    }
