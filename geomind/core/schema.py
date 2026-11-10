"""Task definitions with independent identity fields and runtime bindings."""
from __future__ import annotations

from copy import deepcopy
import re
import hashlib
import json
from typing import Any

from .ids import make_task_id
from .taxonomy import TASK_TO_FAMILY


SCHEMA_VERSION = 2
_CONDITIONS = {"base": "Base", "sem": "Semantic", "semantic": "Semantic",
               "visual": "Visual", "execution": "Execution", "exec": "Execution"}
_FAMILIES = {
    "anchor_reference": "RelationalGrounding",
    "spatial_placement": "RelationalGrounding",
    "constraint_selection": "ConstraintComposition",
    "state_reindexing": "SceneUpdating",
    "historical_identity_tracking": "IdentityTracking",
    "explicit_rule_extrapolation": "GeometricInference",
    "implicit_trend_fitting": "GeometricInference",
    "shape_completion": "GeometricInference",
}
_LAYOUTS = {
    "grasp_anchor_direction": "Cross", "grasp_anchor_distance": "Scatter",
    "select_exclusion_unique": "Line", "select_visual_spatial": "Cross",
    "isometric_extension": "Line", "perfect_midpoint": "Line",
    "point_symmetry": "Line", "square_completion": "Square",
    "trend_continuation": "Line", "grid_rowcol_grounding": "Grid",
    "ordinal_insert_static": "Line", "reindex_extreme": "Line",
    "reindex_insert": "Line", "reindex_neighbor": "Line", "reindex_subset": "Line",
    "place_directional": "Pair", "place_inside": "Containment", "place_on_top": "Stack",
    "track_single": "Line", "track_multi": "Line", "track_clutter": "Clutter",
}


def normalize_condition(value: str) -> str:
    try:
        return _CONDITIONS[str(value).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported task condition: {value!r}") from exc


def make_registration_id(task_id: str, *, condition: str, version: int,
                         interface: str = "Standard", task_variant: str = "",
                         parameters: dict[str, Any] | None = None) -> str:
    """Keep independent variants/parameters from overwriting one runtime key."""
    stem = task_id + task_variant
    if parameters:
        fingerprint = hashlib.sha256(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8].upper()
        stem += "P" + fingerprint
    key = make_task_id(task_name=stem, variant=normalize_condition(condition), version=version)
    if interface == "Vla": key = re.sub(r"_v(\d+)$", r"Vla_v\1", key)
    return key


def upgrade_task_config(
    payload: dict[str, Any], *, family: str | None = None,
    condition: str | None = None, layout: str | None = None,
) -> dict[str, Any]:
    """Convert a legacy execution payload without changing its runtime bindings."""
    if "runtime" in payload:
        result = deepcopy(payload)
        runtime = result["runtime"]
        if runtime.get("legacy_task_id") == runtime.get("canonical_task_id"):
            runtime.pop("legacy_task_id", None)
        if family is not None: result["family"] = family
        if condition is not None: result["condition"] = normalize_condition(condition)
        if layout is not None: result["layout"] = layout
        return result
    runtime = deepcopy(payload)
    if runtime.get("legacy_task_id") == runtime.get("canonical_task_id"):
        runtime.pop("legacy_task_id", None)
    registration_id = str(runtime.pop("task_id"))
    task_name = str(runtime.pop("task_name"))
    protocol_family = str(runtime.pop("family"))
    version = runtime.pop("version", 0)
    aliases = list(runtime.pop("aliases", []))
    public_variant = str(runtime.pop("public_variant", ""))
    parameters = deepcopy(runtime.pop("parameters", {}))
    runtime.pop("task_contract", None)
    runtime["registration_id"] = registration_id
    runtime["protocol_family"] = protocol_family
    runtime["interface"] = "Vla" if str(runtime.get("task_version", "")).lower() == "vla" else "Standard"
    if condition is None:
        # VLA task_version describes the interface; condition is a separate axis.
        if re.search(r"(?:SemVla|Sem)_v\d+$", registration_id): condition = "Semantic"
        elif re.search(r"(?:VisualVla|Visual)_v\d+$", registration_id): condition = "Visual"
        elif str(runtime.get("variant", "")).lower() in _CONDITIONS: condition = runtime["variant"]
        elif "AugSem_" in registration_id or "BaseSem_" in registration_id: condition = "Semantic"
        elif "AugVisual_" in registration_id or "BaseVisual_" in registration_id: condition = "Visual"
        else: condition = "Base"
    result = {
        "task_id": task_name,
        "family": family or TASK_TO_FAMILY.get(task_name, _FAMILIES.get(protocol_family, protocol_family)),
        "condition": normalize_condition(condition),
        "layout": layout or _LAYOUTS.get(str(runtime.get("canonical_task_id", "")), "Custom"),
        "version": version,
        "schema_version": SCHEMA_VERSION,
        "runtime": runtime,
        "aliases": list(dict.fromkeys([registration_id, *aliases])),
    }
    if public_variant: result["variant"] = public_variant
    if parameters: result["parameters"] = parameters
    return result


def normalize_task_config(definition: dict[str, Any], *, config_root=None) -> dict[str, Any]:
    """Adapt the public schema to the established runtime payload."""
    if definition.get("schema_version") == 3:
        from .episode_compat import normalize_legacy_episode
        return normalize_legacy_episode(definition, config_root)
    if "runtime" not in definition:
        if definition.get("schema_version") == SCHEMA_VERSION or (
            {"condition", "layout"} <= definition.keys() and "public_task_id" not in definition
        ):
            raise ValueError("Missing task definition runtime bindings.")
        return deepcopy(definition)
    for key in ("task_id", "family", "condition", "layout", "version"):
        if key not in definition:
            raise ValueError(f"Missing task definition field: {key}")
    task_id = str(definition["task_id"])
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", task_id):
        raise ValueError("task_id must contain only the task name, without prefixes or _vN suffixes.")
    expected_family = TASK_TO_FAMILY.get(task_id)
    if expected_family is not None and definition["family"] != expected_family:
        raise ValueError(f"{task_id} belongs to {expected_family}, got {definition['family']!r}.")
    version = definition["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("version must be a nonnegative integer.")
    condition = normalize_condition(definition["condition"])
    public_variant = str(definition.get("variant", ""))
    parameters = definition.get("parameters", {})
    if public_variant and not re.fullmatch(r"[A-Z][A-Za-z0-9]*", public_variant):
        raise ValueError("variant must be a CamelCase task variant.")
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object.")
    if not str(definition["family"]).strip() or not str(definition["layout"]).strip():
        raise ValueError("family and layout must be nonempty.")
    payload = deepcopy(definition["runtime"])
    if not isinstance(payload, dict):
        raise ValueError("runtime must be an object.")
    # Keep the legacy execution payload compatible without duplicating the ID
    # in public task definitions. Explicit old values still reach validation.
    if payload.get("canonical_task_id"):
        payload.setdefault("legacy_task_id", payload["canonical_task_id"])
    if not payload.get("protocol_family"):
        raise ValueError("runtime.protocol_family is required for oracle dispatch.")
    interface = str(payload.get("interface", "Standard"))
    if interface not in {"Standard", "Vla"}:
        raise ValueError("runtime.interface must be Standard or Vla.")
    variant = str(payload.get("task_version") or ("vla" if interface == "Vla" else condition.lower()))
    registration_id = make_registration_id(task_id, condition=condition, version=version,
        interface=interface, task_variant=public_variant, parameters=parameters)
    registration_id = str(payload.get("registration_id") or registration_id)
    registration_version = re.search(r"_v(\d+)$", registration_id)
    if registration_version is None:
        raise ValueError("runtime.registration_id must end with _vN.")
    # The independent version field is authoritative, even for legacy keys.
    registration_id = re.sub(r"_v\d+$", f"_v{version}", registration_id)
    if (variant == "vla") != (interface == "Vla"):
        raise ValueError("runtime.interface conflicts with runtime.task_version.")
    payload.update({
        "task_id": registration_id,
        "registration_id": registration_id,
        "task_name": task_id,
        "family": payload["protocol_family"],
        "version": version,
        "variant": str(payload.get("variant") or variant),
        "task_version": variant,
        "split": str(payload.get("split") or "base"),
        "aliases": list(definition.get("aliases", [])),
        "public_task_id": task_id,
        "task_family": definition["family"],
        "condition": condition,
        "layout": definition["layout"],
        "public_variant": public_variant,
        "parameters": deepcopy(parameters),
        "task_contract": {
            "task_id": task_id,
            "canonical_task_id": payload.get("canonical_task_id", ""),
            "variant": public_variant,
            "parameters": deepcopy(parameters),
            "interface": interface,
        },
    })
    return payload
