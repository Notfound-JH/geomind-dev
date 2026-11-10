from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geomind_lab.mimic.state_reindexing_benchmark_utils import (
    HISTORICAL_IDENTITY_TRACKING_FAMILY,
    STATE_REINDEXING_FAMILY,
)
from geomind_lab.mimic.utils.config_semantics_adapter import (
    EXPLICIT_RULE_EXTRAPOLATION_FAMILY,
    IMPLICIT_TREND_FITTING_FAMILY,
    augment_record_with_semantics,
    canonicalize_task_id,
    normalize_annotation_entry,
)
from geomind_lab.mimic.utils.family_runner_registry import (
    DISPATCH_STATUS_DECLARED_NOT_RUNNABLE,
    DISPATCH_STATUS_DECLARED_RUNNER_UNIMPLEMENTED,
    DISPATCH_STATUS_RUNNABLE,
    DISPATCH_STATUS_UNKNOWN_RUNNER,
    GEOMETRIC_STRUCTURE_RUNNER_FAMILY,
    RUNNER_DISPATCH_MODE_PERMISSIVE,
    RUNNER_DISPATCH_MODE_STRICT,
    STATE_REFRESH_RUNNER_FAMILY,
    UNKNOWN_RUNNER_FAMILY,
    registered_runner_families,
    resolve_runner_family,
    resolve_runner_family_for_entry,
    runner_family_config,
    runner_family_is_implemented,
    split_agent_names_by_runner as split_agent_names_by_runner_family,
)
from geomind_lab.mimic.utils.task_metadata_registry import (
    build_task_metadata,
    resolve_task_id as resolve_benchmark_task_id,
    supported_task_versions,
)
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import load_annotated_tasks


STATE_REFRESH_RUNNER_KEY = STATE_REFRESH_RUNNER_FAMILY
GEOMETRIC_STRUCTURE_RUNNER_KEY = GEOMETRIC_STRUCTURE_RUNNER_FAMILY
UNKNOWN_RUNNER_KEY = UNKNOWN_RUNNER_FAMILY

STATE_REFRESH_TASK_FAMILIES = frozenset(
    {
        STATE_REINDEXING_FAMILY,
        HISTORICAL_IDENTITY_TRACKING_FAMILY,
    }
)
GEOMETRIC_STRUCTURE_TASK_FAMILIES = frozenset(
    {
        EXPLICIT_RULE_EXTRAPOLATION_FAMILY,
        IMPLICIT_TREND_FITTING_FAMILY,
        "shape_completion",
    }
)

STATE_REFRESH_AGENT_NAMES = frozenset(runner_family_config(STATE_REFRESH_RUNNER_KEY).get("agent_names", ()))
GEOMETRIC_STRUCTURE_AGENT_NAMES = frozenset(runner_family_config(GEOMETRIC_STRUCTURE_RUNNER_KEY).get("agent_names", ()))
ANNOTATION_BACKED_MODULES = frozenset(
    {
        "post_action_relocation",
        "geometric_inference",
        # Transitional compatibility step: anchor_reference instances are annotation-backed for preview,
        # but benchmark dispatch should still be gated by task metadata and runner implementation.
        "anchor_reference",
        "constraint_selection",
        "spatial_placement",
    }
)


def load_benchmark_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def runner_key_for_family(family: str) -> str:
    return resolve_runner_family(family=family)


def split_agent_names_by_runner(agent_names: list[str]) -> dict[str, list[str]]:
    return split_agent_names_by_runner_family(agent_names)


def _serialize_semantics_field(value: Any) -> str | None:
    if value in (None, "", [], {}, ()):
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _registered_grouped_lists() -> dict[str, list[str]]:
    grouped = {runner_family: [] for runner_family in registered_runner_families()}
    grouped[UNKNOWN_RUNNER_KEY] = []
    return grouped


def build_task_catalog_entries(annotations_path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = load_annotated_tasks(annotations_path)
    entries: list[dict[str, Any]] = []
    for layout_key, layout in payload.items():
        if not isinstance(layout, dict):
            continue
        tasks = layout.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        for task_index, task_payload in enumerate(tasks):
            if not isinstance(task_payload, dict):
                continue
            annotation = task_payload.get("annotation", {})
            if not isinstance(annotation, dict):
                continue
            protocol = annotation.get("protocol", {})
            if not isinstance(protocol, dict):
                continue
            module_name = str(protocol.get("module_name", "")).strip()
            if module_name not in ANNOTATION_BACKED_MODULES:
                continue

            semantic_view = normalize_annotation_entry(
                layout_key=layout_key,
                task_index=task_index,
                layout=layout,
                task_payload=task_payload,
            )
            raw_task_name = str(annotation.get("task_id") or f"{layout_key}__task_{task_index}")
            instance_semantics = semantic_view.get("instance_semantics", {})
            instance_id = (
                str(instance_semantics.get("instance_id", "")).strip()
                if isinstance(instance_semantics, dict)
                else ""
            ) or f"{layout_key}:{task_index}"
            entry = augment_record_with_semantics(
                {
                    "task_name": raw_task_name,
                    "task_id": raw_task_name,
                    "family": semantic_view.get("family"),
                    "variant": semantic_view.get("variant"),
                    "task_type": semantic_view.get("task_type"),
                    "task_version": semantic_view.get("task_version"),
                    "scene_id": int(layout.get("scene_id", 1)),
                    "layout_key": str(layout_key),
                    "task_index": int(task_index),
                    "interface": "Vla" if str(layout_key).lower().endswith("_vla") else "Standard",
                    "diagnostic_tags": semantic_view.get("diagnostic_tags", []),
                    "difficulty_factors": semantic_view.get("difficulty_factors", {}),
                }
            )
            preview_runnable = bool(module_name in {"anchor_reference", "constraint_selection"})
            if preview_runnable:
                entry["task_name"] = instance_id
                entry["instance_id"] = instance_id
                entry["annotation_task_id"] = raw_task_name
            runner_family = resolve_runner_family_for_entry(entry)
            entry.update(
                {
                    "runner_family": runner_family,
                    "runner_key": runner_family,
                    "runner_implemented": runner_family_is_implemented(runner_family),
                    "raw_task_name": raw_task_name,
                    "preview_runnable": preview_runnable,
                    "is_runnable": False if preview_runnable else bool(entry.get("is_runnable", False)),
                    "task_semantics": semantic_view.get("task_semantics", {}),
                    "scene_semantics": semantic_view.get("scene_semantics", {}),
                    "instance_semantics": instance_semantics if isinstance(instance_semantics, dict) else {},
                }
            )
            for key in ("task_semantics", "scene_semantics", "instance_semantics"):
                serialized = _serialize_semantics_field(entry.get(key))
                if serialized is not None:
                    entry[f"{key}_json"] = serialized
            entries.append(entry)
    return entries


def build_task_catalog(annotations_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for entry in build_task_catalog_entries(annotations_path):
        task_name = str(entry.get("task_name", ""))
        layout_key = str(entry.get("layout_key", ""))
        task_index = int(entry.get("task_index", 0))
        task_id = str(entry.get("task_id", ""))
        catalog[task_name] = entry
        catalog[f"{layout_key}:{task_index}"] = entry
        if layout_key and layout_key not in catalog:
            catalog[layout_key] = entry
        canonical_task_id = str(entry.get("canonical_task_id", "")).strip()
        if canonical_task_id and canonical_task_id not in catalog:
            catalog[canonical_task_id] = entry
        if task_id and task_id not in catalog:
            catalog[task_id] = entry
    return catalog


def _matches_metadata_filters(
    entry: dict[str, Any],
    *,
    task_versions: list[str] | None = None,
    phases: list[str] | None = None,
) -> bool:
    # Generic benchmark queries select the standard interface. VLA is an
    # explicit interface request, not another default condition variant.
    is_vla = entry.get("interface") == "Vla" or entry.get("task_version") == "vla"
    allowed_versions = {str(value).strip() for value in (task_versions or []) if str(value).strip()}
    if is_vla:
        if "vla" not in allowed_versions:
            return False
    elif allowed_versions and str(entry.get("task_version", "")).strip() not in allowed_versions:
        return False
    if phases:
        allowed_phases = {str(value).strip() for value in phases if str(value).strip()}
        if allowed_phases and str(entry.get("phase", "")).strip() not in allowed_phases:
            return False
    return True


def _build_registry_only_entries(
    requested_name: str,
    *,
    task_versions: list[str] | None = None,
    phases: list[str] | None = None,
) -> list[dict[str, Any]]:
    resolved_task_id = resolve_benchmark_task_id(requested_name)
    if not resolved_task_id or tuple(supported_task_versions(resolved_task_id)) == ():
        return []

    requested_versions = [str(value).strip() for value in (task_versions or []) if str(value).strip()]
    versions = requested_versions or list(supported_task_versions(resolved_task_id))
    entries: list[dict[str, Any]] = []
    for task_version in versions:
        metadata = build_task_metadata(resolved_task_id, task_version=task_version)
        runner_family = resolve_runner_family(
            task_id=metadata.get("task_id"),
            family=metadata.get("family"),
            semantic_module=metadata.get("semantic_module"),
            skill_type=metadata.get("skill_type"),
        )
        entry = {
            "task_name": "",
            "task_id": str(metadata.get("task_id", "")),
            "canonical_task_id": "",
            "family": str(metadata.get("family", "")),
            "variant": str(metadata.get("task_id", "")),
            "semantic_module": str(metadata.get("semantic_module", "")),
            "task_version": str(metadata.get("task_version", "")),
            "skill_type": str(metadata.get("skill_type", "")),
            "phase": str(metadata.get("phase", "")),
            "parent_task_id": str(metadata.get("parent_task_id", "")),
            "paired_control_id": str(metadata.get("paired_control_id", "")),
            "enabled_probes": metadata.get("enabled_probes", []),
            "evaluator_key": str(metadata.get("evaluator_key", "")),
            "stress_factors": metadata.get("stress_factors", []),
            "notes": str(metadata.get("notes", "")),
            "runner_family": runner_family,
            "runner_key": runner_family,
            "runner_implemented": runner_family_is_implemented(runner_family),
            "is_runnable": bool(metadata.get("is_runnable", False)),
            "raw_task_name": requested_name,
        }
        if _matches_metadata_filters(entry, task_versions=task_versions, phases=phases):
            entries.append(entry)
    return entries


def resolve_task_entries(
    requested_name: str,
    *,
    annotations_path: str | Path | None = None,
    task_versions: list[str] | None = None,
    phases: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested = str(requested_name).strip()
    if not requested:
        return []
    if not task_versions and (requested.split(":", 1)[0].lower().endswith("_vla")
                              or "vla_v" in requested.lower()):
        task_versions = ["vla"]

    entries = build_task_catalog_entries(annotations_path)
    matches: list[dict[str, Any]] = []
    if ":" in requested:
        maybe_layout_key, maybe_task_index = requested.rsplit(":", 1)
        if maybe_task_index.isdigit():
            for entry in entries:
                if (
                    str(entry.get("layout_key", "")) == maybe_layout_key
                    and int(entry.get("task_index", -1)) == int(maybe_task_index)
                ):
                    if _matches_metadata_filters(entry, task_versions=task_versions, phases=phases):
                        matches.append(entry)
            return matches or _build_registry_only_entries(
                requested,
                task_versions=task_versions,
                phases=phases,
            )

    for entry in entries:
        if requested in {
            str(entry.get("task_name", "")),
            str(entry.get("layout_key", "")),
            str(entry.get("task_id", "")),
        }:
            if not _matches_metadata_filters(entry, task_versions=task_versions, phases=phases):
                continue
            matches.append(entry)
    if matches:
        return matches

    requested_canonical = canonicalize_task_id(requested)
    if requested_canonical:
        seen: set[tuple[str, int]] = set()
        for entry in entries:
            key = (str(entry.get("layout_key", "")), int(entry.get("task_index", -1)))
            if key in seen:
                continue
            if str(entry.get("canonical_task_id", "")) == requested_canonical:
                if not _matches_metadata_filters(entry, task_versions=task_versions, phases=phases):
                    continue
                matches.append(entry)
                seen.add(key)
    if matches:
        return matches

    return _build_registry_only_entries(
        requested,
        task_versions=task_versions,
        phases=phases,
    )


def _dispatch_reason_for_entry(entry: dict[str, Any]) -> tuple[str, str]:
    runner_family = str(entry.get("runner_family", "")).strip() or UNKNOWN_RUNNER_KEY
    task_id = str(entry.get("task_id", "")).strip() or str(entry.get("task_name", "")).strip() or "<unknown>"
    task_version = str(entry.get("task_version", "")).strip() or "<unknown>"
    if runner_family == UNKNOWN_RUNNER_KEY:
        return (
            DISPATCH_STATUS_UNKNOWN_RUNNER,
            f"No child runner is registered for task_id={task_id} task_version={task_version}.",
        )
    if not bool(entry.get("runner_implemented", False)):
        return (
            DISPATCH_STATUS_DECLARED_RUNNER_UNIMPLEMENTED,
            f"Runner family '{runner_family}' is declared but not implemented for task_id={task_id} task_version={task_version}.",
        )
    if not bool(entry.get("is_runnable", False)):
        return (
            DISPATCH_STATUS_DECLARED_NOT_RUNNABLE,
            f"Task metadata is declared but no runnable annotation-backed instance is available for task_id={task_id} task_version={task_version}.",
        )
    return (DISPATCH_STATUS_RUNNABLE, "")


def _build_dispatch_descriptor(requested_name: str, entry: dict[str, Any]) -> dict[str, Any]:
    payload = dict(entry)
    runner_family = str(payload.get("runner_family", "")).strip() or resolve_runner_family_for_entry(payload)
    payload["runner_family"] = runner_family
    payload["runner_key"] = runner_family
    payload["runner_implemented"] = runner_family_is_implemented(runner_family)
    dispatch_status, dispatch_reason = _dispatch_reason_for_entry(payload)
    payload["requested_name"] = str(requested_name).strip()
    payload["dispatch_status"] = dispatch_status
    payload["dispatch_reason"] = dispatch_reason
    return payload


def _descriptor_dedupe_key(entry: dict[str, Any]) -> tuple[str, int, str, str]:
    layout_key = str(entry.get("layout_key", "")).strip()
    task_index = int(entry.get("task_index", -1))
    task_id = str(entry.get("task_id", "")).strip()
    task_version = str(entry.get("task_version", "")).strip()
    if layout_key:
        return (layout_key, task_index, "", "")
    return ("", -1, task_id, task_version)


def plan_runner_dispatch(
    task_names: list[str],
    agent_names: list[str],
    *,
    annotations_path: str | Path | None = None,
    task_versions: list[str] | None = None,
    phases: list[str] | None = None,
    dispatch_mode: str = RUNNER_DISPATCH_MODE_PERMISSIVE,
) -> dict[str, Any]:
    mode = str(dispatch_mode).strip() or RUNNER_DISPATCH_MODE_PERMISSIVE
    if mode not in {RUNNER_DISPATCH_MODE_PERMISSIVE, RUNNER_DISPATCH_MODE_STRICT}:
        raise ValueError(f"Unsupported runner dispatch mode: {dispatch_mode!r}")

    task_groups = _registered_grouped_lists()
    task_seen: dict[str, set[str]] = {runner_family: set() for runner_family in task_groups}
    agent_groups = split_agent_names_by_runner(agent_names)
    unknown_tasks: list[str] = []
    selected_task_descriptors: list[dict[str, Any]] = []
    runnable_task_descriptors: list[dict[str, Any]] = []
    skipped_task_descriptors: list[dict[str, Any]] = []
    seen_descriptor_keys: set[tuple[str, int, str, str]] = set()

    for raw_name in task_names:
        requested = str(raw_name).strip()
        if not requested:
            continue
        entries = resolve_task_entries(
            requested,
            annotations_path=annotations_path,
            task_versions=task_versions,
            phases=phases,
        )
        if not entries:
            unknown_tasks.append(requested)
            continue
        for entry in entries:
            descriptor = _build_dispatch_descriptor(requested, entry)
            dedupe_key = _descriptor_dedupe_key(descriptor)
            if dedupe_key in seen_descriptor_keys:
                continue
            seen_descriptor_keys.add(dedupe_key)
            selected_task_descriptors.append(descriptor)
            runner_family = str(descriptor.get("runner_family", UNKNOWN_RUNNER_KEY))
            if str(descriptor.get("dispatch_status", "")) == DISPATCH_STATUS_RUNNABLE:
                runnable_task_descriptors.append(descriptor)
                task_name = str(descriptor.get("task_name", "")).strip()
                if task_name and task_name not in task_seen[runner_family]:
                    task_groups[runner_family].append(task_name)
                    task_seen[runner_family].add(task_name)
            else:
                skipped_task_descriptors.append(descriptor)

    return {
        "dispatch_mode": mode,
        "task_groups": task_groups,
        "agent_groups": agent_groups,
        "unknown_tasks": unknown_tasks,
        "unknown_agents": list(agent_groups.get(UNKNOWN_RUNNER_KEY, [])),
        "selected_task_descriptors": selected_task_descriptors,
        "runnable_task_descriptors": runnable_task_descriptors,
        "skipped_task_descriptors": skipped_task_descriptors,
    }


def resolve_task_names(
    task_names: list[str],
    *,
    annotations_path: str | Path | None = None,
    task_versions: list[str] | None = None,
    phases: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for raw_name in task_names:
        matches = resolve_task_entries(
            str(raw_name),
            annotations_path=annotations_path,
            task_versions=task_versions,
            phases=phases,
        )
        if not matches:
            unknown.append(str(raw_name).strip())
            continue
        for entry in matches:
            task_name = str(entry.get("task_name", "")).strip()
            if not task_name or task_name in seen:
                continue
            resolved.append(task_name)
            seen.add(task_name)
    return resolved, unknown


def split_task_names_by_runner(
    task_names: list[str],
    *,
    annotations_path: str | Path | None = None,
    task_versions: list[str] | None = None,
    phases: list[str] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    dispatch = plan_runner_dispatch(
        task_names,
        agent_names=[],
        annotations_path=annotations_path,
        task_versions=task_versions,
        phases=phases,
        dispatch_mode=RUNNER_DISPATCH_MODE_PERMISSIVE,
    )
    return dispatch["task_groups"], dispatch["unknown_tasks"]


__all__ = [
    "DISPATCH_STATUS_DECLARED_NOT_RUNNABLE",
    "DISPATCH_STATUS_DECLARED_RUNNER_UNIMPLEMENTED",
    "DISPATCH_STATUS_RUNNABLE",
    "DISPATCH_STATUS_UNKNOWN_RUNNER",
    "EXPLICIT_RULE_EXTRAPOLATION_FAMILY",
    "HISTORICAL_IDENTITY_TRACKING_FAMILY",
    "IMPLICIT_TREND_FITTING_FAMILY",
    "RUNNER_DISPATCH_MODE_PERMISSIVE",
    "RUNNER_DISPATCH_MODE_STRICT",
    "GEOMETRIC_STRUCTURE_AGENT_NAMES",
    "GEOMETRIC_STRUCTURE_RUNNER_KEY",
    "GEOMETRIC_STRUCTURE_TASK_FAMILIES",
    "STATE_REFRESH_AGENT_NAMES",
    "STATE_REFRESH_RUNNER_KEY",
    "STATE_REFRESH_TASK_FAMILIES",
    "STATE_REINDEXING_FAMILY",
    "UNKNOWN_RUNNER_KEY",
    "build_task_catalog",
    "build_task_catalog_entries",
    "load_benchmark_manifest",
    "plan_runner_dispatch",
    "registered_runner_families",
    "resolve_task_entries",
    "resolve_task_names",
    "runner_family_config",
    "runner_key_for_family",
    "split_agent_names_by_runner",
    "split_task_names_by_runner",
]
