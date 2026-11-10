"""Migrate active v2 registrations to v3 episodes; retain isolated runtime adapters."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from geomind.core.taxonomy import TASK_TO_FAMILY
from geomind.episodes.resolver import load_json


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def migrate(root: Path, *, replace_records: bool = False) -> dict:
    from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import (
        load_annotated_tasks, resolve_task_request,
    )
    layouts = load_annotated_tasks()
    for task, family in TASK_TO_FAMILY.items():
        write(root / "task_definitions" / f"{task}.json", {
            "task_id": task, "domain": "SpatialMemory" if family in {"SceneUpdating", "IdentityTracking"}
            else "SpatialReasoning", "family": family,
        })
    changed, checked = [], []
    for folder in ("tasks", "vla_tasks"):
        for path in sorted((root / folder).glob("*.json")):
            definition = load_json(path)
            if "task_id" not in definition:
                continue
            if definition.get("schema_version") == 3:
                from geomind.core.episode_compat import normalize_legacy_episode
                normalize_legacy_episode(definition, root)
                if "release_version" not in definition:
                    definition["release_version"] = "1.0"
                    write(path, definition)
                checked.append(path.name)
                continue
            runtime = definition["runtime"]
            identity = runtime["registration_id"]
            archive = root / "legacy" / "registrations" / f"{identity}.json"
            if archive.exists() and load_json(archive) != definition and not replace_records:
                raise ValueError(f"Refusing to replace a different compatibility record: {archive}")
            write(archive, definition)
            placeholder = runtime.get("implementation_status") == "placeholder"
            episode = {
                "schema_version": 3, "release_version": "1.0", "episode_id": identity, "task_id": definition["task_id"],
                "execution": "Unavailable" if placeholder else "LegacyIsaacLab",
                "condition": definition["condition"], "interface": runtime["interface"],
                "version": definition["version"], "parameters": deepcopy(definition.get("parameters", {})),
                "aliases": definition.get("aliases", []), "seed": 0,
            }
            if "variant" in definition:
                episode["variant"] = definition["variant"]
            if not placeholder:
                key = runtime.get("annotation_layout_key", "")
                index = runtime.get("annotation_task_index", 0)
                if not key:
                    request = resolve_task_request(task_id=runtime["canonical_task_id"],
                        variant=runtime["task_version"], scene_id=runtime["scene_id"])
                    key, index = request.layout_key, request.task_index
                    runtime["annotation_task_index"] = index
                    # The archived adapter may still choose its original inferred layout;
                    # the public episode now has an explicit layout reference.
                layout = layouts.get(key)
                if not isinstance(layout, dict):
                    raise ValueError(f"Cannot resolve initialization layout {key!r} for {identity}")
                tasks = layout.get("tasks", [])
                annotation = tasks[index].get("annotation", {}) if index < len(tasks) else {}
                source_adaptation = annotation.get("scene_adaptation", {})
                adaptation = {k: deepcopy(v) for k, v in source_adaptation.items()
                              if k in {"runtime_objects", "background_overrides", "object_mode"}}
                # Roles belong to the old protocol; initialization stores only physical objects.
                for obj in adaptation.get("runtime_objects", []):
                    obj.pop("role", None)
                new_layout = {
                    "layout_id": key, "format": "LegacyInitialization", "category": definition["layout"],
                    "source_layout": key, "scene_structure": deepcopy(layout.get("scene_structure", {})),
                    "initialization": adaptation,
                }
                destination = root / "layouts" / f"{key}.json"
                if destination.exists():
                    previous = load_json(destination)
                    if previous["initialization"] != new_layout["initialization"]:
                        raise ValueError(f"Layout has task-dependent initialization: {key}")
                write(destination, new_layout)
                episode.update(scene_id=f"scene_{runtime['scene_id']}", layout_id=key)
            write(path, episode)
            changed.append(identity)
    return {"migrated": changed, "already_v3": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    args = parser.parse_args()
    print(json.dumps(migrate(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
