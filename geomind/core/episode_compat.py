"""Explicit bindings from compact v3 registrations to their actual runners.

Historical execution records live under configs/legacy/registrations. They are
not the public task definition and are never interpreted by the native GT API.
Native assembled registrations have separate runtime records and never pretend
to be old manager-based IsaacLab environments.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from geomind.episodes.resolver import ContractError, _check, _fields, _id, _integer, load_json


def config_root() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def legacy_definition(episode: dict, root: Path | None = None) -> dict:
    root = Path(root) if root is not None else config_root()
    common = {"schema_version", "episode_id", "task_id", "execution", "condition", "interface",
              "version", "parameters", "aliases", "seed"}
    _fields(episode, common, {"scene_id", "layout_id", "variant", "release_version"})
    _check(type(episode["schema_version"]) is int and episode["schema_version"] == 3, "Expected schema_version=3")
    for key in ("episode_id", "task_id"):
        _id(episode[key])
    _check(isinstance(episode["execution"], str) and episode["execution"] in {"LegacyIsaacLab", "AssembledNative", "Unavailable"},
           "Unsupported compatibility execution")
    _check(episode.get("release_version", "1.0") == "1.0", "Legacy registrations belong to release 1.0")
    _check(_integer(episode["version"]), "Invalid episode version")
    _check(_integer(episode["seed"]), "Invalid episode seed")
    _check(isinstance(episode["parameters"], dict), "parameters must be an object")
    _check(isinstance(episode["aliases"], list) and all(isinstance(a, str) and a for a in episode["aliases"]),
           "aliases must be nonempty strings")
    _check(isinstance(episode["interface"], str) and episode["interface"] in {"Standard", "Vla"}, "Invalid interface")
    _check(isinstance(episode["condition"], str) and episode["condition"] in {"Base", "Semantic", "Visual", "Execution"},
           "Invalid condition")
    task = load_json(root / "task_definitions" / f"{episode['task_id']}.json")
    _check(task["task_id"] == episode["task_id"], "Task reference mismatch")
    records = 'native' if episode['execution']=='AssembledNative' else 'legacy'
    record = load_json(root / records / "registrations" / f"{episode['episode_id']}.json")
    _check(record["task_id"] == episode["task_id"], "Legacy task binding mismatch")
    _check(episode["parameters"].keys() == record.get("parameters", {}).keys(),
           "Legacy execution accepts only its existing parameter fields; use native episodes for new task parameters")
    result = deepcopy(record)
    result.update(task_id=episode["task_id"], family=task["family"], condition=episode["condition"],
                  version=episode["version"], aliases=episode["aliases"], parameters=episode["parameters"])
    if "variant" in episode:
        result["variant"] = episode["variant"]
    runtime = result["runtime"]
    runtime["interface"] = episode["interface"]
    # Public experiment condition and old execution protocol version are
    # independent (e.g. SelectByDistance Semantic still uses protocol base).
    runtime["registration_id"] = episode["episode_id"]
    runtime.setdefault("backend_kwargs", {})["seed"] = episode["seed"]
    if episode['execution']=='AssembledNative':
        _check(runtime.get('implementation_status')=='implemented'
               and runtime.get('protocol_family')=='canonical_extension'
               and runtime.get('backend')=='assembled_native',
               'Native assembled registration must bind an implemented native recipe')
        _check(episode.get('scene_id')=='scene_65', 'Native assembled recipe requires the registered source scene 65')
        _check(isinstance(episode.get('layout_id'),str) and episode['layout_id'].startswith('native_'),
               'Native assembled registration must identify its recipe layout')
        _id(episode['layout_id'])
        runtime['scene_id']=65
        runtime['backend_kwargs']['native_assembly']=True
        runtime['native_assembly']['registration_id']=episode['episode_id']
        return result
    placeholder = runtime.get("implementation_status") == "placeholder"
    _check((episode["execution"] == "Unavailable") == placeholder, "Execution capability mismatch")
    if placeholder:
        _check("scene_id" not in episode and "layout_id" not in episode,
               "Unavailable legacy episodes cannot pretend to have an instantiated scene/layout")
    else:
        _check({"scene_id", "layout_id"} <= episode.keys(), "Legacy execution requires scene and layout references")
        _id(episode["scene_id"])
        _id(episode["layout_id"])
        scene = load_json(root / "scenes" / f"{episode['scene_id']}.json")
        layout = load_json(root / "layouts" / f"{episode['layout_id']}.json")
        _check(f"scene_{scene['scene_id']}" == episode["scene_id"], "Legacy scene reference mismatch")
        _check(layout["layout_id"] == episode["layout_id"], "Layout reference mismatch")
        _check(layout["format"] == "LegacyInitialization", "Legacy adapter requires a legacy initialization layout")
        runtime["scene_id"] = scene["scene_id"]
        runtime["annotation_layout_key"] = layout["source_layout"]
        result["layout"] = layout["category"]
    return result


def normalize_legacy_episode(episode: dict, root: Path | None = None) -> dict:
    from .schema import normalize_task_config
    return normalize_task_config(legacy_definition(episode, root))
