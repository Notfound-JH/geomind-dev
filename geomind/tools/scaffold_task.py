from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

from geomind.core.schema import upgrade_task_config, make_registration_id, normalize_task_config
from geomind.core.resolver import resolve_task_request
from geomind_lab.mimic.utils.task_metadata_registry import (
    build_task_metadata,
    resolve_task_id as resolve_legacy_task_id,
    supported_task_versions,
)


_TASK_VERSION_TO_VARIANT = {
    "base": "Base",
    "semantic": "Sem",
    "visual": "Visual",
    "execution": "Execution",
}


def _normalize_task_version(value: str) -> str:
    normalized = str(value).strip().lower()
    return {
        "base": "base",
        "sem": "semantic",
        "semantic": "semantic",
        "visual": "visual",
        "execution": "execution",
        "exec": "execution",
    }.get(normalized, normalized)


def _normalize_variant(value: str) -> str:
    normalized = str(value).strip().lower()
    return {
        "base": "Base",
        "semantic": "Sem",
        "sem": "Sem",
        "visual": "Visual",
        "execution": "Execution",
        "exec": "Execution",
    }.get(normalized, normalized.capitalize())


def _resolve_semantic_field(
    *,
    field_name: str,
    cli_value: str,
    metadata_value: str,
    required_when_missing: bool = False,
) -> str:
    local_value = str(cli_value).strip()
    canonical_value = str(metadata_value).strip()
    if local_value and canonical_value and local_value.lower() != canonical_value.lower():
        raise ValueError(
            f"{field_name} mismatch with legacy metadata: local={local_value!r}, legacy={canonical_value!r}"
        )
    resolved = canonical_value or local_value
    if required_when_missing and not resolved:
        raise ValueError(f"Missing required semantic field: {field_name}")
    return resolved


def legacy_main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold a GeoMind task config with legacy-aligned semantics.")
    parser.add_argument(
        "--canonical-task-id",
        "--legacy-task-id",
        dest="canonical_task_id",
        required=True,
        help="Legacy/canonical task id, e.g. place_directional",
    )
    parser.add_argument(
        "--task-id-prefix",
        default="",
        help="Deprecated compatibility option; public task IDs no longer include a category prefix.",
    )
    parser.add_argument("--split", required=True, help="base|aug")
    parser.add_argument(
        "--task-version",
        required=True,
        help="Required semantic version axis: base|semantic|visual|execution",
    )
    parser.add_argument("--variant", default="", help="Optional task_id variant override; must match task_version")
    parser.add_argument("--task-name", required=True, help="CamelCase task name")
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--family", default="", help="Public task family, e.g. RelationalGrounding")
    parser.add_argument("--protocol-family", default="", help="Runtime protocol family; must match canonical metadata")
    parser.add_argument("--condition", default="", help="Base|Semantic|Visual|Execution; defaults to task-version")
    parser.add_argument("--layout", default="", help="Layout category, e.g. Line|Grid|Cross; inferred for known tasks")
    parser.add_argument("--task-variant", default="", help="Public variant, e.g. Inside or OnTop")
    parser.add_argument("--parameters", type=json.loads, default=None, help="Task parameter JSON object")
    parser.add_argument("--evaluator-key", default="", help="Only used when canonical metadata is unavailable")
    parser.add_argument("--semantic-module", default="", help="Only used when canonical metadata is unavailable")
    parser.add_argument("--skill-type", default="", help="Only used when canonical metadata is unavailable")
    parser.add_argument("--phase", default="", help="Only used when canonical metadata is unavailable")
    parser.add_argument("--annotation-layout-key", default="")
    parser.add_argument("--annotation-task-index", type=int, default=0)
    parser.add_argument("--object-set", default="")
    parser.add_argument("--assets-group", default="")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--no-validate", action="store_true", default=False)
    args = parser.parse_args()

    requested_task_version = _normalize_task_version(args.task_version)
    if requested_task_version not in _TASK_VERSION_TO_VARIANT:
        raise ValueError(
            "Unsupported task_version for task_id schema. Supported values: "
            f"{', '.join(sorted(_TASK_VERSION_TO_VARIANT))}; got {requested_task_version!r}"
        )

    variant_cap = _TASK_VERSION_TO_VARIANT[requested_task_version]
    if str(args.variant).strip():
        normalized_variant = _normalize_variant(args.variant)
        if normalized_variant != variant_cap:
            raise ValueError(
                f"--variant must match --task-version: task_version={requested_task_version!r}, "
                f"variant={normalized_variant!r}"
            )

    canonical_task_id = resolve_legacy_task_id(args.canonical_task_id) or str(args.canonical_task_id).strip()
    declared_versions = tuple(supported_task_versions(canonical_task_id))
    if not declared_versions:
        raise ValueError(
            f"Unknown canonical task metadata for {canonical_task_id!r}. "
            "Refuse to scaffold compressed/local semantics. Please register canonical metadata first."
        )
    metadata = build_task_metadata(canonical_task_id, task_version=requested_task_version)
    metadata_available = bool(str(metadata.get("family", "")).strip())
    if str(metadata.get("task_version", "")).strip().lower() != requested_task_version:
        supported_versions = ", ".join(declared_versions)
        raise ValueError(
            f"{canonical_task_id!r} does not support task_version={requested_task_version!r}. "
            f"Supported: {supported_versions}"
        )

    family = _resolve_semantic_field(
        field_name="family",
        cli_value=args.protocol_family,
        metadata_value=str(metadata.get("family", "")),
        required_when_missing=True,
    )
    evaluator_key = _resolve_semantic_field(
        field_name="evaluator_key",
        cli_value=args.evaluator_key,
        metadata_value=str(metadata.get("evaluator_key", "")),
        required_when_missing=True,
    )
    semantic_module = _resolve_semantic_field(
        field_name="semantic_module",
        cli_value=args.semantic_module,
        metadata_value=str(metadata.get("semantic_module", "")),
        required_when_missing=True,
    )
    skill_type = _resolve_semantic_field(
        field_name="skill_type",
        cli_value=args.skill_type,
        metadata_value=str(metadata.get("skill_type", "")),
        required_when_missing=True,
    )
    phase = _resolve_semantic_field(
        field_name="phase",
        cli_value=args.phase,
        metadata_value=str(metadata.get("phase", "")),
        required_when_missing=True,
    )
    instruction = str(args.instruction).strip() or str(metadata.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("Missing instruction. Provide --instruction or register it in legacy metadata.")

    if args.parameters is not None and not isinstance(args.parameters, dict):
        raise ValueError("--parameters must be a JSON object")
    task_id = make_registration_id(str(args.task_name).strip(),
        condition=args.condition or requested_task_version, version=int(args.version),
        task_variant=args.task_variant, parameters=args.parameters)

    payload = {
        "task_id": task_id,
        "canonical_task_id": canonical_task_id,
        "family": family,
        "split": str(args.split).strip().lower(),
        "variant": requested_task_version,
        "task_version": requested_task_version,
        "task_name": str(args.task_name).strip(),
        "version": int(args.version),
        "scene_id": int(args.scene_id),
        "instruction": instruction,
        "evaluator_key": evaluator_key,
        "semantic_module": semantic_module,
        "skill_type": skill_type,
        "phase": phase,
        "annotation_layout_key": str(args.annotation_layout_key),
        "annotation_task_index": int(args.annotation_task_index),
        "object_set": str(args.object_set),
        "assets_group": str(args.assets_group),
        "tags": [],
        "aliases": [canonical_task_id, task_id],
        "meta": {
            "source_registry": "geomind_lab.mimic.utils.task_metadata_registry" if metadata_available else "manual",
        },
    }
    out_path = Path(args.output) if args.output else Path("geomind") / "configs" / "legacy" / "registrations" / f"{task_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    definition = upgrade_task_config(
        payload, family=args.family or None,
        condition=args.condition or requested_task_version, layout=args.layout or None,
    )
    if args.task_variant: definition["variant"] = args.task_variant
    if args.parameters is not None:
        if not isinstance(args.parameters, dict): raise ValueError("--parameters must be a JSON object")
        definition["parameters"] = args.parameters
    normalize_task_config(definition)
    out_path.write_text(json.dumps(definition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_validate:
        resolved = resolve_task_request(task_id)
        print(f"validated_resolved_task_id={resolved.resolved_task_id}")
        print(f"validated_runtime_mode={resolved.env_cfg.runtime_mode}")
        print(f"validated_evaluator_key={resolved.spec.evaluator_key}")
    print(f"created={out_path}")
    print(f"task_id={definition['task_id']}")
    print(f"registration_id={task_id}")
    print(f"canonical_task_id={canonical_task_id}")
    return 0


def main() -> int:
    if "--legacy" in sys.argv:
        sys.argv.remove("--legacy")
        return legacy_main()
    from geomind.episodes import EpisodeResolver
    from geomind.episodes.resolver import ContractError, _check, _id, load_json
    from geomind.tools.migrate_episode_configs import write
    parser = argparse.ArgumentParser(description="Create a validated native v3 episode. Old runtime options require --legacy.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--layout-id", required=True)
    parser.add_argument("--parameters-file", type=Path, required=True)
    parser.add_argument("--response", choices=["SelectObject", "ReportPoint"], required=True)
    parser.add_argument("--position-tolerance-m", type=float)
    parser.add_argument("--protocol-id")
    parser.add_argument("--physical-execution-file", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    try:
        episode = {"schema_version": 3, "release_version": "2.0", "episode_id": args.episode_id, "task_id": args.task_id,
                   "scene_id": args.scene_id, "layout_id": args.layout_id,
                   "parameters": load_json(args.parameters_file), "response": args.response, "seed": args.seed}
        if args.protocol_id:
            episode["protocol_id"] = args.protocol_id
        if args.physical_execution_file:
            episode["physical_execution"] = load_json(args.physical_execution_file)
        if args.position_tolerance_m is not None:
            episode["evaluation"] = {"position_tolerance_m": args.position_tolerance_m}
        _check(args.response != "ReportPoint" or "evaluation" in episode,
               "ReportPoint execution requires --position-tolerance-m")
        for identity in (args.episode_id, args.task_id, args.scene_id, args.layout_id):
            _id(identity)
        if args.protocol_id:
            _id(args.protocol_id)
        # Validate in a staging directory before publishing the active file.
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            refs = [("task_definitions", args.task_id), ("scenes", args.scene_id), ("layouts", args.layout_id)]
            if args.protocol_id:
                refs.append(("protocols", args.protocol_id))
            for folder, identity in refs:
                destination = staging / folder / f"{identity}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(args.root / folder / f"{identity}.json", destination)
            write(staging / "episodes" / f"{args.episode_id}.json", episode)
            resolved = EpisodeResolver(staging).resolve(args.episode_id)
        destination = args.root / "episodes" / f"{args.episode_id}.json"
        _check(not destination.exists(), f"Episode already exists: {destination}")
        write(destination, episode)
        print(json.dumps({"created": str(destination), "contract_hash": resolved.contract_hash,
                          "configuration_valid": True, "ground_truth_computed": False}))
    except (ContractError, OSError) as exc:
        parser.exit(2, f"Episode error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
