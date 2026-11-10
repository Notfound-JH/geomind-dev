#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.mimic.state_reindexing_benchmark_utils import (  # noqa: E402
    HISTORICAL_IDENTITY_TRACKING_FAMILY,
    POST_ACTION_RELOCATION_MODULE_NAME,
    STATE_REINDEXING_FAMILY,
)
from geomind_lab.mimic.utils.config_semantics_adapter import normalize_annotation_entry  # noqa: E402
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import load_annotated_tasks  # noqa: E402
from geomind_lab.utils.asset_utils import resolve_assets_root  # noqa: E402
from geomind_lab.utils.scene_utils import extract_xform_tree, parse_usda_scene  # noqa: E402


DEFAULT_FAMILIES = (STATE_REINDEXING_FAMILY, HISTORICAL_IDENTITY_TRACKING_FAMILY)
DEFAULT_VARIANTS = (
    "edge_translation",
    "internal_insertion",
    "single_step_tracking",
    "multi_step_shell_game",
    "clutter_filtering",
)
DEFAULT_OBJECT_COUNTS = (3, 4, 5)
DEFAULT_DISTRACTOR_COUNTS = (0, 2, 3)
DEFAULT_VISUAL_MODES = ("distinct_objects", "identical_objects")
DEFAULT_INTERVENTION_TYPES = ("edge_move", "insertion", "multi_step_relocation")

GENERIC_CONTAINER_SPEC = {
    "basket": 1,
    "basket_position": [[0.3, -0.35]],
    "plate": 0,
    "plate_position": [],
    "bowl": 0,
    "bowl_position": [],
}
GENERIC_SHELF_SPEC = {
    "shelf": 0,
    "shelf_position": [],
}

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "edge_translation": {
        "family": STATE_REINDEXING_FAMILY,
        "script_name": "leftmost_to_right_of_rightmost",
        "target_rank": "leftmost",
        "order_axis": "x",
        "task_category": "Post Action Relocation",
        "query_family": "state_reindex_query",
        "query_text": "Pick up the current leftmost object.",
        "instruction": "Pick up the current leftmost object.",
        "intervention_type": "edge_move",
        "intervention_steps": 1,
        "supported_object_counts": (3, 4, 5),
    },
    "internal_insertion": {
        "family": STATE_REINDEXING_FAMILY,
        "script_name": "rightmost_to_between_first_and_second",
        "target_rank": "index_1",
        "order_axis": "x",
        "task_category": "Post Action Relocation",
        "query_family": "state_reindex_query",
        "query_text": "Pick up the object that is currently second from the start.",
        "instruction": "Pick up the object that is currently second from the start.",
        "intervention_type": "insertion",
        "intervention_steps": 1,
        "supported_object_counts": (3, 4, 5),
    },
    "single_step_tracking": {
        "family": HISTORICAL_IDENTITY_TRACKING_FAMILY,
        "script_name": "middle_to_left_of_leftmost",
        "target_rank": "middle",
        "order_axis": "y",
        "task_category": "Historical Identity Tracking",
        "query_family": "historical_identity_query",
        "query_text": "Pick the object that was originally in the middle.",
        "instruction": "Pick the object that was originally in the middle.",
        "intervention_type": "edge_move",
        "intervention_steps": 1,
        "supported_object_counts": (3,),
    },
    "multi_step_shell_game": {
        "family": HISTORICAL_IDENTITY_TRACKING_FAMILY,
        "script_name": "leftmost_to_middle_then_rightmost_to_left_of_leftmost",
        "target_rank": "leftmost",
        "order_axis": "y",
        "task_category": "Historical Identity Tracking",
        "query_family": "historical_identity_query",
        "query_text": "Pick the object that was originally leftmost.",
        "instruction": "Pick the object that was originally leftmost.",
        "intervention_type": "multi_step_relocation",
        "intervention_steps": 2,
        "supported_object_counts": (5,),
    },
    "clutter_filtering": {
        "family": STATE_REINDEXING_FAMILY,
        "script_name": "leftmost_to_right_of_rightmost",
        "target_rank": "leftmost",
        "order_axis": "x",
        "task_category": "Post Action Relocation",
        "query_family": "filtered_rank_query",
        "query_text": "Among the marked target objects, pick up the current leftmost one.",
        "instruction": "Among the marked target objects, pick up the current leftmost one.",
        "intervention_type": "edge_move",
        "intervention_steps": 1,
        "supported_object_counts": (3, 4, 5),
        "requires_distractors": True,
    },
    "semantic_isolation": {
        "family": STATE_REINDEXING_FAMILY,
        "script_name": "leftmost_to_right_of_rightmost",
        "target_rank": "leftmost",
        "order_axis": "x",
        "task_category": "Post Action Relocation",
        "query_family": "semantic_rank_query",
        "query_text": "Within the designated semantic target set, pick up the current leftmost object.",
        "instruction": "Within the designated semantic target set, pick up the current leftmost object.",
        "intervention_type": "edge_move",
        "intervention_steps": 1,
        "supported_object_counts": (3, 4, 5),
        "requires_distractors": True,
        "allowed_visual_modes": ("distinct_objects",),
    },
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate merge-ready GeoMind state-reindexing / identity-tracking task matrix records."
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(DEFAULT_FAMILIES),
        help="Families to generate.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Semantic task variants to generate.",
    )
    parser.add_argument(
        "--object-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_OBJECT_COUNTS),
        help="Target-subset object counts.",
    )
    parser.add_argument(
        "--distractor-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_DISTRACTOR_COUNTS),
        help="Distractor counts. The total scene object count is object_count + distractor_count.",
    )
    parser.add_argument(
        "--visual-modes",
        nargs="+",
        default=list(DEFAULT_VISUAL_MODES),
        help="Visual modes to stamp into protocol metadata.",
    )
    parser.add_argument(
        "--intervention-types",
        nargs="+",
        default=list(DEFAULT_INTERVENTION_TYPES),
        help="Intervention types to include.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation.",
    )
    return parser


def _sort_by_suffix(names: list[str]) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        match = re.search(r"_(\d+)$", str(name))
        return (int(match.group(1)) if match else 0, str(name))

    return sorted((str(name) for name in names), key=key)


def _object_prefix(name: str) -> str:
    return re.sub(r"_\d+$", "", str(name))


def _position_xy(x: float, y: float) -> dict[str, list[float]]:
    return {"position": [round(float(x), 6), round(float(y), 6)]}


def _walk_nodes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    walked: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        walked.append(node)
        for child in node.get("children", []):
            if isinstance(child, dict):
                walk(child)

    for item in items:
        if isinstance(item, dict):
            walk(item)
    return walked


def _load_annotation_scene_inventory() -> list[dict[str, Any]]:
    payload = load_annotated_tasks()
    deduped: dict[tuple[int, tuple[str, ...], str], dict[str, Any]] = {}
    for layout in payload.values():
        if not isinstance(layout, dict):
            continue
        scene_structure = layout.get("scene_structure", {})
        if not isinstance(scene_structure, dict):
            continue
        object_positions = scene_structure.get("object_positions", {})
        if not isinstance(object_positions, dict):
            continue
        object_names = _sort_by_suffix(list(object_positions.keys()))
        if not object_names:
            continue
        scene_id = int(layout.get("scene_id", 1))
        object_asset = str(scene_structure.get("object_asset", "")).strip() or _object_prefix(object_names[0]).upper()
        key = (scene_id, tuple(object_names), object_asset)
        deduped[key] = {
            "scene_id": scene_id,
            "scene_path": None,
            "object_names": tuple(object_names),
            "object_count": len(object_names),
            "object_asset": object_asset,
        }
    return sorted(
        deduped.values(),
        key=lambda candidate: (int(candidate["object_count"]), int(candidate["scene_id"])),
    )


def _load_stage_scene_inventory() -> list[dict[str, Any]]:
    stage_root = resolve_assets_root() / "stage"
    if not stage_root.is_dir():
        raise FileNotFoundError(f"Unable to resolve GeoMind stage directory under {stage_root}")

    candidates: list[dict[str, Any]] = []
    for scene_dir in sorted((path for path in stage_root.iterdir() if path.is_dir()), key=lambda path: int(path.name)):
        scene_id = int(scene_dir.name)
        scene_path = scene_dir / f"{scene_id}_new.usda"
        if not scene_path.is_file():
            scene_path = scene_dir / f"{scene_id}.usda"
        if not scene_path.is_file():
            continue

        try:
            nodes = parse_usda_scene(scene_path)
            tree = extract_xform_tree(nodes)
        except Exception:
            continue

        object_names: list[str] = []
        for node in _walk_nodes(tree):
            if str(node.get("name", "")) != "object":
                continue
            children = node.get("children", [])
            object_names = _sort_by_suffix([str(child.get("name", "")) for child in children if isinstance(child, dict)])
            break

        if not object_names:
            continue

        candidates.append(
            {
                "scene_id": scene_id,
                "scene_path": str(scene_path),
                "object_names": tuple(object_names),
                "object_count": len(object_names),
                "object_asset": _object_prefix(object_names[0]).upper(),
            }
        )

    if not candidates:
        raise ValueError("No runnable GeoMind scene templates were discovered from geomind_assets/stage.")
    return candidates


def _load_scene_inventory() -> list[dict[str, Any]]:
    annotation_candidates = _load_annotation_scene_inventory()
    if annotation_candidates:
        return annotation_candidates
    return _load_stage_scene_inventory()


def _select_scene_template(candidates: list[dict[str, Any]], total_object_count: int) -> dict[str, Any] | None:
    viable = [candidate for candidate in candidates if int(candidate["object_count"]) >= int(total_object_count)]
    if not viable:
        return None
    viable.sort(key=lambda candidate: (int(candidate["object_count"]), int(candidate["scene_id"])))
    return viable[0]


def _build_object_positions(
    *,
    object_names: list[str],
    target_count: int,
    distractor_count: int,
    order_axis: str,
) -> tuple[dict[str, dict[str, list[float]]], list[str], list[str]]:
    target_names = [str(name) for name in object_names[:target_count]]
    distractor_names = [str(name) for name in object_names[target_count : target_count + distractor_count]]
    positions: dict[str, dict[str, list[float]]] = {}

    if order_axis == "y":
        target_x = 0.40
        target_y0 = -0.06
        target_gap = 0.11
        for index, object_name in enumerate(target_names):
            positions[object_name] = _position_xy(target_x, target_y0 + target_gap * float(index))

        for index, object_name in enumerate(distractor_names):
            lane = -1.0 if index % 2 == 0 else 1.0
            row = float(index // 2)
            positions[object_name] = _position_xy(
                target_x + 0.11 * lane,
                target_y0 + 0.05 + row * 0.16,
            )
    else:
        target_y = 0.12
        target_x0 = 0.28
        target_gap = 0.07
        for index, object_name in enumerate(target_names):
            positions[object_name] = _position_xy(target_x0 + target_gap * float(index), target_y)

        for index, object_name in enumerate(distractor_names):
            lane = -1.0 if index % 2 == 0 else 1.0
            row = float(index // 2)
            positions[object_name] = _position_xy(
                target_x0 + 0.05 + row * 0.14,
                target_y + 0.10 * lane,
            )

    return positions, target_names, distractor_names


def _scene_layout_name(*, distractor_count: int) -> str:
    return "Interleaved Line" if int(distractor_count) > 0 else "Line"


def _build_task_name(
    *,
    family: str,
    variant: str,
    object_count: int,
    distractor_count: int,
    visual_mode: str,
) -> str:
    name = f"{family}_{variant}_{int(object_count)}obj"
    if int(distractor_count) > 0:
        name += f"_{int(distractor_count)}dist"
    if visual_mode == "distinct_objects":
        name += "_distinct"
    else:
        name += "_identical"
    return name


def _build_assumptions(
    *,
    family: str,
    variant: str,
    visual_mode: str,
    order_axis: str,
    distractor_count: int,
    intervention_steps: int,
) -> list[str]:
    axis_name = "x" if order_axis == "x" else "y"
    assumptions = [
        f"Target objects are operationally ranked along the tabletop {axis_name} axis.",
        "Ground truth is resolved at runtime from the existing post_action_relocation protocol chain.",
        "The generated object ids are kept remap-compatible with existing suffix-based object remapping.",
    ]
    if family == HISTORICAL_IDENTITY_TRACKING_FAMILY:
        assumptions.append("Ground truth is locked to the initial ordering cached at reset.")
    else:
        assumptions.append("Ground truth is resolved from the post-intervention ordering instead of a static goal sequence.")
    if int(intervention_steps) > 1:
        assumptions.append("The intervention consists of multiple scripted relocations before the final query.")
    if int(distractor_count) > 0:
        assumptions.append("Visible distractors are excluded from ranking via target_subset_spec.")
    if visual_mode == "distinct_objects":
        assumptions.append("visual_mode is preserved as benchmark metadata while keeping the current scene/remap chain unchanged.")
    if variant == "semantic_isolation":
        assumptions.append("semantic_isolation is represented as a frozen protocol tag and does not introduce a new engine path.")
    return assumptions


def _build_validation_checks(*, distractor_count: int) -> list[str]:
    checks = [
        "all object_id referenced by object_positions exist in the scene",
        "goal_sequence is intentionally empty because the protocol evaluator resolves GT at runtime",
        "ignore z height by design",
    ]
    if int(distractor_count) > 0:
        checks.insert(
            1,
            "all object_id referenced by target_subset_spec exist in the scene",
        )
    return checks


def _build_protocol(
    *,
    family: str,
    variant: str,
    query_family: str,
    query_text: str,
    order_axis: str,
    script_name: str,
    target_rank: str,
    object_count: int,
    distractor_count: int,
    visual_mode: str,
    intervention_type: str,
    intervention_steps: int,
    target_names: list[str],
) -> dict[str, Any]:
    diagnostic_tags: list[str] = ["post_intervention_reindex"]
    if family == HISTORICAL_IDENTITY_TRACKING_FAMILY:
        diagnostic_tags.append("historical_identity_tracking")
    if variant == "internal_insertion":
        diagnostic_tags.append("internal_topology_update")
    if int(distractor_count) > 0 or variant == "clutter_filtering":
        diagnostic_tags.append("distractor_filtering")
    if int(intervention_steps) > 1:
        diagnostic_tags.append("multi_step_temporal_tracking")
    if variant == "semantic_isolation":
        diagnostic_tags.append("semantic_isolation")

    protocol = {
        "module_name": POST_ACTION_RELOCATION_MODULE_NAME,
        "track_name": family,
        "family": family,
        "variant": variant,
        "query_family": query_family,
        "query_text": query_text,
        "object_count": int(object_count),
        "distractor_count": int(distractor_count),
        "visual_mode": visual_mode,
        "intervention_type": intervention_type,
        "intervention_steps": int(intervention_steps),
        "noise_level": "clean",
        "diagnostic_tags": diagnostic_tags,
        "difficulty_factors": {
            "object_count": int(object_count),
            "distractor_count": int(distractor_count),
            "intervention_steps": int(intervention_steps),
            "visual_mode": visual_mode,
        },
        "intervention": {
            "type": "scripted_reorder",
            "script_name": script_name,
        },
        "gt_spec": {
            "resolver": "initial_rank" if family == HISTORICAL_IDENTITY_TRACKING_FAMILY else "rank_after_intervention",
            "order_axis": order_axis,
            "target_rank": target_rank,
        },
        "control_mode": None,
    }
    if int(distractor_count) > 0:
        protocol["target_subset_spec"] = {"object_ids": list(target_names)}
    return protocol


def _build_layout_record(
    *,
    template: dict[str, Any],
    variant_spec: dict[str, Any],
    variant: str,
    object_count: int,
    distractor_count: int,
    visual_mode: str,
) -> dict[str, Any]:
    total_object_count = int(object_count) + int(distractor_count)
    order_axis = str(variant_spec["order_axis"])
    scene_structure_layout = _scene_layout_name(distractor_count=int(distractor_count))
    object_positions, target_names, _ = _build_object_positions(
        object_names=list(template["object_names"]),
        target_count=int(object_count),
        distractor_count=int(distractor_count),
        order_axis=order_axis,
    )
    task_name = _build_task_name(
        family=str(variant_spec["family"]),
        variant=variant,
        object_count=int(object_count),
        distractor_count=int(distractor_count),
        visual_mode=visual_mode,
    )
    protocol = _build_protocol(
        family=str(variant_spec["family"]),
        variant=variant,
        query_family=str(variant_spec["query_family"]),
        query_text=str(variant_spec["query_text"]),
        order_axis=order_axis,
        script_name=str(variant_spec["script_name"]),
        target_rank=str(variant_spec["target_rank"]),
        object_count=int(object_count),
        distractor_count=int(distractor_count),
        visual_mode=visual_mode,
        intervention_type=str(variant_spec["intervention_type"]),
        intervention_steps=int(variant_spec["intervention_steps"]),
        target_names=target_names,
    )
    assumptions = _build_assumptions(
        family=str(variant_spec["family"]),
        variant=variant,
        visual_mode=visual_mode,
        order_axis=order_axis,
        distractor_count=int(distractor_count),
        intervention_steps=int(variant_spec["intervention_steps"]),
    )
    record = {
        "layout_key": task_name,
        "layout": {
            "scene_id": int(template["scene_id"]),
            "scene_structure": {
                "layout": scene_structure_layout,
                "num_of_objects": total_object_count,
                "object_asset": str(template["object_asset"]),
                "object_positions": object_positions,
                "container": dict(GENERIC_CONTAINER_SPEC),
                "shelf": dict(GENERIC_SHELF_SPEC),
            },
            "tasks": [
                {
                    "task_category": str(variant_spec["task_category"]),
                    "instruction": str(variant_spec["instruction"]),
                    "annotation": {
                        "task_id": task_name,
                        "reference_frame": "world",
                        "targets": [],
                        "goal_sequence": [],
                        "assumptions": assumptions,
                        "validation_checks": _build_validation_checks(distractor_count=int(distractor_count)),
                        "protocol": protocol,
                    },
                }
            ],
        },
    }
    return record


def _attach_semantics(record: dict[str, Any]) -> dict[str, Any]:
    layout = record.get("layout", {})
    if not isinstance(layout, dict):
        return record

    tasks = layout.get("tasks", [])
    if not isinstance(tasks, list):
        return record

    scene_semantics = layout.get("scene_semantics")
    for task_index, task_payload in enumerate(tasks):
        if not isinstance(task_payload, dict):
            continue
        annotation = task_payload.get("annotation", {})
        if not isinstance(annotation, dict):
            continue
        semantic_view = normalize_annotation_entry(
            layout_key=record.get("layout_key"),
            task_index=task_index,
            layout=layout,
            task_payload=task_payload,
        )
        annotation.setdefault("canonical_task_id", semantic_view.get("canonical_task_id"))
        annotation.setdefault("task_semantics", semantic_view.get("task_semantics"))
        annotation.setdefault("instance_semantics", semantic_view.get("instance_semantics"))
        annotation.setdefault("diagnostic_tags", semantic_view.get("diagnostic_tags"))
        annotation.setdefault("difficulty_factors", semantic_view.get("difficulty_factors"))
        if scene_semantics is None:
            scene_semantics = semantic_view.get("scene_semantics")

    if scene_semantics is not None:
        layout.setdefault("scene_semantics", scene_semantics)
    return record


def generate_matrix(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    scene_inventory = _load_scene_inventory()
    families = {str(value) for value in args.families}
    variants = [str(value) for value in args.variants]
    object_counts = [int(value) for value in args.object_counts]
    distractor_counts = [int(value) for value in args.distractor_counts]
    visual_modes = [str(value) for value in args.visual_modes]
    intervention_types = {str(value) for value in args.intervention_types}

    records: list[dict[str, Any]] = []
    skipped: list[str] = []

    for variant in variants:
        variant_spec = VARIANT_SPECS.get(variant)
        if variant_spec is None:
            skipped.append(f"variant={variant}: unsupported variant")
            continue
        if str(variant_spec["family"]) not in families:
            continue
        if str(variant_spec["intervention_type"]) not in intervention_types:
            continue

        allowed_visual_modes = tuple(variant_spec.get("allowed_visual_modes", visual_modes))
        allowed_visual_mode_set = {str(value) for value in allowed_visual_modes}

        for object_count in object_counts:
            if int(object_count) not in set(int(value) for value in variant_spec["supported_object_counts"]):
                skipped.append(f"variant={variant} object_count={object_count}: unsupported by current scripted logic")
                continue

            for distractor_count in distractor_counts:
                if variant_spec.get("requires_distractors") and int(distractor_count) <= 0:
                    continue
                total_object_count = int(object_count) + int(distractor_count)
                template = _select_scene_template(scene_inventory, total_object_count)
                if template is None:
                    skipped.append(
                        f"variant={variant} object_count={object_count} distractor_count={distractor_count}: no scene template with >= {total_object_count} objects"
                    )
                    continue

                for visual_mode in visual_modes:
                    if str(visual_mode) not in allowed_visual_mode_set:
                        continue
                    records.append(
                        _attach_semantics(
                            _build_layout_record(
                                template=template,
                                variant_spec=variant_spec,
                                variant=variant,
                                object_count=int(object_count),
                                distractor_count=int(distractor_count),
                                visual_mode=str(visual_mode),
                            )
                        )
                    )

    records.sort(key=lambda item: str(item.get("layout_key", "")))
    return records, skipped


def main() -> None:
    args = build_argparser().parse_args()
    records, skipped = generate_matrix(args)
    serialized = json.dumps(records, indent=int(args.indent), ensure_ascii=False) + "\n"
    if skipped:
        sys.stderr.write(f"[generate_state_reindexing_matrix] generated={len(records)} skipped={len(skipped)}\n")
        for message in skipped[:20]:
            sys.stderr.write(f"  - {message}\n")
        if len(skipped) > 20:
            sys.stderr.write(f"  ... {len(skipped) - 20} more skipped combinations\n")
    if args.output == "-":
        sys.stdout.write(serialized)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
