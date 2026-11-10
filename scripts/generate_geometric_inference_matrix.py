#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.mimic.utils.config_semantics_adapter import normalize_annotation_entry
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import load_annotated_tasks


GEOMETRIC_INFERENCE_MODULE_NAME = "geometric_inference"
EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME = "explicit_rule_extrapolation"
IMPLICIT_TREND_FITTING_TRACK_NAME = "implicit_trend_fitting"
PCA_TREND_RULE_TYPE = "pca_trend_continuation"

DEFAULT_EXPLICIT_SPACINGS = (0.08, 0.10, 0.12)
DEFAULT_IMPLICIT_SPACINGS = (0.08, 0.10, 0.12)
DEFAULT_IMPLICIT_ANCHOR_COUNTS = (3, 4, 5)
DEFAULT_IMPLICIT_NOISE_LEVELS = ("low_noise", "medium_noise", "high_noise")

NOISE_SPECS = {
    "low_noise": {"sigma_parallel": 0.004, "sigma_perpendicular": 0.008},
    "medium_noise": {"sigma_parallel": 0.006, "sigma_perpendicular": 0.015},
    "high_noise": {"sigma_parallel": 0.008, "sigma_perpendicular": 0.024},
}
PARALLEL_PATTERN = (0.0, -0.35, 0.25, -0.2, 0.4, -0.3, 0.18, -0.12)
PERP_PATTERN = (0.0, 0.85, -0.65, 0.5, -0.9, 0.7, -0.4, 0.25)
WEAK_DESTRUCTION_SCALE = 1.65


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a minimal geometric inference task matrix as merge-ready JSON records."
    )
    parser.add_argument(
        "--family",
        choices=("explicit", "implicit", "all"),
        default="all",
        help="Which geometric family to generate.",
    )
    parser.add_argument(
        "--explicit-spacings",
        type=float,
        nargs="+",
        default=list(DEFAULT_EXPLICIT_SPACINGS),
        help="Spacing values for explicit-rule tasks.",
    )
    parser.add_argument(
        "--implicit-spacings",
        type=float,
        nargs="+",
        default=list(DEFAULT_IMPLICIT_SPACINGS),
        help="Spacing values for implicit-trend tasks.",
    )
    parser.add_argument(
        "--implicit-anchor-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_IMPLICIT_ANCHOR_COUNTS),
        help="Anchor counts for implicit-trend tasks.",
    )
    parser.add_argument(
        "--implicit-noise-levels",
        type=str,
        nargs="+",
        default=list(DEFAULT_IMPLICIT_NOISE_LEVELS),
        help="Noise levels for implicit-trend tasks.",
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
        stem, _, suffix = name.rpartition("_")
        if stem and suffix.isdigit():
            return (int(suffix), name)
        return (0, name)

    return sorted(names, key=key)


def _coerce_spacing_slug(spacing: float) -> str:
    return f"s{int(round(float(spacing) * 100.0)):03d}"


def _template_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for layout_key, layout in sorted(payload.items()):
        if not isinstance(layout, dict):
            continue
        scene_structure = layout.get("scene_structure", {})
        if not isinstance(scene_structure, dict):
            continue
        object_positions = scene_structure.get("object_positions", {})
        if not isinstance(object_positions, dict) or not object_positions:
            continue
        candidates.append(
            {
                "layout_key": str(layout_key),
                "scene_id": int(layout.get("scene_id", 1)),
                "scene_structure": deepcopy(scene_structure),
                "object_names": _sort_by_suffix(list(object_positions.keys())),
                "object_count": len(object_positions),
            }
        )
    if not candidates:
        raise ValueError("No annotated layouts with object_positions were found.")
    return candidates


def _select_template(candidates: list[dict[str, Any]], required_object_count: int) -> dict[str, Any]:
    viable = [candidate for candidate in candidates if int(candidate["object_count"]) >= int(required_object_count)]
    if not viable:
        raise ValueError(f"No template scene has at least {required_object_count} objects.")
    viable.sort(key=lambda candidate: (int(candidate["object_count"]), str(candidate["layout_key"])))
    return viable[0]


def _position_xy(x: float, y: float) -> dict[str, list[float]]:
    return {"position": [round(float(x), 6), round(float(y), 6)]}


def _build_line_positions(
    *,
    object_names: list[str],
    anchor_count: int,
    spacing: float,
    variant: str,
    noise_level: str,
) -> tuple[dict[str, dict[str, list[float]]], str, list[str]]:
    if len(object_names) < int(anchor_count) + 1:
        raise ValueError("object_names must contain at least one target plus all anchors.")

    target_object = str(object_names[0])
    anchor_names = [str(name) for name in object_names[1 : 1 + int(anchor_count)]]
    distractor_names = [str(name) for name in object_names[1 + int(anchor_count) :]]

    spacing = float(spacing)
    if variant == "perfect_midpoint":
        anchor_start_y = -0.5 * spacing
    elif variant == "isometric_extension":
        anchor_start_y = -1.0 * spacing
    else:
        anchor_start_y = -0.5 * spacing * float(anchor_count)

    center_x = 0.40
    target_x = 0.56
    target_y = max(min(anchor_start_y - 0.10, -0.12), -0.30)
    distractor_x = 0.56
    distractor_y0 = max(anchor_start_y, -0.30)

    noise_spec = NOISE_SPECS.get(str(noise_level), {"sigma_parallel": 0.0, "sigma_perpendicular": 0.0})
    sigma_parallel = float(noise_spec["sigma_parallel"])
    sigma_perpendicular = float(noise_spec["sigma_perpendicular"])
    if variant == "weak_destruction_control":
        sigma_perpendicular *= WEAK_DESTRUCTION_SCALE

    object_positions = {target_object: _position_xy(target_x, target_y)}
    for index, object_name in enumerate(anchor_names):
        parallel_scale = PARALLEL_PATTERN[index % len(PARALLEL_PATTERN)]
        perpendicular_scale = PERP_PATTERN[index % len(PERP_PATTERN)]
        object_positions[object_name] = _position_xy(
            center_x + sigma_perpendicular * perpendicular_scale,
            anchor_start_y + float(index) * spacing + sigma_parallel * parallel_scale,
        )

    for index, object_name in enumerate(distractor_names):
        object_positions[object_name] = _position_xy(
            distractor_x + 0.04 * float(index % 2),
            distractor_y0 + 0.11 * float(index),
        )

    return object_positions, target_object, anchor_names


def _base_scene_structure(template: dict[str, Any], object_positions: dict[str, Any]) -> dict[str, Any]:
    template_scene_structure = deepcopy(template["scene_structure"])
    scene_structure = {
        "layout": "Line",
        "num_of_objects": len(object_positions),
        "object_asset": str(template_scene_structure.get("object_asset", "GENERIC_OBJECT")),
        "object_positions": object_positions,
        "container": deepcopy(template_scene_structure.get("container", {})),
        "shelf": deepcopy(template_scene_structure.get("shelf", {})),
    }
    return scene_structure


def _build_annotation(
    *,
    layout_key: str,
    family: str,
    variant: str,
    track_name: str,
    scene_id: int,
    scene_structure: dict[str, Any],
    anchor_count: int,
    spacing: float,
    noise_level: str,
    sigma_parallel: float,
    sigma_perpendicular: float,
    target_object: str,
    anchor_names: list[str],
) -> dict[str, Any]:
    diagnostic_tags: list[str] = []
    if family == EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME:
        instruction = (
            "Put the grasped object exactly at the midpoint between anchor A and anchor B."
            if variant == "perfect_midpoint"
            else "Put the object on the extension of the line from A to B with the same spacing."
        )
        query_text = (
            "把手中的方块精确地放在 A 和 B 的正中间。"
            if variant == "perfect_midpoint"
            else "把方块放在 A 和 B 连线的延长线上，靠近 B 的外侧，并保持相同间距。"
        )
        gt_rule_type = variant
        query_family = "explicit_rule_query"
        control_mode = "deterministic_rule"
        assumptions = [
            "Ground truth is resolved directly from the two anchors in the tabletop XY plane.",
            "The target object is treated as pre-attached to the gripper for the minimal placement protocol.",
            "This generated layout keeps explicit-rule noise at zero by construction.",
        ]
        diagnostic_tags.append("center_interpolation" if variant == "perfect_midpoint" else "directional_extrapolation")
    else:
        instruction = "Continue the geometric trend by placing the grasped object at the next position."
        query_text = "顺着现有方块的排列走势，放在下一个位置。"
        gt_rule_type = PCA_TREND_RULE_TYPE
        query_family = "implicit_trend_query"
        control_mode = "weak_destruction" if variant == "weak_destruction_control" else "backbone"
        assumptions = [
            "Ground truth is operationalized by the existing PCA trend continuation solver.",
            "Parallel and perpendicular perturbations are injected deterministically from the configured noise level.",
            "The target object is treated as pre-attached to the gripper for the minimal placement protocol.",
        ]
        diagnostic_tags.append("global_trend_extraction")
        if variant == "weak_destruction_control":
            diagnostic_tags.append("structure_destroyed_control")

    protocol = {
        "module_name": GEOMETRIC_INFERENCE_MODULE_NAME,
        "track_name": track_name,
        "family": family,
        "variant": variant,
        "subtask_name": variant,
        "query_family": query_family,
        "query_text": query_text,
        "anchor_count": int(anchor_count),
        "anchor_layout": "line",
        "spacing": float(spacing),
        "noise_level": str(noise_level),
        "sigma_parallel": float(sigma_parallel),
        "sigma_perpendicular": float(sigma_perpendicular),
        "diagnostic_tags": diagnostic_tags,
        "difficulty_factors": {
            "anchor_count": int(anchor_count),
            "spacing": float(spacing),
            "noise_level": str(noise_level),
            "sigma_parallel": float(sigma_parallel),
            "sigma_perpendicular": float(sigma_perpendicular),
        },
        "pre_attached_obj": str(target_object),
        "success_tolerance_l2": 0.05,
        "success_tolerance_orthogonal": 0.02,
        "anchors": [
            {"anchor_id": chr(ord("A") + index), "object_id": str(anchor_name)}
            for index, anchor_name in enumerate(anchor_names)
        ],
        "target_object": {
            "object_id": str(target_object),
            "carrier": "gripper",
            "preloaded": True,
        },
        "gt_spec": {
            "rule_type": gt_rule_type,
        },
        "control_mode": control_mode,
    }
    if track_name == IMPLICIT_TREND_FITTING_TRACK_NAME:
        protocol["gt_spec"].update(
            {
                "axis_method": "pca_principal_axis",
                "step_method": "projected_average_spacing",
            }
        )

    return {
        "scene_id": int(scene_id),
        "scene_structure": scene_structure,
        "tasks": [
            {
                "task_category": "Geometric Inference",
                "instruction": instruction,
                "annotation": {
                    "task_id": layout_key,
                    "reference_frame": "world",
                    "targets": [],
                    "goal_sequence": [],
                    "assumptions": assumptions,
                    "validation_checks": [
                        "all object_id referenced by anchors and target_object exist in the scene",
                        "goal_sequence is intentionally empty because the protocol evaluator resolves GT from anchors at runtime",
                        "ignore z height by design",
                    ],
                    "protocol": protocol,
                },
            }
        ],
    }


def _attach_semantics(layout_key: str, layout: dict[str, Any]) -> dict[str, Any]:
    tasks = layout.get("tasks", [])
    if not isinstance(tasks, list):
        return layout

    scene_semantics = layout.get("scene_semantics")
    for task_index, task_payload in enumerate(tasks):
        if not isinstance(task_payload, dict):
            continue
        annotation = task_payload.get("annotation", {})
        if not isinstance(annotation, dict):
            continue
        semantic_view = normalize_annotation_entry(
            layout_key=layout_key,
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
    return layout


def _build_explicit_records(candidates: list[dict[str, Any]], spacings: list[float]) -> list[dict[str, Any]]:
    template = _select_template(candidates, required_object_count=3)
    records: list[dict[str, Any]] = []
    for variant in ("perfect_midpoint", "isometric_extension"):
        for spacing in spacings:
            object_positions, target_object, anchor_names = _build_line_positions(
                object_names=list(template["object_names"]),
                anchor_count=2,
                spacing=spacing,
                variant=variant,
                noise_level="zero_noise",
            )
            layout_key = f"geometric_inference_{variant}_line_2_{_coerce_spacing_slug(spacing)}"
            scene_structure = _base_scene_structure(template, object_positions)
            records.append(
                {
                    "layout_key": layout_key,
                    "layout": _attach_semantics(
                        layout_key,
                        _build_annotation(
                            layout_key=layout_key,
                            family=EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME,
                            variant=variant,
                            track_name=EXPLICIT_RULE_EXTRAPOLATION_TRACK_NAME,
                            scene_id=int(template["scene_id"]),
                            scene_structure=scene_structure,
                            anchor_count=2,
                            spacing=float(spacing),
                            noise_level="zero_noise",
                            sigma_parallel=0.0,
                            sigma_perpendicular=0.0,
                            target_object=target_object,
                            anchor_names=anchor_names,
                        ),
                    ),
                }
            )
    return records


def _build_implicit_records(
    candidates: list[dict[str, Any]],
    spacings: list[float],
    anchor_counts: list[int],
    noise_levels: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for anchor_count in anchor_counts:
        template = _select_template(candidates, required_object_count=int(anchor_count) + 1)
        for variant in ("trend_continuation", "weak_destruction_control"):
            for spacing in spacings:
                for noise_level in noise_levels:
                    if noise_level not in NOISE_SPECS:
                        raise ValueError(f"Unsupported implicit noise level: {noise_level!r}")
                    noise_spec = NOISE_SPECS[noise_level]
                    object_positions, target_object, anchor_names = _build_line_positions(
                        object_names=list(template["object_names"]),
                        anchor_count=int(anchor_count),
                        spacing=spacing,
                        variant=variant,
                        noise_level=noise_level,
                    )
                    layout_key = (
                        f"geometric_inference_{variant}_line_{int(anchor_count)}_{noise_level}_{_coerce_spacing_slug(spacing)}"
                    )
                    scene_structure = _base_scene_structure(template, object_positions)
                    sigma_perpendicular = float(noise_spec["sigma_perpendicular"])
                    if variant == "weak_destruction_control":
                        sigma_perpendicular *= WEAK_DESTRUCTION_SCALE
                    records.append(
                        {
                            "layout_key": layout_key,
                            "layout": _attach_semantics(
                                layout_key,
                                _build_annotation(
                                    layout_key=layout_key,
                                    family=IMPLICIT_TREND_FITTING_TRACK_NAME,
                                    variant=variant,
                                    track_name=IMPLICIT_TREND_FITTING_TRACK_NAME,
                                    scene_id=int(template["scene_id"]),
                                    scene_structure=scene_structure,
                                    anchor_count=int(anchor_count),
                                    spacing=float(spacing),
                                    noise_level=noise_level,
                                    sigma_parallel=float(noise_spec["sigma_parallel"]),
                                    sigma_perpendicular=sigma_perpendicular,
                                    target_object=target_object,
                                    anchor_names=anchor_names,
                                ),
                            ),
                        }
                    )
    return records


def generate_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = load_annotated_tasks()
    candidates = _template_candidates(payload)
    records: list[dict[str, Any]] = []
    if args.family in {"explicit", "all"}:
        records.extend(_build_explicit_records(candidates, [float(value) for value in args.explicit_spacings]))
    if args.family in {"implicit", "all"}:
        records.extend(
            _build_implicit_records(
                candidates,
                [float(value) for value in args.implicit_spacings],
                [int(value) for value in args.implicit_anchor_counts],
                [str(value) for value in args.implicit_noise_levels],
            )
        )
    return records


def main() -> None:
    args = build_argparser().parse_args()
    records = generate_matrix(args)
    serialized = json.dumps(records, indent=int(args.indent), ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(serialized)
        return
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
