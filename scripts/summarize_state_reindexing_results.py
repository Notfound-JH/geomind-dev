#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.mimic.state_reindexing_benchmark_utils import (  # noqa: E402
    DISCRETE_SELECTION_TASK_TYPE,
    build_state_refresh_summary_metadata,
)
from geomind_lab.mimic.utils.config_semantics_adapter import (  # noqa: E402
    augment_record_with_semantics,
    normalize_annotation_entry,
    serialize_semantic_payload,
)
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import load_annotated_tasks  # noqa: E402


DEFAULT_GROUP_BY = (
    "semantic_module",
    "task_id",
    "task_version",
    "skill_type",
    "phase",
    "agent_name",
    "object_count",
    "distractor_count",
    "visual_mode",
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate metadata-first records from the state-refresh / identity-tracking family backend into a stable table."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more JSON/JSONL files or directories.",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=list(DEFAULT_GROUP_BY),
        help="Grouping columns for the output table.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional CSV export path.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Optional annotated_tasks.json override used to enrich metadata.",
    )
    return parser


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _collect_input_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*") if child.suffix.lower() in {".json", ".jsonl"}))
            continue
        if path.is_file():
            files.append(path)
    return files


def _load_payload(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    extracted: list[dict[str, Any]] = []
    _extract_atomic_records(payload, extracted)
    return extracted


def _extract_atomic_records(payload: Any, sink: list[dict[str, Any]]) -> None:
    if isinstance(payload, list):
        for value in payload:
            _extract_atomic_records(value, sink)
        return
    if not isinstance(payload, dict):
        return
    for list_key in ("task_results", "task_summaries"):
        nested = payload.get(list_key)
        if isinstance(nested, list):
            for value in nested:
                _extract_atomic_records(value, sink)
            return
    sink.append(dict(payload))


def _serialize_metadata(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _build_annotation_index(path: str | None) -> dict[tuple[str, int], dict[str, Any]]:
    payload = load_annotated_tasks(path)
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for layout_key, layout in payload.items():
        if not isinstance(layout, dict):
            continue
        scene_structure = layout.get("scene_structure", {})
        object_positions = scene_structure.get("object_positions", {}) if isinstance(scene_structure, dict) else {}
        total_object_count = len(object_positions) if isinstance(object_positions, dict) else None
        tasks = layout.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        for task_index, task_payload in enumerate(tasks):
            if not isinstance(task_payload, dict):
                continue
            annotation = task_payload.get("annotation", {})
            if not isinstance(annotation, dict):
                annotation = {}
            protocol = annotation.get("protocol", {})
            if not isinstance(protocol, dict):
                protocol = {}
            if str(protocol.get("module_name", "")) != "post_action_relocation":
                continue
            semantic_view = normalize_annotation_entry(
                layout_key=layout_key,
                task_index=task_index,
                layout=layout,
                task_payload=task_payload,
            )
            task_name = str(annotation.get("task_id") or f"{layout_key}__task_{task_index}")
            metadata = build_state_refresh_summary_metadata(
                protocol=protocol,
                total_object_count=total_object_count,
                task_name=task_name,
            )
            metadata.update(
                {
                    "task_name": task_name,
                    "canonical_task_id": semantic_view.get("canonical_task_id"),
                    "family": semantic_view.get("family") or metadata.get("family"),
                    "variant": semantic_view.get("variant") or metadata.get("variant"),
                    "task_type": semantic_view.get("task_type") or DISCRETE_SELECTION_TASK_TYPE,
                    "track_name": str(protocol.get("track_name", "")),
                    "core_skill": semantic_view.get("task_semantics", {}).get("core_skill"),
                    "query_semantics": semantic_view.get("task_semantics", {}).get("query_semantics"),
                    "gt_mode": semantic_view.get("task_semantics", {}).get("gt_mode"),
                    "canonical_role": semantic_view.get("task_semantics", {}).get("canonical_role"),
                    "scene_name": semantic_view.get("scene_semantics", {}).get("scene_name"),
                    "scene_capability_tags": semantic_view.get("scene_semantics", {}).get("capability_tags", []),
                    "scene_risk_tags": semantic_view.get("scene_semantics", {}).get("risk_tags", []),
                    "instance_id": semantic_view.get("instance_semantics", {}).get("instance_id"),
                    "instance_layout": semantic_view.get("instance_semantics", {}).get("layout"),
                    "instance_axis_variant": semantic_view.get("instance_semantics", {}).get("axis_variant"),
                    "task_semantics": semantic_view.get("task_semantics", {}),
                    "scene_semantics": semantic_view.get("scene_semantics", {}),
                    "instance_semantics": semantic_view.get("instance_semantics", {}),
                    "diagnostic_tags": semantic_view.get("diagnostic_tags", metadata.get("diagnostic_tags", [])),
                    "difficulty_factors": semantic_view.get("difficulty_factors", metadata.get("difficulty_factors", {})),
                }
            )
            index[(str(layout_key), int(task_index))] = metadata
    return index


def _enrich_record(record: dict[str, Any], annotation_index: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(record)
    layout_key = str(enriched.get("layout_key", ""))
    task_index = _safe_int(enriched.get("task_index"))
    annotation_meta = annotation_index.get((layout_key, int(task_index))) if layout_key and task_index is not None else None
    if annotation_meta is not None:
        for key, value in annotation_meta.items():
            if enriched.get(key) in (None, "", "mixed"):
                enriched[key] = value

    if not enriched.get("task_name"):
        enriched["task_name"] = str(
            enriched.get("task_label")
            or enriched.get("task_id")
            or enriched.get("layout_key")
            or enriched.get("summary_json")
            or "unknown_task"
        )
    if not enriched.get("agent_name"):
        enriched["agent_name"] = str(enriched.get("agent_alias") or enriched.get("agent") or "unknown_agent")
    if not enriched.get("family"):
        enriched["family"] = str(enriched.get("track_name", ""))
    if not enriched.get("visual_mode"):
        enriched["visual_mode"] = "identical_objects"
    if not enriched.get("task_type"):
        enriched["task_type"] = DISCRETE_SELECTION_TASK_TYPE

    if enriched.get("episode_count") in (None, "") and enriched.get("num_episodes") not in (None, ""):
        enriched["episode_count"] = enriched.get("num_episodes")
    if enriched.get("valid_rate") in (None, "") and enriched.get("valid_episode_rate") not in (None, ""):
        enriched["valid_rate"] = enriched.get("valid_episode_rate")
    if (
        enriched.get("intervention_success_rate") in (None, "")
        and enriched.get("intervention_success_count") not in (None, "")
        and _safe_int(enriched.get("episode_count")) not in (None, 0)
    ):
        enriched["intervention_success_rate"] = (
            float(enriched["intervention_success_count"]) / float(enriched["episode_count"])
        )
    if (
        enriched.get("accidental_displacement_rate") in (None, "")
        and enriched.get("accidental_displacement_count") not in (None, "")
        and _safe_int(enriched.get("episode_count")) not in (None, 0)
    ):
        enriched["accidental_displacement_rate"] = (
            float(enriched["accidental_displacement_count"]) / float(enriched["episode_count"])
        )
    enriched = augment_record_with_semantics(enriched)
    for field_name in (
        "enabled_probes",
        "stress_factors",
        "diagnostic_tags",
        "difficulty_factors",
        "task_metadata",
        "scene_capability_tags",
        "scene_risk_tags",
        "task_semantics",
        "scene_semantics",
        "instance_semantics",
    ):
        value = serialize_semantic_payload(enriched.get(field_name))
        if value is not None:
            enriched[field_name] = value
    return enriched


class ScalarAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0

    def add_value(self, value: float) -> None:
        self.count += 1
        self.total += float(value)

    def mean(self) -> float | None:
        if self.count <= 0:
            return None
        return self.total / float(self.count)


class GroupAccumulator:
    def __init__(self, group_values: dict[str, Any]) -> None:
        self.group_values = dict(group_values)
        self.episode_count = 0
        self.valid_count = 0
        self.intervention_success_count = 0
        self.success_count = 0
        self.accidental_displacement_count = 0
        self.final_frame_only_gap = ScalarAccumulator()
        self.historical_tracking_gap = ScalarAccumulator()
        self.clutter_gap = ScalarAccumulator()
        self.diagnostic_tags: set[str] = set()
        self.difficulty_factors: set[str] = set()

    def add_record(self, record: dict[str, Any]) -> None:
        diagnostic_tags = _serialize_metadata(record.get("diagnostic_tags"))
        if diagnostic_tags is not None:
            self.diagnostic_tags.add(diagnostic_tags)
        difficulty_factors = _serialize_metadata(record.get("difficulty_factors"))
        if difficulty_factors is not None:
            self.difficulty_factors.add(difficulty_factors)

        is_episode = "episode_index" in record or ("selected_object_index" in record and "failure_category" in record)

        if is_episode:
            self.episode_count += 1
            self.valid_count += int(bool(record.get("valid_episode")))
            self.intervention_success_count += int(bool(record.get("intervention_success")))
            self.success_count += int(bool(record.get("success")))
            self.accidental_displacement_count += int(bool(record.get("accidental_displacement")))
            self._maybe_add_gap_values(record)
            return

        episode_count = _safe_int(record.get("episode_count"))
        if episode_count is None:
            episode_count = _safe_int(record.get("num_episodes"))
        if episode_count is None:
            episode_count = 0
        self.episode_count += int(episode_count)

        valid_count = _safe_int(record.get("valid_episode_count"))
        if valid_count is None:
            valid_rate = _safe_float(record.get("valid_rate"))
            if valid_rate is None:
                valid_rate = _safe_float(record.get("valid_episode_rate"))
            valid_count = int(round(float(valid_rate) * float(episode_count))) if valid_rate is not None else 0
        self.valid_count += int(valid_count)

        intervention_success_count = _safe_int(record.get("intervention_success_count"))
        if intervention_success_count is None:
            intervention_success_rate = _safe_float(record.get("intervention_success_rate"))
            intervention_success_count = (
                int(round(float(intervention_success_rate) * float(episode_count)))
                if intervention_success_rate is not None
                else 0
            )
        self.intervention_success_count += int(intervention_success_count)

        success_count = _safe_int(record.get("success_count"))
        if success_count is None:
            success_rate = _safe_float(record.get("success_rate"))
            success_count = int(round(float(success_rate) * float(episode_count))) if success_rate is not None else 0
        self.success_count += int(success_count)

        accidental_displacement_count = _safe_int(record.get("accidental_displacement_count"))
        if accidental_displacement_count is None:
            accidental_displacement_rate = _safe_float(record.get("accidental_displacement_rate"))
            accidental_displacement_count = (
                int(round(float(accidental_displacement_rate) * float(episode_count)))
                if accidental_displacement_rate is not None
                else 0
            )
        self.accidental_displacement_count += int(accidental_displacement_count)
        self._maybe_add_gap_values(record)

    def _maybe_add_gap_values(self, record: dict[str, Any]) -> None:
        for field_name, accumulator in (
            ("final_frame_only_gap", self.final_frame_only_gap),
            ("historical_tracking_gap", self.historical_tracking_gap),
            ("clutter_gap", self.clutter_gap),
        ):
            value = _safe_float(record.get(field_name))
            if value is not None:
                accumulator.add_value(value)

    def to_row(self) -> dict[str, Any]:
        row = dict(self.group_values)
        row["episode_count"] = int(self.episode_count)
        row["valid_rate"] = float(self.valid_count) / float(self.episode_count) if self.episode_count > 0 else None
        row["intervention_success_rate"] = (
            float(self.intervention_success_count) / float(self.episode_count) if self.episode_count > 0 else None
        )
        row["success_rate"] = (
            float(self.success_count) / float(self.episode_count) if self.episode_count > 0 else None
        )
        row["accidental_displacement_rate"] = (
            float(self.accidental_displacement_count) / float(self.episode_count) if self.episode_count > 0 else None
        )
        for field_name, accumulator in (
            ("final_frame_only_gap", self.final_frame_only_gap),
            ("historical_tracking_gap", self.historical_tracking_gap),
            ("clutter_gap", self.clutter_gap),
        ):
            if accumulator.count > 0:
                row[field_name] = accumulator.mean()
        if self.diagnostic_tags:
            row["diagnostic_tags"] = " | ".join(sorted(self.diagnostic_tags))
        if self.difficulty_factors:
            row["difficulty_factors"] = " | ".join(sorted(self.difficulty_factors))
        return row


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        print("No state-refresh family backend records found.")
        return

    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(_format_cell(row.get(column))))

    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    print(header)
    print(divider)
    for row in rows:
        print("  ".join(_format_cell(row.get(column)).ljust(widths[column]) for column in columns))


def summarize_records(
    *,
    records: list[dict[str, Any]],
    group_by: list[str],
    annotation_index: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], GroupAccumulator] = {}
    for raw_record in records:
        record = _enrich_record(raw_record, annotation_index)
        group_values = {column: record.get(column) for column in group_by}
        group_key = tuple(group_values.get(column) for column in group_by)
        accumulator = groups.get(group_key)
        if accumulator is None:
            accumulator = GroupAccumulator(group_values)
            groups[group_key] = accumulator
        accumulator.add_record(record)

    rows = [
        groups[key].to_row()
        for key in sorted(groups.keys(), key=lambda values: tuple("" if value is None else str(value) for value in values))
    ]
    summarized_rows: list[dict[str, Any]] = []
    for row in rows:
        summarized_rows.append(_enrich_record(row, annotation_index))
    return summarized_rows


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def main() -> None:
    args = build_argparser().parse_args()
    files = _collect_input_files([str(value) for value in args.inputs])
    annotation_index = _build_annotation_index(args.annotations)

    records: list[dict[str, Any]] = []
    for path in files:
        records.extend(_load_payload(path))

    rows = summarize_records(
        records=records,
        group_by=[str(value) for value in args.group_by],
        annotation_index=annotation_index,
    )
    metric_columns = [
        "raw_task_name",
        "canonical_task_id",
        "task_id",
        "task_type",
        "semantic_module",
        "task_version",
        "skill_type",
        "phase",
        "parent_task_id",
        "paired_control_id",
        "evaluator_key",
        "core_skill",
        "query_semantics",
        "gt_mode",
        "canonical_role",
        "enabled_probes",
        "stress_factors",
        "notes",
        "scene_name",
        "scene_capability_tags",
        "scene_risk_tags",
        "instance_id",
        "instance_layout",
        "instance_axis_variant",
        "episode_count",
        "valid_rate",
        "intervention_success_rate",
        "success_rate",
        "accidental_displacement_rate",
        "final_frame_only_gap",
        "historical_tracking_gap",
        "clutter_gap",
        "task_metadata",
        "task_semantics",
        "scene_semantics",
        "instance_semantics",
        "diagnostic_tags",
        "difficulty_factors",
    ]
    columns = [
        column
        for column in list(args.group_by) + metric_columns
        if any(column in row for row in rows) or column in args.group_by
    ]
    _print_table(rows, columns)

    if args.csv_out:
        write_csv(args.csv_out, rows, columns)


if __name__ == "__main__":
    main()
