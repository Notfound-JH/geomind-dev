#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.mimic.utils.joint_summary_utils import CONTINUOUS_PLACEMENT_TASK_TYPE
from geomind_lab.mimic.utils.config_semantics_adapter import (
    GEOMETRIC_INFERENCE_MODULE_NAME,
    augment_record_with_semantics,
    normalize_annotation_entry,
    serialize_semantic_payload,
)
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import load_annotated_tasks


DEFAULT_GROUP_BY = ("semantic_module", "task_id", "task_version", "skill_type", "phase", "agent_name", "noise_level", "anchor_count")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate metadata-first records from the geometric structure-inference family backend into a stable table."
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
    task_results = payload.get("task_results")
    if isinstance(task_results, list):
        for task_result in task_results:
            _extract_atomic_records(task_result, sink)
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
            if str(protocol.get("module_name", "")) != GEOMETRIC_INFERENCE_MODULE_NAME:
                continue
            gt_spec = protocol.get("gt_spec", {})
            if not isinstance(gt_spec, dict):
                gt_spec = {}
            semantic_view = normalize_annotation_entry(
                layout_key=layout_key,
                task_index=task_index,
                layout=layout,
                task_payload=task_payload,
            )
            anchor_count = _safe_int(protocol.get("anchor_count"))
            anchors = protocol.get("anchors", [])
            if anchor_count is None and isinstance(anchors, list):
                anchor_count = len(anchors)
            task_name = annotation.get("task_id") or f"{layout_key}__task_{task_index}"
            index[(str(layout_key), int(task_index))] = {
                "task_name": str(task_name),
                "canonical_task_id": semantic_view.get("canonical_task_id"),
                "family": str(semantic_view.get("family") or protocol.get("family", protocol.get("track_name", ""))),
                "variant": str(
                    semantic_view.get("variant")
                    or protocol.get("variant", protocol.get("subtask_name", gt_spec.get("rule_type", "")))
                ),
                "noise_level": str(protocol.get("noise_level", "")),
                "anchor_count": anchor_count,
                "anchor_layout": str(protocol.get("anchor_layout", "")),
                "spacing": protocol.get("spacing"),
                "track_name": str(protocol.get("track_name", "")),
                "rule_type": str(gt_spec.get("rule_type", "")),
                "task_type": semantic_view.get("task_type") or CONTINUOUS_PLACEMENT_TASK_TYPE,
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
                "diagnostic_tags": semantic_view.get("diagnostic_tags", protocol.get("diagnostic_tags", [])),
                "difficulty_factors": semantic_view.get("difficulty_factors", protocol.get("difficulty_factors", {})),
            }
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
    if not enriched.get("family"):
        enriched["family"] = str(enriched.get("track_name", ""))
    if not enriched.get("variant"):
        enriched["variant"] = str(enriched.get("subtask_name") or enriched.get("rule_type", ""))
    if not enriched.get("agent_name"):
        enriched["agent_name"] = str(enriched.get("agent", ""))
    if not enriched.get("task_type"):
        enriched["task_type"] = CONTINUOUS_PLACEMENT_TASK_TYPE
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


def _distribution_to_moments(*, mean: float, std: float | None, count: int) -> tuple[float, float]:
    variance = 0.0 if std is None else float(std) ** 2
    total = float(mean) * float(count)
    total_sq = (variance + float(mean) ** 2) * float(count)
    return (total, total_sq)


class ScalarAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0

    def add_value(self, value: float) -> None:
        self.count += 1
        self.total += float(value)
        self.total_sq += float(value) * float(value)

    def add_distribution(self, *, mean: float, std: float | None, count: int) -> None:
        if int(count) <= 0:
            return
        total, total_sq = _distribution_to_moments(mean=float(mean), std=std, count=int(count))
        self.count += int(count)
        self.total += total
        self.total_sq += total_sq

    def mean(self) -> float | None:
        if self.count <= 0:
            return None
        return self.total / float(self.count)

    def std(self) -> float | None:
        if self.count <= 0:
            return None
        mean = self.mean()
        if mean is None:
            return None
        variance = max(self.total_sq / float(self.count) - float(mean) ** 2, 0.0)
        return math.sqrt(variance)


class GroupAccumulator:
    def __init__(self, group_values: dict[str, Any]) -> None:
        self.group_values = dict(group_values)
        self.episode_count = 0
        self.valid_count = 0
        self.success_count = 0
        self.direction_correct_count = 0
        self.direction_correct_denom = 0
        self.trend_consistent_count = 0
        self.trend_consistent_denom = 0
        self.l2_error = ScalarAccumulator()
        self.orthogonal_distance = ScalarAccumulator()
        self.between_score = ScalarAccumulator()
        self.diagnostic_tags: set[str] = set()
        self.difficulty_factors: set[str] = set()

    def add_record(self, record: dict[str, Any]) -> None:
        diagnostic_tags = _serialize_metadata(record.get("diagnostic_tags"))
        if diagnostic_tags is not None:
            self.diagnostic_tags.add(diagnostic_tags)
        difficulty_factors = _serialize_metadata(record.get("difficulty_factors"))
        if difficulty_factors is not None:
            self.difficulty_factors.add(difficulty_factors)

        is_episode = "episode_index" in record or "pred_pos_xy" in record

        if is_episode:
            self.episode_count += 1
            if _safe_bool(record.get("valid_episode")):
                self.valid_count += 1
            if _safe_bool(record.get("success")):
                self.success_count += 1

            l2_error = _safe_float(record.get("l2_error"))
            if l2_error is not None and _safe_bool(record.get("valid_episode")):
                self.l2_error.add_value(l2_error)
            orthogonal_distance = _safe_float(record.get("orthogonal_distance"))
            if orthogonal_distance is not None and _safe_bool(record.get("valid_episode")):
                self.orthogonal_distance.add_value(orthogonal_distance)
            between_score = _safe_float(record.get("between_score"))
            if between_score is not None and _safe_bool(record.get("valid_episode")):
                self.between_score.add_value(between_score)

            direction_correct = _safe_bool(record.get("direction_correct"))
            if direction_correct is not None:
                self.direction_correct_denom += 1
                self.direction_correct_count += int(direction_correct)

            trend_consistent = _safe_bool(record.get("trend_consistent"))
            if trend_consistent is not None:
                self.trend_consistent_denom += 1
                self.trend_consistent_count += int(trend_consistent)
            return

        episode_count = _safe_int(record.get("episode_count"))
        if episode_count is None:
            episode_count = _safe_int(record.get("num_episodes"))
        if episode_count is None:
            episode_count = 0
        self.episode_count += int(episode_count)

        valid_count = _safe_int(record.get("valid_episode_count"))
        if valid_count is None:
            valid_rate = _safe_float(record.get("valid_episode_rate"))
            valid_count = int(round(float(valid_rate) * float(episode_count))) if valid_rate is not None else 0
        self.valid_count += int(valid_count)

        success_count = _safe_int(record.get("success_count"))
        if success_count is None:
            success_rate = _safe_float(record.get("success_rate"))
            success_count = int(round(float(success_rate) * float(episode_count))) if success_rate is not None else 0
        self.success_count += int(success_count)

        metric_count = _safe_int(record.get("metric_episode_count"))
        if metric_count is None:
            metric_count = int(valid_count)

        mean_l2 = _safe_float(record.get("l2_error_mean"))
        std_l2 = _safe_float(record.get("l2_error_std"))
        if mean_l2 is None:
            mean_l2 = _safe_float(record.get("mean_l2_error"))
            std_l2 = _safe_float(record.get("std_l2_error"))
        if mean_l2 is not None and int(metric_count) > 0:
            self.l2_error.add_distribution(mean=mean_l2, std=std_l2, count=int(metric_count))

        mean_orthogonal = _safe_float(record.get("orthogonal_distance_mean"))
        std_orthogonal = _safe_float(record.get("orthogonal_distance_std"))
        if mean_orthogonal is None:
            mean_orthogonal = _safe_float(record.get("mean_orthogonal_distance"))
            std_orthogonal = _safe_float(record.get("std_orthogonal_distance"))
        if mean_orthogonal is not None and int(metric_count) > 0:
            self.orthogonal_distance.add_distribution(
                mean=mean_orthogonal,
                std=std_orthogonal,
                count=int(metric_count),
            )

        mean_between = _safe_float(record.get("between_score_mean"))
        std_between = _safe_float(record.get("between_score_std"))
        if mean_between is None:
            mean_between = _safe_float(record.get("mean_between_score"))
            std_between = _safe_float(record.get("std_between_score"))
        if mean_between is not None and int(metric_count) > 0:
            self.between_score.add_distribution(mean=mean_between, std=std_between, count=int(metric_count))

        direction_count = _safe_int(record.get("direction_correct_count"))
        direction_rate = _safe_float(record.get("direction_correct_rate"))
        if direction_count is not None:
            denom = int(episode_count) if int(episode_count) > 0 else int(direction_count)
            self.direction_correct_count += int(direction_count)
            self.direction_correct_denom += int(denom)
        elif direction_rate is not None and int(episode_count) > 0:
            self.direction_correct_count += int(round(float(direction_rate) * float(episode_count)))
            self.direction_correct_denom += int(episode_count)

        trend_count = _safe_int(record.get("trend_consistent_count"))
        trend_rate = _safe_float(record.get("trend_consistent_rate"))
        if trend_count is not None:
            denom = int(episode_count) if int(episode_count) > 0 else int(trend_count)
            self.trend_consistent_count += int(trend_count)
            self.trend_consistent_denom += int(denom)
        elif trend_rate is not None and int(episode_count) > 0:
            self.trend_consistent_count += int(round(float(trend_rate) * float(episode_count)))
            self.trend_consistent_denom += int(episode_count)

    def to_row(self) -> dict[str, Any]:
        row = dict(self.group_values)
        row["episode_count"] = int(self.episode_count)
        row["valid_rate"] = (
            float(self.valid_count) / float(self.episode_count) if self.episode_count > 0 else None
        )
        row["success_rate"] = (
            float(self.success_count) / float(self.episode_count) if self.episode_count > 0 else None
        )
        row["mean_l2_error"] = self.l2_error.mean()
        row["std_l2_error"] = self.l2_error.std()
        row["mean_orthogonal_distance"] = self.orthogonal_distance.mean()
        row["std_orthogonal_distance"] = self.orthogonal_distance.std()
        if self.between_score.count > 0:
            row["mean_between_score"] = self.between_score.mean()
        if self.direction_correct_denom > 0:
            row["direction_correct_rate"] = float(self.direction_correct_count) / float(self.direction_correct_denom)
        if self.trend_consistent_denom > 0:
            row["trend_consistent_rate"] = float(self.trend_consistent_count) / float(self.trend_consistent_denom)
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
        print("No geometric structure-inference backend records found.")
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

    rows = [groups[key].to_row() for key in sorted(groups.keys(), key=lambda values: tuple("" if value is None else str(value) for value in values))]
    return [_enrich_record(row, annotation_index) for row in rows]


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
        "success_rate",
        "mean_l2_error",
        "std_l2_error",
        "mean_orthogonal_distance",
        "std_orthogonal_distance",
        "mean_between_score",
        "direction_correct_rate",
        "trend_consistent_rate",
        "task_metadata",
        "task_semantics",
        "scene_semantics",
        "instance_semantics",
        "diagnostic_tags",
        "difficulty_factors",
    ]
    columns = [column for column in list(args.group_by) + metric_columns if any(column in row for row in rows) or column in args.group_by]
    _print_table(rows, columns)

    if args.csv_out:
        write_csv(args.csv_out, rows, columns)


if __name__ == "__main__":
    main()
