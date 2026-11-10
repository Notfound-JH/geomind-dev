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

from geomind_lab.mimic.utils.joint_summary_utils import normalize_joint_row  # noqa: E402


DEFAULT_GROUP_BY = ("semantic_module", "task_id", "task_version", "skill_type", "phase", "agent_name", "task_type")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a metadata-first joint GeoMind benchmark summary across family backend outputs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more benchmark output directories or summary files (CSV/JSON/JSONL).",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=list(DEFAULT_GROUP_BY),
        help="Grouping columns for the joint output table.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional CSV export path.",
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


def _collect_input_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            preferred = []
            for candidate in (
                path / "aggregate_state_refresh.csv",
                path / "aggregate_geometric_structure.csv",
                path / "state_refresh" / "aggregate.csv",
                path / "geometric_structure" / "aggregate.csv",
            ):
                if candidate.is_file():
                    preferred.append(candidate)
            if preferred:
                files.extend(preferred)
                continue
            files.extend(
                sorted(child for child in path.rglob("*") if child.suffix.lower() in {".csv", ".json", ".jsonl"})
            )
            continue
        if path.is_file():
            files.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        deduped.append(path)
        seen.add(resolved)
    return deduped


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


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


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    _extract_atomic_records(payload, rows)
    return rows


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_rows(path)
    if suffix in {".json", ".jsonl"}:
        return _load_json_rows(path)
    return []


class WeightedScalarAccumulator:
    def __init__(self) -> None:
        self.total_weight = 0.0
        self.total_value = 0.0

    def add(self, value: float, *, weight: float) -> None:
        if weight <= 0:
            return
        self.total_weight += float(weight)
        self.total_value += float(value) * float(weight)

    def mean(self) -> float | None:
        if self.total_weight <= 0.0:
            return None
        return self.total_value / self.total_weight


class GroupAccumulator:
    def __init__(self, group_values: dict[str, Any]) -> None:
        self.group_values = dict(group_values)
        self.episode_count = 0
        self.valid_total = 0.0
        self.success_total = 0.0
        self.intervention_success_total = 0.0
        self.accidental_displacement_total = 0.0
        self.has_intervention_success = False
        self.has_accidental_displacement = False
        self.mean_l2_error = WeightedScalarAccumulator()
        self.mean_orthogonal_distance = WeightedScalarAccumulator()
        self.direction_correct_rate = WeightedScalarAccumulator()
        self.trend_consistent_rate = WeightedScalarAccumulator()
        self.diagnostic_tags: set[str] = set()
        self.difficulty_factors: set[str] = set()

    def add_row(self, row: dict[str, Any]) -> None:
        episode_count = _safe_int(row.get("episode_count")) or 0
        valid_rate = _safe_float(row.get("valid_rate"))
        success_rate = _safe_float(row.get("success_rate"))
        intervention_success_rate = _safe_float(row.get("intervention_success_rate"))
        accidental_displacement_rate = _safe_float(row.get("accidental_displacement_rate"))

        self.episode_count += int(episode_count)
        if valid_rate is not None:
            self.valid_total += float(valid_rate) * float(episode_count)
        if success_rate is not None:
            self.success_total += float(success_rate) * float(episode_count)
        if intervention_success_rate is not None:
            self.has_intervention_success = True
            self.intervention_success_total += float(intervention_success_rate) * float(episode_count)
        if accidental_displacement_rate is not None:
            self.has_accidental_displacement = True
            self.accidental_displacement_total += float(accidental_displacement_rate) * float(episode_count)

        metric_weight = float(episode_count)
        valid_weight = float(valid_rate) * float(episode_count) if valid_rate is not None else float(episode_count)
        for field_name, accumulator, weight in (
            ("mean_l2_error", self.mean_l2_error, valid_weight),
            ("mean_orthogonal_distance", self.mean_orthogonal_distance, valid_weight),
            ("direction_correct_rate", self.direction_correct_rate, metric_weight),
            ("trend_consistent_rate", self.trend_consistent_rate, metric_weight),
        ):
            value = _safe_float(row.get(field_name))
            if value is not None:
                accumulator.add(value, weight=weight)

        if row.get("diagnostic_tags") not in (None, ""):
            self.diagnostic_tags.add(str(row["diagnostic_tags"]))
        if row.get("difficulty_factors") not in (None, ""):
            self.difficulty_factors.add(str(row["difficulty_factors"]))

    def to_row(self) -> dict[str, Any]:
        row = dict(self.group_values)
        episode_count = int(self.episode_count)
        row["episode_count"] = episode_count
        row["valid_rate"] = self.valid_total / float(episode_count) if episode_count > 0 else None
        row["success_rate"] = self.success_total / float(episode_count) if episode_count > 0 else None
        if self.has_intervention_success:
            row["intervention_success_rate"] = (
                self.intervention_success_total / float(episode_count) if episode_count > 0 else None
            )
        if self.has_accidental_displacement:
            row["accidental_displacement_rate"] = (
                self.accidental_displacement_total / float(episode_count) if episode_count > 0 else None
            )

        for field_name, accumulator in (
            ("mean_l2_error", self.mean_l2_error),
            ("mean_orthogonal_distance", self.mean_orthogonal_distance),
            ("direction_correct_rate", self.direction_correct_rate),
            ("trend_consistent_rate", self.trend_consistent_rate),
        ):
            value = accumulator.mean()
            if value is not None:
                row[field_name] = value

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
        print("No metadata-enriched GeoMind benchmark rows found.")
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


def summarize_rows(rows: list[dict[str, Any]], group_by: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], GroupAccumulator] = {}
    for raw_row in rows:
        normalized = normalize_joint_row(raw_row)
        group_values = {column: normalized.get(column) for column in group_by}
        group_key = tuple(group_values.get(column) for column in group_by)
        accumulator = groups.get(group_key)
        if accumulator is None:
            accumulator = GroupAccumulator(group_values)
            groups[group_key] = accumulator
        accumulator.add_row(normalized)
    return [
        normalize_joint_row(groups[key].to_row())
        for key in sorted(groups.keys(), key=lambda values: tuple("" if value is None else str(value) for value in values))
    ]


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
    input_files = _collect_input_files([str(value) for value in args.inputs])

    raw_rows: list[dict[str, Any]] = []
    for path in input_files:
        raw_rows.extend(_load_records(path))

    rows = summarize_rows(raw_rows, group_by=[str(value) for value in args.group_by])
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
        "intervention_success_rate",
        "accidental_displacement_rate",
        "mean_l2_error",
        "mean_orthogonal_distance",
        "direction_correct_rate",
        "trend_consistent_rate",
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
