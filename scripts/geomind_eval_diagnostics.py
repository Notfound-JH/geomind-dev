#!/usr/bin/env python3
"""
GeoMind 评估日志诊断指标：SGE、SCS、MTA（不替代 success rate，用于解释失败来源）。

Spatial Goal Error (SGE)：episode 结束时末端或物体相对任务空间目标的误差。
Spatial Constraint Satisfaction (SCS)：reasoning-heavy 任务上空间谓词满足比例。
Memory Target Accuracy (MTA)：memory-related 任务上最终操作/选择对象是否与更新后 GT 目标一致。

依赖：标准库、numpy、pandas。不修改 trial_results.jsonl。

默认会从 ``geomind/configs/legacy/layouts/standard.json`` 与 ``geomind/configs/legacy/layouts/vla.json``
合并读取布局：按 ``result.json`` 的 ``annotation_layout_key`` / ``layout_ref`` 解析 ``protocol.gt_spec``、
目标物体 id 与 ``scene_structure`` / ``runtime_objects`` 的参考位姿，写入 ``gt_target_object_id``、``gt_target_pos``
等（不覆盖日志里已有字段）。可用 ``--no-layout-enrich`` 关闭。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 显式阈值与坐标约定（与 GeoMind / IsaacLab 工作区一致时再调整）
# ---------------------------------------------------------------------------
RELATION_MARGIN = 0.02  # m，left_of / right_of 等沿轴分离裕量
NEAR_THRESHOLD = 0.05  # m
ALIGNMENT_THRESHOLD = 0.03  # m，点到参考直线的距离
BETWEEN_THRESHOLD = 0.04  # m，到 anchor 线段的横向距离上限
SYMMETRY_THRESHOLD = 0.05  # m，与期望对称点的偏差
INSIDE_MARGIN = 0.0  # m，inside 判定时可向内收缩边界

# FRONT_AXIS: 0=x, 1=y, 2=z；FRONT_SIGN: 沿该轴“前”为正方向时为 +1，否则 -1
FRONT_AXIS = 1
FRONT_SIGN = 1.0

# 任务族默认：与 current_task_mapping 中未显式标记时的行为一致
REASONING_HEAVY_FAMILIES = frozenset(
    {
        "Relational Grounding",
        "Constraint Composition",
        "Geometric Inference",
    }
)
MEMORY_RELATED_FAMILIES = frozenset(
    {
        "Scene Updating",
        "Identity Tracking",
    }
)

# 额外将 Reindex / Track 类任务视为 memory-related（MTA 适用）
_TRACK_TASK_ID_RE = re.compile(r"^Track", re.IGNORECASE)
_REINDEX_SCENE_RE = re.compile(r"^Reindex", re.IGNORECASE)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MAPPING = _REPO_ROOT / "geomind" / "configs" / "current_task_mapping.json"
_DEFAULT_TASK_CONFIG_DIR = _REPO_ROOT / "geomind" / "configs" / "tasks"
# 合并顺序：后加载文件中的同名 layout_key 覆盖前者（通常 VLA 布局更贴近 eval 日志）
_DEFAULT_LAYOUT_JSONS: tuple[Path, ...] = (
    _REPO_ROOT / "geomind" / "configs" / "legacy" / "layouts" / "standard.json",
    _REPO_ROOT / "geomind" / "configs" / "legacy" / "layouts" / "vla.json",
)
# 布局里多为桌面 xy；无 z 时用该参考高度（米）拼成 3D，仅作诊断近似
DEFAULT_TABLETOP_OBJECT_Z_M = 0.784


def load_results(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = jsonl_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            rows.append({"_parse_error": str(exc), "_raw_line": line[:200]})
    return rows


def load_task_mapping(mapping_path: Path) -> dict[str, Any]:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return {"tasks": [], "raw": data}
    return {"tasks": tasks, "raw": data}


def _mapping_by_task_id(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in tasks:
        tid = str(t.get("task_id", "")).strip()
        if tid:
            out[tid] = t
    return out


def normalize_task_id_for_mapping(task_id: str) -> str:
    """VLA 日志中的 …Vla_v0 与 mapping 中 …_v0 对齐。"""
    return re.sub(r"Vla_v(\d+)$", r"_v\1", str(task_id).strip())


def infer_task_metadata(
    episode: dict[str, Any],
    task_mapping: dict[str, Any],
) -> dict[str, Any]:
    """合并 taxonomy、suite、增广轴及 reasoning / memory 标志。"""
    by_id = _mapping_by_task_id(list(task_mapping.get("tasks") or []))
    raw_tid = str(episode.get("task_id") or episode.get("task_name") or "").strip()
    norm = normalize_task_id_for_mapping(raw_tid)
    row = by_id.get(norm) or by_id.get(raw_tid) or {}

    taxonomy_family = str(row.get("taxonomy_family") or "").strip()
    suite_raw = str(row.get("suite") or episode.get("suite") or "").strip().lower()
    if suite_raw in {"augmented", "aug"}:
        suite_label = "Aug"
    elif suite_raw == "base":
        suite_label = "Base"
    elif not suite_raw and not row:
        suite_label = "Base"
    elif row.get("suite") == "augmented":
        suite_label = "Aug"
    elif row.get("suite") == "base":
        suite_label = "Base"
    else:
        suite_label = "Base"

    aug = str(row.get("augmentation_axis") or episode.get("augmentation_type") or "").strip()
    if aug == "none":
        aug = ""

    dim = str(row.get("dimension") or "").strip().lower()
    reasoning_heavy = bool(row.get("reasoning_heavy")) or (taxonomy_family in REASONING_HEAVY_FAMILIES)
    tid_for_pat = raw_tid or norm
    memory_related = bool(row.get("memory_related")) or (taxonomy_family in MEMORY_RELATED_FAMILIES) or (dim == "spatial_memory")
    # Reindex 中仅 Scene Updating 族视为 scene-update / memory；Track* 视为 identity tracking。
    if not memory_related and tid_for_pat:
        if _TRACK_TASK_ID_RE.match(tid_for_pat):
            memory_related = True
        elif _REINDEX_SCENE_RE.match(tid_for_pat) and taxonomy_family == "Scene Updating":
            memory_related = True

    return {
        "task_id_raw": raw_tid,
        "task_id_mapping_key": norm,
        "task_family": taxonomy_family or str(episode.get("task_family") or episode.get("family") or ""),
        "suite": suite_label,
        "augmentation_type": aug,
        "taxonomy_row": row,
        "reasoning_heavy": reasoning_heavy,
        "memory_related": memory_related,
    }


def inspect_schema(results: list[dict[str, Any]], sample: int = 200) -> dict[str, Any]:
    """汇总若干行 jsonl 的键并分类计数。"""
    key_counts: Counter[str] = Counter()
    type_hints: dict[str, Counter[str]] = {}
    n = min(len(results), sample)
    for rec in results[:n]:
        if not isinstance(rec, dict):
            continue
        for k, v in rec.items():
            key_counts[k] += 1
            t = type(v).__name__
            type_hints.setdefault(k, Counter())[t] += 1
    return {
        "num_records_sampled": n,
        "key_frequency": dict(key_counts.most_common()),
        "key_types": {k: dict(v) for k, v in sorted(type_hints.items())},
    }


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return None


def merge_episode_with_result_json(episode: dict[str, Any]) -> dict[str, Any]:
    """浅合并 result.json（不读 trace，除非调用方另行处理）。"""
    merged = dict(episode)
    rj = episode.get("result_json")
    if isinstance(rj, str) and rj.strip():
        extra = _safe_read_json(Path(rj))
        if isinstance(extra, dict):
            merged["_result_json"] = extra
            # 常用嵌套扁平化前缀，便于探测
            ev = extra.get("evaluation")
            if isinstance(ev, dict):
                merged["_evaluation_flat"] = {f"evaluation.{k}": v for k, v in ev.items()}
            eb = extra.get("execution_binding")
            if isinstance(eb, dict):
                merged["_execution_binding_flat"] = {f"execution_binding.{k}": v for k, v in eb.items()}
    return merged


def load_layout_bundle(layout_paths: Iterable[Path] | None = None) -> dict[str, Any]:
    """合并多份 legacy_layouts（顶层 key = layout_key）。"""
    merged: dict[str, Any] = {}
    paths = list(layout_paths) if layout_paths is not None else list(_DEFAULT_LAYOUT_JSONS)
    for p in paths:
        data = _safe_read_json(p)
        if isinstance(data, dict):
            merged.update(data)
    return merged


def _candidate_layout_keys(layout_key: str) -> list[str]:
    k = str(layout_key or "").strip()
    if not k:
        return []
    out: list[str] = [k]
    if k.endswith("_vla"):
        out.append(k[:-4])
    else:
        out.append(f"{k}_vla")
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _resolve_layout_entry(layout_bundle: dict[str, Any], layout_key: str) -> tuple[str | None, dict[str, Any] | None]:
    for cand in _candidate_layout_keys(layout_key):
        entry = layout_bundle.get(cand)
        if isinstance(entry, dict) and entry.get("tasks") is not None:
            return cand, entry
    return None, None


def _annotation_task_index(episode: dict[str, Any], task_config: dict[str, Any] | None) -> int:
    res = episode.get("_result_json")
    if isinstance(res, dict):
        if isinstance(res.get("annotation_task_index"), int):
            return int(res["annotation_task_index"])
        lr = res.get("layout_ref")
        if isinstance(lr, dict) and isinstance(lr.get("task_index"), int):
            return int(lr["task_index"])
    if isinstance(task_config, dict) and isinstance(task_config.get("annotation_task_index"), int):
        return int(task_config["annotation_task_index"])
    return 0


def _episode_layout_key(episode: dict[str, Any], task_config: dict[str, Any] | None) -> str:
    res = episode.get("_result_json")
    if isinstance(res, dict):
        k = str(res.get("annotation_layout_key") or "").strip()
        if k:
            return k
        lr = res.get("layout_ref")
        if isinstance(lr, dict):
            k2 = str(lr.get("layout_key") or "").strip()
            if k2:
                return k2
    if isinstance(task_config, dict):
        k3 = str(task_config.get("annotation_layout_key") or "").strip()
        if k3:
            return k3
    return ""


def _vec3_from_xy_or_xyz(
    pos: Any,
    default_z: float,
) -> np.ndarray | None:
    if isinstance(pos, dict):
        if "xy" in pos and isinstance(pos["xy"], (list, tuple)) and len(pos["xy"]) >= 2:
            z = float(pos.get("z", default_z))
            return np.array([float(pos["xy"][0]), float(pos["xy"][1]), z], dtype=np.float64)
        if "position" in pos:
            return _vec3_from_xy_or_xyz(pos["position"], default_z)
        return None
    if isinstance(pos, (list, tuple)):
        if len(pos) >= 3:
            try:
                return np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
            except (TypeError, ValueError):
                return None
        if len(pos) >= 2:
            try:
                return np.array([float(pos[0]), float(pos[1]), float(default_z)], dtype=np.float64)
            except (TypeError, ValueError):
                return None
    return None


def _positions_from_layout_entry(
    layout_entry: dict[str, Any],
    task_payload: dict[str, Any],
    default_z: float,
) -> dict[str, np.ndarray]:
    """scene_structure.object_positions + 当前 task 的 scene_adaptation.runtime_objects.pose。"""
    poses: dict[str, np.ndarray] = {}
    scene = layout_entry.get("scene_structure")
    if isinstance(scene, dict):
        obj_pos = scene.get("object_positions")
        if isinstance(obj_pos, dict):
            for oid, val in obj_pos.items():
                if isinstance(val, dict) and "position" in val:
                    v = _vec3_from_xy_or_xyz(val["position"], default_z)
                else:
                    v = _vec3_from_xy_or_xyz(val, default_z)
                if v is not None:
                    poses[str(oid)] = v
    sa = task_payload.get("scene_adaptation")
    if isinstance(sa, dict):
        for obj in sa.get("runtime_objects") or []:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name") or "").strip()
            if not name:
                continue
            pose = obj.get("pose")
            if isinstance(pose, dict):
                v = _vec3_from_xy_or_xyz(pose, default_z)
                if v is not None:
                    poses[name] = v
    return poses


def _infer_gt_target_object_id_from_annotation(annotation: dict[str, Any]) -> str | None:
    proto = annotation.get("protocol")
    if not isinstance(proto, dict):
        return None
    gt = proto.get("gt_spec")
    if isinstance(gt, dict):
        for key in ("target_object_id", "expected_target"):
            v = gt.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    to = proto.get("target_object")
    if isinstance(to, dict):
        oid = to.get("object_id")
        if isinstance(oid, str) and oid.strip():
            return oid.strip()
    targets = annotation.get("targets")
    if isinstance(targets, list) and targets:
        t0 = targets[0]
        if isinstance(t0, str) and t0.strip():
            return t0.strip()
    return None


def enrich_episode_from_layout_bundle(
    episode: dict[str, Any],
    task_config: dict[str, Any] | None,
    layout_bundle: dict[str, Any],
    *,
    default_z: float | None = None,
) -> None:
    """
    根据 annotation_layout_key + task_index 从 legacy_layouts 解析 protocol / 物体参考位姿，
    写入 episode 顶层键（不覆盖日志里已有非空值）：gt_target_object_id、gt_target_pos、_layout_pos_map 等。
    """
    if not layout_bundle:
        return
    z_ref = float(default_z if default_z is not None else DEFAULT_TABLETOP_OBJECT_Z_M)
    layout_key = _episode_layout_key(episode, task_config)
    if not layout_key:
        return
    resolved_key, entry = _resolve_layout_entry(layout_bundle, layout_key)
    if not entry or not isinstance(entry.get("tasks"), list):
        episode["_layout_enrichment"] = {"status": "layout_not_found", "requested_key": layout_key}
        return
    tidx = _annotation_task_index(episode, task_config)
    tasks = entry.get("tasks") or []
    if tidx < 0 or tidx >= len(tasks):
        episode["_layout_enrichment"] = {
            "status": "task_index_out_of_range",
            "layout_key": resolved_key,
            "task_index": tidx,
        }
        return
    task_payload = tasks[tidx]
    if not isinstance(task_payload, dict):
        return
    annotation = task_payload.get("annotation")
    if not isinstance(annotation, dict):
        episode["_layout_enrichment"] = {"status": "missing_annotation", "layout_key": resolved_key}
        return
    protocol = annotation.get("protocol")
    if not isinstance(protocol, dict):
        protocol = {}

    pos_map = _positions_from_layout_entry(entry, task_payload, z_ref)
    gt_id = _infer_gt_target_object_id_from_annotation(annotation)
    gt_pos = pos_map.get(gt_id) if gt_id else None

    episode["_layout_enrichment"] = {
        "status": "ok",
        "layout_key_resolved": resolved_key,
        "layout_key_requested": layout_key,
        "task_index": tidx,
        "gt_target_object_id_inferred": gt_id,
    }
    episode["_layout_protocol"] = protocol
    episode["_layout_annotation"] = annotation
    episode["_layout_pos_map"] = {k: v.tolist() for k, v in pos_map.items()}

    gt_spec = protocol.get("gt_spec") if isinstance(protocol.get("gt_spec"), dict) else {}
    episode["_layout_gt_xy_only"] = bool(gt_spec.get("xy_only"))

    if gt_id and not (str(episode.get("gt_target_object_id") or "").strip()):
        episode["gt_target_object_id"] = gt_id
    if gt_pos is not None and not episode.get("gt_target_pos"):
        episode["gt_target_pos"] = [float(gt_pos[0]), float(gt_pos[1]), float(gt_pos[2])]


def extract_last_eef_pos_from_trace(trace_path: Path) -> list[float] | None:
    """从 trace.json 文本中解析最后一次出现的 eef_pos（避免整文件 json 解析过大文件）。"""
    try:
        text = trace_path.read_text(encoding="utf-8")
    except OSError:
        return None
    key = '"eef_pos"'
    idx = text.rfind(key)
    if idx < 0:
        return None
    lb = text.find("[", idx)
    rb = text.find("]", lb + 1)
    if lb < 0 or rb < 0:
        return None
    chunk = text[lb : rb + 1]
    try:
        arr = json.loads(chunk)
        if isinstance(arr, list) and len(arr) >= 3:
            return [float(arr[0]), float(arr[1]), float(arr[2])]
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _as_vec3(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("xyz", "position", "pos"):
            if k in x:
                return _as_vec3(x[k])
        return None
    if isinstance(x, (list, tuple)) and len(x) >= 3:
        try:
            return np.array([float(x[0]), float(x[1]), float(x[2])], dtype=np.float64)
        except (TypeError, ValueError):
            return None
    return None


def _nested_get(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _episode_get(ep: dict[str, Any], *names: str) -> Any:
    """按候选键名读取 episode（含 _result_json）。"""
    for name in names:
        if name in ep and ep[name] is not None:
            return ep[name]
    res = ep.get("_result_json")
    if isinstance(res, dict):
        for name in names:
            if name in res and res[name] is not None:
                return res[name]
        ev = res.get("evaluation")
        if isinstance(ev, dict):
            for name in names:
                if name in ev:
                    return ev[name]
    return None


def point_to_axis_aligned_box_distance(p: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> float:
    """点到轴对齐盒子的 L2 距离：内为 0。"""
    c = np.minimum(np.maximum(p, mn), mx)
    return float(np.linalg.norm(p - c))


def point_to_sphere_distance(p: np.ndarray, center: np.ndarray, radius: float) -> float:
    d = float(np.linalg.norm(p - center))
    return max(0.0, d - radius)


def compute_sge(episode: dict[str, Any], metadata: dict[str, Any], config: dict[str, Any]) -> tuple[float | None, str]:
    """
    SGE：优先 gt_target_pos / final_eef；否则 move_to 类可用 evaluation.goal_distance 作为与目标几何距离的代理。
    """
    warnings: list[str] = []
    gt_pos = _as_vec3(_episode_get(episode, "gt_target_pos", "target_pos", "goal_pos"))
    gt_region = _episode_get(episode, "gt_target_region", "target_region")

    final_eef = _as_vec3(_episode_get(episode, "final_eef_pos", "eef_pos_final"))
    if final_eef is None and config.get("trace_eef"):
        tp = episode.get("episode_dir") or episode.get("output_root")
        if isinstance(tp, str):
            trace_path = Path(tp) / "episode_0000" / "trace.json"
            if not trace_path.is_file():
                trace_path = Path(tp) / "trace.json"
            v = extract_last_eef_pos_from_trace(trace_path)
            if v is not None:
                final_eef = np.array(v, dtype=np.float64)

    manip_pos = _as_vec3(_episode_get(episode, "final_manipulated_object_pos", "manipulated_object_pos"))

    use_manip = bool(metadata.get("use_manipulated_for_sge"))  # 预留：默认 False

    ref_point = manip_pos if (use_manip and manip_pos is not None) else final_eef

    if gt_pos is not None and ref_point is not None:
        if (episode.get("_layout_enrichment") or {}).get("status") == "ok":
            warnings.append("sge_gt_target_from_legacy_layout")
        if episode.get("_layout_gt_xy_only"):
            warnings.append("sge_gt_spec_xy_only_used_tabletop_z_constant")
        return float(np.linalg.norm(ref_point - gt_pos)), ";".join([w for w in warnings if w])

    if isinstance(gt_region, dict) and ref_point is not None:
        kind = str(gt_region.get("type", gt_region.get("shape", ""))).lower()
        try:
            if kind in {"box", "aabb"}:
                mn = _as_vec3(gt_region.get("min", gt_region.get("lower")))
                mx = _as_vec3(gt_region.get("max", gt_region.get("upper")))
                if mn is not None and mx is not None:
                    inside = np.all(ref_point >= mn) and np.all(ref_point <= mx)
                    if inside:
                        return 0.0, ""
                    return point_to_axis_aligned_box_distance(ref_point, mn, mx), ""
            if kind in {"sphere", "ball"}:
                c = _as_vec3(gt_region.get("center", gt_region.get("origin")))
                r = float(gt_region.get("radius", gt_region.get("r", 0.0)))
                if c is not None and r > 0:
                    d = point_to_sphere_distance(ref_point, c, r)
                    return d, ""
            if kind in {"circle", "disk"}:
                c = _as_vec3(gt_region.get("center"))
                r = float(gt_region.get("radius", 0.0))
                if c is not None and r > 0 and ref_point.shape[0] >= 2:
                    d2 = float(np.linalg.norm(ref_point[:2] - c[:2]))
                    return max(0.0, d2 - r), ""
        except (TypeError, ValueError) as exc:
            warnings.append(f"gt_target_region_parse_error:{exc}")
        return None, ";".join(warnings + ["unsupported_or_invalid_gt_target_region"])

    # goal_distance 代理（与 run_vla_eval.extract_distance 一致）
    res = episode.get("_result_json")
    dist = None
    if isinstance(res, dict):
        ev = res.get("evaluation")
        if isinstance(ev, dict):
            # goal_distance_l2_m 是未经各向异性重标度的欧氏距离（米）。带 orthogonal_r/axis 的
            # 几何任务里 goal_distance 已不是长度量，只能用于成功判定，不能当 SGE。
            for k in ("goal_distance_l2_m", "goal_distance", "l2_error", "final_l2_error"):
                if k in ev and ev[k] is not None:
                    try:
                        dist = float(ev[k])
                        break
                    except (TypeError, ValueError):
                        pass
    if dist is None and episode.get("distance_m") is not None:
        try:
            dist = float(episode["distance_m"])
        except (TypeError, ValueError):
            pass
    if dist is not None:
        warnings.append("sge_from_goal_distance_proxy_no_gt_target_pos")
        return dist, ";".join(warnings)

    return None, "missing_gt_target_pos_and_goal_distance;missing_final_eef_for_region"


def check_constraint(
    final_state: dict[str, np.ndarray],
    constraint: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool | None, str]:
    """
    对单条空间谓词做判定。不支持或缺少实体坐标时返回 (None, reason)。
    final_state: object_id -> 世界系 3D 位置。
    """
    ctype = str(constraint.get("type") or constraint.get("relation") or constraint.get("name") or "").lower()
    if not ctype:
        return None, "empty_constraint_type"

    def _pos(oid: Any) -> np.ndarray | None:
        if oid is None:
            return None
        return final_state.get(str(oid))

    margin = float(config.get("relation_margin", RELATION_MARGIN))
    axis = int(config.get("front_axis", FRONT_AXIS))
    sign = float(config.get("front_sign", FRONT_SIGN))

    if ctype in {"left_of"}:
        s, r = _pos(constraint.get("subject")), _pos(constraint.get("reference") or constraint.get("object_b"))
        if s is None or r is None:
            return None, "missing_subject_or_reference"
        return bool(s[0] < r[0] - margin), ""

    if ctype in {"right_of"}:
        s, r = _pos(constraint.get("subject")), _pos(constraint.get("reference") or constraint.get("object_b"))
        if s is None or r is None:
            return None, "missing_subject_or_reference"
        return bool(s[0] > r[0] + margin), ""

    if ctype in {"front_of"}:
        s, r = _pos(constraint.get("subject")), _pos(constraint.get("reference") or constraint.get("object_b"))
        if s is None or r is None:
            return None, "missing_subject_or_reference"
        return bool(sign * (s[axis] - r[axis]) > margin), ""

    if ctype in {"behind"}:
        s, r = _pos(constraint.get("subject")), _pos(constraint.get("reference") or constraint.get("object_b"))
        if s is None or r is None:
            return None, "missing_subject_or_reference"
        return bool(sign * (s[axis] - r[axis]) < -margin), ""

    if ctype in {"near", "close"}:
        s, r = _pos(constraint.get("subject")), _pos(constraint.get("reference") or constraint.get("object_b"))
        if s is None or r is None:
            return None, "missing_subject_or_reference"
        thr = float(constraint.get("threshold", config.get("near_threshold", NEAR_THRESHOLD)))
        return bool(float(np.linalg.norm(s - r)) < thr), ""

    if ctype in {"inside", "within"}:
        s = _pos(constraint.get("subject"))
        reg = constraint.get("region") or constraint.get("reference_region")
        if s is None:
            return None, "missing_subject"
        if isinstance(reg, dict):
            t = str(reg.get("type", "box")).lower()
            if t in {"box", "aabb"}:
                mn = _as_vec3(reg.get("min"))
                mx = _as_vec3(reg.get("max"))
                if mn is not None and mx is not None:
                    m = float(config.get("inside_margin", INSIDE_MARGIN))
                    return bool(np.all(s >= mn + m) and np.all(s <= mx - m)), ""
        ref = _pos(constraint.get("reference") or constraint.get("container"))
        if ref is not None:
            half = float(constraint.get("half_extent", 0.05))
            mn, mx = ref - half, ref + half
            return bool(np.all(s >= mn) and np.all(s <= mx)), ""
        return None, "inside_missing_region"

    if ctype in {"between"}:
        s = _pos(constraint.get("subject"))
        a = _pos(constraint.get("anchor_a") or constraint.get("a"))
        b = _pos(constraint.get("anchor_b") or constraint.get("b"))
        if s is None or a is None or b is None:
            return None, "missing_between_entities"
        ab = b - a
        L2 = float(np.dot(ab, ab)) ** 0.5
        if L2 < 1e-9:
            return None, "degenerate_between_segment"
        t = float(np.dot(s - a, ab) / (L2 * L2))
        tcl = min(1.0, max(0.0, t))
        proj = a + tcl * ab
        perp = float(np.linalg.norm(s - proj))
        thr = float(constraint.get("threshold", config.get("between_threshold", BETWEEN_THRESHOLD)))
        along_ok = 0.0 <= t <= 1.0
        return bool(along_ok and perp < thr), ""

    if ctype in {"aligned", "colinear"}:
        s = _pos(constraint.get("subject"))
        a = _pos(constraint.get("anchor_a") or constraint.get("a"))
        b = _pos(constraint.get("anchor_b") or constraint.get("b"))
        if s is None or a is None or b is None:
            return None, "missing_aligned_entities"
        ab = b - a
        L2 = float(np.dot(ab, ab)) ** 0.5
        if L2 < 1e-9:
            return None, "degenerate_alignment_line"
        t = float(np.dot(s - a, ab)) / (L2 * L2)
        proj = a + t * ab
        d = float(np.linalg.norm(s - proj))
        thr = float(constraint.get("threshold", config.get("alignment_threshold", ALIGNMENT_THRESHOLD)))
        return bool(d < thr), ""

    if ctype in {"symmetric", "symmetry"}:
        s = _pos(constraint.get("subject"))
        center = _as_vec3(constraint.get("center") or constraint.get("axis_origin"))
        axis_vec = _as_vec3(constraint.get("axis") or constraint.get("axis_dir"))
        ref_obj = constraint.get("reference_object_id")
        refp = _pos(ref_obj) if ref_obj is not None else _as_vec3(constraint.get("reference_point"))
        if s is None or center is None or axis_vec is None or refp is None:
            return None, "missing_symmetry_parameters"
        axis_vec = axis_vec / (float(np.linalg.norm(axis_vec)) + 1e-9)
        # 将 reference 点关于过 center、方向 axis 的平面对称
        v = refp - center
        proj = np.dot(v, axis_vec) * axis_vec
        mirrored = refp - 2.0 * proj
        thr = float(constraint.get("threshold", config.get("symmetry_threshold", SYMMETRY_THRESHOLD)))
        return bool(float(np.linalg.norm(s - mirrored)) < thr), ""

    return None, f"unsupported_constraint_type:{ctype}"


def _gather_constraints(episode: dict[str, Any], task_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    extra_protocols: list[dict[str, Any]] = []
    lp = episode.get("_layout_protocol")
    if isinstance(lp, dict):
        extra_protocols.append(lp)
    for source in (episode, task_config or {}):
        if not isinstance(source, dict):
            continue
        c = source.get("constraints")
        if isinstance(c, list):
            out.extend([x for x in c if isinstance(x, dict)])
        sc = source.get("spatial_constraints")
        if isinstance(sc, list):
            out.extend([x for x in sc if isinstance(x, dict)])
        proto = source.get("protocol")
        if isinstance(proto, dict):
            extra_protocols.append(proto)
    for proto in extra_protocols:
        gt = proto.get("gt_spec") or proto.get("gt")
        if isinstance(gt, dict):
            c2 = gt.get("constraints") or gt.get("spatial_constraints")
            if isinstance(c2, list):
                out.extend([x for x in c2 if isinstance(x, dict)])
    return out


def _load_task_config_for_episode(
    episode: dict[str, Any],
    metadata: dict[str, Any],
    task_config_dir: Path,
) -> dict[str, Any] | None:
    row = metadata.get("taxonomy_row") or {}
    fname = str(row.get("file") or "").strip()
    candidates: list[Path] = []
    if fname:
        candidates.append(task_config_dir / fname)
    tid = metadata.get("task_id_mapping_key") or metadata.get("task_id_raw")
    if tid:
        candidates.append(task_config_dir / f"{tid}.json")
    for p in candidates:
        data = _safe_read_json(p)
        if isinstance(data, dict):
            if "runtime" in data:
                if str(_REPO_ROOT) not in sys.path:
                    sys.path.insert(0, str(_REPO_ROOT))
                from geomind.core.schema import normalize_task_config
                return normalize_task_config(data)
            return data
    return None


def _poses_from_episode(episode: dict[str, Any]) -> dict[str, np.ndarray]:
    """尽力从 episode / result 提取 object_id -> pos；无遥测时用布局中的静态参考位姿兜底。"""
    poses: dict[str, np.ndarray] = {}
    raw = _episode_get(episode, "final_object_poses", "object_poses_final")
    if isinstance(raw, dict):
        for oid, val in raw.items():
            v = _as_vec3(val)
            if v is not None:
                poses[str(oid)] = v
    if not poses:
        lmap = episode.get("_layout_pos_map")
        if isinstance(lmap, dict):
            for oid, val in lmap.items():
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    try:
                        poses[str(oid)] = np.array(
                            [float(val[0]), float(val[1]), float(val[2])],
                            dtype=np.float64,
                        )
                    except (TypeError, ValueError):
                        continue
    return poses


def compute_scs(
    episode: dict[str, Any],
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float | None, str]:
    """SCS：仅对 reasoning_heavy episode 有意义；无约束列表时 NaN。"""
    if not metadata.get("reasoning_heavy"):
        return None, "not_reasoning_heavy_task"
    tc = config.get("task_config")
    if not isinstance(tc, dict):
        tc = None
    cons = _gather_constraints(episode, tc)
    if not cons:
        return None, "missing_constraints"
    raw_poses = _episode_get(episode, "final_object_poses", "object_poses_final")
    raw_ok = isinstance(raw_poses, dict) and bool(raw_poses)
    final_state = _poses_from_episode(episode)
    if not final_state:
        return None, "missing_final_object_poses_for_scs"
    warn_extra = ""
    if not raw_ok and isinstance(episode.get("_layout_pos_map"), dict) and episode["_layout_pos_map"]:
        warn_extra = "scs_using_layout_static_positions"

    satisfied = 0
    resolved = 0
    reasons: list[str] = []
    for c in cons:
        ok, r = check_constraint(final_state, c, config)
        if ok is None:
            reasons.append(r)
            continue
        resolved += 1
        if ok:
            satisfied += 1
    if resolved <= 0:
        msg = "all_constraints_unsupported_or_missing_pos:" + ",".join(reasons[:8])
        return None, msg
    return satisfied / float(resolved), warn_extra


def _nearest_object_id_from_poses(eef: np.ndarray, poses: dict[str, np.ndarray]) -> str | None:
    best = None
    best_d = math.inf
    for oid, p in poses.items():
        d = float(np.linalg.norm(p - eef))
        if d < best_d:
            best_d = d
            best = oid
    return best


def _largest_displacement_object(initial: dict[str, np.ndarray], final: dict[str, np.ndarray]) -> str | None:
    best = None
    best_d = -1.0
    for oid, p0 in initial.items():
        p1 = final.get(oid)
        if p1 is None:
            continue
        d = float(np.linalg.norm(p1 - p0))
        if d > best_d:
            best_d = d
            best = oid
    return best


def compute_mta(episode: dict[str, Any], metadata: dict[str, Any], config: dict[str, Any]) -> tuple[float | None, str | None, str | None, str]:
    """
    返回 (mta, pred_id, gt_id, warning)。
    MTA 为 0/1 或 NaN（用 None 表示，写入 CSV 为 NaN）。
    """
    if not metadata.get("memory_related"):
        return None, None, None, "not_memory_related_task"

    gt = _episode_get(episode, "gt_target_object_id", "target_object_gt_id")
    if not gt and isinstance(episode.get("_result_json"), dict):
        eb = episode["_result_json"].get("execution_binding") or {}
        if isinstance(eb, dict) and eb.get("target_object_id"):
            gt = eb.get("target_object_id")
    if not gt:
        return None, None, None, "missing_gt_target_object_id"

    pred = _episode_get(episode, "manipulated_object_id", "selected_object_id", "pred_target_object_id")
    warn_extra = ""
    if not pred:
        eef = _as_vec3(_episode_get(episode, "final_eef_pos"))
        poses = _poses_from_episode(episode)
        if eef is not None and poses:
            pred = _nearest_object_id_from_poses(eef, poses)
            warn_extra = "pred_from_nearest_object_to_eef"
        else:
            ini = {}
            raw_i = _episode_get(episode, "initial_object_poses")
            if isinstance(raw_i, dict):
                for oid, val in raw_i.items():
                    v = _as_vec3(val)
                    if v is not None:
                        ini[str(oid)] = v
            fin = _poses_from_episode(episode)
            if ini and fin:
                pred = _largest_displacement_object(ini, fin)
                warn_extra = "pred_from_largest_displacement_fallback"
            else:
                return None, None, str(gt), "cannot_infer_pred_target_object_id"

    ok = str(pred) == str(gt)
    w = warn_extra
    return (1.0 if ok else 0.0), str(pred), str(gt), w


def compute_success_rates(results: list[dict[str, Any]]) -> float:
    """与常见汇总一致：对 trial 行的 ``success`` 字段取平均（不在此重算阈值）。"""
    vals: list[float] = []
    for r in results:
        if not isinstance(r, dict) or "_parse_error" in r:
            continue
        if "success" in r:
            vals.append(1.0 if bool(r.get("success")) else 0.0)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def build_per_episode_table(
    results: list[dict[str, Any]],
    task_mapping: dict[str, Any],
    *,
    task_config_dir: Path,
    load_trace_eef: bool,
    layout_bundle: dict[str, Any] | None,
    use_layout_enrich: bool,
    tabletop_z: float = DEFAULT_TABLETOP_OBJECT_Z_M,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    miss_counter: Counter[str] = Counter()
    layout_status_counter: Counter[str] = Counter()
    cfg_base = {
        "relation_margin": RELATION_MARGIN,
        "near_threshold": NEAR_THRESHOLD,
        "alignment_threshold": ALIGNMENT_THRESHOLD,
        "between_threshold": BETWEEN_THRESHOLD,
        "symmetry_threshold": SYMMETRY_THRESHOLD,
        "front_axis": FRONT_AXIS,
        "front_sign": FRONT_SIGN,
        "inside_margin": INSIDE_MARGIN,
        "trace_eef": load_trace_eef,
    }

    for raw in results:
        if not isinstance(raw, dict) or "_parse_error" in raw:
            continue
        ep = merge_episode_with_result_json(raw)
        meta = infer_task_metadata(ep, task_mapping)
        tc = _load_task_config_for_episode(ep, meta, task_config_dir)
        if use_layout_enrich and layout_bundle:
            enrich_episode_from_layout_bundle(ep, tc, layout_bundle, default_z=tabletop_z)
            st = str((ep.get("_layout_enrichment") or {}).get("status", "unknown"))
            layout_status_counter[st] += 1
        else:
            layout_status_counter["skipped"] += 1
        cfg = {**cfg_base, "task_config": tc}

        agent = str(ep.get("remote_policy_model") or ep.get("agent") or "").strip() or str(ep.get("method") or "")

        success = bool(ep.get("success")) if "success" in ep else float("nan")

        sge, sge_w = compute_sge(ep, meta, cfg)
        if sge is None:
            for token in sge_w.split(";"):
                miss_counter[f"sge:{token}"] += 1

        scs, scs_w = compute_scs(ep, meta, cfg)
        if meta["reasoning_heavy"]:
            if scs is None:
                miss_counter[f"scs:{scs_w}"] += 1

        mta, pred_id, gt_id, mta_w = compute_mta(ep, meta, cfg)
        if meta["memory_related"]:
            if mta is None:
                miss_counter[f"mta:{mta_w or ''}"] += 1

        # 省略「不适用」类提示，突出数据缺失与代理指标说明
        skip_tokens = {"not_reasoning_heavy_task", "not_memory_related_task"}
        warns = [w for w in (sge_w, scs_w, mta_w) if w and w not in skip_tokens]
        row = {
            "method": agent,
            "agent": agent,
            "suite": meta["suite"],
            "task_id": meta["task_id_raw"],
            "task_name": meta["task_id_raw"],
            "task_family": meta["task_family"],
            "augmentation_type": meta["augmentation_type"],
            "success": success,
            "success_float": 1.0 if success is True else (0.0 if success is False else float("nan")),
            "sge": sge,
            "scs": scs,
            "mta": mta,
            "pred_target_object_id": pred_id,
            "gt_target_object_id": gt_id,
            "missing_reason_or_warning": "; ".join(warns),
            "reasoning_heavy": meta["reasoning_heavy"],
            "memory_related": meta["memory_related"],
            "result_json": ep.get("result_json", ""),
            "success_level": ep.get("success_level", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    meta_report = {
        "missing_histogram": dict(miss_counter.most_common(50)),
        "layout_enrichment_status": dict(layout_status_counter),
    }
    return df, meta_report


def aggregate_metrics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """按 task / family / aug_type 聚合 mean 与 success_rate。"""
    def _group(keys: list[str]) -> pd.DataFrame:
        g = df.groupby(keys, dropna=False)
        return g.agg(
            success_rate=("success_float", "mean"),
            mean_sge=("sge", "mean"),
            mean_scs=("scs", "mean"),
            mean_mta=("mta", "mean"),
            num_episodes=("task_id", "count"),
        ).reset_index()

    by_task = _group(["method", "suite", "task_id"])
    by_family = _group(["method", "suite", "task_family"])
    by_aug = _group(["method", "augmentation_type"])
    return {"by_task": by_task, "by_family": by_family, "by_aug": by_aug}


def save_outputs(
    per_episode_df: pd.DataFrame,
    aggregate_dfs: dict[str, pd.DataFrame],
    output_dir: Path,
    missing_report: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_df = per_episode_df.rename(
        columns={"missing_reason_or_warning": "missing_reason / warning"},
    )
    out_df.to_csv(output_dir / "per_episode_diagnostics.csv", index=False)
    aggregate_dfs["by_task"].to_csv(output_dir / "aggregate_diagnostics_by_task.csv", index=False)
    aggregate_dfs["by_family"].to_csv(output_dir / "aggregate_diagnostics_by_family.csv", index=False)
    aggregate_dfs["by_aug"].to_csv(output_dir / "aggregate_diagnostics_by_aug_type.csv", index=False)
    lines = [
        "GeoMind diagnostics missing-field / skip summary",
        json.dumps(missing_report, ensure_ascii=False, indent=2),
    ]
    (output_dir / "diagnostics_missing_fields_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_schema_summary(schema: dict[str, Any]) -> None:
    print("=== trial_results.jsonl schema (sampled keys) ===")
    print(json.dumps(schema, ensure_ascii=False, indent=2))


def _field_presence_report(results: list[dict[str, Any]], merged_sample: int = 50) -> dict[str, Any]:
    """检查用户关心的字段在原始行与合并 result 后的出现频率。"""
    want = [
        "task_name",
        "task_id",
        "task_family",
        "suite",
        "augmentation_type",
        "success",
        "final_eef_pos",
        "final_object_poses",
        "initial_object_poses",
        "gt_target_object_id",
        "gt_target_pos",
        "gt_target_region",
        "constraints",
        "manipulated_object_id",
    ]
    raw_counts = {k: 0 for k in want}
    merged_counts = {k: 0 for k in want}
    n = 0
    for r in results[:merged_sample]:
        if not isinstance(r, dict):
            continue
        n += 1
        m = merge_episode_with_result_json(r)
        for k in want:
            if k in r and r.get(k) is not None:
                raw_counts[k] += 1
            if _episode_get(m, k) is not None:
                merged_counts[k] += 1
    return {"sample_size": n, "raw_line_presence": raw_counts, "after_result_json_merge_presence": merged_counts}


def main() -> int:
    p = argparse.ArgumentParser(description="GeoMind trial_results.jsonl 诊断指标（SGE / SCS / MTA）")
    p.add_argument("--trial-results", type=Path, required=True, help="trial_results.jsonl 路径")
    p.add_argument("--mapping", type=Path, default=_DEFAULT_MAPPING, help="current_task_mapping.json")
    p.add_argument("--output-dir", type=Path, required=True, help="输出目录（将新建）")
    p.add_argument("--task-config-dir", type=Path, default=_DEFAULT_TASK_CONFIG_DIR, help="任务 JSON 目录")
    p.add_argument(
        "--load-trace-eef",
        action="store_true",
        help="从 trace.json 解析最后一次 eef_pos（大文件时较慢）",
    )
    p.add_argument("--schema-sample", type=int, default=200, help="inspect_schema 抽样行数")
    p.add_argument(
        "--layout-json",
        type=Path,
        action="append",
        default=None,
        help="legacy_layouts JSON，可重复传入；默认合并 tasks+vla_tasks 两份",
    )
    p.add_argument(
        "--no-layout-enrich",
        action="store_true",
        help="禁用从 legacy_layouts 解析 gt / 位姿",
    )
    p.add_argument(
        "--tabletop-z",
        type=float,
        default=float(DEFAULT_TABLETOP_OBJECT_Z_M),
        help="布局仅有 xy 时用于拼 3D 的参考 z（米）",
    )
    args = p.parse_args()

    results = load_results(args.trial_results)
    schema = inspect_schema(results, sample=args.schema_sample)
    _print_schema_summary(schema)

    presence = _field_presence_report(results)
    print("=== requested field presence (raw vs merged) ===")
    print(json.dumps(presence, ensure_ascii=False, indent=2))

    mapping = load_task_mapping(args.mapping)
    layout_paths: list[Path] | None = list(args.layout_json) if args.layout_json else None
    layout_bundle: dict[str, Any] | None = None
    use_layout = not bool(args.no_layout_enrich)
    if use_layout:
        layout_bundle = load_layout_bundle(layout_paths)
    df, miss = build_per_episode_table(
        results,
        mapping,
        task_config_dir=args.task_config_dir,
        load_trace_eef=args.load_trace_eef,
        layout_bundle=layout_bundle,
        use_layout_enrich=use_layout,
        tabletop_z=float(args.tabletop_z),
    )
    aggs = aggregate_metrics(df)

    n_total = len(df)
    n_sge = int(df["sge"].notna().sum())
    n_scs = int(df["scs"].notna().sum())
    n_mta = int(df["mta"].notna().sum())

    overall_success = compute_success_rates(results)

    missing_report = {
        "field_presence": presence,
        "aggregate_missing_histogram": miss.get("missing_histogram", {}),
        "layout_enrichment_status": miss.get("layout_enrichment_status", {}),
        "overall_success_rate_trials": overall_success,
        "notes": [
            "SCS/MTA 为诊断指标，不替代 success。",
            "SGE 在缺少 gt_target_pos 时可能回退为 evaluation.goal_distance / distance_m。",
            "若启用布局解析：gt_target_* 来自 legacy_layouts 中 protocol.gt_spec + 物体位姿，可能与仿真扰动后真值略有偏差。",
        ],
    }
    save_outputs(df, aggs, args.output_dir, missing_report)

    print("=== summary ===")
    print(f"总 episode 数: {n_total}")
    print(f"可计算 SGE 的 episode 数: {n_sge}")
    print(f"可计算 SCS 的 episode 数: {n_scs}")
    print(f"可计算 MTA 的 episode 数: {n_mta}")
    print(f"整体 success 率（trial 行 success 字段平均）: {overall_success:.4f}")
    print("各指标缺失主要原因（计数）:")
    for k, v in miss.get("missing_histogram", {}).items():
        print(f"  {k}: {v}")
    print("布局解析状态（episode 计数）:")
    for k, v in miss.get("layout_enrichment_status", {}).items():
        print(f"  {k}: {v}")
    print(f"输出目录: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
