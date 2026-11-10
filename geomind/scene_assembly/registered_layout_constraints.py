"""Optional grid export constraints, separate from the frozen task pipeline.

The legacy randomizer remains responsible for its one seeded sample. This
post-sample repair preserves authored row/column identities while keeping each
row inside the existing GT resolver's quantization bucket. It does not change
the task rule, sample again, mutate inputs, or certify geometry/physics.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _contract(protocol):
    _require(isinstance(protocol, Mapping), "Grid protocol must be an object")
    gt = protocol.get("gt_spec", {})
    _require(isinstance(gt, Mapping) and gt.get("resolver") == "grid_rowcol_slot",
             "Row quantization repair is only available for grid_rowcol_slot")
    rows, cols = gt.get("grid_num_rows"), gt.get("grid_num_cols")
    _require(type(rows) is int and rows > 0 and type(cols) is int and cols > 0,
             "Grid rows and columns must be positive integers")
    row_axis, col_axis = gt.get("row_axis", "x"), gt.get("col_axis", "y")
    _require(row_axis in {"x", "y"} and col_axis in {"x", "y"} and row_axis != col_axis,
             "Grid row and column axes must be distinct x/y axes")
    step = gt.get("grid_quant_step", .028)
    _require(type(step) in (int, float) and math.isfinite(step) and step > 0,
             "Grid quantization step must be finite and positive")
    return gt, rows, cols, int(row_axis == "y"), int(col_axis == "y"), float(step)


def _objects(runtime_objects):
    _require(isinstance(runtime_objects, list) and bool(runtime_objects), "Grid runtime objects must be a nonempty list")
    by_name = {}
    for obj in runtime_objects:
        _require(isinstance(obj, dict), "Grid runtime descriptor must be an object")
        name = obj.get("name")
        _require(isinstance(name, str) and name and name not in by_name, "Grid object names must be unique")
        point = obj.get("pose", {}).get("xy")
        _require(isinstance(point, (list, tuple)) and len(point) == 2
                 and all(type(value) in (int, float) and math.isfinite(value) for value in point),
                 "Grid objects require finite authored/randomized pose.xy")
        by_name[name] = obj
    return by_name


def _cell_names(protocol, objects, rows, cols):
    names = protocol.get("target_subset_spec", {}).get("object_ids") or protocol.get("candidate_object_ids") or list(objects)
    _require(isinstance(names, list) and len(names) == rows * cols and len(set(names)) == len(names)
             and set(names) <= set(objects), "Grid cell set does not match rows*columns")
    return names


def validate_grid_layout(runtime_objects, protocol):
    """Use the original grid assignment predicate, without solving a new task."""
    import torch
    from geomind_lab.mimic.envs.state_reindexing_logic import (
        _grid_min_center_separation, _grid_slot_assignment_for_env,
    )
    gt, rows, cols, ri, ci, step = _contract(protocol)
    objects = _objects(runtime_objects)
    names = _cell_names(protocol, objects, rows, cols)
    xy = torch.tensor([objects[name]["pose"]["xy"] for name in names], dtype=torch.float64)
    row_ids, col_ids, valid = _grid_slot_assignment_for_env(xy, num_rows=rows, num_cols=cols,
        row_axis="y" if ri else "x", col_axis="y" if ci else "x", quant_step=step)
    _require(valid, "Grid geometry does not satisfy the original quantized row/column assignment")
    separation = _grid_min_center_separation(xy)
    minimum = float(gt.get("min_center_separation", .024))
    _require(math.isfinite(minimum) and minimum > 0 and separation + 1e-9 >= minimum,
             "Grid cell centres violate the original minimum separation")
    return {"passed": True, "original_assignment_predicate_used": True,
        "grid_num_rows": rows, "grid_num_cols": cols, "grid_quant_step_m": step,
        "minimum_center_separation_m": separation,
        "slots": {name: {"row": int(row_ids[i]), "column": int(col_ids[i])} for i, name in enumerate(names)}}


def repair_grid_row_quantization(authored_runtime_objects, randomized_runtime_objects, protocol,
                                 *, max_correction_m=None):
    """Return repaired copies and an explicit bounded post-sampling record.

    Authored rows/columns determine membership; randomized geometry is never
    re-clustered to manufacture another cell identity. Row means move to their
    nearest original quantization bucket centre. Within-row offsets stay exact
    unless they exceed one quarter bucket, when one common scale compresses
    them. Every object's total correction is bounded by half a bucket unless a
    caller explicitly supplies another positive bound. Column values never move.
    """
    from geomind_lab.tasks.manager_based.manipulation.geomind.layout_randomizer import _cluster_axis
    _, rows, cols, ri, ci, step = _contract(protocol)
    authored = _objects(authored_runtime_objects)
    randomized = _objects(randomized_runtime_objects)
    _require(set(authored) == set(randomized), "Authored and randomized grid identities differ")
    names = _cell_names(protocol, authored, rows, cols)
    bound = step / 2 if max_correction_m is None else max_correction_m
    _require(type(bound) in (int, float) and math.isfinite(bound) and bound > 0,
             "Grid correction bound must be finite and positive")
    row_labels, _ = _cluster_axis([authored[name]["pose"]["xy"][ri] for name in names], rows)
    col_labels, _ = _cluster_axis([authored[name]["pose"]["xy"][ci] for name in names], cols)
    _require(row_labels is not None and col_labels is not None, "Cannot recover authored grid groups")
    row_members = [[name for i, name in enumerate(names) if row_labels[i] == row] for row in range(rows)]
    col_by_name = dict(zip(names, col_labels))
    _require(all(len(group) == cols and {col_by_name[name] for name in group} == set(range(cols)) for group in row_members),
             "Authored grid cannot be grouped into one cell per row/column")
    # The authored source, unlike the randomized sample, must already describe
    # distinct thin axis-aligned bands. Widest-gap clustering alone is too weak.
    for axis, labels, count in ((ri, row_labels, rows), (ci, col_labels, cols)):
        bands = [[authored[name]["pose"]["xy"][axis] for i, name in enumerate(names) if labels[i] == label] for label in range(count)]
        _require(all(max(band)-min(band) <= step/2 for band in bands), "Authored grid bands are too wide to identify safely")
        _require(all(min(bands[i+1])-max(bands[i]) > step/2 for i in range(count-1)),
                 "Authored grid bands are not distinctly ordered")
    means = [sum(randomized[name]["pose"]["xy"][ri] for name in group)/len(group) for group in row_members]
    buckets = [round(mean/step)*step for mean in means]
    _require(all(means[i+1] > means[i] for i in range(rows-1))
             and all(buckets[i+1] > buckets[i] for i in range(rows-1)),
             "Randomized authored rows overlap or share their nearest quantization bucket")
    result = deepcopy(randomized_runtime_objects)
    output = _objects(result)
    corrections = []
    row_records = []
    allowed_offset = step / 4
    for row, group in enumerate(row_members):
        ordered = sorted(group, key=lambda name: col_by_name[name])
        col_values = [randomized[name]["pose"]["xy"][ci] for name in ordered]
        _require(all(col_values[i+1]-col_values[i] > 1e-6 for i in range(cols-1)),
                 "Randomized column order differs from the authored cell order")
        residual = [randomized[name]["pose"]["xy"][ri]-means[row] for name in group]
        largest = max(abs(value) for value in residual)
        factor = min(1., allowed_offset/largest) if largest else 1.
        row_records.append({"authored_row": row+1, "object_names": ordered,
            "randomized_mean_m": means[row], "nearest_bucket_center_m": buckets[row],
            "mean_translation_m": buckets[row]-means[row], "within_row_compression_factor": factor})
        for name, offset in zip(group, residual):
            before = list(randomized[name]["pose"]["xy"])
            after = list(before); after[ri] = buckets[row] + offset*factor
            correction = after[ri]-before[ri]
            _require(abs(correction) <= bound+1e-12, "Required grid correction exceeds the explicit movement bound")
            output[name]["pose"]["xy"] = after
            corrections.append({"object_name": name, "authored_row": row+1,
                "authored_column": col_by_name[name]+1, "before_xy_m": before, "after_xy_m": after,
                "delta_xy_m": [after[i]-before[i] for i in range(2)],
                "reason": "align authored row mean with original GT quantization bucket" +
                    ("; bounded compression of within-row offsets" if factor < 1 else "; within-row jitter preserved")})
    checked = validate_grid_layout(result, protocol)
    expected_slots = {name: {"row": row+1, "column": col_by_name[name]+1}
                      for row, group in enumerate(row_members) for name in group}
    _require(checked["slots"] == expected_slots, "Repair changed authored row/column semantics")
    return result, {"kind": "registered_grid_quantization_constraint", "applied": True,
        "sampling_policy": "postprocess one existing seeded legacy randomizer sample; no resampling",
        "seed_changed": False, "gt_rule_changed": False, "randomizer_additional_calls": 0,
        "row_axis": "y" if ri else "x", "column_axis": "y" if ci else "x",
        "grid_quant_step_m": step, "max_correction_m": float(bound),
        "max_within_row_deviation_m": allowed_offset, "rows": row_records,
        "object_corrections": corrections, "original_predicate_validation": checked,
        "column_coordinates_changed": False,
        "geometry_support_clearance_and_physics": "unverified; exporter and physical workers must recheck"}
