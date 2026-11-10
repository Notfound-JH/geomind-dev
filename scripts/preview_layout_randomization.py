#!/usr/bin/env python3
"""Preview the effect of layout randomization without launching Isaac Sim.

Examples
--------
    # one layout, jitter + global shift
    python scripts/preview_layout_randomization.py --layout-key place_inside_scene65_skill \
        --jitter --shift --jitter-seed 7 --shift-seed 9

    # resolve the layout from a task id like the debug scripts do
    python scripts/preview_layout_randomization.py --task-id place_inside --variant skill --jitter --shift

    # sanity-sweep every annotated layout for N seeds and report the displacement range
    python scripts/preview_layout_randomization.py --all --trials 20 --jitter --shift
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind_lab.tasks.manager_based.manipulation.geomind import task_specs as ts  # noqa: E402


def _apply_env(args: argparse.Namespace, *, enabled: bool) -> None:
    for name in list(os.environ):
        if name.startswith("GEOMIND_LAYOUT_"):
            del os.environ[name]
    if not enabled:
        return
    os.environ["GEOMIND_LAYOUT_JITTER_ENABLE"] = "1" if args.jitter else "0"
    os.environ["GEOMIND_LAYOUT_SHIFT_ENABLE"] = "1" if args.shift else "0"
    if args.jitter_seed is not None:
        os.environ["GEOMIND_LAYOUT_JITTER_SEED"] = str(args.jitter_seed)
    if args.shift_seed is not None:
        os.environ["GEOMIND_LAYOUT_SHIFT_SEED"] = str(args.shift_seed)
    if args.jitter_xy is not None:
        os.environ["GEOMIND_LAYOUT_JITTER_XY"] = str(args.jitter_xy)
    if args.shift_max is not None:
        os.environ["GEOMIND_LAYOUT_SHIFT_MAX"] = str(args.shift_max)
    if args.min_clearance is not None:
        os.environ["GEOMIND_LAYOUT_MIN_CLEARANCE"] = str(args.min_clearance)
    if args.bounds:
        os.environ["GEOMIND_LAYOUT_BOUNDS"] = args.bounds


def _load(layout_key: str, scene_id: int, task_index: int, object_mode: str) -> dict:
    scene_profile = ts._build_lightweight_scene_profile(layout_key, scene_id)
    spec = ts.load_task_spec(layout_key, task_index, scene_profile)
    return ts.apply_object_mode(spec, object_mode=object_mode)


def _positions(spec: dict) -> dict[str, tuple[float, float]]:
    return {
        name: (float(entry["position"][0]), float(entry["position"][1]))
        for name, entry in (spec.get("object_positions") or {}).items()
        if isinstance(entry, dict) and len(entry.get("position", ())) >= 2
    }


def _pair_distances(positions: dict[str, tuple[float, float]]) -> dict[tuple[str, str], float]:
    names = sorted(positions)
    return {
        (a, b): math.dist(positions[a], positions[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }


def preview_one(args: argparse.Namespace, layout_key: str, scene_id: int) -> None:
    _apply_env(args, enabled=False)
    base = _load(layout_key, scene_id, args.task_index, args.object_mode)
    base_xy = _positions(base)

    _apply_env(args, enabled=True)
    new = _load(layout_key, scene_id, args.task_index, args.object_mode)
    new_xy = _positions(new)
    record = new.get("applied_layout_randomization")

    print(f"layout_key : {layout_key}  (scene {scene_id}, task_index {args.task_index})")
    if record is None:
        print("randomization: DISABLED (no switch turned on)")
        return
    print(f"strategy   : {record['strategy']}   resolver: {record['resolver'] or '(none)'}")
    print(f"jitter     : enabled={record['jitter_enabled']} seed={record['jitter_seed']} xy=±{record['jitter_xy']}")
    print(f"shift      : enabled={record['shift_enabled']} seed={record['shift_seed']} max={record['shift_max']}")
    print(f"global shift applied: ({record['global_shift'][0]:+.4f}, {record['global_shift'][1]:+.4f})")
    print(f"safe region: x[{record['bounds'][0]:.3f}, {record['bounds'][1]:.3f}]  y[{record['bounds'][2]:.3f}, {record['bounds'][3]:.3f}]")
    groups = [g for g in record["rigid_groups"] if len(g) > 1]
    if groups:
        print(f"rigid groups (move together): {groups}")
    print()

    width = max(len(name) for name in base_xy) if base_xy else 10
    print(f"{'object':<{width}}  {'before':>17}  {'after':>17}  {'move':>7}")
    for name in sorted(base_xy):
        before, after = base_xy[name], new_xy.get(name)
        if after is None:
            continue
        print(
            f"{name:<{width}}  ({before[0]:7.4f},{before[1]:8.4f})  "
            f"({after[0]:7.4f},{after[1]:8.4f})  {math.dist(before, after):7.4f}"
        )

    before_pairs = _pair_distances(base_xy)
    after_pairs = _pair_distances(new_xy)
    shared = sorted(set(before_pairs) & set(after_pairs), key=lambda k: after_pairs[k])
    if shared:
        print("\nclosest pairs after randomization:")
        for a, b in shared[:5]:
            print(f"  {a} <-> {b}: {before_pairs[(a, b)]:.4f} -> {after_pairs[(a, b)]:.4f}")


def sweep(args: argparse.Namespace) -> int:
    # Use the same merged view the runtime sees: annotated_tasks.json plus the
    # geomind/ legacy + VLA layout overlays.
    annotations = ts.load_annotated_tasks()
    failures = 0
    uncovered = 0
    for layout_key, entry in annotations.items():
        scene_id = int(entry.get("scene_id", 65))
        try:
            _apply_env(args, enabled=False)
            base = _load(layout_key, scene_id, 0, args.object_mode)
        except Exception as exc:  # noqa: BLE001 - preview tool, report and continue
            print(f"{layout_key:<58} SKIP ({type(exc).__name__}: {exc})")
            continue
        if not base.get("runtime_objects"):
            uncovered += 1
            print(f"{layout_key:<58} NO-RUNTIME-OBJECTS (scene_profile fallback, not randomized)")
            continue
        base_xy = _positions(base)

        moves: list[float] = []
        strategies: set[str] = set()
        for trial in range(args.trials):
            args.jitter_seed = 1000 + trial
            args.shift_seed = 7000 + trial
            _apply_env(args, enabled=True)
            spec = _load(layout_key, scene_id, 0, args.object_mode)
            record = spec.get("applied_layout_randomization")
            if record is None:
                failures += 1
                print(f"{layout_key:<58} NO-SAMPLE at trial {trial}")
                break
            strategies.add(record["strategy"])
            new_xy = _positions(spec)
            moves.append(max(math.dist(base_xy[n], new_xy[n]) for n in base_xy if n in new_xy))
        if moves:
            print(
                f"{layout_key:<58} {'/'.join(sorted(strategies)):<12} "
                f"move min={min(moves):.4f} max={max(moves):.4f}"
            )
    print(f"\nlayouts swept: {len(annotations)}   randomized: {len(annotations) - uncovered - failures}   "
          f"not covered: {uncovered}   failures: {failures}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--layout-key", help="annotation layout key, e.g. place_inside_scene65_skill")
    source.add_argument("--task-id", help="registry task id, e.g. place_inside")
    source.add_argument("--all", action="store_true", help="sweep every annotated layout")

    parser.add_argument("--variant", default="base", help="task variant when using --task-id")
    parser.add_argument("--scene-id", type=int, help="override the scene id")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--object-mode", default="geometric")

    parser.add_argument("--jitter", action="store_true", help="enable the per-object jitter stage")
    parser.add_argument("--shift", action="store_true", help="enable the global translation stage")
    parser.add_argument("--jitter-seed", type=int)
    parser.add_argument("--shift-seed", type=int)
    parser.add_argument("--jitter-xy", type=float)
    parser.add_argument("--shift-max", type=float)
    parser.add_argument("--min-clearance", type=float)
    parser.add_argument("--bounds", help="x_lo,x_hi,y_lo,y_hi")
    parser.add_argument("--trials", type=int, default=10, help="samples per layout when using --all")

    args = parser.parse_args()
    if not args.jitter and not args.shift:
        parser.error("enable at least one of --jitter / --shift")

    if args.all:
        return sweep(args)

    if args.task_id:
        request = ts.resolve_task_request(task_id=args.task_id, variant=args.variant)
        layout_key = request.layout_key
        scene_id = int(request.scene_id)
        args.task_index = int(getattr(request, "task_index", args.task_index) or 0)
    else:
        layout_key = args.layout_key
        scene_id = args.scene_id if args.scene_id is not None else 65
    if args.scene_id is not None:
        scene_id = args.scene_id

    preview_one(args, layout_key, scene_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
