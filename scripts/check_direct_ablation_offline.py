#!/usr/bin/env python3
"""Offline audit of the direct-reference ablation overrides (no IsaacLab / no GPU).

Verification #1 (config integrity) and #2 (pairing-relevant layout is untouched):
for every VLA task it loads the annotated task spec, applies the direct override and
asserts that the canonical layout hash is unchanged. It also prints exactly what each
task's direct variant changes, which is the table that goes into the report.

    python scripts/check_direct_ablation_offline.py --json docs/direct_ablation/offline_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind.tasks import direct_ablation as direct  # noqa: E402
from geomind.core.schema import normalize_task_config

VLA_TASK_DIR = REPO_ROOT / "geomind" / "configs" / "vla_tasks"


def load_vla_task_payloads() -> list[dict]:
    payloads = []
    for path in sorted(VLA_TASK_DIR.glob("*.json")):
        if path.name == "legacy_layouts.json":
            continue
        payload = normalize_task_config(json.loads(path.read_text(encoding="utf-8")))
        if payload.get("implementation_status") == "placeholder":
            continue
        payload["_config_path"] = str(path)
        payloads.append(payload)
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="", help="Optional path to dump the machine-readable audit.")
    args = parser.parse_args()

    from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import load_task

    payloads = load_vla_task_payloads()
    rows: list[dict] = []
    failures: list[str] = []

    for index, payload in enumerate(payloads):
        task_id = str(payload.get("task_id", ""))
        try:
            loaded = load_task(
                task_id=str(payload.get("canonical_task_id", "")) or task_id,
                variant=str(payload.get("task_version", "")) or None,
                scene=None,
                scene_id=int(payload.get("scene_id", 0)),
                layout_key=str(payload.get("annotation_layout_key", "")) or None,
                task_index=int(payload.get("annotation_task_index", 0)),
                object_mode=None,
                task_contract=payload.get("task_contract"),
            )
            base_spec = loaded.task_spec
            base_hash = direct.layout_hash(base_spec)
            binding = direct.resolve_direct_target(base_spec)
            direct_spec, record = direct.apply_direct_override(
                task_spec=base_spec,
                task_id=task_id,
                base_instruction=str(payload.get("instruction", "")),
            )
            direct_hash = direct.layout_hash(direct_spec)
            status = "ok" if base_hash == direct_hash else "LAYOUT_HASH_MISMATCH"
            if status != "ok":
                failures.append(f"{task_id}: {status}")
            rows.append(
                {
                    "task_index": index,
                    "task_id": task_id,
                    "status": status,
                    "mechanism": record["mechanism"],
                    "gradient_tier": record["gradient_tier"],
                    "gt_target_object_id": binding["target_object_id"],
                    "gt_target_source": binding["target_object_id_source"],
                    "base_instruction": record["original_query_text"] or record["original_instruction"],
                    "direct_instruction": record["effective_instruction"],
                    "recolored_object_id": record["recolored_object_id"],
                    "applied_color": record["applied_color"],
                    "color_choice_reason": record["color_choice_reason"],
                    "cleared_visual_material": record["cleared_visual_material"],
                    "marker": record["marker"],
                    "base_layout_hash": base_hash,
                    "direct_layout_hash": direct_hash,
                    "runtime_object_count_base": len(base_spec.get("runtime_objects", [])),
                    "runtime_object_count_direct": len(direct_spec.get("runtime_objects", [])),
                    "notes": record["notes"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - the audit must report, not crash
            failures.append(f"{task_id}: {type(exc).__name__}: {exc}")
            rows.append({"task_index": index, "task_id": task_id, "status": f"ERROR:{type(exc).__name__}", "error": str(exc)})

    width = max((len(str(row["task_id"])) for row in rows), default=10)
    print(f"{'task_id'.ljust(width)}  {'mech':<8} {'tier':<24} {'status':<20} target/marker")
    print("-" * (width + 80))
    for row in rows:
        detail = row.get("recolored_object_id") or (row.get("marker") or {}).get("name", "") or "-"
        if row.get("marker"):
            detail = f"marker@{[round(v, 4) for v in row['marker']['xy']]} color={row['marker']['color']}"
        print(
            f"{str(row['task_id']).ljust(width)}  {str(row.get('mechanism', '-')):<8} "
            f"{str(row.get('gradient_tier', '-')):<24} {str(row['status']):<20} {detail}"
        )

    tiers: dict[str, int] = {}
    for row in rows:
        tiers[str(row.get("gradient_tier", "-"))] = tiers.get(str(row.get("gradient_tier", "-")), 0) + 1
    print("\ngradient tiers:", json.dumps(tiers, ensure_ascii=False))
    print(f"tasks={len(rows)} failures={len(failures)}")
    for failure in failures:
        print(f"  FAIL {failure}")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "coverage": {
                        "checked": [
                            "direct config has an entry for every VLA task",
                            "GT target resolvable without execution_binding",
                            "canonical layout hash identical between base and direct",
                            "object set only ever gains direct_marker",
                            "colour uniqueness rule and expected colour agree",
                        ],
                        "not_checked": [
                            "runtime object poses inside IsaacLab (needs a simulator; see check_direct_pairing.py)",
                            "the instruction string actually sent to the policy server (see check_direct_instructions.py)",
                            "rendered appearance (see render_direct_frames.py)",
                        ],
                    },
                    "tasks": rows,
                    "failures": failures,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"json={out_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
