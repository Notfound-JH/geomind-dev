#!/usr/bin/env python3
"""Verification #3: the direct instruction actually reaches the policy server.

Runs fully offline. For every VLA task it applies the direct override, feeds the
resulting instruction through the exact observation -> payload -> OpenPI-prompt chain
used at rollout time, and asserts the string that would be sent matches the override
and contains no residual spatial-relation wording from the base instruction.

    python scripts/check_direct_instructions.py --json docs/direct_ablation/instruction_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geomind.core.observation import build_raw_observation  # noqa: E402
from geomind.tasks import direct_ablation as direct  # noqa: E402
from geomind.core.schema import normalize_task_config

VLA_TASK_DIR = REPO_ROOT / "geomind" / "configs" / "vla_tasks"

# Wording that would mean the "direct" instruction still requires a spatial inference.
BANNED_SUBSTRINGS = (
    "left",
    "right",
    "closest",
    "nearest",
    "between",
    "midway",
    "row ",
    "column",
    "second",
    "third",
    "fourth from",
    "leftmost",
    "rightmost",
    "started",
    "symmetric",
    "extension",
    "continue",
    "reorder",
    "among",
)
# Tasks whose base instruction is already direct and therefore intentionally unchanged.
UNCHANGED_OK = {"PlaceByTopologyInsideVla_v0", "PlaceByTopologyOnTopVla_v0"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import (
        load_task,
        resolve_task_instruction,
        resolve_task_protocol,
    )

    sys.path.insert(0, str(REPO_ROOT))
    import geomind_client  # noqa: PLC0415 - repo-root hook module

    rows: list[dict] = []
    failures: list[str] = []

    for path in sorted(VLA_TASK_DIR.glob("*.json")):
        if path.name == "legacy_layouts.json":
            continue
        payload = normalize_task_config(json.loads(path.read_text(encoding="utf-8")))
        if payload.get("implementation_status") == "placeholder":
            continue
        task_id = str(payload.get("task_id", ""))
        loaded = load_task(
            task_id=str(payload.get("canonical_task_id", "")) or task_id,
            task_contract=payload.get("task_contract"),
            variant=str(payload.get("task_version", "")) or None,
            scene_id=int(payload.get("scene_id", 0)),
            layout_key=str(payload.get("annotation_layout_key", "")) or None,
            task_index=int(payload.get("annotation_task_index", 0)),
        )
        direct_spec, record = direct.apply_direct_override(
            task_spec=loaded.task_spec,
            task_id=task_id,
            base_instruction=str(payload.get("instruction", "")),
        )
        # Source 1: what the env cfg would publish (protocol.query_text -> task_instruction).
        env_cfg_instruction = resolve_task_instruction(direct_spec, resolve_task_protocol(direct_spec))
        # Source 2: what the delegate hands to the observation builder.
        effective = str(record["effective_instruction"])
        raw_obs = build_raw_observation(
            instruction=effective,
            step=0,
            max_steps=1,
            state={},
            cameras={},
            action_info={"joint_action": [], "joint_velocity": [], "ee_action": []},
            meta={"task_id": task_id, "direct_ablation": record},
        )
        sent = geomind_client.build_payload(env=None, raw_obs=raw_obs, step=0, trace=[])
        prompt = str(sent.get("instruction") or raw_obs.get("instruction") or "")

        problems = []
        if prompt != effective:
            problems.append(f"payload_instruction_mismatch:{prompt!r}")
        if env_cfg_instruction != effective:
            problems.append(f"query_text_mismatch:{env_cfg_instruction!r}")
        if task_id not in UNCHANGED_OK:
            lowered = prompt.lower()
            hits = [token for token in BANNED_SUBSTRINGS if token in lowered]
            if hits:
                problems.append(f"residual_spatial_wording:{hits}")
        if problems:
            failures.append(f"{task_id}: {'; '.join(problems)}")

        rows.append(
            {
                "task_id": task_id,
                "mechanism": record["mechanism"],
                "base_instruction": str(payload.get("instruction", "")),
                "direct_instruction": effective,
                "payload_instruction": prompt,
                "env_cfg_task_instruction": env_cfg_instruction,
                "instruction_sources_patched": record["instruction_sources_patched"],
                "problems": problems,
            }
        )

    for row in rows:
        flag = "ok  " if not row["problems"] else "FAIL"
        print(f'{flag} {row["task_id"]:<48} {row["payload_instruction"]!r}')
    print(f"\ntasks={len(rows)} failures={len(failures)}")
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
                            "protocol.query_text -> resolve_task_instruction agrees with the override",
                            "delegate instruction -> build_raw_observation -> build_payload['instruction']",
                            "no residual spatial-relation wording in the direct instruction",
                        ],
                        "not_checked": [
                            "the live HTTP/WebSocket frame (needs a running policy server)",
                            "server-side prompt tokenisation",
                        ],
                        "note": (
                            "_build_openpi_infer_element sets prompt = payload['instruction'], so asserting "
                            "payload['instruction'] is equivalent to asserting the OpenPI prompt."
                        ),
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
