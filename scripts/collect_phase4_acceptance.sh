#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${GEOMIND_PYTHON:-}" ]]; then
  PYTHON_BIN="${GEOMIND_PYTHON}"
else
  source "${SCRIPT_DIR}/lib/geomind_conda.sh"
  geomind_activate_conda "${CONDA_ENV_NAME:-isaac_curobo}" || exit 1
  PYTHON_BIN="$(command -v python)"
fi
LAYOUTS="${LAYOUTS:-2}"
ROLLOUTS_PER_LAYOUT="${ROLLOUTS_PER_LAYOUT:-2}"
META_SEED="${META_SEED:-20260806}"
LAYOUT_KEY="${LAYOUT_KEY:-grasp_anchor_direction_scene65_base}"
SCENE_ID="${SCENE_ID:-65}"
TASK_INDEX="${TASK_INDEX:-0}"
SAMPLE_HZ="${SAMPLE_HZ:-10}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/phase4_acceptance_$(date +%Y%m%d_%H%M%S)}"
PHASE4_DRY_RUN="${PHASE4_DRY_RUN:-0}"

RUNNER_ARGS=(
  --layouts "${LAYOUTS}"
  --rollouts-per-layout "${ROLLOUTS_PER_LAYOUT}"
  --meta-seed "${META_SEED}"
  --out-dir "${OUT_DIR}"
  --layout-key "${LAYOUT_KEY}"
  --scene-id "${SCENE_ID}"
  --task-index "${TASK_INDEX}"
  --sample-hz "${SAMPLE_HZ}"
  --segment-source event_kinematics_v2
  --vary-trajectory-seed
  --headless
  --verify-phase2
  --verify-phase3
)

if [[ "${PHASE4_DRY_RUN}" == "1" ]]; then
  RUNNER_ARGS+=(--dry-run)
fi

for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    PHASE4_DRY_RUN=1
  fi
done

cd "${REPO_ROOT}"

echo "Phase4 acceptance collection"
echo "  out_dir=${OUT_DIR}"
echo "  layouts=${LAYOUTS} rollouts_per_layout=${ROLLOUTS_PER_LAYOUT}"
echo "  layout_key=${LAYOUT_KEY} scene_id=${SCENE_ID} task_index=${TASK_INDEX}"
echo "  segment_source=event_kinematics_v2"

"${PYTHON_BIN}" scripts/collect_anchor_reference_trajectory_dataset.py "${RUNNER_ARGS[@]}" "$@"

if [[ "${PHASE4_DRY_RUN}" == "1" ]]; then
  echo "PHASE4_ACCEPTANCE: DRY_RUN_ONLY"
  echo "No H5 files were generated or checked."
  exit 0
fi

"${PYTHON_BIN}" - "${OUT_DIR}" "${LAYOUTS}" "${ROLLOUTS_PER_LAYOUT}" <<'PY'
import json
import sys
from pathlib import Path

import h5py

root = Path(sys.argv[1])
expected_count = int(sys.argv[2]) * int(sys.argv[3])
errors: list[str] = []

manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"PHASE4_ACCEPTANCE: FAIL\n  - missing manifest: {manifest_path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
h5_files = sorted(root.glob("layout_*/trajectory*.h5"))

if len(h5_files) != expected_count:
    errors.append(f"expected {expected_count} H5 files, found {len(h5_files)}")

required_datasets = (
    "events/object_grasped",
    "events/grasp_transition",
    "events/raw_grasp_transition_source_index",
    "events/raw_grasp_transition_timestamp_s",
    "events/raw_grasp_transition_kind",
    "labels/atomic_id",
    "labels/segment_id",
    "segments/start_step",
    "segments/end_step",
    "segments/start_reason",
    "segments/end_reason",
)

for path in h5_files:
    with h5py.File(path, "r") as handle:
        for dataset in required_datasets:
            if dataset not in handle:
                errors.append(f"{path}: missing {dataset}")

        if "metadata" not in handle:
            errors.append(f"{path}: missing metadata group")
            continue

        metadata = handle["metadata"]

        def read_meta(name: str):
            if name not in metadata:
                errors.append(f"{path}: missing metadata/{name}")
                return None
            value = metadata[name][()]
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value

        if read_meta("segmenter_version") != "event_kinematics_v2":
            errors.append(f"{path}: segmenter_version is not event_kinematics_v2")

        diagnostics_raw = read_meta("segmenter_diagnostics")
        if diagnostics_raw is not None:
            diagnostics = json.loads(str(diagnostics_raw))
            projection = diagnostics.get("event_projection", {})
            if int(projection.get("dropped_event_count", 0)) != 0:
                errors.append(f"{path}: dropped grasp/release events: {projection}")

        atomic_ids = [int(value) for value in handle["labels/atomic_id"][:].tolist()]
        if any(value < 0 or value > 9 for value in atomic_ids):
            errors.append(f"{path}: atomic label outside v1 label range")

        raw_kinds = [int(value) for value in handle["events/raw_grasp_transition_kind"][:].tolist()]
        projected_events = [int(value) for value in handle["events/grasp_transition"][:].tolist() if int(value) != 0]
        if raw_kinds and projected_events != raw_kinds:
            errors.append(f"{path}: projected event order differs: raw={raw_kinds}, projected={projected_events}")

        segment_starts = [int(value) for value in handle["segments/start_step"][:].tolist()]
        segment_ends = [int(value) for value in handle["segments/end_step"][:].tolist()]
        if any(start > end for start, end in zip(segment_starts, segment_ends)):
            errors.append(f"{path}: invalid segment range")

phase2 = manifest.get("phase2_check", {})
phase3 = manifest.get("phase3_check", {})
if not phase2.get("passed", False):
    errors.append("phase2_check_report.json did not pass")
if not phase3.get("passed", False):
    errors.append("phase3_check_report.json did not pass")

if errors:
    print("PHASE4_ACCEPTANCE: FAIL")
    for error in errors:
        print(f"  - {error}")
    raise SystemExit(1)

print("PHASE4_ACCEPTANCE: PASS")
print(f"output_dir: {root}")
print(f"h5_count: {len(h5_files)}")
print("checks: phase2, phase3, event order, no dropped events, segment ranges, atomic label range")
PY