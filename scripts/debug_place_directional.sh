#!/usr/bin/env bash
set -euo pipefail

GEOMIND_DEFAULT_MDL_SYSTEM_PATH="${GEOMIND_GRSCENES_HOME:-${GEOMIND_ASSET_BUNDLE_ROOT:-${GEOMIND_REPO_ROOT:-$PWD}/external/geomind-assets-v1}/grscenes_home}/Materials"
if [[ -z "${MDL_SYSTEM_PATH:-}" ]]; then
  if [[ -d "${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}" ]]; then
    export MDL_SYSTEM_PATH="${GEOMIND_DEFAULT_MDL_SYSTEM_PATH}"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/geomind_debug_task_common.sh"

export TASK_ID="${TASK_ID:-place_directional}"
export TASK_VERSION="${TASK_VERSION:-${PRESET:-base}}"
geomind_debug_scan_request_args 0 "$@"
geomind_debug_sync_task_version_to_variant
geomind_resolve_task_context "${REPO_ROOT}" "${TASK_ID}"

export TASK_INDEX="${TASK_INDEX:-0}"
export AGENT="${AGENT:-oracle}"
export AGENT_EPISODES="${AGENT_EPISODES:-1}"
export RESET_SETTLE_STEPS="${RESET_SETTLE_STEPS:-24}"
export POST_PLACE_SETTLE_STEPS="${POST_PLACE_SETTLE_STEPS:-24}"
export HEADLESS="${HEADLESS:-0}"
export DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
export DEVICE="${DEVICE:-}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
geomind_default_output_dir "${REPO_ROOT}" "${TIMESTAMP}"
export SUMMARY_JSON_OUT="${SUMMARY_JSON_OUT:-${OUTPUT_DIR}/summary.json}"
export RUNNER_FAMILY="${RUNNER_FAMILY:-spatial_placement}"

if [[ "${HEADLESS}" == "1" ]]; then
  export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
fi

geomind_debug_print_core_context "GeoMind Place Directional"

# Resolved runtime clutter (viz_clutter_*): names and spawned asset / primitive params.
# Uses same LAYOUT_KEY / TASK_INDEX / SCENE_ID / OBJECT_MODE / GEOMIND_VISUAL_CLUTTER_SEED as the run below.
geomind_debug_print_place_directional_clutter() {
  local repo="${REPO_ROOT}"
  PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}" python3 - "${repo}" <<'PY'
import os
import sys

repo_root = sys.argv[1]
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

layout_key = os.environ.get("LAYOUT_KEY", "").strip()
task_index = int(os.environ.get("TASK_INDEX", "0"))
scene_id = int(os.environ.get("SCENE_ID", "65"))
object_mode = (os.environ.get("OBJECT_MODE") or "").strip() or "geometric"
seed = os.environ.get("GEOMIND_VISUAL_CLUTTER_SEED", "").strip() or "(unset → random per process)"
fixed_kind = os.environ.get("GEOMIND_VISUAL_CLUTTER_KIND", "").strip() or "(unset → random per slot)"

from geomind_lab.tasks.manager_based.manipulation.geomind import task_specs as ts

sp = ts._build_lightweight_scene_profile(layout_key, scene_id)
spec = ts.load_task_spec(layout_key, task_index, sp)
spec = ts.apply_object_mode(spec, object_mode=object_mode)

rows: list[str] = []
for obj in spec.get("runtime_objects") or []:
    if not isinstance(obj, dict):
        continue
    name = str(obj.get("name", "")).strip()
    if not (name.startswith("viz_clutter_") or name.startswith("plate_trinket_")):
        continue
    kind = str(obj.get("kind", "")).strip()
    if kind == "usd":
        cat = str(obj.get("asset_category", "")).strip()
        key = str(obj.get("asset_key", "")).strip()
        sc = obj.get("scale")
        rows.append(f"  {name}: usd  {cat}/{key}  scale={sc}")
    elif kind == "primitive_cylinder":
        rows.append(
            f"  {name}: primitive_cylinder  r={obj.get('radius')}  h={obj.get('height')}"
        )
    elif kind == "primitive_cone":
        rows.append(
            f"  {name}: primitive_cone  r={obj.get('radius')}  h={obj.get('height')}"
        )
    elif kind == "primitive_cuboid":
        rows.append(f"  {name}: primitive_cuboid  size={obj.get('size')}")
    else:
        rows.append(f"  {name}: {kind!r}  {obj}")

print("------------------------------------------------------------------------")
print("[clutter] GEOMIND_VISUAL_CLUTTER_SEED=%s" % (seed,))
print("[clutter] GEOMIND_VISUAL_CLUTTER_KIND=%s" % (fixed_kind,))
print("[clutter] layout_key=%s  task_index=%s  scene_id=%s  object_mode=%s" % (layout_key, task_index, scene_id, object_mode))
if not rows:
    print("[clutter] (no viz_clutter_* runtime objects in this layout)")
else:
    for line in rows:
        print(line)
print("------------------------------------------------------------------------")
PY
}

geomind_debug_print_place_directional_clutter

exec bash "${SCRIPT_DIR}/debug_protocol_spatial_placement.sh" "${GEOMIND_FORWARD_ARGS[@]}" "$@"
