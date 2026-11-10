#!/usr/bin/env bash
# Unified launcher for per-task debug wrappers under scripts/.
#
# Usage:
#   bash scripts/debug_task.sh --list
#   bash scripts/debug_task.sh --list-aliases
#   bash scripts/debug_task.sh [--task] <task_id_or_alias> [-- extra args to entry script]
#
# Common environment variables are inherited (not parsed here), e.g.:
#   HEADLESS=1 TASK_VERSION=stress SCENE_ID=65 LAYOUT_KEY=... PRESET=standard
#
# Registry: scripts/debug_task_registry.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY_JSON="${SCRIPT_DIR}/debug_task_registry.json"

usage() {
  cat <<'EOF'
Usage: bash scripts/debug_task.sh [OPTIONS] [TASK_ID_OR_ALIAS] [-- EXTRA_ARGS...]

Options:
  --list                 List all registered task ids (tab: id, category, entry).
  --list-aliases         List alias -> canonical task_id mappings.
  --list-categories      List tasks grouped by category.
  --task <task_id>       Explicit task id or alias (optional if first positional is used).
  -h, --help             Show this help.

Examples:
  bash scripts/debug_task.sh place1
  bash scripts/debug_task.sh place_directional
  HEADLESS=1 bash scripts/debug_task.sh refresh1
  TASK_VERSION=skill bash scripts/debug_task.sh place2

Registry:
  scripts/debug_task_registry.json  (requires python3 to parse)
EOF
}

if [[ ! -f "${REGISTRY_JSON}" ]]; then
  echo "error: registry missing: ${REGISTRY_JSON}" >&2
  exit 1
fi

PY_MAIN="$(cat <<'PY'
import json
import os
import sys
from pathlib import Path

# Short aliases -> canonical task_id (see scripts/README.md)
ALIASES: dict[str, str] = {
    "anchor1": "grasp_anchor_direction",
    "anchor2": "grasp_anchor_distance",
    "place1": "place_directional",
    "place2": "place_on_top",
    "place3": "place_inside",
    "filter1": "select_visual_spatial",
    "filter2": "select_exclusion_unique",
    "refresh1": "reindex_extreme",
    "refresh2": "reindex_insert",
    "refresh3": "reindex_subset",
    "refresh4": "reindex_neighbor",
    "refresh5": "ordinal_insert_static",
    "refresh6": "grid_rowcol_grounding",
    "track1": "track_single",
    "track2": "track_multi",
    "track3": "track_clutter",
    "structure1": "perfect_midpoint",
    "structure2": "isometric_extension",
    "structure3": "point_symmetry",
    "structure4": "trend_continuation",
    "structure5": "weak_destruction_control",
    "structure6": "square_completion",
    "filter1c": "select_visual_spatial_control",
    "track1c": "track_single_control",
    "place3c": "place_inside_boundary_control",
    "structure6c": "square_completion_decoy_control",
    "refresh4c": "reindex_neighbor_control",
    "refresh3c": "reindex_subset_control",
}


def _load_registry(script_dir: str) -> dict:
    reg_path = os.path.join(script_dir, "debug_task_registry.json")
    with open(reg_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    script_dir = sys.argv[1]
    cmd = sys.argv[2]
    data = _load_registry(script_dir)
    tasks = data.get("tasks") or []
    cats = data.get("categories") or {}

    if cmd == "list":
        for t in sorted(tasks, key=lambda x: (str(x.get("category", "")), str(x.get("id", "")))):
            tid = t.get("id", "")
            cat = t.get("category", "")
            entry = t.get("entry", "")
            print(f"{tid}\t{cat}\t{entry}")
        return

    if cmd == "list-aliases":
        for alias in sorted(ALIASES.keys()):
            print(f"{alias}\t{ALIASES[alias]}")
        return

    if cmd == "list-categories":
        by_cat: dict[str, list] = {}
        for t in tasks:
            c = str(t.get("category", ""))
            by_cat.setdefault(c, []).append(t)
        for c in sorted(by_cat.keys()):
            meta = cats.get(c) or {}
            label_zh = meta.get("label_zh", "")
            label_en = meta.get("label_en", "")
            header = c
            if label_zh or label_en:
                header = f"{c} ({label_zh} / {label_en})".strip()
            print(header)
            for t in sorted(by_cat[c], key=lambda x: str(x.get("id", ""))):
                print(f"  - {t.get('id')}: {t.get('entry')}")
            print()
        return

    if cmd == "normalize":
        raw = sys.argv[3].strip()
        if raw in ALIASES:
            print(f"{ALIASES[raw]}\t{raw}")
            return
        print(f"{raw}\t")
        return

    if cmd == "resolve":
        want = sys.argv[3]
        for t in tasks:
            if str(t.get("id", "")) == want:
                entry = t.get("entry")
                if not entry:
                    print("error: task has no entry", file=sys.stderr)
                    sys.exit(3)
                print(entry)
                if t.get("backend"):
                    print(f"# backend: {t['backend']}", file=sys.stderr)
                return
        print(
            f"error: unknown task_id {want!r}. Try: bash scripts/debug_task.sh --list or --list-aliases",
            file=sys.stderr,
        )
        sys.exit(2)

    if cmd == "print_loaded":
        canonical = sys.argv[3]
        alias_token = sys.argv[4] if len(sys.argv) > 4 else ""
        repo_root = str(Path(script_dir).resolve().parent)

        alias_line = alias_token if alias_token.strip() else "<none>"
        scene = os.environ.get("SCENE_ID")
        layout = os.environ.get("LAYOUT_KEY")
        tv_env = os.environ.get("TASK_VERSION")

        md: dict = {}
        try:
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from geomind_lab.mimic.utils.task_metadata_registry import build_task_metadata

            md = build_task_metadata(canonical, task_version=tv_env)
        except Exception:
            md = {}

        def pick(key: str, default: str = "<unknown>") -> str:
            v = md.get(key) if md else None
            if v is None or v == "":
                return default
            return str(v)

        tv_show = tv_env if tv_env else pick("task_version", "<unknown>")
        print("[Task Loaded]")
        print(f"alias: {alias_line}")
        print(f"task_id: {canonical}")
        print(f"semantic_module: {pick('semantic_module')}")
        print(f"task_version: {tv_show}")
        print(f"phase: {pick('phase')}")
        print(f"instruction: {pick('instruction')}")
        print(f"scene_id: {scene if scene else '<unset>'}")
        print(f"layout_key: {layout if layout else '<unset>'}")
        return

    print("error: bad command", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
PY
)"

run_python() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -c "$PY_MAIN" "$@"
    return
  fi
  echo "error: python3 is required to read debug_task_registry.json" >&2
  exit 1
}

LIST=0
LISTALIAS=0
LISTCAT=0
TASK_RAW=""
PASS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list)
      LIST=1
      shift
      ;;
    --list-aliases)
      LISTALIAS=1
      shift
      ;;
    --list-categories)
      LISTCAT=1
      shift
      ;;
    --task)
      if [[ -z "${2:-}" ]]; then
        echo "error: --task requires a value" >&2
        exit 2
      fi
      TASK_RAW="$2"
      shift 2
      ;;
    --)
      shift
      PASS=( "$@" )
      break
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      echo "hint: bash scripts/debug_task.sh --help" >&2
      exit 2
      ;;
    *)
      if [[ -z "${TASK_RAW}" ]]; then
        TASK_RAW="$1"
      else
        echo "error: unexpected argument: $1 (use -- to pass args to the entry script)" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ "${LIST}" -eq 1 ]]; then
  run_python "${SCRIPT_DIR}" list
  exit 0
fi

if [[ "${LISTALIAS}" -eq 1 ]]; then
  run_python "${SCRIPT_DIR}" list-aliases
  exit 0
fi

if [[ "${LISTCAT}" -eq 1 ]]; then
  run_python "${SCRIPT_DIR}" list-categories
  exit 0
fi

if [[ -z "${TASK_RAW}" ]]; then
  echo "error: missing task_id or alias. Examples: bash scripts/debug_task.sh place1" >&2
  echo "hint: bash scripts/debug_task.sh --list-aliases" >&2
  exit 2
fi

# shellcheck disable=SC2207
IFS=$'\t' read -r TASK_ID ALIAS_TOKEN < <(run_python "${SCRIPT_DIR}" normalize "${TASK_RAW}")
export TASK_ID

set +e
ENTRY="$(run_python "${SCRIPT_DIR}" resolve "${TASK_ID}")"
resolve_rc=$?
set -e
if [[ "${resolve_rc}" -ne 0 ]]; then
  exit "${resolve_rc}"
fi
if [[ -z "${ENTRY}" ]]; then
  echo "error: registry resolve returned empty entry for task_id=${TASK_ID}" >&2
  exit 3
fi

ENTRY_PATH="${SCRIPT_DIR}/${ENTRY}"
if [[ ! -f "${ENTRY_PATH}" ]]; then
  echo "error: entry script not found for task_id=${TASK_ID}: ${ENTRY_PATH}" >&2
  echo "hint: fix scripts/debug_task_registry.json or restore the missing file." >&2
  exit 3
fi

run_python "${SCRIPT_DIR}" print_loaded "${TASK_ID}" "${ALIAS_TOKEN}"

if [[ "${HEADLESS:-0}" == "1" ]]; then
  export HEADLESS_CAMERA_VIDEO="${HEADLESS_CAMERA_VIDEO:-1}"
  echo "headless_camera_video=${HEADLESS_CAMERA_VIDEO}"
fi

exec bash "${ENTRY_PATH}" "${PASS[@]}"
