from __future__ import annotations

from collections.abc import Mapping
from typing import Any


STATE_REFRESH_RUNNER_FAMILY = "state_refresh"
GEOMETRIC_STRUCTURE_RUNNER_FAMILY = "geometric_structure"

ALIGNMENT_RUNNER_FAMILY = "alignment"
ROUTING_RUNNER_FAMILY = "routing"
SPATIAL_PLACEMENT_RUNNER_FAMILY = "spatial_placement"
ANCHOR_REFERENCE_RUNNER_FAMILY = "anchor_reference"
CONSTRAINT_SELECTION_RUNNER_FAMILY = "constraint_selection"
TOPOLOGICAL_RUNNER_FAMILY = "topological"
UNKNOWN_RUNNER_FAMILY = "unknown"

RUNNER_DISPATCH_MODE_PERMISSIVE = "permissive"
RUNNER_DISPATCH_MODE_STRICT = "strict"
RUNNER_DISPATCH_MODES = (
    RUNNER_DISPATCH_MODE_PERMISSIVE,
    RUNNER_DISPATCH_MODE_STRICT,
)

DISPATCH_STATUS_RUNNABLE = "runnable"
DISPATCH_STATUS_DECLARED_NOT_RUNNABLE = "declared_not_runnable"
DISPATCH_STATUS_DECLARED_RUNNER_UNIMPLEMENTED = "declared_runner_unimplemented"
DISPATCH_STATUS_UNKNOWN_RUNNER = "unknown_runner"


def _runner(
    *,
    child_runner_script_name: str,
    summarizer_script_name: str,
    output_subdir: str,
    summary_group_by_key: str,
    default_agent_group_key: str,
    agent_episode_env: str,
    default_agent_episodes: int,
    implemented: bool,
    task_families: tuple[str, ...] = (),
    task_ids: tuple[str, ...] = (),
    skill_types: tuple[str, ...] = (),
    semantic_modules: tuple[str, ...] = (),
    agent_names: tuple[str, ...] = (),
    notes: str = "",
) -> dict[str, Any]:
    return {
        "child_runner_script_name": child_runner_script_name,
        "summarizer_script_name": summarizer_script_name,
        "output_subdir": output_subdir,
        "summary_group_by_key": summary_group_by_key,
        "default_agent_group_key": default_agent_group_key,
        "agent_episode_env": agent_episode_env,
        "default_agent_episodes": int(default_agent_episodes),
        "implemented": bool(implemented),
        "task_families": tuple(str(value) for value in task_families if str(value).strip()),
        "task_ids": tuple(str(value) for value in task_ids if str(value).strip()),
        "skill_types": tuple(str(value) for value in skill_types if str(value).strip()),
        "semantic_modules": tuple(str(value) for value in semantic_modules if str(value).strip()),
        "agent_names": tuple(str(value) for value in agent_names if str(value).strip()),
        "notes": str(notes),
    }


_STATE_REFRESH_RUNNER_CFG = _runner(
    child_runner_script_name="run_state_reindexing_matrix.sh",
    summarizer_script_name="summarize_state_reindexing_results.py",
    output_subdir="state_refresh",
    summary_group_by_key="state_refresh_group_by",
    default_agent_group_key="state_refresh",
    agent_episode_env="STATE_REFRESH_AGENT_EPISODES",
    default_agent_episodes=5,
    implemented=True,
    task_families=("state_reindexing", "historical_identity_tracking"),
    agent_names=(
        "oracle",
        "random",
        "state_refresh_core_oracle",
        "state_refresh_core_random",
        "historical_oracle",
        "historical_final_frame_only",
    ),
    notes="Batch runner for state reindexing and historical identity tracking.",
)
_GEOMETRIC_STRUCTURE_RUNNER_CFG = _runner(
    child_runner_script_name="run_geometric_inference_matrix.sh",
    summarizer_script_name="summarize_geometric_inference_results.py",
    output_subdir="geometric_structure",
    summary_group_by_key="geometric_structure_group_by",
    default_agent_group_key="geometric_structure",
    agent_episode_env="GEOMETRIC_STRUCTURE_AGENT_EPISODES",
    default_agent_episodes=1,
    implemented=True,
    task_families=("explicit_rule_extrapolation", "implicit_trend_fitting"),
    task_ids=("square_completion",),
    agent_names=(
        "oracle",
        "random",
        "explicit_midpoint_oracle",
        "explicit_extension_oracle",
        "explicit_random",
        "implicit_trend_oracle",
        "implicit_trend_random",
    ),
    notes="Batch runner for explicit geometric rules and implicit trend fitting.",
)

RUNNER_FAMILY_REGISTRY: dict[str, dict[str, Any]] = {
    STATE_REFRESH_RUNNER_FAMILY: _STATE_REFRESH_RUNNER_CFG,
    GEOMETRIC_STRUCTURE_RUNNER_FAMILY: _GEOMETRIC_STRUCTURE_RUNNER_CFG,
    ALIGNMENT_RUNNER_FAMILY: _runner(
        child_runner_script_name="",
        summarizer_script_name="",
        output_subdir="alignment",
        summary_group_by_key="joint_group_by",
        default_agent_group_key="alignment",
        agent_episode_env="ALIGNMENT_AGENT_EPISODES",
        default_agent_episodes=1,
        implemented=False,
        task_families=("shape_completion",),
        skill_types=("alignment",),
        notes="Reserved child runner family for alignment-heavy structure tasks.",
    ),
    ROUTING_RUNNER_FAMILY: _runner(
        child_runner_script_name="",
        summarizer_script_name="",
        output_subdir="routing",
        summary_group_by_key="joint_group_by",
        default_agent_group_key="routing",
        agent_episode_env="ROUTING_AGENT_EPISODES",
        default_agent_episodes=1,
        implemented=False,
        skill_types=("routing",),
        notes="Reserved child runner family for placement-routing upgrades.",
    ),
    SPATIAL_PLACEMENT_RUNNER_FAMILY: _runner(
        child_runner_script_name="",
        summarizer_script_name="",
        output_subdir="spatial_placement",
        summary_group_by_key="joint_group_by",
        default_agent_group_key="spatial_placement",
        agent_episode_env="SPATIAL_PLACEMENT_AGENT_EPISODES",
        default_agent_episodes=1,
        implemented=False,
        task_families=("spatial_placement",),
        notes="Reserved child runner family for base/stress spatial placement tasks.",
    ),
    ANCHOR_REFERENCE_RUNNER_FAMILY: _runner(
        child_runner_script_name="",
        summarizer_script_name="",
        output_subdir="anchor_reference",
        summary_group_by_key="joint_group_by",
        default_agent_group_key="anchor_reference",
        agent_episode_env="ANCHOR_REFERENCE_AGENT_EPISODES",
        default_agent_episodes=1,
        implemented=False,
        task_families=("anchor_reference",),
        notes="Reserved child runner family for T1 anchor-reference tasks.",
    ),
    CONSTRAINT_SELECTION_RUNNER_FAMILY: _runner(
        child_runner_script_name="",
        summarizer_script_name="",
        output_subdir="constraint_selection",
        summary_group_by_key="joint_group_by",
        default_agent_group_key="constraint_selection",
        agent_episode_env="CONSTRAINT_SELECTION_AGENT_EPISODES",
        default_agent_episodes=1,
        implemented=False,
        task_families=("constraint_selection",),
        notes="Reserved child runner family for T3 constraint-selection tasks.",
    ),
    TOPOLOGICAL_RUNNER_FAMILY: _runner(
        child_runner_script_name="",
        summarizer_script_name="",
        output_subdir="topological",
        summary_group_by_key="joint_group_by",
        default_agent_group_key="topological",
        agent_episode_env="TOPOLOGICAL_AGENT_EPISODES",
        default_agent_episodes=1,
        implemented=False,
        semantic_modules=("topological_reasoning",),
        notes="Reserved placeholder for future topology- or graph-style task families.",
    ),
}


RUNNER_FAMILY_ORDER = tuple(RUNNER_FAMILY_REGISTRY.keys())

_TASK_ID_TO_RUNNER_FAMILY: dict[str, str] = {}
_SKILL_TYPE_TO_RUNNER_FAMILY: dict[str, str] = {}
_TASK_FAMILY_TO_RUNNER_FAMILY: dict[str, str] = {}
_SEMANTIC_MODULE_TO_RUNNER_FAMILY: dict[str, str] = {}
_AGENT_TO_RUNNER_FAMILIES: dict[str, list[str]] = {}

for _runner_family, _config in RUNNER_FAMILY_REGISTRY.items():
    for _task_id in _config.get("task_ids", ()):
        _TASK_ID_TO_RUNNER_FAMILY.setdefault(str(_task_id), _runner_family)
    for _skill_type in _config.get("skill_types", ()):
        _SKILL_TYPE_TO_RUNNER_FAMILY.setdefault(str(_skill_type), _runner_family)
    for _task_family in _config.get("task_families", ()):
        _TASK_FAMILY_TO_RUNNER_FAMILY.setdefault(str(_task_family), _runner_family)
    for _semantic_module in _config.get("semantic_modules", ()):
        _SEMANTIC_MODULE_TO_RUNNER_FAMILY.setdefault(str(_semantic_module), _runner_family)
    for _agent_name in _config.get("agent_names", ()):
        _AGENT_TO_RUNNER_FAMILIES.setdefault(str(_agent_name), []).append(_runner_family)


def _as_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def registered_runner_families() -> tuple[str, ...]:
    return RUNNER_FAMILY_ORDER


def runner_family_config(runner_family: str) -> dict[str, Any]:
    key = _as_str(runner_family)
    config = RUNNER_FAMILY_REGISTRY.get(key, {})
    return dict(config) if isinstance(config, Mapping) else {}


def runner_family_is_implemented(runner_family: str) -> bool:
    return bool(runner_family_config(runner_family).get("implemented", False))


def resolve_runner_family(
    *,
    task_id: Any = None,
    family: Any = None,
    semantic_module: Any = None,
    skill_type: Any = None,
    preferred_runner_family: Any = None,
) -> str:
    preferred = _as_str(preferred_runner_family)
    if preferred in RUNNER_FAMILY_REGISTRY:
        return preferred

    resolved_task_id = _as_str(task_id)
    if resolved_task_id and resolved_task_id in _TASK_ID_TO_RUNNER_FAMILY:
        return _TASK_ID_TO_RUNNER_FAMILY[resolved_task_id]

    resolved_skill_type = _as_str(skill_type)
    if resolved_skill_type and resolved_skill_type in _SKILL_TYPE_TO_RUNNER_FAMILY:
        return _SKILL_TYPE_TO_RUNNER_FAMILY[resolved_skill_type]

    resolved_family = _as_str(family)
    if resolved_family and resolved_family in _TASK_FAMILY_TO_RUNNER_FAMILY:
        return _TASK_FAMILY_TO_RUNNER_FAMILY[resolved_family]

    resolved_semantic_module = _as_str(semantic_module)
    if resolved_semantic_module and resolved_semantic_module in _SEMANTIC_MODULE_TO_RUNNER_FAMILY:
        return _SEMANTIC_MODULE_TO_RUNNER_FAMILY[resolved_semantic_module]

    return UNKNOWN_RUNNER_FAMILY


def resolve_runner_family_for_entry(entry: Mapping[str, Any] | None) -> str:
    payload = entry or {}
    return resolve_runner_family(
        task_id=payload.get("task_id"),
        family=payload.get("family"),
        semantic_module=payload.get("semantic_module"),
        skill_type=payload.get("skill_type"),
        preferred_runner_family=payload.get("runner_family"),
    )


def runner_families_for_agent(agent_name: Any) -> tuple[str, ...]:
    return tuple(_AGENT_TO_RUNNER_FAMILIES.get(_as_str(agent_name), ()))


def split_agent_names_by_runner(agent_names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {runner_family: [] for runner_family in registered_runner_families()}
    grouped[UNKNOWN_RUNNER_FAMILY] = []
    seen: dict[str, set[str]] = {
        runner_family: set() for runner_family in grouped
    }
    for raw_name in agent_names:
        agent_name = _as_str(raw_name)
        if not agent_name:
            continue
        runner_families = runner_families_for_agent(agent_name)
        if not runner_families:
            if agent_name not in seen[UNKNOWN_RUNNER_FAMILY]:
                grouped[UNKNOWN_RUNNER_FAMILY].append(agent_name)
                seen[UNKNOWN_RUNNER_FAMILY].add(agent_name)
            continue
        for runner_family in runner_families:
            if agent_name in seen[runner_family]:
                continue
            grouped[runner_family].append(agent_name)
            seen[runner_family].add(agent_name)
    return grouped


__all__ = [
    "ALIGNMENT_RUNNER_FAMILY",
    "ANCHOR_REFERENCE_RUNNER_FAMILY",
    "CONSTRAINT_SELECTION_RUNNER_FAMILY",
    "GEOMETRIC_STRUCTURE_RUNNER_FAMILY",
    "DISPATCH_STATUS_DECLARED_NOT_RUNNABLE",
    "DISPATCH_STATUS_DECLARED_RUNNER_UNIMPLEMENTED",
    "DISPATCH_STATUS_RUNNABLE",
    "DISPATCH_STATUS_UNKNOWN_RUNNER",
    "ROUTING_RUNNER_FAMILY",
    "RUNNER_DISPATCH_MODE_PERMISSIVE",
    "RUNNER_DISPATCH_MODE_STRICT",
    "RUNNER_DISPATCH_MODES",
    "RUNNER_FAMILY_REGISTRY",
    "SPATIAL_PLACEMENT_RUNNER_FAMILY",
    "STATE_REFRESH_RUNNER_FAMILY",
    "TOPOLOGICAL_RUNNER_FAMILY",
    "UNKNOWN_RUNNER_FAMILY",
    "registered_runner_families",
    "resolve_runner_family",
    "resolve_runner_family_for_entry",
    "runner_family_config",
    "runner_family_is_implemented",
    "runner_families_for_agent",
    "split_agent_names_by_runner",
]
