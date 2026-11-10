"""Franka-specific GeoMind task config entry points."""

from __future__ import annotations

import gymnasium as gym

from .franka_env_cfg import FrankaGeoMindEnvCfg
from .geomind_ik_rel_env_cfg import FrankaGeoMindIKRelEnvCfg
from .geomind_ik_rel_visuomotor_env_cfg import (
    FrankaGeoMindIKRelVisuomotorEnvCfg,
    FrankaGeoMindIKRelVisuomotorSkillgenEnvCfg,
)
from .task_registry import (
    DEFAULT_TASK_REGISTRY_ENTRY,
    FrankaGeoMindScene1Line3Task0EnvCfg,
    TaskRegistryEntry,
    build_task_registry_entries,
)


def register_tasks() -> tuple[str, ...]:
    """Register the minimal fixed GeoMind task set with Gym."""

    registered_env_ids: list[str] = []
    for entry in build_task_registry_entries():
        if entry.env_id not in gym.registry:
            gym.register(
                id=entry.env_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={"env_cfg_entry_point": entry.cfg_entry_point},
                disable_env_checker=True,
            )
        registered_env_ids.append(entry.env_id)
    return tuple(registered_env_ids)


SUPPORTED_GEOMIND_TASKS = register_tasks()
DEFAULT_GEOMIND_TASK_ENV_ID = DEFAULT_TASK_REGISTRY_ENTRY.env_id

__all__ = [
    "DEFAULT_GEOMIND_TASK_ENV_ID",
    "FrankaGeoMindEnvCfg",
    "FrankaGeoMindIKRelEnvCfg",
    "FrankaGeoMindIKRelVisuomotorEnvCfg",
    "FrankaGeoMindIKRelVisuomotorSkillgenEnvCfg",
    "FrankaGeoMindScene1Line3Task0EnvCfg",
    "SUPPORTED_GEOMIND_TASKS",
    "TaskRegistryEntry",
    "build_task_registry_entries",
    "register_tasks",
]
