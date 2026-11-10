from __future__ import annotations

from dataclasses import dataclass

from isaaclab.utils import configclass

from .geomind_ik_rel_visuomotor_env_cfg import FrankaGeoMindIKRelVisuomotorEnvCfg


DEFAULT_TASK_ID = "reindex_extreme"
DEFAULT_TASK_VARIANT = "standard"
DEFAULT_OBJECT_MODE = "geometric"
DEFAULT_SCENE_ID = 65
DEFAULT_TASK_LAYOUT_KEY = "state_reindex_blocks_line_4"
DEFAULT_TASK_INDEX = 0
DEFAULT_TASK_CATEGORY = "generic"
DEFAULT_TASK_MODE = "vla"
DEFAULT_ENV_ID = "GeoMind-ReindexExtreme-Standard-Franka-IK-Rel-Visuomotor-v0"


@configclass
class FrankaGeoMindReindexExtremeEnvCfg(FrankaGeoMindIKRelVisuomotorEnvCfg):
    """Minimal fixed GeoMind env pinned to a current canonical debug task."""

    task_id: str = DEFAULT_TASK_ID
    task_variant: str = DEFAULT_TASK_VARIANT
    object_mode: str = DEFAULT_OBJECT_MODE
    task_layout_key: str = DEFAULT_TASK_LAYOUT_KEY
    task_index: int = DEFAULT_TASK_INDEX
    scene_id: int = DEFAULT_SCENE_ID
    task_category: str = DEFAULT_TASK_CATEGORY
    task_mode: str = DEFAULT_TASK_MODE

    def __post_init__(self):
        # Set selectors before parent init so scene/task resolution uses the fixed task.
        self.task_id = DEFAULT_TASK_ID
        self.task_variant = DEFAULT_TASK_VARIANT
        self.object_mode = DEFAULT_OBJECT_MODE
        self.task_layout_key = DEFAULT_TASK_LAYOUT_KEY
        self.task_index = DEFAULT_TASK_INDEX
        self.scene_id = DEFAULT_SCENE_ID
        self.task_category = DEFAULT_TASK_CATEGORY
        self.task_mode = DEFAULT_TASK_MODE
        super().__post_init__()


# Compatibility alias for the previous bootstrap env name.
FrankaGeoMindScene1Line3Task0EnvCfg = FrankaGeoMindReindexExtremeEnvCfg


@dataclass(frozen=True)
class TaskRegistryEntry:
    env_id: str
    cfg_entry_point: str
    task_id: str
    task_variant: str
    scene_id: int
    task_layout_key: str
    task_index: int
    object_mode: str = DEFAULT_OBJECT_MODE
    task_mode: str = DEFAULT_TASK_MODE


DEFAULT_TASK_REGISTRY_ENTRY = TaskRegistryEntry(
    env_id=DEFAULT_ENV_ID,
    cfg_entry_point=f"{__name__}:FrankaGeoMindReindexExtremeEnvCfg",
    task_id=DEFAULT_TASK_ID,
    task_variant=DEFAULT_TASK_VARIANT,
    scene_id=DEFAULT_SCENE_ID,
    task_layout_key=DEFAULT_TASK_LAYOUT_KEY,
    task_index=DEFAULT_TASK_INDEX,
)


def build_task_registry_entries() -> tuple[TaskRegistryEntry, ...]:
    """Return the minimal fixed task set for the current migration stage."""

    return (DEFAULT_TASK_REGISTRY_ENTRY,)
