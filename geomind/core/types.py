from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParsedTaskId:
    raw: str
    family: str
    split: str
    variant: str
    task_name: str
    version: int


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    canonical_task_id: str
    family: str
    split: str
    variant: str
    task_version: str
    task_name: str
    version: int
    scene_id: int
    instruction: str
    evaluator_key: str
    public_task_id: str = ""
    task_family: str = ""
    condition: str = "Base"
    layout: str = "Custom"
    interface: str = "Standard"
    task_variant: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    semantic_module: str = ""
    skill_type: str = ""
    phase: str = ""
    legacy_task_id: str = ""
    annotation_layout_key: str = ""
    annotation_task_index: int = 0
    object_set: str = ""
    assets_group: str = ""
    supported_task_versions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    implementation_status: str = "implemented"
    release_version: str = "1.0"

    @property
    def domain(self) -> str:
        return "SpatialMemory" if self.task_family in {"SceneUpdating", "IdentityTracking"} else "SpatialReasoning"


@dataclass(frozen=True)
class EnvCfg:
    task: TaskSpec
    max_episode_steps: int = 200
    num_envs: int = 1
    device: str = "cpu"
    use_fabric: bool | None = None
    cameras: tuple[str, ...] = ("ego_view", "table_cam")
    backend: str = "mock"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    scene_ref: dict[str, Any] = field(default_factory=dict)
    layout_ref: dict[str, Any] = field(default_factory=dict)
    runtime_mode: str = "temporary_mock_backend"


@dataclass(frozen=True)
class ResolvedTaskRequest:
    requested_task_id: str
    resolved_task_id: str
    spec: TaskSpec
    env_cfg: EnvCfg
    config_path: Path


@dataclass(frozen=True)
class RunResult:
    task_id: str
    success: bool
    episode_steps: int
    reward_sum: float
    done_reason: str
    output_dir: Path
    details: dict[str, Any] = field(default_factory=dict)
