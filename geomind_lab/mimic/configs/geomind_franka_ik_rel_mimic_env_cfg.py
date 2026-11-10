from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from geomind_lab.tasks.manager_based.manipulation.geomind.config.franka import (
    FrankaGeoMindIKRelVisuomotorSkillgenEnvCfg,
)


def configure_default_datagen(datagen_config) -> None:
    defaults = {
        "name": "geomind_mimic",
        "generation_guarantee": True,
        "generation_keep_failed": True,
        "generation_num_trials": 10,
        "generation_select_src_per_subtask": True,
        "generation_transform_first_robot_pose": False,
        "generation_interpolate_from_last_target_pose": True,
        "max_num_failures": 25,
        "seed": 1,
    }
    for key, value in defaults.items():
        if hasattr(datagen_config, key):
            setattr(datagen_config, key, value)


def build_place_subtask_configs(task_spec: dict, eef_name: str = "franka") -> dict[str, list[SubTaskConfig]]:
    goal_sequence = list(task_spec.get("goal_sequence", []))
    first_step = goal_sequence[0] if goal_sequence else {}
    goals = list(first_step.get("goals", []))
    target_object = goals[0].get("object_id") if goals else None
    if target_object is None:
        protocol = dict(task_spec.get("protocol", {}) or {})
        target_payload = dict(protocol.get("target_object", {}) or {})
        target_object = target_payload.get("object_id")
    if target_object is None:
        object_positions = dict(task_spec.get("object_positions", {}))
        target_object = next(iter(object_positions.keys()), None)
    if target_object is None:
        return {eef_name: []}

    subtasks = [
        SubTaskConfig(
            object_ref=str(target_object),
            subtask_term_signal="grasp_1",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.03,
            num_interpolation_steps=5,
            description=f"Grasp {target_object}",
            next_subtask_description="Move the grasped object to the target region",
        ),
        SubTaskConfig(
            object_ref=str(target_object),
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.03,
            num_interpolation_steps=5,
            description=f"Place {target_object} at the goal",
        ),
    ]
    return {eef_name: subtasks}


@configclass
class GeoMindFrankaIKRelMimicEnvCfg(FrankaGeoMindIKRelVisuomotorSkillgenEnvCfg, MimicEnvCfg):
    """Generic mimic config for GeoMind Franka IK-relative tasks."""

    def __post_init__(self):
        super().__post_init__()

        self.task_mode = "skillgen"
        configure_default_datagen(self.datagen_config)

        if getattr(self.datagen_config, "task_name", None) is None:
            task_metadata = dict(getattr(self, "task_metadata", {}) or {})
            resolved_task_id = str(task_metadata.get("task_id") or getattr(self, "task_id", "") or self.task_layout_key)
            resolved_variant = str(task_metadata.get("variant") or getattr(self, "task_variant", "") or "default")
            resolved_scene = str(
                task_metadata.get("scene_name")
                or task_metadata.get("scene")
                or getattr(self, "scene_name", "")
                or f"scene_{self.scene_id}"
            )
            resolved_object_mode = str(task_metadata.get("object_mode") or getattr(self, "object_mode", "") or "default")
            self.datagen_config.task_name = (
                f"geomind-{resolved_task_id}-{resolved_variant}-{resolved_scene}-{resolved_object_mode}-task-{self.task_index + 1}"
            )

        if not getattr(self, "subtask_configs", None):
            self.subtask_configs = build_place_subtask_configs(self.task_spec)
