from __future__ import annotations

from typing import Any

from ...core.types import TaskSpec


class BaseFamilyBuilder:
    family_name = "base"

    def build_context(self, task: TaskSpec) -> dict[str, Any]:
        return {
            "family": task.family,
            "task_id": task.public_task_id,
            "task_family": task.task_family,
            "condition": task.condition,
            "layout": task.layout,
            "version": task.version,
            "task_variant": task.task_variant,
            "parameters": dict(task.parameters),
            "task_name": task.task_name,
            "variant": task.variant,
            "scene_id": task.scene_id,
        }
