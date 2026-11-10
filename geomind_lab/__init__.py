"""GeoMind task package built on top of IsaacLab."""


def register_tasks() -> tuple[str, ...]:
    """Register GeoMind Gym tasks after the environment has been bootstrapped."""

    from geomind_lab.tasks.manager_based.manipulation.geomind.config import franka

    return franka.register_tasks()


def default_task_id() -> str:
    """Return the default fixed GeoMind env id used during migration."""

    from geomind_lab.tasks.manager_based.manipulation.geomind.config import franka

    return franka.DEFAULT_GEOMIND_TASK_ENV_ID


__all__ = ["bootstrap", "cli", "data", "default_task_id", "mimic", "register_tasks", "scene", "tasks", "utils"]
