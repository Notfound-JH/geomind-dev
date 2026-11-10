from __future__ import annotations

from typing import Any

import gymnasium as gym

from ..resolver import resolve_task_request
from ..types import EnvCfg
from .isaaclab_native_delegate import GeoMindIsaacLabNativeDelegate
from .isaaclab_env import GeoMindIsaacLabEnv


def _build_runtime_env(*, task_id: str, cfg: EnvCfg):
    if cfg.task.implementation_status == "placeholder":
        raise NotImplementedError(f"Task {task_id!r} is registered as a placeholder and has no implementation yet.")
    if cfg.runtime_mode == "temporary_mock_delegate":
        return GeoMindIsaacLabEnv(task_id=task_id, cfg=cfg)
    if cfg.runtime_mode == "isaaclab_native_delegate":
        return GeoMindIsaacLabNativeDelegate(task_id=task_id, cfg=cfg)
    raise ValueError(
        "No runtime delegate configured for "
        f"task_id={task_id!r}, backend={cfg.backend!r}, runtime_mode={cfg.runtime_mode!r}"
    )


class GeoMindTaskEnv(gym.Env):
    """Registered IsaacLab-first task entry.

    This class is the stable gym registration surface. Runtime execution is delegated
    to a temporary mock adapter for now, so the registration path is separated from
    the temporary implementation that still stands in for IsaacLab-native envs.
    """

    metadata = {"render_modes": []}

    def __init__(self, task_id: str, cfg: EnvCfg | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.task_id = str(task_id)
        self.cfg = cfg or resolve_task_request(
            self.task_id,
            max_episode_steps=kwargs.get("max_episode_steps"),
            backend=kwargs.get("backend"),
        ).env_cfg
        self._runtime_env = _build_runtime_env(task_id=self.task_id, cfg=self.cfg)
        self.action_space = self._runtime_env.action_space
        self.observation_space = getattr(self._runtime_env, "observation_space", gym.spaces.Dict({}))

    @property
    def direct_ablation_record(self) -> dict[str, Any]:
        return dict(getattr(self._runtime_env, "direct_ablation_record", {}) or {})

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return self._runtime_env.reset(seed=seed, options=options)

    def step(self, action):
        return self._runtime_env.step(action)

    def close(self):
        close_fn = getattr(self._runtime_env, "close", None)
        if callable(close_fn):
            close_fn()
