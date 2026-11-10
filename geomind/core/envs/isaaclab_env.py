from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ..observation import build_raw_observation
from ..types import EnvCfg
from ..resolver import resolve_task_request
from .mock_backend import MockBackendEnv


class GeoMindIsaacLabEnv(gym.Env):
    """Temporary runtime delegate for the registered task entry.

    This env is intentionally no longer the registered gym surface. It remains as a
    compatibility adapter that translates `EnvCfg` into the temporary mock backend
    until an IsaacLab-native delegate replaces it.
    """

    metadata = {"render_modes": []}

    def __init__(self, task_id: str, cfg: EnvCfg | None = None, **kwargs: Any) -> None:
        super().__init__()

        self.task_id = str(task_id)
        if cfg is None:
            resolved = resolve_task_request(
                self.task_id,
                max_episode_steps=kwargs.get("max_episode_steps"),
                backend=kwargs.get("backend"),
            )
            self.cfg = resolved.env_cfg
        else:
            self.cfg = cfg
        if self.cfg.runtime_mode != "temporary_mock_delegate":
            raise ValueError(
                f"GeoMindIsaacLabEnv only supports temporary_mock_delegate, got {self.cfg.runtime_mode!r}"
            )
        self._backend = self._build_backend(self.cfg)
        self.action_space = self._backend.action_space
        self.observation_space = gym.spaces.Dict({})
        self._episode_step = 0
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _build_backend(self, cfg: EnvCfg):
        if cfg.backend == "mock":
            return MockBackendEnv(cfg=cfg)
        raise ValueError(f"Unsupported backend: {cfg.backend!r}")

    def _adapt_observation(self, backend_obs: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        return build_raw_observation(
            instruction=self.cfg.task.instruction,
            step=self._episode_step,
            max_steps=self.cfg.max_episode_steps,
            state={"backend_obs": backend_obs},
            cameras={name: None for name in self.cfg.cameras},
            action_info={"joint_action": self._last_action.tolist(), "joint_velocity": [], "ee_action": []},
            meta={
                "task_id": self.cfg.task.task_id,
                "canonical_task_id": self.cfg.task.canonical_task_id,
                "family": self.cfg.task.family,
                "variant": self.cfg.task.variant,
                "task_version": self.cfg.task.task_version,
                "evaluator_key": self.cfg.task.evaluator_key,
                "scene_id": self.cfg.task.scene_id,
                "scene_name": self.cfg.scene_ref.get("name", ""),
                "layout_key": self.cfg.layout_ref.get("layout_key", ""),
                "max_episode_steps": self.cfg.max_episode_steps,
                "runtime_mode": self.cfg.runtime_mode,
                "runtime_delegate": self.__class__.__name__,
                **dict(info or {}),
            },
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._episode_step = 0
        backend_obs, info = self._backend.reset(seed=seed, options=options)
        return self._adapt_observation(backend_obs, info), info

    def step(self, action):
        self._episode_step += 1
        self._last_action = np.asarray(action, dtype=np.float32)
        backend_obs, reward, terminated, truncated, info = self._backend.step(action)
        obs = self._adapt_observation(backend_obs, info)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        close_fn = getattr(self._backend, "close", None)
        if callable(close_fn):
            close_fn()
