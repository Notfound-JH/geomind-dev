from __future__ import annotations

import importlib
import inspect
from typing import Any

import gymnasium as gym

from .registry import registry
from .types import EnvCfg
from .resolver import resolve_task_request


def _resolve_entry_point(entry_point: Any):
    if callable(entry_point):
        return entry_point
    if isinstance(entry_point, str):
        module_name, attr_name = entry_point.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise ValueError(f"Unsupported entry point type: {type(entry_point)!r}")


def _invoke_factory(factory, *, task_id: str, overrides: dict[str, Any]):
    signature = inspect.signature(factory)
    kwargs: dict[str, Any] = dict(overrides)
    if "task_id" in signature.parameters:
        kwargs["task_id"] = task_id
    return factory(**kwargs)


def load_cfg_from_registry(task_name: str, entry_point_key: str, **overrides: Any) -> Any:
    spec = gym.spec(task_name.split(":")[-1])
    cfg_entry_point = spec.kwargs.get(entry_point_key)
    if cfg_entry_point is None:
        raise ValueError(f"Missing '{entry_point_key}' in gym registry for {task_name!r}")
    factory = _resolve_entry_point(cfg_entry_point)
    return _invoke_factory(factory, task_id=task_name, overrides=overrides)


def load_env_cfg_for_task(task_id: str, **overrides: Any) -> EnvCfg:
    resolved = resolve_task_request(
        task_id,
        max_episode_steps=overrides.get("max_episode_steps"),
        backend=overrides.get("backend"),
        condition=overrides.get("condition"), layout=overrides.get("layout"),
        version=overrides.get("version"), interface=overrides.get("interface"),
        task_variant=overrides.get("task_variant"), parameters=overrides.get("parameters"),
    )
    cfg = resolved.env_cfg
    # Remaining overrides keep IsaacLab-style cfg override behavior.
    mutable = dict(cfg.__dict__)
    mutable.update(
        {
            "num_envs": int(overrides.get("num_envs", cfg.num_envs)),
            "device": str(overrides.get("device", cfg.device)),
            "use_fabric": overrides.get("use_fabric", cfg.use_fabric),
            "cameras": tuple(overrides.get("cameras", cfg.cameras)),
            "backend_kwargs": {**cfg.backend_kwargs, **dict(overrides.get("backend_kwargs", {}))},
        }
    )
    mutable.pop("task", None)
    return EnvCfg(task=cfg.task, **mutable)


def parse_env_cfg(
    task_name: str,
    *,
    device: str = "cpu",
    num_envs: int | None = None,
    use_fabric: bool | None = None,
    max_episode_steps: int | None = None,
    backend: str | None = None,
    condition: str | None = None,
    layout: str | None = None,
    version: int | None = None,
    interface: str | None = None,
    task_variant: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> EnvCfg:
    resolved = resolve_task_request(
        task_name,
        max_episode_steps=max_episode_steps,
        backend=backend,
        condition=condition, layout=layout, version=version, interface=interface,
        task_variant=task_variant, parameters=parameters,
    )
    registry.register_gym_envs()
    cfg = load_cfg_from_registry(
        resolved.resolved_task_id,
        "env_cfg_entry_point",
        max_episode_steps=max_episode_steps,
        backend=backend,
    )
    if not isinstance(cfg, EnvCfg):
        raise RuntimeError(f"Configuration for {task_name!r} is not EnvCfg.")
    return EnvCfg(task=cfg.task, **{
        **{k: v for k, v in cfg.__dict__.items() if k != "task"},
        "device": device,
        "num_envs": num_envs if num_envs is not None else cfg.num_envs,
        "use_fabric": use_fabric if use_fabric is not None else cfg.use_fabric,
    })


def create_env(task_id: str, **cfg_overrides: Any):
    requested_backend = cfg_overrides.get("backend")
    native_physics = False
    if requested_backend == "isaaclab":
        from pathlib import Path
        from geomind.episodes.physical_contract import PHYSICS_TASKS
        root = Path(cfg_overrides.get("root") or Path(__file__).resolve().parents[1] / "configs")
        candidate = root / "episodes" / f"{task_id}.json"
        if not candidate.is_file() and task_id in PHYSICS_TASKS:
            candidate = root / "episodes" / f"{task_id}_v0.json"
        if candidate.is_file():
            from geomind.episodes.resolver import load_json
            native_physics = load_json(candidate).get("task_id") in PHYSICS_TASKS
        if native_physics and cfg_overrides.get("root") is None and task_id not in PHYSICS_TASKS:
            # An explicit versioned Standard registration selects its declared
            # manipulation route. Bare native names and explicitly supplied
            # episode roots retain the original release-2.0 episode API.
            try:
                registration = registry.get(task_id)
            except KeyError:
                registration = None
            if (registration is not None and registration.task_id == task_id
                    and registration.meta.get("native_assembly")):
                native_physics = False
    if requested_backend == "semantic" or native_physics:
        from geomind.episodes import create_episode_env
        allowed = {"backend", "root", "device", "headless"} if native_physics else {"backend", "root"}
        unsupported = set(cfg_overrides) - allowed
        if unsupported:
            raise ValueError(f"Semantic episodes require parameters in config, unsupported overrides: {sorted(unsupported)}")
        return create_episode_env(task_id, root=cfg_overrides.get("root"), backend=requested_backend,
                                  device=cfg_overrides.get("device", "cuda:0"), headless=cfg_overrides.get("headless", True))
    registry.register_gym_envs()
    resolved = resolve_task_request(
        task_id,
        max_episode_steps=cfg_overrides.get("max_episode_steps"),
        backend=cfg_overrides.get("backend"),
        condition=cfg_overrides.get("condition"), layout=cfg_overrides.get("layout"),
        version=cfg_overrides.get("version"), interface=cfg_overrides.get("interface"),
        task_variant=cfg_overrides.get("task_variant"), parameters=cfg_overrides.get("parameters"),
    )
    if resolved.env_cfg.runtime_mode=='assembled_native_pipeline':
        raise ValueError('This task requires the measured assembly pipeline, not an unbound manager environment. '
                         'Run python -m geomind.tools.regenerate_registered_tasks --tasks '
                         + resolved.resolved_task_id + ' --max-steps 12000 --output NEW_DIRECTORY')
    env_cfg = load_env_cfg_for_task(
        resolved.resolved_task_id,
        max_episode_steps=resolved.env_cfg.max_episode_steps,
        backend=resolved.env_cfg.backend,
        num_envs=cfg_overrides.get("num_envs", resolved.env_cfg.num_envs),
        device=cfg_overrides.get("device", resolved.env_cfg.device),
        use_fabric=cfg_overrides.get("use_fabric", resolved.env_cfg.use_fabric),
        cameras=cfg_overrides.get("cameras", resolved.env_cfg.cameras),
        backend_kwargs=cfg_overrides.get("backend_kwargs", resolved.env_cfg.backend_kwargs),
    )
    env = gym.make(resolved.resolved_task_id, cfg=env_cfg)
    # Minimal safety assertion for cfg flow consistency.
    unwrapped_cfg = getattr(getattr(env, "unwrapped", env), "cfg", None)
    if isinstance(unwrapped_cfg, EnvCfg):
        assert int(unwrapped_cfg.max_episode_steps) == int(env_cfg.max_episode_steps), "max_episode_steps cfg mismatch"
    return env
