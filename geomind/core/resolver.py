from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geomind.tasks.evaluators import has_evaluator, list_evaluator_keys

from .registry import registry
from .types import EnvCfg, ResolvedTaskRequest


def _geomind_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_scene_index() -> dict[int, dict[str, Any]]:
    scenes_dir = _geomind_root() / "configs" / "scenes"
    scene_index: dict[int, dict[str, Any]] = {}
    for path in sorted(scenes_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        scene_id = payload.get("scene_id")
        try:
            scene_index[int(scene_id)] = dict(payload)
        except (TypeError, ValueError):
            continue
    return scene_index


def _load_layout_index() -> dict[str, Any]:
    from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import load_annotated_tasks
    # Use the same annotation + overlay index as the native task loader.
    return load_annotated_tasks()


def _resolve_layout_ref(layouts: dict[str, Any], *, layout_key: str, task_index: int) -> dict[str, Any]:
    if not layout_key:
        return {}
    layout_payload = layouts.get(layout_key)
    if not isinstance(layout_payload, dict):
        raise ValueError(f"Missing annotation layout: {layout_key!r}")
    tasks = layout_payload.get("tasks", [])
    if not isinstance(tasks, list) or task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"Invalid annotation task index: layout={layout_key!r}, task_index={task_index}")
    task_payload = tasks[task_index]
    if not isinstance(task_payload, dict):
        raise ValueError(f"Annotation task payload must be object: layout={layout_key!r}, task_index={task_index}")
    return {
        "layout_key": layout_key,
        "task_index": int(task_index),
        "scene_id": int(layout_payload.get("scene_id", 0)),
        "instruction": str(task_payload.get("instruction", "")),
        "annotation": dict(task_payload.get("annotation", {})) if isinstance(task_payload.get("annotation", {}), dict) else {},
    }


def _build_env_cfg(spec, *, payload: dict[str, Any], max_episode_steps: int | None, backend: str | None) -> EnvCfg:
    if spec.implementation_status == "placeholder":
        raise NotImplementedError(f"Task {spec.task_id!r} is registered as a placeholder and has no implementation yet.")
    if payload.get('backend')=='assembled_native':
        if backend not in (None,'assembled_native','isaaclab'):
            raise ValueError(f'Task {spec.task_id!r} requires the native assembled manipulation runner; backend={backend!r} is not supported')
        native=dict(payload['native_assembly'])
        kwargs={**payload.get('backend_kwargs',{}),**native,
                'registration_id':spec.task_id,'native_assembly':True,
                'task_contract':dict(payload.get('task_contract',{}))}
        return EnvCfg(task=spec,max_episode_steps=int(max_episode_steps or 12000),
            num_envs=1,device='cuda:0',cameras=('table_cam','wrist_cam'),backend='assembled_native',
            backend_kwargs=kwargs,scene_ref=_load_scene_index()[spec.scene_id],
            layout_ref={'native_recipe_module':native['recipe_module'],'registration_id':spec.task_id},
            runtime_mode='assembled_native_pipeline')
    scene_index = _load_scene_index()
    if spec.scene_id not in scene_index:
        raise ValueError(f"Scene ref missing for scene_id={spec.scene_id}. Please add geomind/configs/scenes entry.")
    if not has_evaluator(spec.evaluator_key):
        raise ValueError(
            f"Unknown evaluator_key={spec.evaluator_key!r}. Known: {', '.join(list_evaluator_keys())}"
        )
    layouts = _load_layout_index()
    layout_ref = _resolve_layout_ref(
        layouts,
        layout_key=spec.annotation_layout_key,
        task_index=spec.annotation_task_index,
    )
    if layout_ref and int(layout_ref.get("scene_id", spec.scene_id)) not in (0, spec.scene_id):
        raise ValueError(
            f"Layout scene mismatch for {spec.task_id}: scene_id={spec.scene_id}, layout_scene={layout_ref.get('scene_id')}"
        )

    cameras = tuple(str(item) for item in payload.get("cameras", ("ego_view", "table_cam")))
    backend_kwargs = dict(payload.get("backend_kwargs", {}))
    backend_kwargs["task_contract"] = dict(payload.get("task_contract", {}))
    resolved_max_steps = int(max_episode_steps if max_episode_steps is not None else payload.get("max_episode_steps", 200))
    resolved_backend = str(backend if backend is not None else payload.get("backend", "mock"))
    return EnvCfg(
        task=spec,
        max_episode_steps=resolved_max_steps,
        num_envs=int(payload.get("num_envs", 1)),
        device=str(payload.get("device", "cpu")),
        use_fabric=payload.get("use_fabric"),
        cameras=cameras,
        backend=resolved_backend,
        backend_kwargs=backend_kwargs,
        scene_ref=scene_index[spec.scene_id],
        layout_ref=layout_ref,
        runtime_mode="temporary_mock_delegate" if resolved_backend == "mock" else "isaaclab_native_delegate",
    )


def resolve_task_request(
    task_id: str,
    *,
    max_episode_steps: int | None = None,
    backend: str | None = None,
    condition: str | None = None,
    layout: str | None = None,
    version: int | None = None,
    interface: str | None = None,
    task_variant: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> ResolvedTaskRequest:
    registry.ensure_loaded()
    resolved_task_id = registry.resolve_task_id(
        task_id, condition=condition, layout=layout, version=version, interface=interface,
        task_variant=task_variant, parameters=parameters,
    )
    spec = registry.get(resolved_task_id)
    payload = registry.get_payload(resolved_task_id)
    config_path = registry.get_config_path(resolved_task_id)

    if str(payload.get("task_id", "")).strip() != resolved_task_id:
        raise ValueError(f"Config task_id mismatch in {config_path}: expected {resolved_task_id!r}")

    env_cfg = _build_env_cfg(
        spec,
        payload=payload,
        max_episode_steps=max_episode_steps,
        backend=backend,
    )
    return ResolvedTaskRequest(
        requested_task_id=str(task_id),
        resolved_task_id=resolved_task_id,
        spec=spec,
        env_cfg=env_cfg,
        config_path=config_path,
    )
