from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from geomind.core.loader import create_env
from geomind.core.observation import to_agent_ready_observation
from geomind.core.resolver import resolve_task_request
from geomind.tasks.builders import resolve_builder
from geomind.tasks.evaluators import resolve_evaluator

PolicyFn = Callable[[Any, dict[str, Any], int, list[dict[str, Any]]], np.ndarray]


SUPPORTED_POLICIES = ("oracle", "oracle_protocol", "random_gt", "random", "agent", "model")


def _truthy_env_var(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _termination_is_success_only_from_native_log(info: dict[str, Any] | None) -> bool:
    """When IsaacLab flags terminated, extras.log often lists Episode_Termination/* counters."""
    if not isinstance(info, dict):
        return False
    extras = info.get("native_extras")
    if not isinstance(extras, dict):
        return False
    log = extras.get("log")
    if not isinstance(log, dict):
        return False
    try:
        success_val = float(log.get("Episode_Termination/success") or 0.0)
    except (TypeError, ValueError):
        success_val = 0.0
    if success_val <= 0.0:
        return False
    excluded_suffixes = ("/success", "/time_out")
    for key, raw in log.items():
        if not isinstance(key, str) or not key.startswith("Episode_Termination/"):
            continue
        if any(str(key).endswith(sfx) for sfx in excluded_suffixes):
            continue
        try:
            other = float(raw or 0.0)
        except (TypeError, ValueError):
            continue
        if other > 0.0:
            return False
    return True


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _load_external_policy_hook(module_path: str) -> PolicyFn | None:
    path = str(module_path).strip()
    if not path:
        return None
    module_name, _, attr_name = path.partition(":")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid policy hook module path: {module_path!r}. Use module:callable format.")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name, None)
    if not callable(fn):
        raise ValueError(f"Policy hook {module_path!r} is not callable.")
    return fn


def _extract_policy_action_from_obs(raw_obs: dict[str, Any], action_dim: int) -> np.ndarray | None:
    meta = dict(raw_obs.get("meta", {}))
    payload = meta.get("policy_action")
    if not isinstance(payload, list) or len(payload) < action_dim:
        return None
    return np.asarray(payload[:action_dim], dtype=np.float32)


def _as_xy_vector(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    try:
        vec = np.asarray([float(value[0]), float(value[1])], dtype=np.float32)
    except Exception:
        return None
    if not np.all(np.isfinite(vec)):
        return None
    return vec


def _find_key_recursively(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_key_recursively(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_key_recursively(value, key)
            if found is not None:
                return found
    return None


def _extract_delta_xy_from_native_obs(raw_obs: dict[str, Any]) -> np.ndarray | None:
    native_obs = (raw_obs.get("state") or {}).get("native_obs")
    if native_obs is None:
        return None
    # IsaacLab policy observations usually expose this key.
    for key in ("goal_error", "delta_xy", "target_delta_xy", "goal_delta_xy"):
        candidate = _find_key_recursively(native_obs, key)
        vec = _as_xy_vector(candidate)
        if vec is not None:
            return vec
    return None


def _oracle_action(raw_obs: dict[str, Any], action_dim: int, *, goal_bias_xy: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    action = np.zeros((action_dim,), dtype=np.float32)
    backend_obs = dict((raw_obs.get("state") or {}).get("backend_obs", {}))
    cursor_xy = _as_xy_vector(backend_obs.get("cursor_xy"))
    goal_xy = _as_xy_vector(backend_obs.get("goal_xy"))
    if cursor_xy is not None and goal_xy is not None:
        delta = np.asarray(
            [
                (float(goal_xy[0]) + float(goal_bias_xy[0])) - float(cursor_xy[0]),
                (float(goal_xy[1]) + float(goal_bias_xy[1])) - float(cursor_xy[1]),
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(delta))
        step = delta / max(norm, 1e-6)
        action[:2] = np.clip(step, -1.0, 1.0)
        return action
    native_delta_xy = _extract_delta_xy_from_native_obs(raw_obs)
    if native_delta_xy is not None:
        delta = np.asarray(
            [
                float(native_delta_xy[0]) + float(goal_bias_xy[0]),
                float(native_delta_xy[1]) + float(goal_bias_xy[1]),
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(delta))
        step = delta / max(norm, 1e-6)
        action[:2] = np.clip(step, -1.0, 1.0)
    return action


def build_policy(
    *,
    policy_name: str,
    action_dim: int,
    rng: np.random.Generator,
    interactive: bool = False,
    agent_hook: PolicyFn | None = None,
    model_hook: PolicyFn | None = None,
    oracle_goal_bias_xy: tuple[float, float] = (0.0, 0.0),
) -> PolicyFn:
    name = str(policy_name).strip().lower()
    if name not in SUPPORTED_POLICIES:
        raise ValueError(f"Unknown policy={policy_name!r}, available={', '.join(SUPPORTED_POLICIES)}")

    if name == "agent" and agent_hook is None:
        hook_path = str(__import__("os").environ.get("GEOMIND_AGENT_POLICY_HOOK", "")).strip()
        agent_hook = _load_external_policy_hook(hook_path) if hook_path else None
    if name == "model" and model_hook is None:
        hook_path = str(__import__("os").environ.get("GEOMIND_MODEL_POLICY_HOOK", "")).strip()
        model_hook = _load_external_policy_hook(hook_path) if hook_path else None

    def _policy(env, raw_obs: dict[str, Any], step: int, trace: list[dict[str, Any]]) -> np.ndarray:
        if interactive:
            observed = _extract_policy_action_from_obs(raw_obs, action_dim)
            if observed is not None:
                return observed

        if name == "oracle":
            return _oracle_action(raw_obs, action_dim, goal_bias_xy=oracle_goal_bias_xy)

        if name == "random_gt":
            base = _oracle_action(raw_obs, action_dim, goal_bias_xy=oracle_goal_bias_xy)
            noise = rng.normal(0.0, 0.15, size=(action_dim,)).astype(np.float32)
            return np.clip(base + noise, -1.0, 1.0)

        if name == "random":
            # Keep small noise so trajectory usually runs to max_steps instead of accidental success.
            action = np.zeros((action_dim,), dtype=np.float32)
            if action_dim > 0:
                action[: min(2, action_dim)] = rng.uniform(-0.2, 0.2, size=(min(2, action_dim),)).astype(np.float32)
            return action

        if name == "agent":
            if callable(agent_hook):
                return np.asarray(agent_hook(env, raw_obs, step, trace), dtype=np.float32).reshape(-1)[:action_dim]
            return _oracle_action(raw_obs, action_dim, goal_bias_xy=oracle_goal_bias_xy)

        if name == "model":
            if callable(model_hook):
                return np.asarray(model_hook(env, raw_obs, step, trace), dtype=np.float32).reshape(-1)[:action_dim]
            return _oracle_action(raw_obs, action_dim, goal_bias_xy=oracle_goal_bias_xy)

        return np.zeros((action_dim,), dtype=np.float32)

    return _policy


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _resolve_protocol_runner_script_for_family(family: str) -> str | None:
    normalized = str(family).strip().lower()
    if normalized == "spatial_placement":
        return "debug_protocol_spatial_placement.py"
    if normalized == "anchor_reference":
        return "debug_protocol_anchor_reference.py"
    if normalized == "constraint_selection":
        return "debug_protocol_constraint_selection.py"
    if normalized in {"state_reindexing", "historical_identity_tracking"}:
        return "debug_protocol_state_reindex.py"
    if normalized in {"explicit_rule_extrapolation", "implicit_trend_fitting", "shape_completion"}:
        return "debug_protocol_geometric_inference.py"
    return None


def _extract_success_from_protocol_summary(summary: dict[str, Any], episode0: dict[str, Any]) -> bool:
    episode_success_candidates = (
        "success",
        "zone_satisfied",
        "stack_satisfied",
        "containment_satisfied",
        "valid_episode",
    )
    for key in episode_success_candidates:
        if key in episode0:
            return bool(episode0.get(key))

    summary_rate_candidates = (
        "success_rate",
        "goal_satisfied_rate",
        "zone_satisfied_rate",
        "stack_satisfied_rate",
        "containment_satisfied_rate",
        "valid_episode_rate",
    )
    for key in summary_rate_candidates:
        if key in summary:
            try:
                return float(summary.get(key, 0.0) or 0.0) > 0.0
            except Exception:
                continue
    return False


def _resolve_protocol_agent_name(*, requested_policy_name: str, family: str) -> str:
    policy = str(requested_policy_name).strip().lower()
    normalized_family = str(family).strip().lower()
    if policy in {"oracle", "oracle_protocol"}:
        return "oracle"
    if policy == "random_gt":
        if normalized_family == "spatial_placement":
            return "random_gt"
        return "random"
    return "oracle"


def _run_protocol_policy_for_family(
    *,
    resolved,
    spec,
    max_steps: int,
    requested_policy_name: str,
    seed: int,
    camera_video_dir: str | None = None,
    camera_video_fps: int = 60,
) -> EpisodeResult:
    family = str(spec.family)
    runner_name = _resolve_protocol_runner_script_for_family(family)
    if runner_name is None:
        raise RuntimeError(f"No protocol oracle runner registered for family={family!r}")

    repo_root = Path(__file__).resolve().parents[2]
    runner_script = repo_root / "scripts" / runner_name
    if not runner_script.is_file():
        raise RuntimeError(f"Protocol oracle runner not found: {runner_script}")

    canonical_task_id = str(spec.canonical_task_id or spec.legacy_task_id or spec.task_id).strip()
    if not canonical_task_id:
        raise RuntimeError("Protocol oracle requires canonical task id.")

    with tempfile.TemporaryDirectory(prefix="geomind_protocol_oracle_") as tmp_dir:
        summary_json = Path(tmp_dir) / "summary.json"
        resolved_agent_name = _resolve_protocol_agent_name(requested_policy_name=requested_policy_name, family=family)
        use_reindex_insert_wrapper = family == "state_reindexing" and canonical_task_id == "reindex_insert"
        is_geometric_protocol_runner = runner_name == "debug_protocol_geometric_inference.py"
        annotation_layout_key = str(spec.annotation_layout_key or "")
        geometric_layout_key = annotation_layout_key.strip()
        if is_geometric_protocol_runner and not geometric_layout_key:
            # Keep behavior aligned with debug_isometric_extension.sh: when task config omits
            # annotation_layout_key, let task-id/variant/scene resolve layout instead of forcing
            # an empty --layout-key that falls back to debug defaults.
            geometric_layout_key = str(getattr(resolved.env_cfg, "task_layout_key", "") or "").strip()
        if use_reindex_insert_wrapper:
            wrapper_script = repo_root / "scripts" / "debug_reindex_insert.sh"
            if not wrapper_script.is_file():
                raise RuntimeError(f"Protocol oracle runner not found: {wrapper_script}")
            cmd = [
                "bash",
                str(wrapper_script),
                "--variant",
                str(spec.task_version),
                "--scene-id",
                str(int(spec.scene_id)),
                "--task-index",
                str(int(spec.annotation_task_index)),
                "--summary-json-out",
                str(summary_json),
            ]
            cmd.extend(["--layout-key", str(spec.annotation_layout_key)])
        else:
            cmd = [
                sys.executable,
                str(runner_script),
                "--task-id",
                canonical_task_id,
                "--variant",
                str(spec.task_version),
                "--scene-id",
                str(int(spec.scene_id)),
                "--task-index",
                str(int(spec.annotation_task_index)),
                "--agent",
                resolved_agent_name,
                "--agent-episodes",
                "1",
                "--summary-json-out",
                str(summary_json),
            ]
            if is_geometric_protocol_runner:
                if geometric_layout_key:
                    cmd.extend(["--layout-key", geometric_layout_key])
            else:
                cmd.extend(["--layout-key", annotation_layout_key])
            cmd.extend(["--agent-seed", str(int(seed))])
        protocol_episode_length_s = str(os.environ.get("GEOMIND_PROTOCOL_EPISODE_LENGTH_S", "")).strip()
        if protocol_episode_length_s and runner_name == "debug_protocol_spatial_placement.py":
            cmd.extend(["--episode-length-s", protocol_episode_length_s])
        scene_name = str((resolved.env_cfg.scene_ref or {}).get("name", "")).strip()
        if scene_name:
            cmd.extend(["--scene", scene_name])
        if _truthy_env("HEADLESS", default=True):
            cmd.append("--headless")
        if _truthy_env("DISABLE_FABRIC", default=False):
            cmd.append("--disable-fabric")
        if camera_video_dir:
            cmd.append("--camera-rollout-video")
            cmd.extend(["--camera-rollout-video-fps", str(float(camera_video_fps))])

        child_env = dict(os.environ)
        child_env["GEOMIND_TASK_CONTRACT"] = json.dumps(resolved.env_cfg.backend_kwargs.get("task_contract", {}))
        # In suite runs we don't want protocol debug scripts to hold the viewer idle forever.
        child_env.setdefault("VIEWER_IDLE_AFTER_AGENT", "0")
        if camera_video_dir:
            video_path = Path(camera_video_dir).expanduser()
            output_root = video_path.parent if video_path.name == "camera_videos" else video_path
            child_env["OUTPUT_DIR"] = str(output_root)
            child_env["HEADLESS_CAMERA_VIDEO"] = "1"
        if use_reindex_insert_wrapper:
            # Keep run-suite behavior aligned with scripts/debug_reindex_insert.sh defaults.
            child_env["AGENT"] = resolved_agent_name
            child_env["AGENT_EPISODES"] = "1"
            child_env["AGENT_SEED"] = str(int(seed))
            child_env.setdefault("GEOMIND_INSERT_DEBUG", "1")
            child_env.setdefault("GEOMIND_INSERT_COLOR_TAGS", "1")
        try:
            subprocess.run(cmd, cwd=str(repo_root), check=True, env=child_env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Protocol oracle run failed for family={family!r}.") from exc

        if not summary_json.is_file():
            raise RuntimeError(f"Protocol oracle summary missing: {summary_json}")
        protocol_summary = json.loads(summary_json.read_text(encoding="utf-8"))

    episodes = list(protocol_summary.get("episodes", []))
    episode0 = dict(episodes[0]) if episodes else dict(protocol_summary)
    success = _extract_success_from_protocol_summary(protocol_summary, episode0)
    goal_distance = float(episode0.get("l2_error", 0.0) or 0.0)
    target_object_id = str(episode0.get("target_object_id", "")).strip()
    anchor_object_id = str(episode0.get("anchor_object_id", "")).strip()
    layout_key = str(episode0.get("layout_key", "")).strip() or str(spec.annotation_layout_key)
    scene_name = str((resolved.env_cfg.scene_ref or {}).get("name", ""))
    episode_steps = int(episode0.get("trace_step_count", protocol_summary.get("trace_step_count", 0)) or 0)
    evaluation = {
        "success": bool(success),
        "score": float(1.0 if success else 0.0),
        "goal_distance": float(goal_distance),
        "success_radius": float(0.0),
        "goal_kind": str(episode0.get("goal_kind", "protocol_oracle")),
        "protocol_resolver": str(episode0.get("module_name", "spatial_placement")),
        "trace_len": int(episode_steps),
        "implementation": "protocol_oracle_debug_script",
        "evaluator_key": spec.evaluator_key,
        "evaluator_bucket": "specialized",
        "protocol_family": family,
        "protocol_runner_script": runner_name,
        "protocol_agent": _resolve_protocol_agent_name(requested_policy_name=requested_policy_name, family=family),
        "protocol_episode_result": episode0,
        "protocol_summary": protocol_summary,
    }
    summary = {
        "requested_task_id": resolved.requested_task_id,
        "resolved_task_id": resolved.resolved_task_id,
        "task_id": spec.task_id,
        "canonical_task_id": spec.canonical_task_id,
        "task_version": spec.task_version,
        "phase": spec.phase,
        "semantic_module": spec.semantic_module,
        "skill_type": spec.skill_type,
        "legacy_task_id": spec.legacy_task_id,
        "annotation_layout_key": spec.annotation_layout_key,
        "annotation_task_index": spec.annotation_task_index,
        "scene_id": spec.scene_id,
        "scene_ref": resolved.env_cfg.scene_ref,
        "layout_ref": {
            "layout_key": layout_key,
            "task_index": int(spec.annotation_task_index),
        },
        "runtime_mode": resolved.env_cfg.runtime_mode,
        "evaluator_key": spec.evaluator_key,
        "evaluator_bucket": "specialized",
        "execution_binding": {
            "goal_kind": str(episode0.get("goal_kind", "protocol_oracle")),
            "target_object_id": target_object_id,
            "anchor_object_id": anchor_object_id,
            "layout_key": layout_key,
            "scene_name": scene_name,
            "runtime_object_count": int(episode0.get("runtime_object_count", 0) or 0),
            "distractor_count": int(episode0.get("distractor_count", 0) or 0),
            "protocol_resolver": str(episode0.get("module_name", "spatial_placement")),
        },
        "family_context": resolve_builder(spec.family).build_context(spec),
        "interactive": False,
        "policy_mode": requested_policy_name,
        "action_noise_scale": 0.0,
        "object_goal_perturb_xy": [0.0, 0.0],
        "native_object_pose_perturb_range": 0.0,
        "max_steps": int(max_steps),
        "episode_steps": int(episode_steps),
        "reward_sum": float(0.0),
        "done_reason": "protocol_oracle_finished",
        "evaluation": evaluation,
        "success": bool(success),
    }
    if camera_video_dir:
        video_root = Path(camera_video_dir).expanduser()
        search_root = video_root.parent / "camera_videos" if video_root.name == "camera_videos" else video_root / "camera_videos"
        videos = sorted(search_root.rglob("*.mp4")) if search_root.exists() else []
        summary["camera_video"] = {
            "saved": bool(videos),
            "videos": [
                {"camera": path.stem, "path": str(path), "output_video": str(path)}
                for path in videos
            ],
            "reason": "" if videos else "no_protocol_camera_videos_found",
        }
    trace = []
    return EpisodeResult(summary=summary, trace=trace)


@dataclass(frozen=True)
class EpisodeResult:
    summary: dict[str, Any]
    trace: list[dict[str, Any]]


def run_episode(
    *,
    task_id: str,
    policy_name: str,
    max_steps: int,
    backend: str,
    interactive: bool,
    seed: int = 0,
    action_noise_scale: float = 0.0,
    object_goal_perturb_xy: tuple[float, float] = (0.0, 0.0),
    native_object_pose_perturb_range: float = 0.0,
    camera_video_dir: str | None = None,
    camera_video_fps: int = 60,
    camera_video_names: list[str] | None = None,
    condition: str | None = None,
    layout: str | None = None,
    version: int | None = None,
    interface: str | None = None,
    task_variant: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> EpisodeResult:
    resolved = resolve_task_request(
        task_id, max_episode_steps=max_steps, backend=backend,
        condition=condition, layout=layout, version=version, interface=interface,
        task_variant=task_variant, parameters=parameters,
    )
    spec = resolved.spec
    requested_policy_name = str(policy_name).strip().lower()
    if requested_policy_name in {"oracle", "oracle_protocol", "random_gt"}:
        if str(backend).strip().lower() == "isaaclab":
            runner_name = _resolve_protocol_runner_script_for_family(spec.family)
            if runner_name is not None:
                return _run_protocol_policy_for_family(
                    resolved=resolved,
                    spec=spec,
                    max_steps=max_steps,
                    requested_policy_name=requested_policy_name,
                    seed=seed,
                    camera_video_dir=camera_video_dir,
                    camera_video_fps=camera_video_fps,
                )
            print(
                "[policy_runtime] warning: protocol runner is not registered for family; "
                f"fallback to native {requested_policy_name} policy for task_id={resolved.resolved_task_id} "
                f"family={spec.family} backend={backend}."
            )
        if requested_policy_name in {"oracle", "oracle_protocol"}:
            policy_name = "oracle"
    env = create_env(
        resolved.resolved_task_id,
        max_episode_steps=max_steps,
        backend=backend,
        backend_kwargs={
            "seed": int(seed),
            "enable_object_pose_perturb": bool(native_object_pose_perturb_range > 0.0),
            "object_pose_perturb_range": float(max(native_object_pose_perturb_range, 0.0)),
            "object_pose_perturb_seed": int(seed),
        },
    )
    action_dim = int(np.prod(env.action_space.shape)) if getattr(env.action_space, "shape", None) else 1
    rng = np.random.default_rng(int(seed))
    policy = build_policy(
        policy_name=policy_name,
        action_dim=action_dim,
        rng=rng,
        interactive=interactive,
        oracle_goal_bias_xy=object_goal_perturb_xy,
    )
    evaluator = resolve_evaluator(spec.evaluator_key)
    family_builder = resolve_builder(spec.family)
    trace: list[dict[str, Any]] = []
    reward_sum = 0.0
    zero_action_warned = False
    obs, info = env.reset()
    camera_recorder = None
    camera_video_status: dict[str, Any] = {"saved": False, "reason": "disabled"}
    if camera_video_dir:
        try:
            from geomind_lab.utils.headless_camera_video import create_rollout_camera_recorder

            provisional_recorder = create_rollout_camera_recorder(env, sample_hz=float(camera_video_fps))
            sampling_config = dict(provisional_recorder.sampling_config)
            action_repeat = int(sampling_config.get("action_repeat", 1) or 1)
            sim_step_stride = int(sampling_config.get("sim_step_stride", 1) or 1)
            captures_per_control_step = max((action_repeat + sim_step_stride - 1) // sim_step_stride, 1)
            max_camera_frames = int(max_steps) * int(captures_per_control_step) + 1
            camera_recorder = create_rollout_camera_recorder(
                env,
                max_frames=max_camera_frames,
                sample_hz=float(camera_video_fps),
            ).install()
            camera_recorder.capture_now()
        except Exception as exc:
            camera_recorder = None
            camera_video_status = {
                "saved": False,
                "reason": f"recorder_setup_failed:{exc.__class__.__name__}",
            }
    reset_info = dict(info or {})
    done_reason = "max_steps"
    early_success_step: int | None = None
    early_success_eval: dict[str, Any] | None = None
    ignore_early_success = _truthy_env_var("GEOMIND_IGNORE_SUCCESS_TERMINATION")
    try:
        for step in range(max_steps):
            action = policy(env, obs, step, trace)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if float(action_noise_scale) > 0.0:
                action = action + rng.normal(0.0, float(action_noise_scale), size=action.shape).astype(np.float32)
                action = np.clip(action, -1.0, 1.0)
            if action.shape[0] != action_dim:
                if action.shape[0] < action_dim:
                    action = np.pad(action, (0, action_dim - action.shape[0]), mode="constant")
                else:
                    action = action[:action_dim]
            if (not zero_action_warned) and bool(getattr(resolved.env_cfg, "runtime_mode", "") == "isaaclab_native_delegate"):
                if float(np.linalg.norm(action)) < 1e-6:
                    print(
                        "[policy_runtime] warning: zero action on isaaclab step, "
                        "oracle may not have enough goal signal in observation."
                    )
                    zero_action_warned = True
            obs, reward, terminated, truncated, info = env.step(action)
            reward_value = float(reward)
            reward_sum += reward_value
            info_payload = _to_jsonable(dict(info or {}))
            ignore_success_term = (
                _truthy_env_var("GEOMIND_IGNORE_SUCCESS_TERMINATION")
                and bool(terminated)
                and _termination_is_success_only_from_native_log(dict(info or {}))
            )
            step_record: dict[str, Any] = {
                "step": step,
                "action": action.tolist(),
                "action_noise_scale": float(action_noise_scale),
                "agent_obs": _to_jsonable(to_agent_ready_observation(obs)),
                "reward": reward_value,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "meta": info_payload,
            }
            if ignore_success_term:
                step_record["ignored_success_termination"] = True
            trace.append(step_record)
            if spec.evaluator_key != "always_success":
                current_eval = evaluator.evaluate(trace)
                if bool(current_eval.get("success", False)):
                    if early_success_step is None:
                        early_success_step = int(step)
                        early_success_eval = dict(current_eval)
                    step_record["success"] = True
                    step_record["early_success_evaluation"] = _to_jsonable(current_eval)
                    if ignore_early_success:
                        done_reason = "max_steps_after_success"
                        step_record["ignored_early_success"] = True
                    else:
                        done_reason = "success"
                        break
            if bool(terminated) and not ignore_success_term:
                done_reason = "terminated"
                break
            if bool(truncated):
                done_reason = "truncated"
                break
    finally:
        if camera_recorder is not None:
            camera_recorder.uninstall()
        env.close()
    if camera_recorder is not None and camera_video_dir:
        if camera_video_names is not None:
            wanted = {str(name).strip() for name in camera_video_names if str(name).strip()}
            camera_recorder.frame_buffers = {
                name: frames for name, frames in camera_recorder.frame_buffers.items() if name in wanted
            }
        try:
            saved_paths = camera_recorder.save_mp4s(camera_video_dir, fps=float(camera_video_fps))
            sampling_config = dict(getattr(camera_recorder, "sampling_config", {}) or {})
            camera_video_status = {
                "saved": bool(saved_paths),
                "videos": [
                    {"camera": path.stem, "path": str(path), "output_video": str(path)}
                    for path in saved_paths
                ],
                "frames": int(camera_recorder.frame_count),
                "fps": int(camera_video_fps),
                "target_sample_hz": float(sampling_config.get("target_sample_hz", float(camera_video_fps))),
                "actual_sample_hz": float(sampling_config.get("actual_sample_hz", 0.0) or 0.0),
                "physics_step_hz": float(sampling_config.get("physics_step_hz", 0.0) or 0.0),
                "sim_step_stride": int(sampling_config.get("sim_step_stride", 1) or 1),
                "capture": "sim_step" if bool(sampling_config.get("capture_on_sim_step", False)) else "control_step",
                "reason": "" if saved_paths else "no_camera_frames",
            }
        except Exception as exc:
            camera_video_status = {
                "saved": False,
                "reason": f"video_save_failed:{exc.__class__.__name__}",
                "frames": int(camera_recorder.frame_count),
            }
    eval_out = evaluator.evaluate(trace)
    if early_success_eval is not None and bool(early_success_eval.get("success", False)):
        eval_out = {**eval_out, **early_success_eval}
    eval_out.setdefault("evaluator_key", spec.evaluator_key)
    eval_out.setdefault("evaluator_bucket", str(getattr(evaluator, "bucket", "specialized")))
    summary = {
        "requested_task_id": resolved.requested_task_id,
        "resolved_task_id": resolved.resolved_task_id,
        "task_id": spec.task_id,
        "canonical_task_id": spec.canonical_task_id,
        "task_version": spec.task_version,
        "phase": spec.phase,
        "semantic_module": spec.semantic_module,
        "skill_type": spec.skill_type,
        "legacy_task_id": spec.legacy_task_id,
        "annotation_layout_key": spec.annotation_layout_key,
        "annotation_task_index": spec.annotation_task_index,
        "scene_id": spec.scene_id,
        "scene_ref": resolved.env_cfg.scene_ref,
        "layout_ref": {
            "layout_key": (resolved.env_cfg.layout_ref or {}).get("layout_key", ""),
            "task_index": (resolved.env_cfg.layout_ref or {}).get("task_index", 0),
        },
        "runtime_mode": resolved.env_cfg.runtime_mode,
        "evaluator_key": spec.evaluator_key,
        "evaluator_bucket": str(getattr(evaluator, "bucket", "specialized")),
        "execution_binding": {
            "goal_kind": reset_info.get("goal_kind", ""),
            "target_object_id": reset_info.get("target_object_id", ""),
            "anchor_object_id": reset_info.get("anchor_object_id", ""),
            "layout_key": reset_info.get("layout_key", ""),
            "scene_name": reset_info.get("scene_name", ""),
            "runtime_object_count": reset_info.get("runtime_object_count", 0),
            "distractor_count": reset_info.get("distractor_count", 0),
            "protocol_resolver": reset_info.get("protocol_resolver", ""),
        },
        "family_context": family_builder.build_context(spec),
        "interactive": bool(interactive),
        "policy_mode": policy_name,
        "action_noise_scale": float(action_noise_scale),
        "object_goal_perturb_xy": [float(object_goal_perturb_xy[0]), float(object_goal_perturb_xy[1])],
        "native_object_pose_perturb_range": float(max(native_object_pose_perturb_range, 0.0)),
        "max_steps": max_steps,
        "episode_steps": len(trace),
        "reward_sum": reward_sum,
        "ignore_success_termination_env": _truthy_env_var("GEOMIND_IGNORE_SUCCESS_TERMINATION"),
        "done_reason": done_reason,
        "success_step": early_success_step,
        "evaluation": eval_out,
        "success": bool(eval_out.get("success", False)),
    }
    if camera_video_dir:
        summary["camera_video"] = camera_video_status
    return EpisodeResult(summary=summary, trace=trace)


def save_episode_payload(*, output_dir: Path, result: EpisodeResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(_to_jsonable(result.summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "trace.json").write_text(
        json.dumps(_to_jsonable(result.trace), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def try_save_trajectory_video(*, trace: list[dict[str, Any]], output_path: Path, fps: int = 10) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points: list[tuple[float, float, float, float]] = []
    for step in trace:
        meta = dict(step.get("meta", {}))
        cursor = meta.get("cursor_xy")
        goal = meta.get("goal_xy")
        if (
            isinstance(cursor, list)
            and len(cursor) >= 2
            and isinstance(goal, list)
            and len(goal) >= 2
        ):
            points.append((float(cursor[0]), float(cursor[1]), float(goal[0]), float(goal[1])))
    if not points:
        return {"saved": False, "reason": "missing_cursor_goal_xy"}
    try:
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"saved": False, "reason": f"missing_video_dependencies:{exc.__class__.__name__}"}
    cursor_x = [item[0] for item in points]
    cursor_y = [item[1] for item in points]
    goal_x = [item[2] for item in points]
    goal_y = [item[3] for item in points]
    x_min = min(cursor_x + goal_x) - 0.1
    x_max = max(cursor_x + goal_x) + 0.1
    y_min = min(cursor_y + goal_y) - 0.1
    y_max = max(cursor_y + goal_y) + 0.1
    frames: list[np.ndarray] = []
    for index, (cx, cy, gx, gy) in enumerate(points):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"step={index}")
        ax.plot(goal_x, goal_y, "--", alpha=0.2, color="tab:blue")
        ax.scatter([gx], [gy], marker="*", s=140, color="tab:green", label="goal")
        ax.plot(cursor_x[: index + 1], cursor_y[: index + 1], color="tab:red", linewidth=2, label="cursor")
        ax.scatter([cx], [cy], color="tab:red", s=40)
        ax.legend(loc="upper right")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(frame[:, :, :3])
        plt.close(fig)
    try:
        imageio.mimsave(str(output_path), frames, fps=max(int(fps), 1))
    except Exception as exc:
        return {"saved": False, "reason": f"video_encode_failed:{exc.__class__.__name__}"}
    return {"saved": True, "path": str(output_path), "frames": len(frames), "fps": int(fps)}
