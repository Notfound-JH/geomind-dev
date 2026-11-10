from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from ..observation import build_raw_observation
from ..types import EnvCfg


def _goal_kind_for_family(family: str) -> str:
    normalized = str(family).strip().lower()
    if normalized == "anchor_reference":
        return "anchor_selection"
    if normalized == "spatial_placement":
        return "placement"
    if normalized == "shape_completion":
        return "square_completion"
    if normalized in {"state_reindexing", "historical_identity_tracking"}:
        return "rank_selection"
    return "generic"


def _move_to_resolver(resolver: str) -> bool:
    return str(resolver).strip().lower() in {"move_to_object", "move_to_object_proximity", "object_proximity"}


def _placement_resolver(resolver: str) -> bool:
    return str(resolver).strip().lower() in {
        "directional_region_placement",
        "support_top_placement",
        "containment_placement",
    }


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class _IsaacLabAppRuntime:
    _launcher = None
    _simulation_app = None

    @classmethod
    def ensure_started(cls, *, headless: bool, enable_cameras: bool) -> None:
        if cls._simulation_app is not None:
            return
        try:
            from geomind_lab.bootstrap.isaaclab_path import ensure_isaaclab_source_on_path

            repo_root = Path(__file__).resolve().parents[3]
            source_dir = ensure_isaaclab_source_on_path(repo_root)
            for package_dir in (
                source_dir / "isaaclab",
                source_dir / "isaaclab_tasks",
                source_dir / "isaaclab_mimic",
                source_dir / "isaaclab_assets",
            ):
                package_dir_str = str(package_dir)
                if package_dir_str not in sys.path:
                    sys.path.insert(0, package_dir_str)
            from isaaclab.app import AppLauncher
        except Exception as exc:
            raise RuntimeError(
                "Failed to bootstrap IsaacLab runtime. Ensure Isaac Sim Python env is active "
                "(same as debug scripts, e.g. `conda activate isaac_curobo`) and IsaacLab source is available."
            ) from exc

        parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args([])
        setattr(args, "headless", bool(headless))
        setattr(args, "enable_cameras", bool(enable_cameras))
        cls._launcher = AppLauncher(args)
        cls._simulation_app = cls._launcher.app

    @classmethod
    def close(cls) -> None:
        simulation_app = cls._simulation_app
        cls._launcher = None
        cls._simulation_app = None
        if simulation_app is not None:
            simulation_app.close()


class GeoMindIsaacLabNativeDelegate(gym.Env):
    """IsaacLab native delegate using real simulation backend.

    The runtime config is derived from `geomind` resolver output so the same task/layout
    selection used by runners is forwarded into the IsaacLab mimic environment.
    """

    metadata = {"render_modes": []}

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = str(os.environ.get(name, "")).strip().lower()
        if not value:
            return default
        return value in {"1", "true", "yes", "on"}

    def __init__(self, task_id: str, cfg: EnvCfg | None = None, **_: Any) -> None:
        super().__init__()
        if cfg is None:
            raise ValueError("GeoMindIsaacLabNativeDelegate requires EnvCfg.")
        self.task_id = str(task_id)
        self.cfg = cfg
        if self.cfg.runtime_mode != "isaaclab_native_delegate":
            raise ValueError(
                "GeoMindIsaacLabNativeDelegate only supports isaaclab_native_delegate, got "
                f"{self.cfg.runtime_mode!r}"
            )
        backend_kwargs = dict(self.cfg.backend_kwargs or {})
        headless = self._env_bool("HEADLESS", bool(backend_kwargs.get("headless", True)))
        enable_cameras = self._env_bool("ENABLE_CAMERAS", bool(backend_kwargs.get("enable_cameras", True)))
        backend_kwargs.setdefault("disable_fabric", self._env_bool("DISABLE_FABRIC", bool(backend_kwargs.get("disable_fabric", False))))
        backend_kwargs.setdefault("viewer_eye", tuple(float(v) for v in str(os.environ.get("VIEWER_EYE", "2.2 2.2 1.8")).split()))
        backend_kwargs.setdefault(
            "viewer_lookat", tuple(float(v) for v in str(os.environ.get("VIEWER_LOOKAT", "0.3 0.0 0.8")).split())
        )
        _IsaacLabAppRuntime.ensure_started(headless=headless, enable_cameras=enable_cameras)
        self._native_env = self._build_native_env(backend_kwargs=backend_kwargs)
        self._backend_kwargs = backend_kwargs
        self.action_space = self._native_env.action_space
        self.observation_space = gym.spaces.Dict({})
        self._episode_step = 0
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32) if self.action_space.shape else np.zeros(
            (7,), dtype=np.float32
        )
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._native_object_perturbation_applied = False
        self._goal_metric_issues: list[str] = []

    def _note_goal_metric_issue(self, reason: str) -> None:
        """Record why a goal metric could not be computed instead of silently swallowing it."""
        text = str(reason).strip()
        if text and text not in self._goal_metric_issues:
            self._goal_metric_issues.append(text)

    def _build_native_env(self, *, backend_kwargs: dict[str, Any]):
        from isaaclab.utils import configclass

        from geomind_lab.mimic import GeoMindFrankaIKRelMimicEnv, GeoMindFrankaIKRelMimicEnvCfg
        from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import resolve_task_request

        canonical_task_id = str(self.cfg.task.canonical_task_id or self.cfg.task.task_id)
        scene_name = str((self.cfg.scene_ref or {}).get("name", "") or f"scene_{int(self.cfg.task.scene_id)}")
        requested = resolve_task_request(
            task_id=canonical_task_id,
            variant=self.cfg.task.task_version,
            scene=scene_name,
            scene_id=int(self.cfg.task.scene_id),
            layout_key=str(self.cfg.task.annotation_layout_key or (self.cfg.layout_ref or {}).get("layout_key", "")),
            task_index=int(self.cfg.task.annotation_task_index or (self.cfg.layout_ref or {}).get("task_index", 0)),
        )
        base_cfg_cls = GeoMindFrankaIKRelMimicEnvCfg
        disable_fabric = bool(backend_kwargs.get("disable_fabric", False))
        # `or 1` used to silently rewrite seed 0 into 1. Only fall back when the key is absent.
        raw_seed = backend_kwargs.get("seed", backend_kwargs.get("object_pose_perturb_seed", None))
        runtime_seed = 1 if raw_seed is None else int(raw_seed)
        viewer_eye = tuple(backend_kwargs.get("viewer_eye", (2.2, 2.2, 1.8)))
        viewer_lookat = tuple(backend_kwargs.get("viewer_lookat", (0.3, 0.0, 0.8)))

        direct_spec, direct_record = self._resolve_direct_ablation(
            backend_kwargs=backend_kwargs,
            requested=requested,
            runtime_seed=runtime_seed,
        )
        self._direct_record = direct_record
        if direct_spec is None:
            from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import load_task
            direct_spec = load_task(
                task_id=requested.task_id, variant=requested.variant,
                scene=requested.scene, scene_id=requested.scene_id,
                layout_key=requested.layout_key, task_index=requested.task_index,
                object_mode=requested.object_mode,
                task_contract=backend_kwargs.get("task_contract"),
            ).task_spec
        self._effective_instruction = str(
            direct_record.get("effective_instruction", "") or self.cfg.task.instruction
        )

        def __post_init__(runtime_cfg):
            runtime_cfg.task_id = str(requested.task_id)
            runtime_cfg.task_variant = str(requested.variant)
            runtime_cfg.scene_name = str(requested.scene)
            runtime_cfg.object_mode = str(requested.object_mode)
            runtime_cfg.scene_id = int(requested.scene_id)
            runtime_cfg.task_layout_key = str(requested.layout_key)
            runtime_cfg.task_index = int(requested.task_index)
            runtime_cfg.seed = int(runtime_seed)
            if direct_spec is not None:
                # Pre-seeding task_spec short-circuits geomind_env_cfg's own load_task() call
                # (geomind_env_cfg.py: `if self.task_spec is None`). Everything downstream —
                # object spawning, protocol, goal sequence, terminations — then reads the
                # overridden spec, so instruction + appearance change while the layout does not.
                runtime_cfg.task_spec = direct_spec
            runtime_cfg.scene.num_envs = int(self.cfg.num_envs)
            runtime_cfg.sim.device = str(self.cfg.device)
            runtime_cfg.sim.use_fabric = not disable_fabric
            base_cfg_cls.__post_init__(runtime_cfg)
            runtime_cfg.viewer.eye = tuple(float(v) for v in viewer_eye)
            runtime_cfg.viewer.lookat = tuple(float(v) for v in viewer_lookat)

        runtime_cfg_cls = configclass(
            type(
                "RuntimeGeoMindIsaacLabNativeCfg",
                (base_cfg_cls,),
                {
                    "task_id": str(requested.task_id),
                    "task_variant": str(requested.variant),
                    "scene_name": str(requested.scene),
                    "object_mode": str(requested.object_mode),
                    "scene_id": int(requested.scene_id),
                    "task_layout_key": str(requested.layout_key),
                    "task_index": int(requested.task_index),
                    "__post_init__": __post_init__,
                },
            )
        )
        return GeoMindFrankaIKRelMimicEnv(cfg=runtime_cfg_cls())

    def _resolve_direct_ablation(
        self,
        *,
        backend_kwargs: dict[str, Any],
        requested: Any,
        runtime_seed: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Return ``(task_spec_override_or_None, record)`` for the requested ablation.

        For ``base`` runs nothing is loaded and ``task_spec`` stays ``None``, so the base
        code path is bit-for-bit what it was before this feature existed.
        """
        from geomind.tasks import direct_ablation as direct

        payload = dict(backend_kwargs.get("direct_ablation") or {})
        ablation_id = str(payload.get("ablation_id", direct.BASE_ABLATION_ID) or direct.BASE_ABLATION_ID)
        pair_id = str(payload.get("pair_id", "") or "")
        if not direct.is_direct(ablation_id):
            return None, {
                "ablation_id": direct.BASE_ABLATION_ID,
                "pair_id": pair_id,
                "effective_seed": int(runtime_seed),
                "effective_instruction": str(self.cfg.task.instruction),
            }

        from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import load_task

        loaded = load_task(
            task_id=str(requested.task_id) or None,
            variant=str(requested.variant) or None,
            scene=str(requested.scene) or None,
            scene_id=int(requested.scene_id),
            layout_key=str(requested.layout_key) or None,
            task_index=int(requested.task_index),
            object_mode=str(requested.object_mode) or None,
            task_contract=backend_kwargs.get("task_contract"),
        )
        direct_spec, record = direct.apply_direct_override(
            task_spec=loaded.task_spec,
            task_id=str(self.cfg.task.task_id),
            base_instruction=str(self.cfg.task.instruction),
        )
        record["pair_id"] = pair_id
        record["effective_seed"] = int(runtime_seed)
        record["resolved_layout_key"] = str(requested.layout_key)
        record["resolved_task_index"] = int(requested.task_index)
        return direct_spec, record

    @property
    def direct_ablation_record(self) -> dict[str, Any]:
        return dict(getattr(self, "_direct_record", {}) or {})

    def _to_python(self, value: Any):
        try:
            import torch
        except Exception:
            torch = None
        if isinstance(value, dict):
            return {str(k): self._to_python(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_python(v) for v in value]
        if torch is not None and isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.ndim > 0 and tensor.shape[0] == int(getattr(self._native_env, "num_envs", 1)):
                tensor = tensor[0]
            return tensor.tolist() if tensor.ndim > 0 else float(tensor.item())
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    def _runtime_info(self) -> dict[str, Any]:
        protocol = dict(getattr(self._native_env.cfg, "task_protocol", {}) or {})
        gt_spec = dict(protocol.get("gt_spec", {}) or {})
        resolver = str(gt_spec.get("resolver", "")).strip().lower()
        has_target_zone = isinstance(gt_spec.get("target_zone", {}), dict) and bool(gt_spec.get("target_zone", {}))
        self._goal_metric_issues = []
        goal_sequence_metrics = self._goal_sequence_metrics()
        success_radius = float((dict(self.cfg.layout_ref or {}).get("annotation", {}).get("scene_adaptation", {}).get(
            "preview_proxy_goal", {}
        ).get("tolerance", {}) or {}).get("pos_r", 0.05) or 0.05)
        if has_target_zone:
            success_radius = self._placement_success_radius(gt_spec=gt_spec, default=success_radius)
        elif _move_to_resolver(resolver):
            success_radius = float(
                gt_spec.get("success_radius")
                or gt_spec.get("proximity_radius")
                or gt_spec.get("target_proximity_radius")
                or success_radius
                or 0.085
            )
            clamp_radius = _env_float("GEOMIND_MOVE_TO_SUCCESS_RADIUS_MAX", None)
            if clamp_radius is not None and clamp_radius > 0.0:
                success_radius = min(success_radius, float(clamp_radius))
        elif _placement_resolver(resolver):
            success_radius = self._placement_success_radius(gt_spec=gt_spec, default=success_radius)
        elif goal_sequence_metrics is not None:
            _, success_radius, _ = goal_sequence_metrics
        if has_target_zone:
            goal_distance = self._placement_goal_distance(protocol=protocol, gt_spec=gt_spec)
        elif _move_to_resolver(resolver):
            goal_distance = self._move_to_goal_distance(protocol=protocol, gt_spec=gt_spec)
        elif _placement_resolver(resolver):
            goal_distance = self._placement_goal_distance(protocol=protocol, gt_spec=gt_spec)
        elif goal_sequence_metrics is not None:
            goal_distance, _, _ = goal_sequence_metrics
        else:
            goal_distance = None
        # Every branch except the anisotropic goal_sequence one already reports metres.
        goal_distance_l2_m = goal_distance
        if goal_sequence_metrics is not None and not (
            has_target_zone or _move_to_resolver(resolver) or _placement_resolver(resolver)
        ):
            goal_distance_l2_m = goal_sequence_metrics[2]
        done = bool(self._last_terminated or self._last_truncated)
        distance_achieved = bool(goal_distance is not None and goal_distance <= success_radius)
        distance_backed_goal = bool(
            has_target_zone or _move_to_resolver(resolver) or _placement_resolver(resolver) or goal_sequence_metrics is not None
        )
        # A distance-backed goal whose distance could not be computed is *unknown*, not achieved.
        # Never substitute 0.0 here: doing so turns every terminated episode into a false success.
        goal_metric_valid = bool(goal_distance is not None) if distance_backed_goal else False
        if distance_backed_goal and not goal_metric_valid and not self._goal_metric_issues:
            self._note_goal_metric_issue("goal_distance_unavailable")
        if not distance_backed_goal:
            self._note_goal_metric_issue(f"no_distance_backed_goal_for_resolver:{resolver or '<empty>'}")
        # execution_binding used to come back empty on the VLA path because only the
        # protocol-oracle branch filled it in. These are read-only lookups; they change
        # no metric, they just make the run auditable (and pairing checkable).
        runtime_object_names = [
            str(entry.get("name", "")).strip()
            for entry in (getattr(self._native_env.cfg, "task_spec", {}) or {}).get("runtime_objects", [])
            if isinstance(entry, dict) and str(entry.get("name", "")).strip()
        ]
        target_object_id = self._protocol_object_name(
            protocol, gt_spec, gt_key="target_object_id", protocol_key="target_object"
        )
        if protocol.get("runtime_target_binding") == "protocol_gt":
            state = getattr(self._native_env, "_state_reindexing_runtime", {})
            indices = state.get("gt_object_indices")
            names = list(state.get("object_names", []))
            if indices is not None and 0 <= int(indices[0]) < len(names):
                target_object_id = names[int(indices[0])]
        anchor_object_id = self._protocol_object_name(
            protocol, gt_spec, gt_key="anchor_object_id", protocol_key="anchor_object"
        )
        return {
            "goal_kind": "move_to" if _move_to_resolver(resolver) else _goal_kind_for_family(self.cfg.task.family),
            "protocol_resolver": resolver,
            "selection_resolver": gt_spec.get("selection_resolver", resolver),
            "direct_ablation": self.direct_ablation_record,
            "target_object_id": target_object_id,
            "anchor_object_id": anchor_object_id,
            "runtime_object_count": len(runtime_object_names),
            "distractor_count": max(
                len([name for name in runtime_object_names if name not in {target_object_id, anchor_object_id}]),
                0,
            ),
            "goal_achieved": distance_achieved if distance_backed_goal else bool(done),
            "goal_distance": float(goal_distance) if goal_distance is not None else None,
            "goal_distance_l2_m": float(goal_distance_l2_m) if goal_distance_l2_m is not None else None,
            "goal_metric_valid": goal_metric_valid,
            "goal_metric_issues": list(self._goal_metric_issues),
            "success_radius": success_radius,
            "scene_name": self.cfg.scene_ref.get("name", ""),
            "layout_key": self.cfg.layout_ref.get("layout_key", ""),
            "runtime_mode": self.cfg.runtime_mode,
            "runtime_delegate": self.__class__.__name__,
            "backend": self.cfg.backend,
            "reward": float(self._last_reward),
            "native_object_pose_perturbation_applied": bool(self._native_object_perturbation_applied),
        }

    def _move_to_goal_distance(self, *, protocol: dict[str, Any], gt_spec: dict[str, Any]) -> float | None:
        if protocol.get("runtime_target_binding") == "protocol_gt":
            import torch
            from geomind_lab.mimic.envs.task_contract_logic import protocol_gt_positions
            resolved = protocol_gt_positions(self._native_env, protocol)
            if resolved is None or not bool(resolved[1][0]):
                self._note_goal_metric_issue("history_or_structure_gt_not_validated")
                return None
            target = resolved[0]
            ee = self._native_env.scene["ee_frame"].data.target_pos_w[:, 0, :]
            difference = ee - target
            if gt_spec.get("xy_only", False): difference = difference[:, :2]
            return float(torch.linalg.vector_norm(difference, dim=1)[0].item())
        target_object_id = str(
            gt_spec.get("target_object_id")
            or ((protocol.get("target_object") or {}).get("object_id", ""))
        ).strip()
        if not target_object_id:
            self._note_goal_metric_issue("move_to_target_object_id_missing")
            return None
        try:
            import torch
            from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.common import get_asset_pose_w
        except Exception as exc:
            self._note_goal_metric_issue(f"move_to_import_failed:{type(exc).__name__}")
            return None
        scene = getattr(self._native_env, "scene", None)
        if scene is None or target_object_id not in scene.keys() or "ee_frame" not in scene.keys():
            self._note_goal_metric_issue(f"move_to_scene_missing_asset:{target_object_id}")
            return None
        try:
            target_pos_w, _ = get_asset_pose_w(self._native_env, target_object_id)
            ee_frame = scene["ee_frame"]
            ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
            raw_offset = gt_spec.get("target_offset", (0.0, 0.0, 0.0))
            offset = [0.0, 0.0, 0.0]
            if isinstance(raw_offset, (list, tuple)):
                for index in range(min(3, len(raw_offset))):
                    offset[index] = float(raw_offset[index])
            target_point_w = target_pos_w + torch.tensor(offset, dtype=torch.float32, device=self._native_env.device).unsqueeze(0)
            if bool(gt_spec.get("xy_only", False)):
                distance = torch.linalg.vector_norm(ee_pos_w[:, :2] - target_point_w[:, :2], dim=1)
            else:
                distance = torch.linalg.vector_norm(ee_pos_w - target_point_w, dim=1)
            return float(distance[0].item())
        except Exception as exc:
            self._note_goal_metric_issue(f"move_to_distance_failed:{type(exc).__name__}")
            return None

    def _protocol_object_name(self, protocol: dict[str, Any], gt_spec: dict[str, Any], *, gt_key: str, protocol_key: str) -> str:
        value = str(gt_spec.get(gt_key, "")).strip()
        if value:
            return value
        payload = protocol.get(protocol_key, {})
        if isinstance(payload, dict):
            return str(payload.get("object_id", "")).strip()
        return ""

    def _preview_success_radius(self) -> float | None:
        annotation = dict(self.cfg.layout_ref or {}).get("annotation", {})
        scene_adaptation = dict(annotation or {}).get("scene_adaptation", {})
        preview = dict(scene_adaptation or {}).get("preview_proxy_goal", {})
        tolerance = dict(preview or {}).get("tolerance", {})
        try:
            value = float(dict(tolerance or {}).get("pos_r", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        return value if value > 0.0 else None

    @staticmethod
    def _zone_half_extents(gt_spec: dict[str, Any]) -> tuple[float, float] | None:
        zone = gt_spec.get("target_zone", {})
        half_extents = (zone if isinstance(zone, dict) else {}).get("half_extents", ())
        if not isinstance(half_extents, (list, tuple)) or len(half_extents) < 2:
            return None
        try:
            return (abs(float(half_extents[0])), abs(float(half_extents[1])))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _zone_tolerance() -> float:
        """Allowed overshoot outside the rectangular target zone (metres)."""
        value = _env_float("GEOMIND_PLACEMENT_ZONE_TOLERANCE", 0.01)
        return max(float(value or 0.01), 1.0e-6)

    def _placement_success_radius(self, *, gt_spec: dict[str, Any], default: float) -> float:
        resolver = str(gt_spec.get("resolver", "")).strip().lower()
        if resolver in {"containment_placement", "support_top_placement"}:
            # Both return a violation amount (0.0 = satisfied), so the threshold is a tolerance.
            return self._zone_tolerance()
        if resolver == "directional_region_placement" or (
            isinstance(gt_spec.get("target_zone", {}), dict) and bool(gt_spec.get("target_zone", {}))
        ):
            # `_placement_goal_distance` already returns the distance *outside* the rectangle,
            # so the threshold is an overshoot tolerance, not the zone size itself.
            if self._zone_half_extents(gt_spec) is not None:
                return self._zone_tolerance()
        preview_radius = self._preview_success_radius()
        if preview_radius is not None:
            return float(preview_radius)
        return float(default)

    def _goal_sequence_metrics(self) -> tuple[float, float, float] | None:
        try:
            import torch

            from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.common import (
                goal_tolerance_distance,
            )
        except Exception:
            return None
        task_spec = dict(getattr(self._native_env.cfg, "task_spec", {}) or {})
        goal_sequence = task_spec.get("goal_sequence", [])
        if not isinstance(goal_sequence, list) or not goal_sequence:
            return None
        first_step = goal_sequence[0]
        if not isinstance(first_step, dict):
            return None
        goals = first_step.get("goals", [])
        if not isinstance(goals, list) or not goals:
            return None

        distances: list[float] = []
        radii: list[float] = []
        raw_l2: list[float] = []
        for goal in goals:
            if not isinstance(goal, dict):
                continue
            object_name = str(goal.get("object_id", "")).strip()
            xy = goal.get("xy", [])
            if not object_name or not isinstance(xy, (list, tuple)) or len(xy) < 2:
                continue
            asset_xy = self._asset_position_xy(object_name)
            if asset_xy is None:
                continue
            target_xy = torch.tensor(
                [float(xy[0]), float(xy[1])],
                dtype=torch.float32,
                device=self._native_env.device,
            ).unsqueeze(0)
            tolerance = goal.get("tolerance", {})
            if not isinstance(tolerance, dict):
                tolerance = {}
            radius = float(tolerance.get("pos_r", 0.05) or 0.05)
            delta = asset_xy - target_xy
            distances.append(float(goal_tolerance_distance(delta, tolerance)[0].item()))
            # Anisotropic goals rescale the distance so `<= pos_r` matches the tolerance ellipse,
            # which makes it no longer metres. SGE and other physical-distance consumers need the
            # plain euclidean value, so report it separately instead of overloading goal_distance.
            raw_l2.append(float(torch.linalg.vector_norm(delta, dim=1)[0].item()))
            radii.append(radius)
        if not distances or not radii:
            return None
        return (max(distances), min(radii), max(raw_l2))

    def _asset_position_xy(self, object_name: str):
        object_name = str(object_name).strip()
        if not object_name:
            self._note_goal_metric_issue("asset_name_empty")
            return None
        scene = getattr(self._native_env, "scene", None)
        if scene is None or object_name not in scene.keys():
            self._note_goal_metric_issue(f"scene_missing_asset:{object_name}")
            return None
        try:
            from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.common import get_asset_pose_w

            pos_w, _ = get_asset_pose_w(self._native_env, object_name)
            return pos_w[:, :2]
        except Exception as exc:
            self._note_goal_metric_issue(f"asset_pose_failed:{object_name}:{type(exc).__name__}")
            return None

    def _asset_position_xyz(self, object_name: str):
        object_name = str(object_name).strip()
        if not object_name:
            self._note_goal_metric_issue("asset_name_empty")
            return None
        scene = getattr(self._native_env, "scene", None)
        if scene is None or object_name not in scene.keys():
            self._note_goal_metric_issue(f"scene_missing_asset:{object_name}")
            return None
        try:
            from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.common import get_asset_pose_w

            pos_w, _ = get_asset_pose_w(self._native_env, object_name)
            return pos_w
        except Exception as exc:
            self._note_goal_metric_issue(f"asset_pose_failed:{object_name}:{type(exc).__name__}")
            return None

    def _containment_violation(self, *, target_xyz, gt_spec: dict[str, Any]) -> float | None:
        """How far the object is from being inside the container, in metres (0.0 = inside).

        xy uses the interior half extents (axis-wise), z uses the interior floor / opening planes,
        so an object hovering above the open box no longer counts as contained.
        """
        floor_xyz = self._asset_position_xyz(str(gt_spec.get("container_floor_object_id", "")).strip())
        if floor_xyz is None:
            floor_xyz = self._asset_position_xyz(str(gt_spec.get("container_reference_object_id", "")).strip())
        if floor_xyz is None:
            self._note_goal_metric_issue("containment_floor_unresolved")
            return None
        half_extents = gt_spec.get("interior_half_extents_xy", ())
        if not isinstance(half_extents, (list, tuple)) or len(half_extents) < 2:
            self._note_goal_metric_issue("containment_interior_half_extents_missing")
            return None
        try:
            margin = float(gt_spec.get("xy_inner_margin", 0.0) or 0.0)
            hx = max(abs(float(half_extents[0])) - margin, 0.0)
            hy = max(abs(float(half_extents[1])) - margin, 0.0)
            floor_top_z = float(floor_xyz[0, 2].item()) + float(gt_spec.get("interior_floor_top_offset", 0.0) or 0.0)
            opening_z = float(floor_xyz[0, 2].item()) + float(gt_spec.get("opening_plane_z_offset", 0.0) or 0.0)
            dx = abs(float(target_xyz[0, 0].item()) - float(floor_xyz[0, 0].item()))
            dy = abs(float(target_xyz[0, 1].item()) - float(floor_xyz[0, 1].item()))
            tz = float(target_xyz[0, 2].item())
        except (TypeError, ValueError, IndexError) as exc:
            self._note_goal_metric_issue(f"containment_fields_invalid:{type(exc).__name__}")
            return None
        xy_violation = float((max(dx - hx, 0.0) ** 2 + max(dy - hy, 0.0) ** 2) ** 0.5)
        # Centre-point test only: the object centre must sit between the interior floor and the
        # opening plane. Bounding-box based clearances (max_floor_gap etc.) need asset extents,
        # which the delegate cannot query, so they stay out of the runtime criterion.
        z_violation = max(floor_top_z - tz, tz - opening_z, 0.0)
        return max(xy_violation, z_violation)

    def _support_top_violation(self, *, target_xyz, support_xyz, gt_spec: dict[str, Any]) -> float:
        """0.0 when the object rests over the support top; grows with xy offset or missing lift."""
        allowance = self._preview_success_radius() or 0.02
        dx = float(target_xyz[0, 0].item()) - float(support_xyz[0, 0].item())
        dy = float(target_xyz[0, 1].item()) - float(support_xyz[0, 1].item())
        xy_violation = max(float((dx * dx + dy * dy) ** 0.5) - float(allowance), 0.0)
        # An object on the table next to the support has the same xy signature as one placed on it;
        # requiring the centre to sit above the support centre separates the two.
        z_violation = max(float(support_xyz[0, 2].item()) - float(target_xyz[0, 2].item()), 0.0)
        return max(xy_violation, z_violation)

    def _placement_goal_distance(self, *, protocol: dict[str, Any], gt_spec: dict[str, Any]) -> float | None:
        try:
            import torch
        except Exception as exc:
            self._note_goal_metric_issue(f"placement_import_failed:{type(exc).__name__}")
            return None
        resolver = str(gt_spec.get("resolver", "")).strip().lower()
        target_name = self._protocol_object_name(
            protocol,
            gt_spec,
            gt_key="target_object_id",
            protocol_key="target_object",
        )
        target_xy = self._asset_position_xy(target_name)
        if target_xy is None:
            return None
        if resolver in {"containment_placement", "support_top_placement"}:
            target_xyz = self._asset_position_xyz(target_name)
            if target_xyz is None:
                return None
            if resolver == "containment_placement":
                return self._containment_violation(target_xyz=target_xyz, gt_spec=gt_spec)
            support_xyz = self._asset_position_xyz(
                self._protocol_object_name(protocol, gt_spec, gt_key="support_object_id", protocol_key="support_object")
            )
            if support_xyz is None:
                self._note_goal_metric_issue("support_object_unresolved")
                return None
            return self._support_top_violation(target_xyz=target_xyz, support_xyz=support_xyz, gt_spec=gt_spec)
        center_xy = None
        if resolver == "directional_region_placement":
            anchor_name = self._protocol_object_name(
                protocol,
                gt_spec,
                gt_key="anchor_object_id",
                protocol_key="anchor_object",
            )
            anchor_xy = self._asset_position_xy(anchor_name)
            zone = gt_spec.get("target_zone", {})
            center_offset = (zone if isinstance(zone, dict) else {}).get("center_offset_from_anchor", ())
            if anchor_xy is not None and isinstance(center_offset, (list, tuple)) and len(center_offset) >= 2:
                center_xy = anchor_xy + torch.tensor(
                    [float(center_offset[0]), float(center_offset[1])],
                    dtype=torch.float32,
                    device=self._native_env.device,
                ).unsqueeze(0)
        elif resolver == "support_top_placement":
            support_name = self._protocol_object_name(
                protocol,
                gt_spec,
                gt_key="support_object_id",
                protocol_key="support_object",
            )
            center_xy = self._asset_position_xy(support_name)
        elif resolver == "containment_placement":
            reference_name = str(gt_spec.get("container_reference_object_id", "")).strip()
            center_xy = self._asset_position_xy(reference_name)
        if center_xy is None:
            self._note_goal_metric_issue(f"placement_zone_center_unresolved:{resolver or '<empty>'}")
            return None
        try:
            delta = torch.abs(target_xy - center_xy)
            half_extents = self._zone_half_extents(gt_spec)
            if half_extents is not None:
                # Rectangular zone: distance to the *box*, per axis, instead of an inscribed circle.
                # 0.0 means the object is inside the zone regardless of which axis it is spread along.
                extents = torch.tensor(
                    [float(half_extents[0]), float(half_extents[1])],
                    dtype=torch.float32,
                    device=self._native_env.device,
                ).unsqueeze(0)
                outside = torch.clamp(delta - extents, min=0.0)
                distance = torch.linalg.vector_norm(outside, dim=1)
            else:
                distance = torch.linalg.vector_norm(delta, dim=1)
            return float(distance[0].item())
        except Exception as exc:
            self._note_goal_metric_issue(f"placement_distance_failed:{type(exc).__name__}")
            return None

    def _adapt_observation(self, native_obs: Any, info: dict[str, Any]) -> dict[str, Any]:
        return build_raw_observation(
            instruction=self._effective_instruction,
            step=self._episode_step,
            max_steps=self.cfg.max_episode_steps,
            state={"native_obs": self._to_python(native_obs)},
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
                "max_episode_steps": self.cfg.max_episode_steps,
                "direct_ablation": self.direct_ablation_record,
                **info,
            },
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        self._episode_step = 0
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        # Historically the seed argument was dropped here; determinism comes from
        # cfg.seed set in _build_native_env. Forwarding it would change every existing
        # base result, so it stays opt-in and off by default.
        if seed is not None and self._env_bool("GEOMIND_PASS_SEED_TO_RESET", False):
            obs, extras = self._native_env.reset(seed=int(seed))
        else:
            obs, extras = self._native_env.reset()
        self._native_object_perturbation_applied = self._maybe_apply_native_object_pose_perturbation()
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        info = {**self._runtime_info(), "native_extras": self._to_python(extras)}
        return self._adapt_observation(obs, info), info

    def step(self, action):
        import torch

        self._episode_step += 1
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._native_target_is_preloaded_gripper_object() and self._last_action.size > 0:
            self._last_action[-1] = -1.0
        action_dim = int(self._last_action.shape[0])
        batched_action = torch.as_tensor(self._last_action, device=self._native_env.device, dtype=torch.float32).reshape(
            1, action_dim
        )
        if int(getattr(self._native_env, "num_envs", 1)) > 1:
            batched_action = batched_action.repeat(int(self._native_env.num_envs), 1)
        native_obs, reward, terminated, truncated, extras = self._native_env.step(batched_action)
        reward_value = float(reward[0].item()) if hasattr(reward, "__getitem__") else float(reward)
        terminated_value = bool(terminated[0].item()) if hasattr(terminated, "__getitem__") else bool(terminated)
        truncated_value = bool(truncated[0].item()) if hasattr(truncated, "__getitem__") else bool(truncated)
        self._last_reward = reward_value
        self._last_terminated = terminated_value
        self._last_truncated = truncated_value
        info = {**self._runtime_info(), "native_extras": self._to_python(extras)}
        obs = self._adapt_observation(native_obs, info)
        return obs, reward_value, terminated_value, truncated_value, info

    def close(self):
        close_fn = getattr(self._native_env, "close", None)
        if callable(close_fn):
            close_fn()

    def _resolve_native_target_object_id(self) -> str:
        task_protocol = dict(getattr(self._native_env.cfg, "task_protocol", {}) or {})
        gt_spec = dict(task_protocol.get("gt_spec", {}) or {})
        target_object = dict(task_protocol.get("target_object", {}) or {})
        return str(gt_spec.get("target_object_id") or target_object.get("object_id") or "").strip()

    def _native_target_is_preloaded_gripper_object(self) -> bool:
        task_protocol = dict(getattr(self._native_env.cfg, "task_protocol", {}) or {})
        target_object = task_protocol.get("target_object", {})
        if not isinstance(target_object, dict):
            return False
        if str(target_object.get("carrier", "")).strip().lower() != "gripper":
            return False
        preloaded = target_object.get("preloaded", False)
        if isinstance(preloaded, str):
            preloaded = preloaded.strip().lower() in {"1", "true", "yes", "y", "on"}
        if not bool(preloaded):
            return False
        return bool(str(target_object.get("object_id", "")).strip())

    def _maybe_apply_native_object_pose_perturbation(self) -> bool:
        try:
            import torch
        except Exception:
            return False
        enabled = bool(self._backend_kwargs.get("enable_object_pose_perturb", False))
        perturb_range = float(self._backend_kwargs.get("object_pose_perturb_range", 0.0) or 0.0)
        if not enabled or perturb_range <= 0.0:
            return False
        if self._native_target_is_preloaded_gripper_object():
            return False
        target_object_id = self._resolve_native_target_object_id()
        if not target_object_id:
            return False
        try:
            target_asset = self._native_env.scene[str(target_object_id)]
        except Exception:
            return False
        root_state = getattr(getattr(target_asset, "data", None), "root_state_w", None)
        if root_state is None:
            return False
        try:
            state = root_state.clone()
            num_rows = int(state.shape[0])
            perturb_seed = int(self._backend_kwargs.get("object_pose_perturb_seed", 0) or 0)
            rng = np.random.default_rng(perturb_seed)
            delta_np = rng.uniform(
                -perturb_range,
                perturb_range,
                size=(num_rows, 2),
            ).astype(np.float32)
            delta_xy = torch.as_tensor(
                delta_np,
                device=state.device,
                dtype=state.dtype,
            )
            state[:, 0] = state[:, 0] + delta_xy[:, 0]
            state[:, 1] = state[:, 1] + delta_xy[:, 1]
            state[:, 7:13] = 0.0
            target_asset.write_root_state_to_sim(state)
            self._native_env.scene.write_data_to_sim()
            self._native_env.scene.update(dt=float(getattr(self._native_env, "physics_dt", 0.0) or 0.0))
            return True
        except Exception:
            return False
