from __future__ import annotations

from collections.abc import Sequence
import math
import random
from typing import Any

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import SceneEntityCfg

from geomind_lab.mimic.envs.state_reindexing_logic import (
    DEFAULT_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
    DYNAMIC_SWAP_FIRST_SECOND_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
    FINAL_FRAME_ONLY_CONTROL_MODE,
    MULTI_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
    SEEDED_NEIGHBOR_SWAP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
    SEEDED_NEIGHBOR_SWAP_MIDDLE_DIRECT_OVER_2S_SCRIPT_NAME,
    STATIC_SCENE_NOOP_SCRIPT_NAME,
    SWAP_ADJACENT_PAIRS_OVER_4_SCRIPT_NAME,
    build_state_reindexing_intervention_plan,
    build_state_reindexing_status_labels,
    build_historical_identity_tracking_intervention_plan,
    evaluate_state_reindexing_readout,
    evaluate_state_reindexing_terminal_state,
    evaluate_historical_identity_tracking_terminal_state,
    normalize_post_action_relocation_control_mode,
    resolve_post_action_relocation_control_mode,
    resolve_post_action_relocation_order_axis,
    resolve_state_reindexing_script_name,
    resolve_state_reindexing_target_rank,
)
from geomind_lab.mimic.envs.post_action_relocation_motion import (
    REINDEX_INSERT_SHALLOW_ARC,
    REINDEX_NEIGHBOR_BEZIER_LANES,
    REINDEX_SUBSET_TWO_BLOCK_SWAP,
    TRACK_MULTI_STAGGERED_WAYPOINT,
    TRACK_SINGLE_PAIRED_ARC,
    get_post_action_relocation_motion_function,
    resolve_post_action_relocation_motion_mode,
)
from geomind_lab.tasks.manager_based.manipulation.geomind import mdp as geomind_mdp
from geomind_lab.tasks.manager_based.manipulation.geomind.mdp.spatial_placement_debug_vis import (
    draw_directional_placement_success_zone_debug,
)
from geomind_lab.tasks.manager_based.manipulation.geomind.task_specs import resolve_task_spec_object_names

from geomind_lab.mimic.envs.track_clutter_motion_planner import (
    TRACK_CLUTTER_MAX_KEYFRAMES,
    build_track_clutter_keyframes,
    flatten_final_xy_from_slots,
    interpolate_track_clutter_xy_row,
    repel_active_identity_targets_xy,
    repel_distractor_identity_pair_xy,
    resolve_track_clutter_protocol_grasp_object_index,
)


class GeoMindFrankaIKRelMimicEnv(ManagerBasedRLMimicEnv):
    """Generic GeoMind Mimic wrapper for Franka IK-relative tasks."""

    _PROTOCOL_MODULE_NAME = "post_action_relocation"
    _STATE_REINDEXING_TRACK_NAME = "state_reindexing"
    _OBJECT_POSITION_UPDATE_TRACK_NAME = "object_position_update"
    _HISTORICAL_IDENTITY_TRACKING_TRACK_NAME = "historical_identity_tracking"
    _OBJECT_IDENTITY_TRACKING_TRACK_NAME = "object_identity_tracking"
    _REORDER_WITH_DISTRACTORS_TRACK_NAME = "reorder_with_distractors"
    _TERMINAL_STATE_TRACK_NAMES = frozenset(
        {
            _STATE_REINDEXING_TRACK_NAME,
            _OBJECT_POSITION_UPDATE_TRACK_NAME,
            _REORDER_WITH_DISTRACTORS_TRACK_NAME,
        }
    )
    _INITIAL_STATE_TRACK_NAMES = frozenset(
        {
            _HISTORICAL_IDENTITY_TRACKING_TRACK_NAME,
            _OBJECT_IDENTITY_TRACKING_TRACK_NAME,
        }
    )
    _SUPPORTED_PROTOCOL_TRACK_NAMES = frozenset(
        set(_TERMINAL_STATE_TRACK_NAMES) | set(_INITIAL_STATE_TRACK_NAMES)
    )

    def _eef_name(self) -> str:
        subtask_configs = dict(getattr(self.cfg, "subtask_configs", {}) or {})
        return str(next(iter(subtask_configs.keys()), "franka"))

    def _task_protocol(self) -> dict[str, Any]:
        protocol = getattr(self.cfg, "task_protocol", {})
        return protocol if isinstance(protocol, dict) else {}

    def _protocol_intervention(self) -> dict[str, Any]:
        intervention = self._task_protocol().get("intervention", {})
        return intervention if isinstance(intervention, dict) else {}

    def _protocol_track_name(self) -> str:
        return str(self._task_protocol().get("track_name", ""))

    def _is_track_clutter_task(self) -> bool:
        cfg_tid = str(getattr(self.cfg, "task_id", "") or "").strip().lower()
        if cfg_tid == "track_clutter":
            return True
        task_spec = getattr(self.cfg, "task_spec", {})
        if isinstance(task_spec, dict):
            if str(task_spec.get("task_id", "") or "").strip().lower() == "track_clutter":
                return True
            annotation = task_spec.get("annotation", {})
            if isinstance(annotation, dict) and str(annotation.get("task_id", "") or "").strip().lower() == "track_clutter":
                return True
        return False

    def _protocol_min_reorder_gap_for_intervention(self) -> float:
        """Minimum reorder-axis separation for scripted targets (wider on track_clutter)."""
        g = float(getattr(self.cfg, "protocol_min_reorder_gap", 0.08))
        if self._is_track_clutter_task():
            return max(g * 4.25, 0.44)
        return g

    def _merge_track_clutter_motion_plan(
        self,
        *,
        initial_xy: torch.Tensor,
        step_object_indices: torch.Tensor,
        step_target_xy: torch.Tensor,
        active_object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep only identity-set movers for track_clutter to lower multi-object collision risk."""
        num_envs, ncols = step_object_indices.shape[:2]
        for row in range(num_envs):
            identity_slots: list[tuple[int, torch.Tensor]] = []
            for slot in range(min(3, int(ncols))):
                oid = int(step_object_indices[row, slot].item())
                if oid >= 0:
                    identity_slots.append((oid, step_target_xy[row, slot, :].clone()))
            slot_ptr = 0
            for oid, tgt in identity_slots:
                if slot_ptr >= int(ncols):
                    break
                step_object_indices[row, slot_ptr] = int(oid)
                step_target_xy[row, slot_ptr, :] = tgt
                slot_ptr += 1
            if slot_ptr < int(ncols):
                step_object_indices[row, slot_ptr:] = -1
                step_target_xy[row, slot_ptr:, :] = 0.0
        return step_object_indices, step_target_xy

    def _is_post_action_relocation_task(self) -> bool:
        protocol = self._task_protocol()
        return (
            str(protocol.get("module_name", "")) == self._PROTOCOL_MODULE_NAME
            and self._protocol_track_name() in self._SUPPORTED_PROTOCOL_TRACK_NAMES
        )

    def _is_state_reindexing_task(self) -> bool:
        return self._is_post_action_relocation_task() and self._protocol_track_name() in self._TERMINAL_STATE_TRACK_NAMES

    def _is_reindex_insert_task(self) -> bool:
        return str(getattr(self.cfg, "task_id", "") or "").strip().lower() == "reindex_insert"

    def _is_reindex_neighbor_task(self) -> bool:
        cfg_tid = str(getattr(self.cfg, "task_id", "") or "").strip().lower()
        if cfg_tid == "reindex_neighbor":
            return True
        task_spec = getattr(self.cfg, "task_spec", {})
        if isinstance(task_spec, dict):
            if str(task_spec.get("task_id", "") or "").strip().lower() == "reindex_neighbor":
                return True
            annotation = task_spec.get("annotation", {})
            if isinstance(annotation, dict) and str(annotation.get("task_id", "") or "").strip().lower() == "reindex_neighbor":
                return True
        return False

    def _is_reindex_subset_task(self) -> bool:
        cfg_tid = str(getattr(self.cfg, "task_id", "") or "").strip().lower()
        if cfg_tid == "reindex_subset":
            return True
        task_spec = getattr(self.cfg, "task_spec", {})
        if isinstance(task_spec, dict):
            if str(task_spec.get("task_id", "") or "").strip().lower() == "reindex_subset":
                return True
            annotation = task_spec.get("annotation", {})
            if isinstance(annotation, dict) and str(annotation.get("task_id", "") or "").strip().lower() == "reindex_subset":
                return True
        return False

    def _portable_relocation_motion_mode(self, script_name: str) -> str | None:
        intervention = self._protocol_intervention()
        configured = intervention.get("motion_mode")
        if configured is None and script_name == SEEDED_NEIGHBOR_SWAP_MIDDLE_DIRECT_OVER_2S_SCRIPT_NAME:
            return None
        task_id = str(getattr(self.cfg, "task_id", "") or "").strip().lower()
        defaults = {
            "track_single": TRACK_SINGLE_PAIRED_ARC,
            "track_multi": TRACK_MULTI_STAGGERED_WAYPOINT,
            "reindex_insert": REINDEX_INSERT_SHALLOW_ARC,
            "reindex_neighbor": REINDEX_NEIGHBOR_BEZIER_LANES,
            "reindex_subset": REINDEX_SUBSET_TWO_BLOCK_SWAP,
        }
        default = defaults.get(task_id)
        if configured is None and default is None:
            return None
        if default is None:
            default = TRACK_SINGLE_PAIRED_ARC
        return resolve_post_action_relocation_motion_mode(intervention, default=default)

    def _relocation_motion_mode_label(self, script_name: str) -> str | None:
        mode = self._portable_relocation_motion_mode(script_name)
        if mode is not None:
            return mode
        if script_name == SEEDED_NEIGHBOR_SWAP_MIDDLE_DIRECT_OVER_2S_SCRIPT_NAME:
            return "track_single_middle_direct_legacy"
        return None

    def _is_historical_identity_tracking_task(self) -> bool:
        return self._is_post_action_relocation_task() and self._protocol_track_name() in self._INITIAL_STATE_TRACK_NAMES

    def _state_reindexing_script_name(self) -> str:
        return resolve_state_reindexing_script_name(self._task_protocol())

    def _is_noop_intervention_script(self) -> bool:
        script_name = self._state_reindexing_script_name()
        return script_name == STATIC_SCENE_NOOP_SCRIPT_NAME or script_name.startswith("disabled_noop")

    def _is_dynamic_historical_identity_tracking_script(self) -> bool:
        return self._state_reindexing_script_name() in {
            DEFAULT_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
            DYNAMIC_SWAP_FIRST_SECOND_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
            SEEDED_NEIGHBOR_SWAP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
            SEEDED_NEIGHBOR_SWAP_MIDDLE_DIRECT_OVER_2S_SCRIPT_NAME,
            MULTI_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
        }

    def _is_dynamic_state_reindexing_script(self) -> bool:
        return self._state_reindexing_script_name() == SWAP_ADJACENT_PAIRS_OVER_4_SCRIPT_NAME

    def _uses_dynamic_intervention_motion(self) -> bool:
        if self._is_dynamic_historical_identity_tracking_script() or self._is_dynamic_state_reindexing_script():
            return True
        return self._debug_rightmost_between_slide_enabled()

    def _debug_rightmost_between_slide_enabled(self) -> bool:
        if self._state_reindexing_script_name() != "rightmost_to_between_first_and_second":
            return False
        if bool(self._protocol_intervention().get("vla_visible_motion", False)):
            return True
        return bool(
            getattr(self.cfg, "debug_smooth_planar_insert", False)
            or getattr(self.cfg, "debug_smooth_planar_neighbor", False)
        )

    def _uses_seeded_neighbor_dynamic_swap(self) -> bool:
        return self._state_reindexing_script_name() in {
            SEEDED_NEIGHBOR_SWAP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME,
            SEEDED_NEIGHBOR_SWAP_MIDDLE_DIRECT_OVER_2S_SCRIPT_NAME,
        }

    def _middle_direct_swap_detour_dx(self) -> float:
        raw = self._protocol_intervention().get("middle_detour_dx", 0.08)
        try:
            return max(float(raw), 0.0)
        except (TypeError, ValueError):
            return 0.08

    @staticmethod
    def _quintic_smoothstep(t: torch.Tensor) -> torch.Tensor:
        """Perlin/Ken smootherstep — C² at 0/1 for softer velocity than linear."""
        t = torch.clamp(t, 0.0, 1.0)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _piecewise_two_segment_via_waypoint(
        start_row: torch.Tensor,
        target_row: torch.Tensor,
        waypoint_row: torch.Tensor,
        ep: torch.Tensor,
    ) -> torch.Tensor:
        """Two-segment motion s→w→t with explicit per-batch waypoint ``waypoint_row`` (same shape as s/t)."""
        s = start_row
        t = target_row
        w = waypoint_row
        v0 = w - s
        v1 = t - w
        l0 = torch.linalg.norm(v0, dim=-1).clamp(min=1e-5)
        l1 = torch.linalg.norm(v1, dim=-1).clamp(min=1e-5)
        dist = ep * (l0 + l1)
        pos0 = s + v0 * (dist / l0).clamp(0.0, 1.0).unsqueeze(-1)
        alpha1 = ((dist - l0) / l1).clamp(0.0, 1.0).unsqueeze(-1)
        pos1 = w + v1 * alpha1
        return torch.where(dist.unsqueeze(-1) < l0.unsqueeze(-1), pos0, pos1)

    @staticmethod
    def _quadratic_bezier_xy(
        start_row: torch.Tensor,
        control_row: torch.Tensor,
        target_row: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Quadratic Bezier interpolation for smooth synchronized tabletop motion."""
        t = torch.clamp(u, 0.0, 1.0).unsqueeze(-1)
        omt = 1.0 - t
        return (omt * omt) * start_row + (2.0 * omt * t) * control_row + (t * t) * target_row

    @staticmethod
    def _cubic_bezier_xy(
        start_row: torch.Tensor,
        control_start_row: torch.Tensor,
        control_end_row: torch.Tensor,
        target_row: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Cubic Bezier interpolation with explicit start/end lane controls."""
        t = torch.clamp(u, 0.0, 1.0).unsqueeze(-1)
        omt = 1.0 - t
        return (
            (omt * omt * omt) * start_row
            + (3.0 * omt * omt * t) * control_start_row
            + (3.0 * omt * t * t) * control_end_row
            + (t * t * t) * target_row
        )

    @staticmethod
    def _debug_insert_slide_targets_xy(
        initial_xy: torch.Tensor,
        initial_rank_indices: torch.Tensor,
        active_object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map objects only onto the four initial slots while preserving insert semantics."""
        target_xy = initial_xy.clone()
        planned_moved_mask = torch.zeros_like(active_object_mask, dtype=torch.bool)
        if initial_xy.ndim < 3 or initial_rank_indices.ndim < 2:
            return target_xy, planned_moved_mask

        num_envs, num_objects = tuple(initial_xy.shape[:2])
        if num_objects <= 0:
            return target_xy, planned_moved_mask

        active_counts = active_object_mask.sum(dim=1)

        for row in range(num_envs):
            count = int(active_counts[row].item())
            if count != 4:
                continue
            ranked = [
                int(idx)
                for idx in initial_rank_indices[row, :4].detach().cpu().tolist()
                if 0 <= int(idx) < num_objects
            ]
            if len(ranked) != 4:
                continue

            # Final rank order: r0, rightmost, r1, r2. Every endpoint is one of the four initial slots.
            final_order = [ranked[0], ranked[-1], *ranked[1:-1]]
            slot_xy = initial_xy[row, ranked, :]
            for slot, object_index in enumerate(final_order):
                target_xy[row, object_index, :] = slot_xy[slot]
            displacement = torch.linalg.vector_norm(target_xy[row] - initial_xy[row], dim=-1)
            planned_moved_mask[row] = displacement > 1.0e-4

        return target_xy, planned_moved_mask

    @staticmethod
    def _piecewise_middle_detour_minus_x(
        start_row: torch.Tensor,
        target_row: torch.Tensor,
        ep: torch.Tensor,
        *,
        detour_dx: float,
        table_x_min: float = 0.28,
    ) -> torch.Tensor:
        """Two-segment motion s→w→t with w shifted toward −X to reduce +X anchor collisions."""
        s = start_row
        t = target_row
        device = s.device
        dtype = s.dtype
        x_floor = torch.tensor(float(table_x_min), device=device, dtype=dtype)
        raw_w0 = torch.minimum(s[:, 0], t[:, 0]) - float(detour_dx)
        w0 = torch.maximum(raw_w0, x_floor)
        w1 = 0.5 * (s[:, 1] + t[:, 1])
        w = torch.stack((w0, w1), dim=-1)
        return GeoMindFrankaIKRelMimicEnv._piecewise_two_segment_via_waypoint(s, t, w, ep)

    def _middle_direct_slot1_piecewise_xy(self, start_row: torch.Tensor, target_row: torch.Tensor, ep: torch.Tensor) -> torch.Tensor:
        """Planar middle-slot path for ``middle_direct`` swap: default −X detour, or orbit around a static occluder."""
        intervention = self._protocol_intervention()
        mode = str(intervention.get("middle_motion", "") or "").strip().lower()
        if mode == "orbit_occluder":
            raw_center = intervention.get("occluder_center_xy")
            if isinstance(raw_center, (list, tuple)) and len(raw_center) >= 2:
                try:
                    cx = float(raw_center[0])
                    float(raw_center[1])
                    r_c = max(float(intervention.get("occluder_radius", 0.0192)), 0.0)
                    clr = max(float(intervention.get("orbit_clearance", 0.018)), 0.0)
                except (TypeError, ValueError):
                    pass
                else:
                    # Always bulge toward −X (robot / rear of the row): w0 = min(min(sx,tx), cx−r−clr) − detour.
                    # Works whether the occluder sits on +X (front) or −X (rear): the waypoint stays on the block
                    # row's −X side, never toward +X (viewer front).
                    device = start_row.device
                    dtype = start_row.dtype
                    x_floor = torch.tensor(float(0.28), device=device, dtype=dtype)
                    detour_dx = float(self._middle_direct_swap_detour_dx())
                    m = torch.minimum(start_row[:, 0], target_row[:, 0])
                    west_disc = float(cx - r_c - clr)
                    west_cap = torch.full_like(m, west_disc, device=device, dtype=dtype)
                    w0 = torch.maximum(torch.minimum(m, west_cap) - detour_dx, x_floor)
                    w1 = 0.5 * (start_row[:, 1] + target_row[:, 1])
                    w = torch.stack((w0, w1), dim=-1)
                    return self._piecewise_two_segment_via_waypoint(start_row, target_row, w, ep)
        detour_dx = self._middle_direct_swap_detour_dx()
        return self._piecewise_middle_detour_minus_x(start_row, target_row, ep, detour_dx=detour_dx)

    @staticmethod
    def _enforce_min_pairwise_separation_xy(
        xy_row: torch.Tensor,
        active_mask_row: torch.Tensor,
        *,
        min_sep: float,
        max_iters: int = 2,
    ) -> torch.Tensor:
        """Lightweight per-row XY separation to avoid same-time overlap."""
        out = xy_row.clone()
        n = int(out.shape[0])
        if n < 2:
            return out
        eps = 1e-6
        for _ in range(max(1, int(max_iters))):
            for i in range(n - 1):
                if not bool(active_mask_row[i].item()):
                    continue
                for j in range(i + 1, n):
                    if not bool(active_mask_row[j].item()):
                        continue
                    d = out[i] - out[j]
                    dist = float(torch.linalg.norm(d).item())
                    if dist >= float(min_sep):
                        continue
                    if dist < eps:
                        # deterministic fallback direction
                        d = out.new_tensor([1.0, 0.0]) if ((i + j) % 2 == 0) else out.new_tensor([0.0, 1.0])
                        dist = 1.0
                    direction = d / max(dist, eps)
                    delta = 0.5 * (float(min_sep) - dist) * direction
                    out[i] = out[i] + delta
                    out[j] = out[j] - delta
        return out

    def _dynamic_historical_identity_motion_duration_s(self) -> float:
        intervention = self._protocol_intervention()
        if bool(intervention.get("instant_swap", False)):
            return 1e-3
        try:
            resolved = float(intervention.get("duration_s", 2.0))
        except (TypeError, ValueError):
            resolved = 2.0
        return max(resolved, 1e-3)

    def _reset_stabilization_steps(self) -> int:
        return max(int(getattr(self.cfg, "reset_stabilization_steps", 0)), 0)

    def _protocol_instant_swap(self) -> bool:
        return bool(self._protocol_intervention().get("instant_swap", False))

    def _dynamic_swap_delay_phys_steps(self) -> int:
        """Physics substeps to wait before scripted swap motion (see reset_stabilization_steps × decimation)."""
        if self._protocol_instant_swap():
            return 0
        return int(self._reset_stabilization_steps()) * max(int(self.cfg.decimation), 1)

    def _task_seed(self) -> int:
        try:
            return int(getattr(self.cfg, "seed", 1))
        except (TypeError, ValueError):
            return 1

    def _seeded_dynamic_swap_with_right_mask(
        self,
        env_ids: torch.Tensor,
        episode_counters: torch.Tensor,
    ) -> torch.Tensor:
        intervention = self._protocol_intervention()
        forced = intervention.get("force_neighbor_swap")
        if isinstance(forced, str):
            normalized = forced.strip().lower()
            count = int(env_ids.numel())
            if normalized == "left":
                return torch.zeros((count,), dtype=torch.bool, device=self.device)
            if normalized == "right":
                return torch.ones((count,), dtype=torch.bool, device=self.device)
        env_id_values = env_ids.detach().cpu().tolist()
        episode_values = episode_counters.detach().cpu().tolist()
        base_seed = int(self._task_seed())
        choices = [
            random.Random(base_seed + int(env_id) * 1000003 + int(episode_counter) * 9176).random() >= 0.5
            for env_id, episode_counter in zip(env_id_values, episode_values, strict=False)
        ]
        return torch.as_tensor(choices, dtype=torch.bool, device=self.device)

    def _state_reindexing_target_rank(self) -> str:
        return resolve_state_reindexing_target_rank(self._task_protocol())

    def _protocol_control_mode(self) -> str:
        return resolve_post_action_relocation_control_mode(self._task_protocol())

    def _protocol_order_axis(self) -> str:
        return resolve_post_action_relocation_order_axis(self._task_protocol())

    def _protocol_gt_resolver(self) -> str:
        gt_spec = dict(self._task_protocol().get("gt_spec", {}) or {})
        return str(gt_spec.get("resolver", ""))

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)

    def _protocol_object_names(self) -> list[str]:
        scene_profile = getattr(self.cfg, "_scene_profile", None)
        task_spec = getattr(self.cfg, "task_spec", {})
        resolved_names = resolve_task_spec_object_names(task_spec, scene_profile)
        if isinstance(task_spec, dict):
            annotation = task_spec.get("annotation", {})
            if isinstance(annotation, dict):
                scene_adaptation = annotation.get("scene_adaptation", {})
                if isinstance(scene_adaptation, dict):
                    runtime_objects = scene_adaptation.get("runtime_objects", [])
                    if isinstance(runtime_objects, list):
                        runtime_names = [
                            str(item.get("name"))
                            for item in runtime_objects
                            if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name")
                        ]
                        if runtime_names:
                            merged_names: list[str] = []
                            seen: set[str] = set()
                            for name in list(resolved_names) + runtime_names:
                                if name not in seen:
                                    seen.add(name)
                                    merged_names.append(name)
                            resolved_names = merged_names
        filtered_names: list[str] = []
        for object_name in resolved_names:
            try:
                asset = self.scene[str(object_name)]
            except Exception:
                continue
            data = getattr(asset, "data", None)
            if data is None or not hasattr(data, "root_pos_w"):
                continue
            filtered_names.append(str(object_name))
        return filtered_names

    def _protocol_target_subset_object_names(self, object_names: Sequence[str] | None = None) -> list[str]:
        resolved_object_names = [str(name) for name in (self._protocol_object_names() if object_names is None else object_names)]
        protocol = self._task_protocol()
        subset_spec = protocol.get("target_subset_spec", {})
        if not isinstance(subset_spec, dict):
            return resolved_object_names
        raw_object_ids = subset_spec.get("object_ids", [])
        if not isinstance(raw_object_ids, list):
            return resolved_object_names
        requested = {str(object_id) for object_id in raw_object_ids if isinstance(object_id, str) and object_id}
        if not requested:
            return resolved_object_names
        filtered = [name for name in resolved_object_names if name in requested]
        return filtered if filtered else resolved_object_names

    def _protocol_target_subset_mask(self, object_names: Sequence[str]) -> torch.Tensor:
        subset_names = set(self._protocol_target_subset_object_names(object_names))
        return torch.as_tensor(
            [str(object_name) in subset_names for object_name in object_names],
            device=self.device,
            dtype=torch.bool,
        )

    def _ensure_state_reindexing_runtime(self) -> dict[str, Any]:
        object_names = self._protocol_object_names()
        num_objects = len(object_names)
        motion_slot_count = max(4, num_objects)
        target_subset_object_ids = tuple(self._protocol_target_subset_object_names(object_names))
        target_subset_mask = self._protocol_target_subset_mask(object_names).unsqueeze(0).expand(self.num_envs, -1).clone()
        state = getattr(self, "_state_reindexing_runtime", None)
        if (
            isinstance(state, dict)
            and list(state.get("object_names", ())) == object_names
            and state.get("initial_xy") is not None
            and tuple(state["initial_xy"].shape[:2]) == (self.num_envs, num_objects)
        ):
            if "selected_object_indices" not in state:
                state["selected_object_indices"] = torch.full(
                    (self.num_envs,), -1, dtype=torch.long, device=self.device
                )
            if "planned_moved_mask" not in state:
                state["planned_moved_mask"] = torch.zeros(
                    (self.num_envs, num_objects), dtype=torch.bool, device=self.device
                )
            elif tuple(state["planned_moved_mask"].shape[:2]) != (self.num_envs, num_objects):
                state["planned_moved_mask"] = torch.zeros(
                    (self.num_envs, num_objects), dtype=torch.bool, device=self.device
                )
            if "target_subset_mask" not in state or tuple(state["target_subset_mask"].shape[:2]) != (self.num_envs, num_objects):
                state["target_subset_mask"] = target_subset_mask
            if "max_distractor_planar_displacement" not in state:
                state["max_distractor_planar_displacement"] = torch.zeros(
                    (self.num_envs,), dtype=torch.float32, device=self.device
                )
            if "dynamic_motion_object_indices" not in state or tuple(state["dynamic_motion_object_indices"].shape[:2]) != (
                self.num_envs,
                motion_slot_count,
            ):
                state["dynamic_motion_object_indices"] = torch.full(
                    (self.num_envs, motion_slot_count),
                    -1,
                    dtype=torch.long,
                    device=self.device,
                )
            if "dynamic_motion_start_xy" not in state or tuple(state["dynamic_motion_start_xy"].shape[:3]) != (
                self.num_envs,
                motion_slot_count,
                2,
            ):
                state["dynamic_motion_start_xy"] = torch.zeros(
                    (self.num_envs, motion_slot_count, 2),
                    dtype=torch.float32,
                    device=self.device,
                )
            if "dynamic_motion_prev_planar_xy" not in state or tuple(state["dynamic_motion_prev_planar_xy"].shape[:3]) != (
                self.num_envs,
                motion_slot_count,
                2,
            ):
                state["dynamic_motion_prev_planar_xy"] = torch.zeros(
                    (self.num_envs, motion_slot_count, 2),
                    dtype=torch.float32,
                    device=self.device,
                )
            if "dynamic_motion_target_xy" not in state or tuple(state["dynamic_motion_target_xy"].shape[:3]) != (
                self.num_envs,
                motion_slot_count,
                2,
            ):
                state["dynamic_motion_target_xy"] = torch.zeros(
                    (self.num_envs, motion_slot_count, 2),
                    dtype=torch.float32,
                    device=self.device,
                )
            if "dynamic_motion_fixed_quat" not in state or tuple(state["dynamic_motion_fixed_quat"].shape[:3]) != (
                self.num_envs,
                motion_slot_count,
                4,
            ):
                state["dynamic_motion_fixed_quat"] = torch.zeros(
                    (self.num_envs, motion_slot_count, 4),
                    dtype=torch.float32,
                    device=self.device,
                )
                state["dynamic_motion_fixed_quat"][:, :, 0] = 1.0
            if "dynamic_motion_fixed_z" not in state or tuple(state["dynamic_motion_fixed_z"].shape[:2]) != (
                self.num_envs,
                motion_slot_count,
            ):
                state["dynamic_motion_fixed_z"] = torch.zeros(
                    (self.num_envs, motion_slot_count),
                    dtype=torch.float32,
                    device=self.device,
                )
            if "dynamic_motion_cmd_planar_xy" not in state or tuple(state["dynamic_motion_cmd_planar_xy"].shape[:3]) != (
                self.num_envs,
                motion_slot_count,
                2,
            ):
                state["dynamic_motion_cmd_planar_xy"] = torch.zeros(
                    (self.num_envs, motion_slot_count, 2),
                    dtype=torch.float32,
                    device=self.device,
                )
            if "dynamic_motion_elapsed_s" not in state:
                state["dynamic_motion_elapsed_s"] = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            if "dynamic_motion_duration_s" not in state:
                state["dynamic_motion_duration_s"] = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            if "dynamic_motion_delay_steps_remaining" not in state:
                state["dynamic_motion_delay_steps_remaining"] = torch.zeros(
                    (self.num_envs,), dtype=torch.long, device=self.device
                )
            if "dynamic_swap_with_right" not in state:
                state["dynamic_swap_with_right"] = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            if "dynamic_swap_episode_counter" not in state:
                state["dynamic_swap_episode_counter"] = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
            if "dynamic_motion_enabled" not in state:
                state["dynamic_motion_enabled"] = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            if "dynamic_motion_completed" not in state:
                state["dynamic_motion_completed"] = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            if "track_clutter_identity_slot_begin" not in state:
                state["track_clutter_identity_slot_begin"] = torch.full(
                    (self.num_envs,), -1, dtype=torch.long, device=self.device
                )
            if "track_clutter_kf_active" not in state:
                state["track_clutter_kf_active"] = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            if "track_clutter_kf_p" not in state or tuple(state["track_clutter_kf_p"].shape[:3]) != (
                self.num_envs,
                num_objects,
                TRACK_CLUTTER_MAX_KEYFRAMES,
            ):
                state["track_clutter_kf_p"] = torch.full(
                    (self.num_envs, num_objects, TRACK_CLUTTER_MAX_KEYFRAMES),
                    -1.0,
                    dtype=torch.float32,
                    device=self.device,
                )
            if "track_clutter_kf_xy" not in state or tuple(state["track_clutter_kf_xy"].shape[:4]) != (
                self.num_envs,
                num_objects,
                TRACK_CLUTTER_MAX_KEYFRAMES,
                2,
            ):
                state["track_clutter_kf_xy"] = torch.zeros(
                    (self.num_envs, num_objects, TRACK_CLUTTER_MAX_KEYFRAMES, 2),
                    dtype=torch.float32,
                    device=self.device,
                )
            if "track_clutter_kf_count" not in state or tuple(state["track_clutter_kf_count"].shape[:2]) != (
                self.num_envs,
                num_objects,
            ):
                state["track_clutter_kf_count"] = torch.zeros(
                    (self.num_envs, num_objects),
                    dtype=torch.long,
                    device=self.device,
                )
            if "track_clutter_duration_scale" not in state:
                state["track_clutter_duration_scale"] = torch.ones(
                    (self.num_envs,), dtype=torch.float32, device=self.device
                )
            if "track_clutter_protocol_grasp_object_index" not in state:
                state["track_clutter_protocol_grasp_object_index"] = torch.full(
                    (self.num_envs,), -1, dtype=torch.long, device=self.device
                )
            if "neighbor_protected_root_state" not in state or tuple(state["neighbor_protected_root_state"].shape[:2]) != (
                self.num_envs,
                num_objects,
            ):
                state["neighbor_protected_root_state"] = torch.zeros(
                    (self.num_envs, num_objects, 13),
                    dtype=torch.float32,
                    device=self.device,
                )
                state["neighbor_protected_root_state"][..., 3] = 1.0
            if "neighbor_protected_root_state_valid" not in state:
                state["neighbor_protected_root_state_valid"] = torch.zeros(
                    (self.num_envs,), dtype=torch.bool, device=self.device
                )
            state["track_name"] = self._protocol_track_name()
            state["control_mode"] = self._protocol_control_mode()
            state["order_axis"] = self._protocol_order_axis()
            state["gt_resolver"] = self._protocol_gt_resolver()
            state["query_family"] = str(self._task_protocol().get("query_family", ""))
            state["query_text"] = str(self._task_protocol().get("query_text", ""))
            state["intervention_script_name"] = self._state_reindexing_script_name()
            state["motion_mode"] = self._relocation_motion_mode_label(state["intervention_script_name"])
            state["target_rank"] = self._state_reindexing_target_rank()
            state["target_subset_object_ids"] = target_subset_object_ids
            state["target_subset_mask"] = target_subset_mask
            return state

        state = {
            "module_name": self._PROTOCOL_MODULE_NAME,
            "track_name": self._protocol_track_name(),
            "control_mode": self._protocol_control_mode(),
            "order_axis": self._protocol_order_axis(),
            "gt_resolver": self._protocol_gt_resolver(),
            "query_family": str(self._task_protocol().get("query_family", "")),
            "query_text": str(self._task_protocol().get("query_text", "")),
            "intervention_script_name": self._state_reindexing_script_name(),
            "motion_mode": self._relocation_motion_mode_label(self._state_reindexing_script_name()),
            "target_rank": self._state_reindexing_target_rank(),
            "object_names": tuple(object_names),
            "target_subset_object_ids": target_subset_object_ids,
            "target_subset_mask": target_subset_mask,
            "initial_xy": torch.zeros((self.num_envs, num_objects, 2), dtype=torch.float32, device=self.device),
            "neighbor_protected_root_state": torch.zeros(
                (self.num_envs, num_objects, 13), dtype=torch.float32, device=self.device
            ),
            "neighbor_protected_root_state_valid": torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            ),
            "post_xy": torch.zeros((self.num_envs, num_objects, 2), dtype=torch.float32, device=self.device),
            "initial_rank_indices": torch.full(
                (self.num_envs, num_objects), -1, dtype=torch.long, device=self.device
            ),
            "post_rank_indices": torch.full((self.num_envs, num_objects), -1, dtype=torch.long, device=self.device),
            "moved_object_indices": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
            "planned_moved_mask": torch.zeros((self.num_envs, num_objects), dtype=torch.bool, device=self.device),
            "gt_object_indices": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
            "grasped_object_indices": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
            "selected_object_indices": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
            "success": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "valid_episode": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "intervention_success": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "intervention_failure": torch.ones((self.num_envs,), dtype=torch.bool, device=self.device),
            "gt_resolution_failure": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "grasp_readout_failure": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "valid_episode_task_failed": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "max_non_target_planar_displacement": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            "max_distractor_planar_displacement": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            "accidental_displacement": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "post_intervention_settle_applied": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "dynamic_motion_object_indices": torch.full(
                (self.num_envs, motion_slot_count),
                -1,
                dtype=torch.long,
                device=self.device,
            ),
            "dynamic_motion_start_xy": torch.zeros(
                (self.num_envs, motion_slot_count, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            "dynamic_motion_prev_planar_xy": torch.zeros(
                (self.num_envs, motion_slot_count, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            "dynamic_motion_target_xy": torch.zeros(
                (self.num_envs, motion_slot_count, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            "dynamic_motion_fixed_quat": torch.zeros(
                (self.num_envs, motion_slot_count, 4),
                dtype=torch.float32,
                device=self.device,
            ),
            "dynamic_motion_fixed_z": torch.zeros(
                (self.num_envs, motion_slot_count),
                dtype=torch.float32,
                device=self.device,
            ),
            "dynamic_motion_cmd_planar_xy": torch.zeros(
                (self.num_envs, motion_slot_count, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            "dynamic_motion_elapsed_s": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            "dynamic_motion_duration_s": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            "dynamic_motion_delay_steps_remaining": torch.zeros((self.num_envs,), dtype=torch.long, device=self.device),
            "dynamic_swap_with_right": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "dynamic_swap_episode_counter": torch.zeros((self.num_envs,), dtype=torch.long, device=self.device),
            "dynamic_motion_enabled": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "dynamic_motion_completed": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "track_clutter_identity_slot_begin": torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            ),
            "track_clutter_kf_active": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            "track_clutter_kf_p": torch.full(
                (self.num_envs, num_objects, TRACK_CLUTTER_MAX_KEYFRAMES),
                -1.0,
                dtype=torch.float32,
                device=self.device,
            ),
            "track_clutter_kf_xy": torch.zeros(
                (self.num_envs, num_objects, TRACK_CLUTTER_MAX_KEYFRAMES, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            "track_clutter_kf_count": torch.zeros(
                (self.num_envs, num_objects),
                dtype=torch.long,
                device=self.device,
            ),
            "track_clutter_duration_scale": torch.ones((self.num_envs,), dtype=torch.float32, device=self.device),
            "track_clutter_protocol_grasp_object_index": torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            ),
        }
        state["dynamic_motion_fixed_quat"][:, :, 0] = 1.0
        state["neighbor_protected_root_state"][..., 3] = 1.0
        self._state_reindexing_runtime = state
        self.protocol_runtime = state
        return state

    def _current_object_xy(self, object_names: Sequence[str], env_ids: torch.Tensor) -> torch.Tensor:
        if len(object_names) == 0:
            return torch.zeros((env_ids.numel(), 0, 2), dtype=torch.float32, device=self.device)
        positions = [self.scene[str(object_name)].data.root_pos_w[env_ids, :2] for object_name in object_names]
        return torch.stack(positions, dim=1)

    def _object_uuid(self, env_id: int, object_index: int) -> str | None:
        state = self._ensure_state_reindexing_runtime()
        object_names = list(state["object_names"])
        if not (0 <= int(object_index) < len(object_names)):
            return None
        return f"scene{int(getattr(self.cfg, 'scene_id', 0))}:env{env_id}:{object_names[int(object_index)]}"

    def _rank_indices_to_uuids(self, env_ids: torch.Tensor, rank_indices: torch.Tensor) -> list[list[str | None]]:
        rows: list[list[str | None]] = []
        env_id_list = env_ids.detach().cpu().tolist()
        for row_idx, env_id in enumerate(env_id_list):
            rows.append(
                [
                    self._object_uuid(int(env_id), int(object_index))
                    for object_index in rank_indices[row_idx].tolist()
                    if int(object_index) >= 0
                ]
            )
        return rows

    def _object_indices_to_uuids(self, env_ids: torch.Tensor, object_indices: torch.Tensor) -> list[str | None]:
        env_id_list = env_ids.detach().cpu().tolist()
        return [
            self._object_uuid(int(env_id), int(object_index))
            for env_id, object_index in zip(env_id_list, object_indices.detach().cpu().tolist())
        ]

    def set_state_reindexing_selected_object_indices(
        self,
        selected_object_indices: int | Sequence[int] | torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        if not self._is_post_action_relocation_task():
            return

        state = self._ensure_state_reindexing_runtime()
        selected_env_ids = self._normalize_env_ids(env_ids)
        if selected_env_ids.numel() == 0:
            return

        indices = torch.as_tensor(selected_object_indices, device=self.device, dtype=torch.long)
        if indices.ndim == 0:
            indices = indices.repeat(selected_env_ids.numel())
        else:
            indices = indices.reshape(-1)

        if indices.numel() != selected_env_ids.numel():
            raise ValueError(
                "selected_object_indices size mismatch: "
                f"expected {selected_env_ids.numel()}, got {indices.numel()}"
            )

        state["selected_object_indices"][selected_env_ids] = indices
        self._update_state_reindexing_extras()

    def _xy_rows_by_object(self, xy_rows: torch.Tensor) -> list[dict[str, list[float]]]:
        state = self._ensure_state_reindexing_runtime()
        object_names = list(state["object_names"])
        rows: list[dict[str, list[float]]] = []
        for row in xy_rows.detach().cpu().tolist():
            rows.append(
                {
                    str(object_name): [float(coords[0]), float(coords[1])]
                    for object_name, coords in zip(object_names, row)
                }
            )
        return rows

    def _protocol_log_prefix(self, *, episode_level: bool = False) -> str:
        state = self._ensure_state_reindexing_runtime()
        prefix_root = "Episode_Protocol" if episode_level else "Protocol"
        return f"{prefix_root}/{state['track_name']}/"

    def _classify_protocol_failures(
        self,
        *,
        state: dict[str, Any],
        selected_env_ids: torch.Tensor,
        gt_object_indices: torch.Tensor,
        grasped_object_indices: torch.Tensor,
        selected_object_indices: torch.Tensor,
    ) -> tuple[list[bool], list[bool], list[bool], list[str]]:
        selection_matches_gt = (
            (selected_object_indices >= 0)
            & (gt_object_indices >= 0)
            & (selected_object_indices == gt_object_indices)
        )
        grasp_detected = grasped_object_indices >= 0
        grasped_matches_gt = (
            grasp_detected
            & (gt_object_indices >= 0)
            & (grasped_object_indices == gt_object_indices)
        )

        failure_categories: list[str] = []
        for row_index in range(selected_env_ids.numel()):
            intervention_success = bool(state["intervention_success"][selected_env_ids][row_index].item())
            valid_episode = bool(state["valid_episode"][selected_env_ids][row_index].item())
            success = bool(state["success"][selected_env_ids][row_index].item())
            selected_object_index = int(selected_object_indices[row_index].item())
            if selected_object_index < 0:
                failure_categories.append("pending_selection")
            elif not intervention_success:
                failure_categories.append("intervention_failed")
            elif not valid_episode:
                failure_categories.append("episode_invalid")
            elif not bool(selection_matches_gt[row_index].item()):
                failure_categories.append("wrong_selection")
            elif not bool(grasp_detected[row_index].item()):
                failure_categories.append("no_grasp")
            elif not success:
                failure_categories.append("grasped_wrong_object")
            else:
                failure_categories.append("success")

        return (
            selection_matches_gt.detach().cpu().tolist(),
            grasp_detected.detach().cpu().tolist(),
            grasped_matches_gt.detach().cpu().tolist(),
            failure_categories,
        )

    def _build_agent_protocol_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        control_mode: str | None = None,
    ) -> dict[str, Any]:
        resolved_control_mode = normalize_post_action_relocation_control_mode(
            self._protocol_control_mode() if control_mode is None else control_mode
        )
        agent_snapshot = {
            "env_ids": snapshot.get("env_ids", []),
            "module_name": snapshot.get("module_name"),
            "track_name": snapshot.get("track_name"),
            "order_axis": snapshot.get("order_axis"),
            "gt_resolver": snapshot.get("gt_resolver"),
            "query_family": snapshot.get("query_family"),
            "query_text": snapshot.get("query_text"),
            "motion_mode": snapshot.get("motion_mode"),
            "control_mode": resolved_control_mode,
            "object_names": snapshot.get("object_names", []),
            "target_subset_object_ids": snapshot.get("target_subset_object_ids", []),
            "post_rank_to_uuid": snapshot.get("post_rank_to_uuid", []),
            "post_intervention_rank_to_uuid": snapshot.get("post_intervention_rank_to_uuid", []),
            "post_xy": snapshot.get("post_xy", []),
            "post_intervention_xy_positions": snapshot.get("post_intervention_xy_positions", []),
        }
        if resolved_control_mode != FINAL_FRAME_ONLY_CONTROL_MODE:
            agent_snapshot.update(
                {
                    "intervention_script_name": snapshot.get("intervention_script_name"),
                    "target_rank": snapshot.get("target_rank"),
                    "initial_rank_to_uuid": snapshot.get("initial_rank_to_uuid", []),
                    "initial_xy": snapshot.get("initial_xy", []),
                    "initial_xy_positions": snapshot.get("initial_xy_positions", []),
                }
            )
        return agent_snapshot

    def get_agent_protocol_snapshot(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
        *,
        control_mode: str | None = None,
    ) -> dict[str, Any]:
        if not self._is_post_action_relocation_task():
            return {}
        return self._build_agent_protocol_snapshot(
            self.get_state_reindexing_snapshot(env_ids=env_ids),
            control_mode=control_mode,
        )

    def get_state_reindexing_snapshot(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> dict[str, Any]:
        if not self._is_post_action_relocation_task():
            return {}

        state = self._ensure_state_reindexing_runtime()
        selected_env_ids = self._normalize_env_ids(env_ids)
        gt_object_indices = state["gt_object_indices"][selected_env_ids]
        grasped_object_indices = state["grasped_object_indices"][selected_env_ids]
        selected_object_indices = state["selected_object_indices"][selected_env_ids]
        env_id_list = selected_env_ids.detach().cpu().tolist()
        initial_xy = self._xy_rows_by_object(state["initial_xy"][selected_env_ids])
        post_xy = self._xy_rows_by_object(state["post_xy"][selected_env_ids])
        initial_rank_to_uuid = self._rank_indices_to_uuids(
            selected_env_ids, state["initial_rank_indices"][selected_env_ids]
        )
        post_rank_to_uuid = self._rank_indices_to_uuids(
            selected_env_ids, state["post_rank_indices"][selected_env_ids]
        )
        gt_uuid = self._object_indices_to_uuids(selected_env_ids, gt_object_indices)
        grasped_uuid = self._object_indices_to_uuids(selected_env_ids, grasped_object_indices)
        selected_uuid = self._object_indices_to_uuids(selected_env_ids, selected_object_indices)
        intervention_execution_failure = (~state["intervention_success"][selected_env_ids]).detach().cpu().tolist()
        accidental_displacement = state["accidental_displacement"][selected_env_ids].detach().cpu().tolist()
        max_non_target_planar_displacement = state["max_non_target_planar_displacement"][
            selected_env_ids
        ].detach().cpu().tolist()
        max_distractor_planar_displacement = state["max_distractor_planar_displacement"][
            selected_env_ids
        ].detach().cpu().tolist()
        selection_matches_gt, grasp_detected, grasped_matches_gt, failure_category = self._classify_protocol_failures(
            state=state,
            selected_env_ids=selected_env_ids,
            gt_object_indices=gt_object_indices,
            grasped_object_indices=grasped_object_indices,
            selected_object_indices=selected_object_indices,
        )

        return {
            "env_ids": env_id_list,
            "public_task_contract": dict(self._task_protocol().get("public_task_contract", {})),
            "selection_resolver": self._task_protocol().get("gt_spec", {}).get("selection_resolver"),
            "module_name": state["module_name"],
            "track_name": state["track_name"],
            "control_mode": state["control_mode"],
            "order_axis": state["order_axis"],
            "gt_resolver": state["gt_resolver"],
            "query_family": state["query_family"],
            "query_text": state["query_text"],
            "intervention_script_name": state["intervention_script_name"],
            "motion_mode": state.get("motion_mode"),
            "target_rank": state["target_rank"],
            "task_seed": int(self._task_seed()),
            "protocol_displacement_tol": float(getattr(self.cfg, "protocol_displacement_tol", 0.02)),
            "protocol_min_reorder_gap": float(getattr(self.cfg, "protocol_min_reorder_gap", 0.08)),
            "post_intervention_settle_steps": int(getattr(self.cfg, "post_intervention_settle_steps", 0)),
            "dynamic_motion_enabled": state["dynamic_motion_enabled"][selected_env_ids].detach().cpu().tolist(),
            "dynamic_motion_completed": state["dynamic_motion_completed"][selected_env_ids].detach().cpu().tolist(),
            "dynamic_swap_with_right": state["dynamic_swap_with_right"][selected_env_ids].detach().cpu().tolist(),
            "dynamic_swap_episode_counter": state["dynamic_swap_episode_counter"][
                selected_env_ids
            ].detach().cpu().tolist(),
            "object_names": list(state["object_names"]),
            "target_subset_object_ids": list(state["target_subset_object_ids"]),
            "intervention_success": state["intervention_success"][selected_env_ids].detach().cpu().tolist(),
            "intervention_failure": state["intervention_failure"][selected_env_ids].detach().cpu().tolist(),
            "intervention_execution_failure": intervention_execution_failure,
            "gt_resolution_failure": state["gt_resolution_failure"][selected_env_ids].detach().cpu().tolist(),
            "grasp_readout_failure": state["grasp_readout_failure"][selected_env_ids].detach().cpu().tolist(),
            "valid_episode_task_failed": state["valid_episode_task_failed"][selected_env_ids].detach().cpu().tolist(),
            "success": state["success"][selected_env_ids].detach().cpu().tolist(),
            "valid_episode": state["valid_episode"][selected_env_ids].detach().cpu().tolist(),
            "post_intervention_settle_applied": state["post_intervention_settle_applied"][
                selected_env_ids
            ].detach().cpu().tolist(),
            "moved_object_indices": state["moved_object_indices"][selected_env_ids].detach().cpu().tolist(),
            "gt_object_indices": gt_object_indices.detach().cpu().tolist(),
            "grasped_object_indices": grasped_object_indices.detach().cpu().tolist(),
            "selected_object_indices": selected_object_indices.detach().cpu().tolist(),
            "gt_uuid": gt_uuid,
            "grasped_uuid": grasped_uuid,
            "selected_uuid": selected_uuid,
            "selection_matches_gt": selection_matches_gt,
            "grasp_detected": grasp_detected,
            "grasped_matches_gt": grasped_matches_gt,
            "failure_category": failure_category,
            "initial_rank_to_uuid": initial_rank_to_uuid,
            "post_rank_to_uuid": post_rank_to_uuid,
            "post_intervention_rank_to_uuid": post_rank_to_uuid,
            "initial_xy": initial_xy,
            "initial_xy_positions": initial_xy,
            "post_xy": post_xy,
            "post_intervention_xy_positions": post_xy,
            "accidental_displacement": accidental_displacement,
            "max_non_target_planar_displacement": max_non_target_planar_displacement,
            "max_distractor_planar_displacement": max_distractor_planar_displacement,
            "status_label": build_state_reindexing_status_labels(
                intervention_failure=state["intervention_failure"][selected_env_ids],
                gt_resolution_failure=state["gt_resolution_failure"][selected_env_ids],
                accidental_displacement=state["accidental_displacement"][selected_env_ids],
                grasp_readout_failure=state["grasp_readout_failure"][selected_env_ids],
                valid_episode_task_failed=state["valid_episode_task_failed"][selected_env_ids],
                success=state["success"][selected_env_ids],
            ),
            "confounds": {
                "accidental_displacement": accidental_displacement,
                "max_non_target_planar_displacement": max_non_target_planar_displacement,
                "max_distractor_planar_displacement": max_distractor_planar_displacement,
                "intervention_execution_failure": intervention_execution_failure,
            },
        }

    def _update_state_reindexing_extras(self) -> None:
        if not self._is_post_action_relocation_task():
            return

        self.extras["protocol"] = self.get_state_reindexing_snapshot()

    def _build_state_reindexing_log(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
        *,
        prefix: str | None = None,
    ) -> dict[str, float]:
        if not self._is_post_action_relocation_task():
            return {}

        selected_env_ids = self._normalize_env_ids(env_ids)
        if selected_env_ids.numel() == 0:
            return {}

        resolved_prefix = self._protocol_log_prefix() if prefix is None else prefix
        state = self._ensure_state_reindexing_runtime()
        return {
            f"{resolved_prefix}intervention_success": state["intervention_success"][selected_env_ids].float().mean().item(),
            f"{resolved_prefix}intervention_failure": state["intervention_failure"][selected_env_ids].float().mean().item(),
            f"{resolved_prefix}gt_resolution_failure": state["gt_resolution_failure"][selected_env_ids].float().mean().item(),
            f"{resolved_prefix}grasp_readout_failure": state["grasp_readout_failure"][selected_env_ids].float().mean().item(),
            f"{resolved_prefix}valid_episode_task_failed": state["valid_episode_task_failed"][selected_env_ids].float().mean().item(),
            f"{resolved_prefix}success": state["success"][selected_env_ids].float().mean().item(),
            f"{resolved_prefix}valid_episode": state["valid_episode"][selected_env_ids].float().mean().item(),
            f"{resolved_prefix}accidental_displacement": state["accidental_displacement"][selected_env_ids].float().mean().item(),
            f"{resolved_prefix}intervention_execution_failure": (
                ~state["intervention_success"][selected_env_ids]
            ).float().mean().item(),
            f"{resolved_prefix}max_non_target_planar_displacement": state["max_non_target_planar_displacement"][
                selected_env_ids
            ].mean().item(),
            f"{resolved_prefix}max_distractor_planar_displacement": state["max_distractor_planar_displacement"][
                selected_env_ids
            ].mean().item(),
        }

    def _apply_state_reindexing_intervention(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        state = self._ensure_state_reindexing_runtime()
        object_names = list(state["object_names"])
        num_objects = len(object_names)
        if num_objects == 0:
            return

        script_label = str(state.get("intervention_script_name", "") or "").strip() or self._state_reindexing_script_name()
        use_default_xy_as_reset = (
            self._is_historical_identity_tracking_task()
            and script_label == DEFAULT_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME
            and num_objects > 3
        )
        if use_default_xy_as_reset:
            default_xy_rows = []
            for object_name in object_names:
                asset = self.scene[str(object_name)]
                default_root_state = asset.data.default_root_state[env_ids].clone()
                default_root_state[:, 7:] = 0.0
                asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)
                default_xy_rows.append(default_root_state[:, :2])
            self.scene.write_data_to_sim()
            initial_xy = torch.stack(default_xy_rows, dim=1)
        else:
            initial_xy = self._current_object_xy(object_names, env_ids)
        state["initial_xy"][env_ids] = initial_xy
        state["neighbor_protected_root_state_valid"][env_ids] = False
        if self._is_reindex_neighbor_task():
            protected_root_state = torch.zeros(
                (env_ids.numel(), num_objects, 13),
                dtype=torch.float32,
                device=self.device,
            )
            protected_root_state[..., 3] = 1.0
            for object_index, object_name in enumerate(object_names):
                asset = self.scene[str(object_name)]
                root_state = getattr(getattr(asset, "data", None), "root_state_w", None)
                if root_state is None:
                    continue
                protected_root_state[:, object_index, :] = root_state[env_ids].to(
                    device=self.device,
                    dtype=torch.float32,
                )
            protected_root_state[..., 3:7] = 0.0
            protected_root_state[..., 3] = 1.0
            protected_root_state[..., 7:13] = 0.0
            state["neighbor_protected_root_state"][env_ids] = protected_root_state
            state["neighbor_protected_root_state_valid"][env_ids] = True
        state["post_xy"][env_ids] = initial_xy
        state["post_rank_indices"][env_ids] = -1
        state["moved_object_indices"][env_ids] = -1
        state["planned_moved_mask"][env_ids] = False
        state["gt_object_indices"][env_ids] = -1
        state["grasped_object_indices"][env_ids] = -1
        state["selected_object_indices"][env_ids] = -1
        state["success"][env_ids] = False
        state["intervention_success"][env_ids] = False
        state["intervention_failure"][env_ids] = True
        state["gt_resolution_failure"][env_ids] = False
        state["grasp_readout_failure"][env_ids] = False
        state["valid_episode_task_failed"][env_ids] = False
        state["valid_episode"][env_ids] = False
        state["max_non_target_planar_displacement"][env_ids] = 0.0
        state["max_distractor_planar_displacement"][env_ids] = 0.0
        state["accidental_displacement"][env_ids] = False
        state["post_intervention_settle_applied"][env_ids] = False
        state["dynamic_motion_object_indices"][env_ids] = -1
        state["dynamic_motion_start_xy"][env_ids] = 0.0
        state["dynamic_motion_prev_planar_xy"][env_ids] = 0.0
        state["dynamic_motion_target_xy"][env_ids] = 0.0
        state["dynamic_motion_fixed_quat"][env_ids] = 0.0
        state["dynamic_motion_fixed_quat"][env_ids, :, 0] = 1.0
        state["dynamic_motion_fixed_z"][env_ids] = 0.0
        state["dynamic_motion_elapsed_s"][env_ids] = 0.0
        state["dynamic_motion_duration_s"][env_ids] = 0.0
        state["dynamic_motion_cmd_planar_xy"][env_ids] = 0.0
        state["dynamic_motion_delay_steps_remaining"][env_ids] = 0
        state["dynamic_swap_with_right"][env_ids] = False
        state["dynamic_motion_enabled"][env_ids] = False
        state["dynamic_motion_completed"][env_ids] = False
        state["track_clutter_identity_slot_begin"][env_ids] = -1
        state["track_clutter_kf_active"][env_ids] = False
        state["track_clutter_duration_scale"][env_ids] = 1.0
        state["track_clutter_protocol_grasp_object_index"][env_ids] = -1

        if self._is_noop_intervention_script():
            # Explicit no-op protocol: keep all objects at their settled
            # initial positions and skip any scripted relocation while still
            # caching initial ranks for GT resolution.
            if self._is_historical_identity_tracking_task():
                plan = build_historical_identity_tracking_intervention_plan(
                    initial_xy,
                    min_reorder_gap=self._protocol_min_reorder_gap_for_intervention(),
                    script_name=state["intervention_script_name"],
                    order_axis=state["order_axis"],
                    active_object_mask=state["target_subset_mask"][env_ids],
                )
                state["initial_rank_indices"][env_ids] = plan["initial_rank_indices"]
                state["planned_moved_mask"][env_ids] = plan["planned_moved_mask"]
                state["moved_object_indices"][env_ids] = plan["moved_object_indices"]
            else:
                state["initial_rank_indices"][env_ids] = -1
            return

        if num_objects < 3:
            state["initial_rank_indices"][env_ids] = -1
            return

        active_object_mask = state["target_subset_mask"][env_ids]
        swap_episode_counters = state["dynamic_swap_episode_counter"][env_ids].clone()
        state["dynamic_swap_episode_counter"][env_ids] += 1
        swap_with_right_mask = torch.zeros((env_ids.numel(),), dtype=torch.bool, device=self.device)
        if self._uses_seeded_neighbor_dynamic_swap():
            swap_with_right_mask = self._seeded_dynamic_swap_with_right_mask(env_ids, swap_episode_counters)
        state["dynamic_swap_with_right"][env_ids] = swap_with_right_mask

        if self._is_historical_identity_tracking_task():
            plan = build_historical_identity_tracking_intervention_plan(
                initial_xy,
                min_reorder_gap=self._protocol_min_reorder_gap_for_intervention(),
                script_name=state["intervention_script_name"],
                order_axis=state["order_axis"],
                active_object_mask=active_object_mask,
                swap_with_right_mask=swap_with_right_mask,
            )
            state["initial_rank_indices"][env_ids] = plan["initial_rank_indices"]
            state["moved_object_indices"][env_ids] = plan["moved_object_indices"]
            state["planned_moved_mask"][env_ids] = plan["planned_moved_mask"]

            step_object_indices = plan["step_object_indices"].clone()
            step_target_xy = plan["step_target_xy"].clone()
            if (
                script_label == DEFAULT_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME
                and num_objects > 3
                and self._is_track_clutter_task()
            ):
                step_object_indices, step_target_xy = self._merge_track_clutter_motion_plan(
                    initial_xy=initial_xy,
                    step_object_indices=step_object_indices,
                    step_target_xy=step_target_xy,
                    active_object_mask=active_object_mask,
                )
                state["track_clutter_identity_slot_begin"][env_ids] = (~active_object_mask).sum(dim=1).long()
            if self._is_dynamic_historical_identity_tracking_script():
                if self._is_track_clutter_task():
                    final_xy_flat = flatten_final_xy_from_slots(
                        step_object_indices,
                        step_target_xy,
                        num_objects,
                    )
                    grasp_idx_tc = resolve_track_clutter_protocol_grasp_object_index(
                        initial_rank_indices=plan["initial_rank_indices"],
                        active_object_mask=active_object_mask,
                        target_rank=state["target_rank"],
                    )
                    state["track_clutter_protocol_grasp_object_index"][env_ids] = grasp_idx_tc
                    final_xy_flat = repel_active_identity_targets_xy(
                        final_xy_flat,
                        active_object_mask,
                        identity_initial_xy=initial_xy,
                        min_planar_sep=0.302,
                    )
                    final_xy_flat = repel_distractor_identity_pair_xy(
                        final_xy_flat,
                        active_object_mask,
                        grasp_idx_tc,
                        distractor_min_planar_sep=0.334,
                        identity_initial_xy=initial_xy,
                    )
                    for er in range(step_object_indices.shape[0]):
                        for sl in range(step_object_indices.shape[1]):
                            oid = int(step_object_indices[er, sl].item())
                            if oid >= 0:
                                step_target_xy[er, sl, :] = final_xy_flat[er, oid, :]
                    _raw_seed = getattr(self.cfg, "seed", 0)
                    base_seed = 0 if _raw_seed is None else int(_raw_seed)
                    _raw_scene = getattr(self.cfg, "scene_id", 0)
                    scene_id = 0 if _raw_scene is None else int(_raw_scene)
                    kf_p, kf_xy, kf_count, dur_scale = build_track_clutter_keyframes(
                        initial_xy=initial_xy,
                        final_xy=final_xy_flat,
                        device=self.device,
                        dtype=torch.float32,
                        base_seed=base_seed + 31 * scene_id,
                        active_object_mask=active_object_mask,
                        grasp_object_index=grasp_idx_tc,
                        distractor_pair_min_dist=0.198,
                        identity_pair_min_dist=0.162,
                    )
                    state["track_clutter_kf_p"][env_ids] = kf_p
                    state["track_clutter_kf_xy"][env_ids] = kf_xy
                    state["track_clutter_kf_count"][env_ids] = kf_count
                    state["track_clutter_duration_scale"][env_ids] = dur_scale
                    state["track_clutter_kf_active"][env_ids] = True
                padded_step_object_indices = torch.full_like(
                    state["dynamic_motion_object_indices"][env_ids],
                    -1,
                )
                padded_step_target_xy = torch.zeros_like(state["dynamic_motion_target_xy"][env_ids])
                padded_step_object_indices[:, : step_object_indices.shape[1]] = step_object_indices
                padded_step_target_xy[:, : step_target_xy.shape[1], :] = step_target_xy
                safe_step_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 2)
                start_xy = torch.gather(initial_xy, 1, safe_step_indices)
                valid_step_mask = (padded_step_object_indices >= 0).unsqueeze(-1)
                start_xy = torch.where(valid_step_mask, start_xy, torch.zeros_like(start_xy))
                object_quats = torch.stack(
                    [self.scene[str(object_name)].data.root_quat_w[env_ids] for object_name in object_names],
                    dim=1,
                )
                object_root_pos = torch.stack(
                    [self.scene[str(object_name)].data.root_pos_w[env_ids] for object_name in object_names],
                    dim=1,
                )
                safe_step_quat_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 4)
                fixed_quat = torch.gather(object_quats, 1, safe_step_quat_indices)
                safe_step_pos_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)
                fixed_root_pos = torch.gather(object_root_pos, 1, safe_step_pos_indices)
                fixed_z = fixed_root_pos[..., 2]
                identity_quat = torch.zeros_like(fixed_quat)
                identity_quat[..., 0] = 1.0
                fixed_quat = torch.where(valid_step_mask.expand(-1, -1, 4), fixed_quat, identity_quat)
                fixed_z = torch.where(valid_step_mask.squeeze(-1), fixed_z, torch.zeros_like(fixed_z))
                dynamic_motion_enabled = torch.any(padded_step_object_indices >= 0, dim=1)
                state["dynamic_motion_object_indices"][env_ids] = padded_step_object_indices
                state["dynamic_motion_start_xy"][env_ids] = start_xy
                state["dynamic_motion_prev_planar_xy"][env_ids] = start_xy.clone()
                state["dynamic_motion_target_xy"][env_ids] = padded_step_target_xy
                state["dynamic_motion_cmd_planar_xy"][env_ids] = torch.where(
                    valid_step_mask,
                    start_xy,
                    torch.zeros_like(state["dynamic_motion_cmd_planar_xy"][env_ids]),
                )
                state["dynamic_motion_fixed_quat"][env_ids] = fixed_quat
                state["dynamic_motion_fixed_z"][env_ids] = fixed_z
                state["dynamic_motion_elapsed_s"][env_ids] = 0.0
                base_duration = float(self._dynamic_historical_identity_motion_duration_s())
                if self._is_track_clutter_task():
                    num_active_slots = int(torch.max((padded_step_object_indices >= 0).sum(dim=1)).item())
                    if num_active_slots > 3:
                        base_duration *= 1.0 + 0.28 * float(num_active_slots - 3)
                    dur_tensor = torch.full((env_ids.numel(),), base_duration, dtype=torch.float32, device=self.device)
                    dur_tensor = dur_tensor * state["track_clutter_duration_scale"][env_ids]
                    state["dynamic_motion_duration_s"][env_ids] = torch.clamp(dur_tensor, min=0.55, max=1.35)
                else:
                    state["dynamic_motion_duration_s"][env_ids] = max(base_duration, 1e-3)
                state["dynamic_motion_delay_steps_remaining"][env_ids] = int(self._dynamic_swap_delay_phys_steps())
                state["dynamic_motion_enabled"][env_ids] = dynamic_motion_enabled
                state["dynamic_motion_completed"][env_ids] = ~dynamic_motion_enabled
                return
            num_steps = int(step_object_indices.shape[1]) if step_object_indices.ndim == 2 else 0
            for step_index in range(num_steps):
                current_step_object_indices = step_object_indices[:, step_index]
                current_step_target_xy = step_target_xy[:, step_index]
                for object_index, object_name in enumerate(object_names):
                    mask = current_step_object_indices == object_index
                    if not torch.any(mask):
                        continue
                    selected_env_ids = env_ids[mask]
                    asset = self.scene[object_name]
                    root_state = asset.data.root_state_w[selected_env_ids].clone()
                    root_state[:, 0] = current_step_target_xy[mask, 0]
                    root_state[:, 1] = current_step_target_xy[mask, 1]
                    root_state[:, 7:] = 0.0
                    asset.write_root_state_to_sim(root_state, env_ids=selected_env_ids)
            self.scene.write_data_to_sim()
            return

        plan = build_state_reindexing_intervention_plan(
            initial_xy,
            min_reorder_gap=float(getattr(self.cfg, "protocol_min_reorder_gap", 0.08)),
            script_name=state["intervention_script_name"],
            order_axis=state["order_axis"],
            active_object_mask=active_object_mask,
        )
        initial_rank_indices = plan["initial_rank_indices"]
        moved_object_indices = plan["moved_object_indices"]
        target_xy = plan["target_xy"]
        state["initial_rank_indices"][env_ids] = initial_rank_indices
        state["moved_object_indices"][env_ids] = moved_object_indices
        state["planned_moved_mask"][env_ids] = plan["planned_moved_mask"]

        per_object_target_xy = plan.get("per_object_target_xy")
        insert_script = str(state["intervention_script_name"] or "").strip()
        use_vla_visible_single_object_motion = (
            insert_script == "rightmost_to_between_first_and_second"
            and bool(self._protocol_intervention().get("vla_visible_motion", False))
            and not self._debug_rightmost_between_slide_enabled()
        )
        if use_vla_visible_single_object_motion:
            padded_step_object_indices = torch.full_like(
                state["dynamic_motion_object_indices"][env_ids],
                -1,
            )
            padded_step_target_xy = torch.zeros_like(state["dynamic_motion_target_xy"][env_ids])
            valid_rows = moved_object_indices >= 0
            if torch.any(valid_rows):
                padded_step_object_indices[valid_rows, 0] = moved_object_indices[valid_rows]
                padded_step_target_xy[valid_rows, 0, :] = target_xy[valid_rows]
            safe_step_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 2)
            start_xy = torch.gather(initial_xy, 1, safe_step_indices)
            valid_step_mask = (padded_step_object_indices >= 0).unsqueeze(-1)
            start_xy = torch.where(valid_step_mask, start_xy, torch.zeros_like(start_xy))
            object_quats = torch.stack(
                [self.scene[str(object_name)].data.root_quat_w[env_ids] for object_name in object_names],
                dim=1,
            )
            object_root_pos = torch.stack(
                [self.scene[str(object_name)].data.root_pos_w[env_ids] for object_name in object_names],
                dim=1,
            )
            safe_step_quat_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 4)
            fixed_quat = torch.gather(object_quats, 1, safe_step_quat_indices)
            safe_step_pos_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)
            fixed_root_pos = torch.gather(object_root_pos, 1, safe_step_pos_indices)
            fixed_z = fixed_root_pos[..., 2]
            identity_quat = torch.zeros_like(fixed_quat)
            identity_quat[..., 0] = 1.0
            fixed_quat = torch.where(valid_step_mask.expand(-1, -1, 4), fixed_quat, identity_quat)
            if self._is_reindex_neighbor_task():
                fixed_quat = torch.where(valid_step_mask.expand(-1, -1, 4), identity_quat, fixed_quat)
            fixed_z = torch.where(valid_step_mask.squeeze(-1), fixed_z, torch.zeros_like(fixed_z))
            state["dynamic_motion_object_indices"][env_ids] = padded_step_object_indices
            state["dynamic_motion_start_xy"][env_ids] = start_xy
            state["dynamic_motion_prev_planar_xy"][env_ids] = start_xy.clone()
            state["dynamic_motion_target_xy"][env_ids] = padded_step_target_xy
            state["dynamic_motion_cmd_planar_xy"][env_ids] = torch.where(
                valid_step_mask,
                start_xy,
                torch.zeros_like(state["dynamic_motion_cmd_planar_xy"][env_ids]),
            )
            state["dynamic_motion_fixed_quat"][env_ids] = fixed_quat
            state["dynamic_motion_fixed_z"][env_ids] = fixed_z
            state["dynamic_motion_elapsed_s"][env_ids] = 0.0
            state["dynamic_motion_duration_s"][env_ids] = float(
                self._dynamic_historical_identity_motion_duration_s()
            )
            state["dynamic_motion_delay_steps_remaining"][env_ids] = int(self._dynamic_swap_delay_phys_steps())
            dynamic_motion_enabled = torch.any(padded_step_object_indices >= 0, dim=1)
            state["dynamic_motion_enabled"][env_ids] = dynamic_motion_enabled
            state["dynamic_motion_completed"][env_ids] = ~dynamic_motion_enabled
            return
        use_insert_slide = insert_script == "rightmost_to_between_first_and_second" and self._debug_rightmost_between_slide_enabled()
        if use_insert_slide:
            padded_step_object_indices = torch.full_like(
                state["dynamic_motion_object_indices"][env_ids],
                -1,
            )
            padded_step_target_xy = torch.zeros_like(state["dynamic_motion_target_xy"][env_ids])
            if self._is_reindex_neighbor_task():
                target_xy_all = initial_xy.clone()
                neighbor_moved_mask = torch.zeros_like(state["planned_moved_mask"][env_ids])
                active_counts = active_object_mask.sum(dim=1)
                axis_index = 1 if str(state["order_axis"]).strip().lower() == "y" else 0
                other_axis_index = 1 - axis_index
                object_name_to_index = {str(name): idx for idx, name in enumerate(object_names)}
                protocol = self._task_protocol()
                gt_spec = protocol.get("gt_spec", {}) if isinstance(protocol.get("gt_spec", {}), dict) else {}
                target_object_name = str(
                    (protocol.get("target_object", {}) or {}).get("object_id", "")
                    or gt_spec.get("target_object_id", "")
                    or "right_neighbor_block"
                ).strip()
                red_anchor_index = object_name_to_index.get("red_anchor_block", -1)
                target_object_index = object_name_to_index.get(target_object_name, -1)
                protected_indices = {
                    idx
                    for idx in (
                        object_name_to_index.get("red_anchor_block", -1),
                        object_name_to_index.get("left_front_distractor_block", -1),
                    )
                    if idx >= 0
                }
                gap = max(0.092, min(float(getattr(self.cfg, "protocol_min_reorder_gap", 0.08)) * 1.18, 0.105))
                side_offset = -0.042
                valid_rows = (
                    (active_counts >= 3)
                    & torch.as_tensor(red_anchor_index >= 0, dtype=torch.bool, device=self.device)
                    & torch.as_tensor(target_object_index >= 0, dtype=torch.bool, device=self.device)
                    & torch.as_tensor(target_object_index not in protected_indices, dtype=torch.bool, device=self.device)
                )
                if torch.any(valid_rows):
                    for row in torch.nonzero(valid_rows, as_tuple=False).squeeze(-1).tolist():
                        row_rank = initial_rank_indices[row].detach().cpu().tolist()
                        active_order = [int(idx) for idx in row_rank[: int(active_counts[row].item())] if int(idx) >= 0]
                        if red_anchor_index not in active_order or target_object_index not in active_order:
                            continue
                        red_xy = initial_xy[row, red_anchor_index, :]
                        target_xy_all[row, target_object_index, :] = red_xy
                        target_xy_all[row, target_object_index, axis_index] = red_xy[axis_index] - gap
                        target_xy_all[row, target_object_index, other_axis_index] = red_xy[other_axis_index] + side_offset
                        neighbor_moved_mask[row, target_object_index] = True

                        left_blocker_candidates = [
                            idx
                            for idx in active_order
                            if idx not in protected_indices
                            and idx != target_object_index
                            and float(initial_xy[row, idx, axis_index].item()) < float(red_xy[axis_index].item())
                        ]
                        if left_blocker_candidates:
                            blocker_index = min(
                                left_blocker_candidates,
                                key=lambda idx: abs(float(initial_xy[row, idx, axis_index].item()) - float(red_xy[axis_index].item())),
                            )
                            target_xy_all[row, blocker_index, :] = red_xy
                            target_xy_all[row, blocker_index, axis_index] = red_xy[axis_index] - 2.0 * gap
                            target_xy_all[row, blocker_index, other_axis_index] = red_xy[other_axis_index] + side_offset - 0.032
                            neighbor_moved_mask[row, blocker_index] = True

                    state["planned_moved_mask"][env_ids] = neighbor_moved_mask
                    for row in range(int(neighbor_moved_mask.shape[0])):
                        moved_indices = torch.nonzero(neighbor_moved_mask[row], as_tuple=False).squeeze(-1)
                        if moved_indices.numel() == 0:
                            continue
                        n = min(int(moved_indices.numel()), int(padded_step_object_indices.shape[1]))
                        padded_step_object_indices[row, :n] = moved_indices[:n]
                        padded_step_target_xy[row, :n, :] = target_xy_all[row, moved_indices[:n], :]
            elif self._is_reindex_insert_task():
                valid_rows = moved_object_indices >= 0
                if torch.any(valid_rows):
                    padded_step_object_indices[valid_rows, 0] = moved_object_indices[valid_rows]
                    padded_step_target_xy[valid_rows, 0, :] = target_xy[valid_rows]
            else:
                # Subset control only animates the two swapped red blocks to reduce tabletop collisions.
                active_counts = active_object_mask.sum(dim=1)
                target_xy_all = initial_xy.clone()
                subset_moved_mask = torch.zeros_like(active_object_mask)
                object_name_to_index = {str(name): idx for idx, name in enumerate(object_names)}
                right_red_index = object_name_to_index.get("red_subset_block_2", -1)
                middle_red_index = object_name_to_index.get("red_subset_block_1", -1)
                valid_mask = active_counts >= 3
                if torch.any(valid_mask):
                    valid_rows = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1).tolist()
                    for row in valid_rows:
                        count = int(active_counts[row].item())
                        if count < 3 or right_red_index < 0 or middle_red_index < 0:
                            continue
                        # Swap the right and middle red cubes. Other red/distractor cubes stay pinned.
                        target_xy_all[row, right_red_index, :] = initial_xy[row, middle_red_index, :]
                        target_xy_all[row, middle_red_index, :] = initial_xy[row, right_red_index, :]
                        subset_moved_mask[row, right_red_index] = True
                        subset_moved_mask[row, middle_red_index] = True
                state["planned_moved_mask"][env_ids] = subset_moved_mask
                for row in range(int(subset_moved_mask.shape[0])):
                    moved_indices = torch.nonzero(subset_moved_mask[row], as_tuple=False).squeeze(-1)
                    if moved_indices.numel() == 0:
                        continue
                    n = min(int(moved_indices.numel()), int(padded_step_object_indices.shape[1]))
                    padded_step_object_indices[row, :n] = moved_indices[:n]
                    padded_step_target_xy[row, :n, :] = target_xy_all[row, moved_indices[:n], :]
            safe_step_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 2)
            start_xy = torch.gather(initial_xy, 1, safe_step_indices)
            valid_step_mask = (padded_step_object_indices >= 0).unsqueeze(-1)
            start_xy = torch.where(valid_step_mask, start_xy, torch.zeros_like(start_xy))
            object_quats = torch.stack(
                [self.scene[str(object_name)].data.root_quat_w[env_ids] for object_name in object_names],
                dim=1,
            )
            object_root_pos = torch.stack(
                [self.scene[str(object_name)].data.root_pos_w[env_ids] for object_name in object_names],
                dim=1,
            )
            safe_step_quat_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 4)
            fixed_quat = torch.gather(object_quats, 1, safe_step_quat_indices)
            safe_step_pos_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)
            fixed_root_pos = torch.gather(object_root_pos, 1, safe_step_pos_indices)
            fixed_z = fixed_root_pos[..., 2]
            identity_quat = torch.zeros_like(fixed_quat)
            identity_quat[..., 0] = 1.0
            fixed_quat = torch.where(valid_step_mask.expand(-1, -1, 4), fixed_quat, identity_quat)
            if self._is_reindex_neighbor_task():
                fixed_quat = torch.where(valid_step_mask.expand(-1, -1, 4), identity_quat, fixed_quat)
            fixed_z = torch.where(valid_step_mask.squeeze(-1), fixed_z, torch.zeros_like(fixed_z))
            dynamic_motion_enabled = torch.any(padded_step_object_indices >= 0, dim=1)
            state["dynamic_motion_object_indices"][env_ids] = padded_step_object_indices
            state["dynamic_motion_start_xy"][env_ids] = start_xy
            state["dynamic_motion_prev_planar_xy"][env_ids] = start_xy.clone()
            state["dynamic_motion_target_xy"][env_ids] = padded_step_target_xy
            state["dynamic_motion_cmd_planar_xy"][env_ids] = torch.where(
                valid_step_mask,
                start_xy,
                torch.zeros_like(state["dynamic_motion_cmd_planar_xy"][env_ids]),
            )
            state["dynamic_motion_fixed_quat"][env_ids] = fixed_quat
            state["dynamic_motion_fixed_z"][env_ids] = fixed_z
            state["dynamic_motion_elapsed_s"][env_ids] = 0.0
            base_duration = float(self._dynamic_historical_identity_motion_duration_s())
            if self._is_reindex_insert_task():
                state["dynamic_motion_duration_s"][env_ids] = max(base_duration * 0.45, 0.20)
            else:
                state["dynamic_motion_duration_s"][env_ids] = max(base_duration, 1e-3)
            delay_steps = int(self._reset_stabilization_steps()) * max(int(self.cfg.decimation), 1)
            delay_divisor = 8 if self._is_reindex_insert_task() else 3
            state["dynamic_motion_delay_steps_remaining"][env_ids] = max(1, delay_steps // delay_divisor)
            state["dynamic_motion_enabled"][env_ids] = dynamic_motion_enabled
            state["dynamic_motion_completed"][env_ids] = ~dynamic_motion_enabled
            return

        if self._is_dynamic_state_reindexing_script() and torch.is_tensor(per_object_target_xy):
            active_counts = active_object_mask.sum(dim=1)
            padded_step_object_indices = torch.full_like(
                state["dynamic_motion_object_indices"][env_ids],
                -1,
            )
            padded_step_target_xy = torch.zeros_like(state["dynamic_motion_target_xy"][env_ids])
            valid_mask = (active_counts == 4) & (int(initial_rank_indices.shape[1]) >= 4)
            if torch.any(valid_mask):
                ordered_indices = initial_rank_indices[valid_mask, :4]
                padded_step_object_indices[valid_mask, :4] = ordered_indices
                gather_indices = ordered_indices.unsqueeze(-1).expand(-1, -1, 2)
                padded_step_target_xy[valid_mask, :4, :] = torch.gather(
                    per_object_target_xy[valid_mask], 1, gather_indices
                )
            safe_step_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 2)
            start_xy = torch.gather(initial_xy, 1, safe_step_indices)
            valid_step_mask = (padded_step_object_indices >= 0).unsqueeze(-1)
            start_xy = torch.where(valid_step_mask, start_xy, torch.zeros_like(start_xy))
            object_quats = torch.stack(
                [self.scene[str(object_name)].data.root_quat_w[env_ids] for object_name in object_names],
                dim=1,
            )
            object_root_pos = torch.stack(
                [self.scene[str(object_name)].data.root_pos_w[env_ids] for object_name in object_names],
                dim=1,
            )
            safe_step_quat_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 4)
            fixed_quat = torch.gather(object_quats, 1, safe_step_quat_indices)
            safe_step_pos_indices = padded_step_object_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)
            fixed_root_pos = torch.gather(object_root_pos, 1, safe_step_pos_indices)
            fixed_z = fixed_root_pos[..., 2]
            identity_quat = torch.zeros_like(fixed_quat)
            identity_quat[..., 0] = 1.0
            fixed_quat = torch.where(valid_step_mask.expand(-1, -1, 4), fixed_quat, identity_quat)
            fixed_z = torch.where(valid_step_mask.squeeze(-1), fixed_z, torch.zeros_like(fixed_z))
            dynamic_motion_enabled = torch.any(padded_step_object_indices >= 0, dim=1)
            state["dynamic_motion_object_indices"][env_ids] = padded_step_object_indices
            state["dynamic_motion_start_xy"][env_ids] = start_xy
            state["dynamic_motion_prev_planar_xy"][env_ids] = start_xy.clone()
            state["dynamic_motion_target_xy"][env_ids] = padded_step_target_xy
            state["dynamic_motion_cmd_planar_xy"][env_ids] = torch.where(
                valid_step_mask,
                start_xy,
                torch.zeros_like(state["dynamic_motion_cmd_planar_xy"][env_ids]),
            )
            state["dynamic_motion_fixed_quat"][env_ids] = fixed_quat
            state["dynamic_motion_fixed_z"][env_ids] = fixed_z
            state["dynamic_motion_elapsed_s"][env_ids] = 0.0
            state["dynamic_motion_duration_s"][env_ids] = float(
                max(self._dynamic_historical_identity_motion_duration_s() * 0.55, 0.12)
            )
            state["dynamic_motion_delay_steps_remaining"][env_ids] = (
                int(self._reset_stabilization_steps()) * max(int(self.cfg.decimation), 1)
            )
            state["dynamic_motion_enabled"][env_ids] = dynamic_motion_enabled
            state["dynamic_motion_completed"][env_ids] = ~dynamic_motion_enabled
            return
        if torch.is_tensor(per_object_target_xy):
            for object_index, object_name in enumerate(object_names):
                asset = self.scene[object_name]
                root_state = asset.data.root_state_w[env_ids].clone()
                root_state[:, 0] = per_object_target_xy[:, object_index, 0]
                root_state[:, 1] = per_object_target_xy[:, object_index, 1]
                root_state[:, 7:] = 0.0
                asset.write_root_state_to_sim(root_state, env_ids=env_ids)
            self.scene.write_data_to_sim()
            return

        for object_index, object_name in enumerate(object_names):
            mask = moved_object_indices == object_index
            if not torch.any(mask):
                continue
            selected_env_ids = env_ids[mask]
            asset = self.scene[object_name]
            root_state = asset.data.root_state_w[selected_env_ids].clone()
            root_state[:, 0] = target_xy[mask, 0]
            root_state[:, 1] = target_xy[mask, 1]
            root_state[:, 7:] = 0.0
            asset.write_root_state_to_sim(root_state, env_ids=selected_env_ids)
        self.scene.write_data_to_sim()

    def _update_state_reindexing_dynamic_motion(self, dt: float) -> None:
        if not self._is_post_action_relocation_task():
            return

        state = self._ensure_state_reindexing_runtime()
        delay_steps_before = state["dynamic_motion_delay_steps_remaining"].clone()
        delayed_env_mask = (
            state["dynamic_motion_enabled"]
            & ~state["dynamic_motion_completed"]
            & (delay_steps_before > 0)
        )
        if torch.any(delayed_env_mask):
            state["dynamic_motion_delay_steps_remaining"][delayed_env_mask] = delay_steps_before[delayed_env_mask] - 1
        active_env_ids = torch.nonzero(
            state["dynamic_motion_enabled"] & (delay_steps_before <= 0),
            as_tuple=False,
        ).squeeze(-1)
        if active_env_ids.numel() == 0:
            return

        already_completed = state["dynamic_motion_completed"][active_env_ids]
        elapsed = state["dynamic_motion_elapsed_s"][active_env_ids]
        elapsed = torch.where(already_completed, elapsed, elapsed + float(dt))
        duration = torch.clamp(state["dynamic_motion_duration_s"][active_env_ids], min=1e-3)
        progress = torch.clamp(elapsed / duration, min=0.0, max=1.0)
        state["dynamic_motion_elapsed_s"][active_env_ids] = elapsed
        state["dynamic_motion_completed"][active_env_ids] = progress >= 1.0 - 1e-6
        completed_mask = state["dynamic_motion_completed"][active_env_ids]

        motion_object_indices = state["dynamic_motion_object_indices"][active_env_ids]
        start_xy = state["dynamic_motion_start_xy"][active_env_ids]
        target_xy = state["dynamic_motion_target_xy"][active_env_ids]
        fixed_quat = state["dynamic_motion_fixed_quat"][active_env_ids]
        fixed_z = state["dynamic_motion_fixed_z"][active_env_ids]
        eased_progress = 0.5 - 0.5 * torch.cos(torch.pi * progress)
        ep = eased_progress
        interp_linear = start_xy + (target_xy - start_xy) * ep.view(-1, 1, 1)
        interp_xy = interp_linear
        num_slots = int(motion_object_indices.shape[1])
        script_label = str(state.get("intervention_script_name", "") or "").strip() or self._state_reindexing_script_name()
        middle_direct = script_label == SEEDED_NEIGHBOR_SWAP_MIDDLE_DIRECT_OVER_2S_SCRIPT_NAME
        track_clutter_layout = self._is_track_clutter_task()
        portable_motion_mode = None if track_clutter_layout else self._portable_relocation_motion_mode(script_label)
        if portable_motion_mode is not None:
            motion_function = get_post_action_relocation_motion_function(portable_motion_mode)
            interp_xy = motion_function(
                start_xy,
                target_xy,
                motion_object_indices,
                progress,
                object_names=list(state["object_names"]),
            )
            if portable_motion_mode == REINDEX_NEIGHBOR_BEZIER_LANES:
                    protected_names = {"red_anchor_block", "left_front_distractor_block"}
                    object_names_for_motion = list(state["object_names"])
                    protected_indices = [
                        object_names_for_motion.index(name)
                        for name in protected_names
                        if name in object_names_for_motion
                    ]
                    protected_xy = state["initial_xy"][active_env_ids]
                    static_min_sep = 0.094
                    eps = 1.0e-6
                    for row in range(int(interp_xy.shape[0])):
                        for slot in range(num_slots):
                            moved_idx = int(motion_object_indices[row, slot].item())
                            if moved_idx < 0:
                                continue
                            for protected_idx in protected_indices:
                                if moved_idx == int(protected_idx):
                                    continue
                                delta = interp_xy[row, slot] - protected_xy[row, protected_idx]
                                dist = float(torch.linalg.norm(delta).item())
                                if dist >= static_min_sep:
                                    continue
                                if dist < eps:
                                    delta = interp_xy.new_tensor([-1.0, 0.0])
                                    dist = 1.0
                                direction = delta / max(dist, eps)
                                interp_xy[row, slot] = (
                                    protected_xy[row, protected_idx] + direction * float(static_min_sep)
                                )
        elif (
            track_clutter_layout
            and script_label == DEFAULT_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME
            and num_slots >= 1
        ):
            # Planned paths + clearance; cosine ease-in-out along time (same family as non-clutter scripts).
            kfa = state["track_clutter_kf_active"][active_env_ids]
            if torch.any(kfa):
                kf_p = state["track_clutter_kf_p"]
                kf_xy = state["track_clutter_kf_xy"]
                kf_cnt = state["track_clutter_kf_count"]
                env_list = active_env_ids.tolist()
                ep_list = ep.detach().tolist()
                kfa_list = kfa.tolist()
                for r, env_id in enumerate(env_list):
                    if not kfa_list[r]:
                        continue
                    u_path = float(ep_list[r])
                    for slot in range(num_slots):
                        oid = int(motion_object_indices[r, slot].item())
                        if oid < 0:
                            continue
                        interp_xy[r, slot, :] = interpolate_track_clutter_xy_row(
                            kf_p,
                            kf_xy,
                            kf_cnt,
                            int(env_id),
                            oid,
                            u_path,
                            device=self.device,
                            dtype=interp_xy.dtype,
                        )
        elif script_label == MULTI_STEP_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME and num_slots >= 3:
            # Overlapped staggered phases with strong lane separation:
            # keep >=3 simultaneous movers at mid-progress, but avoid direct overlap.
            lane_offsets_x = start_xy.new_tensor([0.28, 0.24, 0.20, 0.16])
            lane_offsets_y = start_xy.new_tensor([0.14, -0.14, 0.10, -0.10])
            phase_start = start_xy.new_tensor([0.20, 0.35, 0.00, 0.05])
            phase_end = start_xy.new_tensor([0.90, 1.00, 0.70, 0.75])
            x_floor = start_xy.new_tensor(0.18)
            y_min = start_xy.new_tensor(-0.30)
            y_max = start_xy.new_tensor(0.30)
            for slot in range(num_slots):
                slot_active = motion_object_indices[:, slot] >= 0
                if not torch.any(slot_active):
                    continue
                s = start_xy[:, slot, :]
                t = target_xy[:, slot, :]
                lane_idx = min(slot, int(lane_offsets_x.shape[0]) - 1)
                lane_dx = lane_offsets_x[lane_idx]
                lane_dy = lane_offsets_y[lane_idx]
                start_t = phase_start[lane_idx]
                end_t = phase_end[lane_idx]
                slot_ep = torch.clamp((ep - start_t) / torch.clamp(end_t - start_t, min=1e-4), min=0.0, max=1.0)
                w0 = torch.maximum(torch.minimum(s[:, 0], t[:, 0]) - lane_dx, x_floor)
                w1 = torch.clamp(0.5 * (s[:, 1] + t[:, 1]) + lane_dy, min=y_min, max=y_max)
                waypoint = torch.stack((w0, w1), dim=-1)
                routed_xy = self._piecewise_two_segment_via_waypoint(s, t, waypoint, slot_ep)
                interp_xy[:, slot, :] = torch.where(slot_active.unsqueeze(-1), routed_xy, interp_xy[:, slot, :])
        elif script_label == DEFAULT_HISTORICAL_IDENTITY_TRACKING_SCRIPT_NAME and num_slots >= 2 and (
            not track_clutter_layout
        ):
            # Historical default script (non-track_clutter): staggered lanes + waypoint routing.
            lane_count = max(num_slots, 4)
            lane_offsets_x = torch.linspace(0.18, 0.08, steps=lane_count, device=start_xy.device, dtype=start_xy.dtype)
            lane_offsets_y = torch.zeros((lane_count,), device=start_xy.device, dtype=start_xy.dtype)
            lane_offsets_y[0::2] = torch.linspace(
                0.10,
                0.03,
                steps=(lane_count + 1) // 2,
                device=start_xy.device,
                dtype=start_xy.dtype,
            )
            lane_offsets_y[1::2] = -torch.linspace(
                0.10,
                0.03,
                steps=lane_count // 2,
                device=start_xy.device,
                dtype=start_xy.dtype,
            )
            x_floor = start_xy.new_tensor(0.18)
            y_min = start_xy.new_tensor(-0.30)
            y_max = start_xy.new_tensor(0.30)
            max_slots = min(num_slots, lane_count)
            for slot in range(max_slots):
                slot_active = motion_object_indices[:, slot] >= 0
                if not torch.any(slot_active):
                    continue
                s = start_xy[:, slot, :]
                t = target_xy[:, slot, :]
                lane_dx = lane_offsets_x[slot]
                lane_dy = lane_offsets_y[slot]
                start_t = start_xy.new_tensor(0.06 * float(slot))
                end_t = torch.clamp(start_t + start_xy.new_tensor(0.46), max=1.0)
                slot_ep = torch.clamp((ep - start_t) / torch.clamp(end_t - start_t, min=1e-4), min=0.0, max=1.0)
                w0 = torch.maximum(torch.minimum(s[:, 0], t[:, 0]) - 0.65 * lane_dx, x_floor)
                w1 = torch.clamp(0.5 * (s[:, 1] + t[:, 1]) + lane_dy, min=y_min, max=y_max)
                waypoint = torch.stack((w0, w1), dim=-1)
                routed_xy = self._piecewise_two_segment_via_waypoint(s, t, waypoint, slot_ep)
                interp_xy[:, slot, :] = torch.where(slot_active.unsqueeze(-1), routed_xy, interp_xy[:, slot, :])
        elif num_slots >= 2 and not middle_direct:
            two_pi = start_xy.new_full((), 2.0 * float(torch.pi))
            for pair_start in range(0, num_slots - 1, 2):
                pair_end = pair_start + 1
                pair_index = pair_start // 2
                move_clockwise = (pair_index % 2) == 0
                pair_active = (motion_object_indices[:, pair_start] >= 0) & (motion_object_indices[:, pair_end] >= 0)
                if not torch.any(pair_active):
                    continue
                mid = 0.5 * (start_xy[:, pair_start, :] + start_xy[:, pair_end, :])
                radius = torch.linalg.norm(start_xy[:, pair_start, :] - mid, dim=-1)
                arc_ok = pair_active & (radius >= 1e-4)
                for slot in (pair_start, pair_end):
                    theta0 = torch.atan2(
                        start_xy[:, slot, 1] - mid[:, 1],
                        start_xy[:, slot, 0] - mid[:, 0],
                    )
                    theta1 = torch.atan2(
                        target_xy[:, slot, 1] - mid[:, 1],
                        target_xy[:, slot, 0] - mid[:, 0],
                    )
                    if move_clockwise:
                        delta_theta = (theta0 - theta1) % two_pi
                        delta_theta = torch.where(delta_theta < 1e-5, torch.full_like(delta_theta, two_pi), delta_theta)
                        theta_t = theta0 - ep * delta_theta
                    else:
                        delta_theta = (theta1 - theta0) % two_pi
                        delta_theta = torch.where(delta_theta < 1e-5, torch.full_like(delta_theta, two_pi), delta_theta)
                        theta_t = theta0 + ep * delta_theta
                    slot_arc_xy = torch.stack(
                        (
                            mid[:, 0] + radius * torch.cos(theta_t),
                            mid[:, 1] + radius * torch.sin(theta_t),
                        ),
                        dim=-1,
                    )
                    interp_xy[:, slot, :] = torch.where(
                        arc_ok.unsqueeze(-1),
                        slot_arc_xy,
                        interp_linear[:, slot, :],
                    )
        if middle_direct and portable_motion_mode is None and num_slots >= 2:
            # Seeded swap uses slot 0 = neighbor, slot 1 = initial-middle. Skip paired arcs; slot 0 stays linear.
            # Slot 1: default −X detour, or ``middle_motion: orbit_occluder`` (piecewise bulge always toward −X, capped by cx−r−clr).
            mid_slot_ok = motion_object_indices[:, 1] >= 0
            if torch.any(mid_slot_ok):
                routed = self._middle_direct_slot1_piecewise_xy(
                    start_xy[:, 1, :],
                    target_xy[:, 1, :],
                    ep,
                )
                interp_xy[:, 1, :] = torch.where(mid_slot_ok.unsqueeze(-1), routed, interp_xy[:, 1, :])
        if not self._is_track_clutter_task():
            ns = int(interp_xy.shape[1])
            slot_cmd_planar = torch.where(completed_mask.unsqueeze(-1), target_xy, interp_xy)
            state["dynamic_motion_cmd_planar_xy"][active_env_ids, :ns, :] = slot_cmd_planar.detach()
        grasped_object_indices = state["grasped_object_indices"][active_env_ids]
        object_names = list(state["object_names"])

        for slot_index in range(int(motion_object_indices.shape[1])):
            current_slot_indices = motion_object_indices[:, slot_index]
            slot_xy = torch.where(
                completed_mask.unsqueeze(-1),
                target_xy[:, slot_index],
                interp_xy[:, slot_index],
            )
            for object_index, object_name in enumerate(object_names):
                mask = (current_slot_indices == object_index) & (grasped_object_indices != object_index)
                if not torch.any(mask):
                    continue
                selected_env_ids = active_env_ids[mask]
                asset = self.scene[object_name]
                n_cmd = int(selected_env_ids.shape[0])
                root_pose = torch.zeros((n_cmd, 7), device=slot_xy.device, dtype=slot_xy.dtype)
                root_pose[:, 0] = slot_xy[mask, 0]
                root_pose[:, 1] = slot_xy[mask, 1]
                root_pose[:, 2] = fixed_z[mask, slot_index]
                floor_z_w = getattr(self.cfg, "protocol_dynamic_motion_floor_z_w", None)
                if floor_z_w is not None and track_clutter_layout:
                    try:
                        fz = float(floor_z_w)
                        root_pose[:, 2] = torch.maximum(
                            root_pose[:, 2],
                            torch.as_tensor(fz, device=root_pose.device, dtype=root_pose.dtype),
                        )
                    except (TypeError, ValueError):
                        pass
                root_pose[:, 3:7] = fixed_quat[mask, slot_index]
                root_pose[:, 3:7] = torch.nn.functional.normalize(root_pose[:, 3:7], dim=-1, eps=1e-8)

                if track_clutter_layout:
                    # Link pose + link velocity (single API when available). Avoids mixing link pose with CoM velocity,
                    # which otherwise yields spurious angular motion in PhysX.
                    cmd_xy = slot_xy[mask]
                    prev_xy = state["dynamic_motion_prev_planar_xy"][selected_env_ids, slot_index]
                    inv_dt = 1.0 / max(float(dt), 1e-6)
                    proto_grasp = state["track_clutter_protocol_grasp_object_index"][selected_env_ids]
                    is_distractor = (proto_grasp >= 0) & (proto_grasp != int(object_index))
                    v_scale = torch.where(
                        is_distractor,
                        root_pose.new_tensor(0.52),
                        root_pose.new_tensor(1.0),
                    )
                    lin_x = (cmd_xy[:, 0] - prev_xy[:, 0]) * inv_dt * v_scale
                    lin_y = (cmd_xy[:, 1] - prev_xy[:, 1]) * inv_dt * v_scale
                    if hasattr(asset.data, "root_link_state_w"):
                        root_link_state = asset.data.root_link_state_w[selected_env_ids].clone()
                        root_link_state[:, :7] = root_pose
                        root_link_state[:, 7] = lin_x
                        root_link_state[:, 8] = lin_y
                        root_link_state[:, 9] = 0.0
                        root_link_state[:, 10:13] = 0.0
                        asset.write_root_link_state_to_sim(root_link_state, env_ids=selected_env_ids)
                    else:
                        root_velocity_cmd = torch.zeros((n_cmd, 6), device=root_pose.device, dtype=root_pose.dtype)
                        root_velocity_cmd[:, 0] = lin_x
                        root_velocity_cmd[:, 1] = lin_y
                        asset.write_root_pose_to_sim(root_pose, env_ids=selected_env_ids)
                        asset.write_root_velocity_to_sim(root_velocity_cmd, env_ids=selected_env_ids)
                    state["dynamic_motion_prev_planar_xy"][selected_env_ids, slot_index] = cmd_xy.detach()
                else:
                    root_velocity = asset.data.root_state_w[selected_env_ids][:, 7:13].clone()
                    root_velocity.zero_()
                    asset.write_root_pose_to_sim(root_pose, env_ids=selected_env_ids)
                    asset.write_root_velocity_to_sim(root_velocity, env_ids=selected_env_ids)

    def _enforce_dynamic_motion_locked_orientation_after_physics(self) -> None:
        """After PhysX, re-apply the scripted rigid pose for dynamic slide (non-clutter).

        For non-``track_clutter`` tasks, PhysX can drift XY/Z and inject spin between our kinematic writes.
        We snap position (commanded planar XY + locked table Z), orientation, and zero root velocities each
        substep. Track-clutter keeps orientation-only correction because scripted link velocities drive XY.
        """
        if not self._is_post_action_relocation_task():
            return
        state = self._ensure_state_reindexing_runtime()
        lock_completed_rightmost_between_slide = (
            self._debug_rightmost_between_slide_enabled()
            and str(state.get("intervention_script_name", "") or "").strip() == "rightmost_to_between_first_and_second"
        )
        active_mask = state["dynamic_motion_enabled"] & (state["dynamic_motion_delay_steps_remaining"] <= 0)
        if not lock_completed_rightmost_between_slide:
            active_mask = active_mask & ~state["dynamic_motion_completed"]
        if not torch.any(active_mask):
            return
        active_env_ids = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        motion_object_indices = state["dynamic_motion_object_indices"][active_env_ids]
        fixed_quat = state["dynamic_motion_fixed_quat"][active_env_ids]
        grasped_object_indices = state["grasped_object_indices"][active_env_ids]
        object_names = list(state["object_names"])
        track_clutter = self._is_track_clutter_task()

        wrote_any = False
        for slot_index in range(int(motion_object_indices.shape[1])):
            current_slot_indices = motion_object_indices[:, slot_index]
            for object_index, object_name in enumerate(object_names):
                mask = (current_slot_indices == object_index) & (grasped_object_indices != object_index)
                if not torch.any(mask):
                    continue
                selected_env_ids = active_env_ids[mask]
                asset = self.scene[object_name]
                q = torch.nn.functional.normalize(fixed_quat[mask, slot_index], dim=-1, eps=1e-8)
                wrote_any = True

                if track_clutter:
                    if hasattr(asset.data, "root_link_state_w"):
                        rs = asset.data.root_link_state_w[selected_env_ids].clone()
                        rs[:, 3:7] = q
                        rs[:, 10:13] = 0.0
                        asset.write_root_link_state_to_sim(rs, env_ids=selected_env_ids)
                    elif hasattr(asset.data, "root_link_pose_w") and hasattr(asset.data, "root_link_vel_w"):
                        pose = asset.data.root_link_pose_w[selected_env_ids].clone()
                        pose[:, 3:7] = q
                        vel = asset.data.root_link_vel_w[selected_env_ids].clone()
                        vel[:, 3:6] = 0.0
                        asset.write_root_link_pose_to_sim(pose, env_ids=selected_env_ids)
                        asset.write_root_link_velocity_to_sim(vel, env_ids=selected_env_ids)
                    elif hasattr(asset.data, "root_state_w"):
                        rs = asset.data.root_state_w[selected_env_ids].clone()
                        rs[:, 3:7] = q
                        rs[:, 10:13] = 0.0
                        asset.write_root_state_to_sim(rs, env_ids=selected_env_ids)
                    else:
                        n_cmd = int(selected_env_ids.shape[0])
                        root_pose = torch.zeros((n_cmd, 7), device=q.device, dtype=q.dtype)
                        root_pose[:, :3] = asset.data.root_pos_w[selected_env_ids]
                        root_pose[:, 3:7] = q
                        root_velocity = torch.zeros((n_cmd, 6), device=q.device, dtype=q.dtype)
                        asset.write_root_pose_to_sim(root_pose, env_ids=selected_env_ids)
                        asset.write_root_velocity_to_sim(root_velocity, env_ids=selected_env_ids)
                else:
                    cmd = state["dynamic_motion_cmd_planar_xy"][selected_env_ids, slot_index]
                    fz = state["dynamic_motion_fixed_z"][selected_env_ids, slot_index]
                    if hasattr(asset.data, "root_link_state_w"):
                        rs = asset.data.root_link_state_w[selected_env_ids].clone()
                        rs[:, 0] = cmd[:, 0]
                        rs[:, 1] = cmd[:, 1]
                        rs[:, 2] = fz
                        rs[:, 3:7] = q
                        rs[:, 7:13] = 0.0
                        asset.write_root_link_state_to_sim(rs, env_ids=selected_env_ids)
                    elif hasattr(asset.data, "root_link_pose_w") and hasattr(asset.data, "root_link_vel_w"):
                        pose = asset.data.root_link_pose_w[selected_env_ids].clone()
                        pose[:, 0] = cmd[:, 0]
                        pose[:, 1] = cmd[:, 1]
                        pose[:, 2] = fz
                        pose[:, 3:7] = q
                        vel = torch.zeros_like(asset.data.root_link_vel_w[selected_env_ids])
                        asset.write_root_link_pose_to_sim(pose, env_ids=selected_env_ids)
                        asset.write_root_link_velocity_to_sim(vel, env_ids=selected_env_ids)
                    elif hasattr(asset.data, "root_state_w"):
                        rs = asset.data.root_state_w[selected_env_ids].clone()
                        rs[:, 0] = cmd[:, 0]
                        rs[:, 1] = cmd[:, 1]
                        rs[:, 2] = fz
                        rs[:, 3:7] = q
                        rs[:, 7:13] = 0.0
                        asset.write_root_state_to_sim(rs, env_ids=selected_env_ids)
                    else:
                        n_cmd = int(selected_env_ids.shape[0])
                        root_pose = torch.zeros((n_cmd, 7), device=q.device, dtype=q.dtype)
                        root_pose[:, 0] = cmd[:, 0]
                        root_pose[:, 1] = cmd[:, 1]
                        root_pose[:, 2] = fz
                        root_pose[:, 3:7] = q
                        root_velocity = torch.zeros((n_cmd, 6), device=q.device, dtype=q.dtype)
                        asset.write_root_pose_to_sim(root_pose, env_ids=selected_env_ids)
                        asset.write_root_velocity_to_sim(root_velocity, env_ids=selected_env_ids)

        if wrote_any:
            self.scene.write_data_to_sim()

    def _settle_state_reindexing_post_intervention(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        state = self._ensure_state_reindexing_runtime()
        if self._is_noop_intervention_script():
            state["post_intervention_settle_applied"][env_ids] = False
            return
        if self._uses_dynamic_intervention_motion():
            state["post_intervention_settle_applied"][env_ids] = False
            return
        settle_steps = int(getattr(self.cfg, "post_intervention_settle_steps", 0))
        if settle_steps <= 0:
            state["post_intervention_settle_applied"][env_ids] = False
            return

        zero_actions = torch.zeros(self.action_space.shape, device=self.device, dtype=torch.float32)
        self.action_manager.process_action(zero_actions)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(settle_steps):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        state["post_intervention_settle_applied"][env_ids] = True

    def _refresh_state_reindexing_terminal_state(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        state = self._ensure_state_reindexing_runtime()
        object_names = list(state["object_names"])
        num_objects = len(object_names)
        min_required_objects = 3
        if num_objects < min_required_objects:
            state["post_xy"][env_ids] = state["initial_xy"][env_ids]
            state["post_rank_indices"][env_ids] = -1
            state["gt_object_indices"][env_ids] = -1
            state["intervention_success"][env_ids] = False
            state["intervention_failure"][env_ids] = True
            state["gt_resolution_failure"][env_ids] = False
            state["valid_episode"][env_ids] = False
            state["max_non_target_planar_displacement"][env_ids] = 0.0
            state["max_distractor_planar_displacement"][env_ids] = 0.0
            state["accidental_displacement"][env_ids] = False
            return

        post_xy = self._current_object_xy(object_names, env_ids)
        state["post_xy"][env_ids] = post_xy
        active_object_mask = state["target_subset_mask"][env_ids]
        if (
            self._is_reindex_subset_task()
            and str(state.get("intervention_script_name", "") or "").strip() == "rightmost_to_between_first_and_second"
        ):
            initial_xy = state["initial_xy"][env_ids]
            initial_rank_indices = state["initial_rank_indices"][env_ids]
            active_counts = active_object_mask.sum(dim=1)
            name_to_index = {str(name): idx for idx, name in enumerate(object_names)}
            right_red_index = int(name_to_index.get("red_subset_block_2", -1))
            middle_red_index = int(name_to_index.get("red_subset_block_1", -1))
            swap_success = torch.zeros((env_ids.numel(),), dtype=torch.bool, device=self.device)
            if right_red_index >= 0 and middle_red_index >= 0:
                right_red_target = initial_xy[:, middle_red_index, :]
                middle_red_target = initial_xy[:, right_red_index, :]
                right_red_err = torch.linalg.vector_norm(post_xy[:, right_red_index, :] - right_red_target, dim=-1)
                middle_red_err = torch.linalg.vector_norm(
                    post_xy[:, middle_red_index, :] - middle_red_target,
                    dim=-1,
                )
                swap_tol = max(float(getattr(self.cfg, "protocol_displacement_tol", 0.02)) * 1.5, 0.035)
                swap_success = (right_red_err <= swap_tol) & (middle_red_err <= swap_tol)

            static_error = torch.linalg.vector_norm(post_xy - initial_xy, dim=-1)
            static_mask = torch.ones_like(active_object_mask)
            if right_red_index >= 0:
                static_mask[:, right_red_index] = False
            if middle_red_index >= 0:
                static_mask[:, middle_red_index] = False
            static_max_displacement = torch.where(
                static_mask,
                static_error,
                torch.zeros_like(static_error),
            ).amax(dim=1)

            target_index = int(name_to_index.get("red_subset_block_2", -1))
            gt_object_indices = torch.full((env_ids.numel(),), -1, dtype=torch.long, device=self.device)
            if target_index >= 0:
                gt_object_indices[:] = target_index
            static_success = static_max_displacement <= float(getattr(self.cfg, "protocol_displacement_tol", 0.02))
            intervention_success = (active_counts >= 3) & static_success & swap_success & (gt_object_indices >= 0)

            post_rank_indices = initial_rank_indices.clone()
            if right_red_index >= 0 and middle_red_index >= 0:
                for row_index in range(env_ids.numel()):
                    count = int(active_counts[row_index].item())
                    if count < 2:
                        continue
                    row_rank = post_rank_indices[row_index]
                    row_rank[:count] = initial_rank_indices[row_index, :count]
                    right_pos = torch.nonzero(row_rank[:count] == right_red_index, as_tuple=False).squeeze(-1)
                    middle_pos = torch.nonzero(row_rank[:count] == middle_red_index, as_tuple=False).squeeze(-1)
                    if right_pos.numel() > 0 and middle_pos.numel() > 0:
                        right_rank_pos = int(right_pos[0].item())
                        middle_rank_pos = int(middle_pos[0].item())
                        row_rank[right_rank_pos] = middle_red_index
                        row_rank[middle_rank_pos] = right_red_index
            state["post_rank_indices"][env_ids] = post_rank_indices
            state["gt_object_indices"][env_ids] = gt_object_indices
            state["intervention_success"][env_ids] = intervention_success
            state["intervention_failure"][env_ids] = ~intervention_success
            state["gt_resolution_failure"][env_ids] = gt_object_indices < 0
            state["valid_episode"][env_ids] = intervention_success
            state["max_non_target_planar_displacement"][env_ids] = static_max_displacement
            state["max_distractor_planar_displacement"][env_ids] = torch.zeros_like(static_max_displacement)
            state["accidental_displacement"][env_ids] = ~static_success
            return
        if self._is_historical_identity_tracking_task() and self._is_noop_intervention_script():
            initial_rank_indices = state["initial_rank_indices"][env_ids]
            post_rank_indices = initial_rank_indices.clone()
            gt_object_indices = torch.full((env_ids.numel(),), -1, dtype=torch.long, device=self.device)
            target_rank = str(state["target_rank"]).strip().lower()
            active_counts = active_object_mask.sum(dim=1)
            for row_index in range(env_ids.numel()):
                count = int(active_counts[row_index].item())
                rank_position = -1
                if target_rank in {"leftmost", "first", "frontmost"}:
                    rank_position = 0
                elif target_rank in {"rightmost", "last"} and count > 0:
                    rank_position = count - 1
                elif target_rank in {"middle", "center"} and count > 0 and count % 2 == 1:
                    rank_position = count // 2
                elif target_rank == "second_from_start" and count > 1:
                    rank_position = 1
                elif target_rank.startswith("index_"):
                    try:
                        parsed = int(target_rank.split("_", 1)[1])
                    except (TypeError, ValueError):
                        parsed = -1
                    if parsed >= 0 and parsed < count:
                        rank_position = parsed
                if 0 <= rank_position < int(post_rank_indices.shape[1]):
                    resolved_object_index = int(post_rank_indices[row_index, rank_position].item())
                    if resolved_object_index >= 0:
                        gt_object_indices[row_index] = resolved_object_index

            intervention_success = gt_object_indices >= 0
            state["post_rank_indices"][env_ids] = post_rank_indices
            state["gt_object_indices"][env_ids] = gt_object_indices
            state["intervention_success"][env_ids] = intervention_success
            state["intervention_failure"][env_ids] = ~intervention_success
            state["gt_resolution_failure"][env_ids] = ~intervention_success
            state["valid_episode"][env_ids] = intervention_success
            state["max_non_target_planar_displacement"][env_ids] = 0.0
            state["max_distractor_planar_displacement"][env_ids] = 0.0
            state["accidental_displacement"][env_ids] = False
            return

        if self._is_historical_identity_tracking_task():
            terminal_state = evaluate_historical_identity_tracking_terminal_state(
                initial_xy=state["initial_xy"][env_ids],
                post_xy=post_xy,
                initial_rank_indices=state["initial_rank_indices"][env_ids],
                planned_moved_mask=state["planned_moved_mask"][env_ids],
                displacement_tol=float(getattr(self.cfg, "protocol_displacement_tol", 0.02)),
                script_name=state["intervention_script_name"],
                target_rank=state["target_rank"],
                order_axis=state["order_axis"],
                active_object_mask=active_object_mask,
                swap_with_right_mask=state["dynamic_swap_with_right"][env_ids],
            )
        else:
            terminal_state = evaluate_state_reindexing_terminal_state(
                initial_xy=state["initial_xy"][env_ids],
                post_xy=post_xy,
                initial_rank_indices=state["initial_rank_indices"][env_ids],
                moved_object_indices=state["moved_object_indices"][env_ids],
                displacement_tol=float(getattr(self.cfg, "protocol_displacement_tol", 0.02)),
                script_name=state["intervention_script_name"],
                target_rank=state["target_rank"],
                order_axis=state["order_axis"],
                active_object_mask=active_object_mask,
                gt_spec=dict(self._task_protocol().get("gt_spec", {}) or {}),
                planned_moved_mask=state["planned_moved_mask"][env_ids],
                object_names=object_names,
            )
        state["post_rank_indices"][env_ids] = terminal_state["post_rank_indices"]
        state["gt_object_indices"][env_ids] = terminal_state["gt_object_indices"]
        state["intervention_success"][env_ids] = terminal_state["intervention_success"]
        state["intervention_failure"][env_ids] = terminal_state["intervention_failure"]
        state["gt_resolution_failure"][env_ids] = terminal_state["gt_resolution_failure"]
        state["valid_episode"][env_ids] = terminal_state["valid_episode"]
        state["max_non_target_planar_displacement"][env_ids] = terminal_state["max_non_target_planar_displacement"]
        state["max_distractor_planar_displacement"][env_ids] = terminal_state["max_distractor_planar_displacement"]
        state["accidental_displacement"][env_ids] = terminal_state["accidental_displacement"]

    def _update_state_reindexing_readout(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        if not self._is_post_action_relocation_task():
            return

        state = self._ensure_state_reindexing_runtime()
        object_names = list(state["object_names"])
        selected_env_ids = self._normalize_env_ids(env_ids)
        if selected_env_ids.numel() == 0 or len(object_names) == 0:
            return

        grasped_object_indices = torch.full((selected_env_ids.numel(),), -1, dtype=torch.long, device=self.device)
        for object_index, object_name in enumerate(object_names):
            grasped = geomind_mdp.object_grasped(
                self,
                robot_cfg=SceneEntityCfg("robot"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
                object_cfg=SceneEntityCfg(object_name),
            )[selected_env_ids]
            grasped_object_indices = torch.where(
                grasped & (grasped_object_indices < 0),
                torch.full_like(grasped_object_indices, object_index),
                grasped_object_indices,
            )

        state["grasped_object_indices"][selected_env_ids] = grasped_object_indices
        readout = evaluate_state_reindexing_readout(
            gt_object_indices=state["gt_object_indices"][selected_env_ids],
            grasped_object_indices=grasped_object_indices,
            valid_episode=state["valid_episode"][selected_env_ids],
        )
        state["success"][selected_env_ids] = readout["success"]
        state["grasp_readout_failure"][selected_env_ids] = readout["grasp_readout_failure"]
        state["valid_episode_task_failed"][selected_env_ids] = readout["valid_episode_task_failed"]

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)

        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        gripper_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, torch.Tensor] | None = None,
        env_id: int = 0,
        noise: Any = None,
    ) -> torch.Tensor:
        del noise
        eef_name = self._eef_name()

        target_pose = target_eef_pose_dict[eef_name].to(device=self.device, dtype=torch.float32)
        if target_pose.ndim == 3:
            target_pose = target_pose[0]

        cur_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose)
        cur_pos, cur_rot = PoseUtils.unmake_pose(cur_pose)

        action_dim = int(self.action_space.shape[-1]) if hasattr(self.action_space, "shape") else 7
        action = torch.zeros(action_dim, device=self.device, dtype=torch.float32)
        action[:3] = (target_pos - cur_pos).clamp(-1.0, 1.0)

        if action_dim >= 6:
            delta_rot_mat = target_rot.matmul(cur_rot.transpose(-1, -2))
            delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
            action[3:6] = PoseUtils.axis_angle_from_quat(delta_quat).clamp(-1.0, 1.0)
            if action_noise_dict is not None and eef_name in action_noise_dict:
                action[:6] += action_noise_dict[eef_name] * torch.randn_like(action[:6])
                action[:6] = action[:6].clamp(-1.0, 1.0)

        if action_dim >= 7:
            gripper_action = gripper_action_dict.get(eef_name, torch.tensor(0.0, device=self.device))
            action[-1] = float(torch.as_tensor(gripper_action, device=self.device, dtype=torch.float32).reshape(-1)[0])

        return action

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = self._eef_name()

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        cur_pose = self.get_robot_eef_pose(eef_name)
        cur_pos, cur_rot = PoseUtils.unmake_pose(cur_pose)

        target_pos = cur_pos + delta_position

        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, cur_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()
        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {self._eef_name(): actions[..., -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)

        signals = {}
        subtask_terms = dict(self.obs_buf.get("subtask_terms", {}))
        for name, value in subtask_terms.items():
            signals[str(name)] = value[env_ids]
        return signals

    def get_expected_attached_object(self, eef_name: str, subtask_index: int, env_cfg) -> str | None:
        del eef_name
        subtask_configs = list((getattr(env_cfg, "subtask_configs", {}) or {}).get(self._eef_name(), []))
        if not (0 <= int(subtask_index) < len(subtask_configs)):
            return None

        current_cfg = subtask_configs[int(subtask_index)]
        object_ref = getattr(current_cfg, "object_ref", None) or getattr(current_cfg, "object_name", None)
        if object_ref is None:
            return None

        signal_name = str(
            getattr(current_cfg, "subtask_start_signal", None) or getattr(current_cfg, "subtask_term_signal", "")
        ).lower()
        if signal_name.startswith("grasp"):
            return None
        return str(object_ref)

    @staticmethod
    def _truthy_protocol_flag(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _preloaded_gripper_target_object_id(self) -> str:
        protocol = self._task_protocol()
        target_object = protocol.get("target_object", {})
        if not isinstance(target_object, dict):
            return ""
        if str(target_object.get("carrier", "")).strip().lower() != "gripper":
            return ""
        if not self._truthy_protocol_flag(target_object.get("preloaded", False)):
            return ""

        object_id = str(target_object.get("object_id", "")).strip()
        if not object_id:
            return ""
        try:
            return object_id if object_id in self.scene.keys() else ""
        except Exception:
            return ""

    def _preloaded_gripper_joint_value(self, object_name: str) -> float:
        open_val = float(getattr(self.cfg, "gripper_open_val", 0.04))
        close_val = float(getattr(self.cfg, "gripper_close_val", 0.0))
        dims = None
        resolve_dims = getattr(self.cfg, "_resolve_object_runtime_dimensions", None)
        if callable(resolve_dims):
            try:
                dims = resolve_dims(object_name)
            except Exception:
                dims = None
        if dims is None:
            dims = self._preloaded_object_runtime_dimensions_from_task_spec(object_name)
        if dims is None:
            return close_val
        object_width = max(0.0, min(abs(float(dims[0])), abs(float(dims[1]))))
        return min(open_val, max(close_val, 0.5 * object_width))

    def _override_preloaded_gripper_action_close_command(self, object_name: str) -> None:
        action_manager = getattr(self, "action_manager", None)
        get_term = getattr(action_manager, "get_term", None)
        if not callable(get_term):
            return
        try:
            gripper_term = get_term("gripper_action")
        except Exception:
            return
        close_command = getattr(gripper_term, "_close_command", None)
        if close_command is None:
            return
        joint_value = self._preloaded_gripper_joint_value(object_name)
        try:
            close_command[:] = joint_value
        except Exception:
            return
        processed_actions = getattr(gripper_term, "_processed_actions", None)
        if processed_actions is not None:
            try:
                processed_actions[:] = joint_value
            except Exception:
                pass

    def _preloaded_object_runtime_dimensions_from_task_spec(self, object_name: str) -> tuple[float, float, float] | None:
        task_spec = getattr(self.cfg, "task_spec", {})
        if not isinstance(task_spec, dict):
            return None
        runtime_objects = task_spec.get("runtime_objects")
        if runtime_objects is None:
            annotation = task_spec.get("annotation", {})
            scene_adaptation = annotation.get("scene_adaptation", {}) if isinstance(annotation, dict) else {}
            runtime_objects = scene_adaptation.get("runtime_objects", []) if isinstance(scene_adaptation, dict) else []
        if not isinstance(runtime_objects, list):
            return None
        for runtime_object in runtime_objects:
            if not isinstance(runtime_object, dict):
                continue
            if str(runtime_object.get("name", "")).strip() != object_name:
                continue
            size = runtime_object.get("size") or runtime_object.get("dimensions")
            if isinstance(size, (list, tuple)) and len(size) >= 3:
                try:
                    return (abs(float(size[0])), abs(float(size[1])), abs(float(size[2])))
                except (TypeError, ValueError):
                    return None
            radius = runtime_object.get("radius")
            try:
                diameter = 2.0 * abs(float(radius))
            except (TypeError, ValueError):
                return None
            return (diameter, diameter, diameter)
        return None

    def _write_preloaded_gripper_joint_state(self, env_ids: torch.Tensor, object_name: str) -> bool:
        if env_ids.numel() == 0 or "robot" not in self.scene.keys() or not hasattr(self.cfg, "gripper_joint_names"):
            return False
        robot = self.scene["robot"]
        try:
            joint_ids, _ = robot.find_joints(self.cfg.gripper_joint_names)
        except Exception:
            return False
        if len(joint_ids) == 0:
            return False

        joint_value = self._preloaded_gripper_joint_value(object_name)
        try:
            joint_pos = robot.data.joint_pos[env_ids].clone()
            joint_vel = robot.data.joint_vel[env_ids].clone()
            joint_pos[:, joint_ids] = joint_value
            joint_vel[:, joint_ids] = 0.0
            if hasattr(robot, "write_joint_state_to_sim"):
                robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            else:
                robot.write_joint_position_to_sim(joint_pos[:, joint_ids], joint_ids=joint_ids, env_ids=env_ids)
                robot.write_joint_velocity_to_sim(joint_vel[:, joint_ids], joint_ids=joint_ids, env_ids=env_ids)
            return True
        except Exception:
            return False

    def _write_preloaded_target_object_state(self, env_ids: torch.Tensor, object_name: str) -> bool:
        if env_ids.numel() == 0 or object_name not in self.scene.keys() or "ee_frame" not in self.scene.keys():
            return False
        asset = self.scene[object_name]
        ee_frame = self.scene["ee_frame"]
        try:
            frame_pos_w = ee_frame.data.target_pos_w[env_ids]
            if frame_pos_w.shape[1] >= 3:
                target_pos_w = 0.5 * (frame_pos_w[:, 1, :] + frame_pos_w[:, 2, :])
            else:
                target_pos_w = frame_pos_w[:, 0, :].clone()
        except Exception:
            return False

        raw_offset = getattr(self.cfg, "preloaded_gripper_object_world_offset", (0.0, 0.0, 0.0))
        offset = torch.zeros(3, dtype=target_pos_w.dtype, device=target_pos_w.device)
        if isinstance(raw_offset, (list, tuple)):
            for index in range(min(3, len(raw_offset))):
                try:
                    offset[index] = float(raw_offset[index])
                except (TypeError, ValueError):
                    pass
        target_pos_w = target_pos_w + offset.unsqueeze(0)

        try:
            root_state = getattr(getattr(asset, "data", None), "root_state_w", None)
            if root_state is not None:
                state = root_state[env_ids].clone()
                state[:, :3] = target_pos_w
                state[:, 7:13] = 0.0
                asset.write_root_state_to_sim(state, env_ids=env_ids)
                return True

            root_pose = torch.zeros((env_ids.numel(), 7), dtype=target_pos_w.dtype, device=target_pos_w.device)
            root_pose[:, :3] = target_pos_w
            root_pose[:, 3] = 1.0
            root_velocity = torch.zeros((env_ids.numel(), 6), dtype=target_pos_w.dtype, device=target_pos_w.device)
            asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
            return True
        except Exception:
            return False

    def _apply_preloaded_gripper_target(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> None:
        object_name = self._preloaded_gripper_target_object_id()
        if not object_name:
            return
        selected_env_ids = self._normalize_env_ids(env_ids)
        if selected_env_ids.numel() == 0:
            return

        self._override_preloaded_gripper_action_close_command(object_name)
        wrote_gripper = self._write_preloaded_gripper_joint_state(selected_env_ids, object_name)
        wrote_object = self._write_preloaded_target_object_state(selected_env_ids, object_name)
        if wrote_gripper or wrote_object:
            self.scene.write_data_to_sim()
            self.scene.update(dt=float(getattr(self, "physics_dt", 0.0) or 0.0))
        if wrote_object:
            self.extras.setdefault("protocol", {})["preloaded_gripper_target_object_id"] = object_name

    def _lock_preloaded_target_to_gripper(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        object_name = self._preloaded_gripper_target_object_id()
        if not object_name:
            return
        selected_env_ids = self._normalize_env_ids(env_ids)
        if selected_env_ids.numel() == 0:
            return
        self._override_preloaded_gripper_action_close_command(object_name)
        wrote_gripper = self._write_preloaded_gripper_joint_state(selected_env_ids, object_name)
        wrote_object = self._write_preloaded_target_object_state(selected_env_ids, object_name)
        if wrote_gripper or wrote_object:
            self.scene.write_data_to_sim()

    def _force_preloaded_gripper_closed_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self._preloaded_gripper_target_object_id():
            return action
        if not isinstance(action, torch.Tensor) or action.numel() == 0:
            return action
        locked_action = action.clone()
        if locked_action.ndim == 1:
            locked_action[-1] = -1.0
        else:
            locked_action[..., -1] = -1.0
        return locked_action

    def _lock_reindex_neighbor_protected_objects(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        if not self._is_reindex_neighbor_task():
            return
        state = getattr(self, "_state_reindexing_runtime", None)
        if not isinstance(state, dict) or state.get("initial_xy") is None:
            return
        selected_env_ids = self._normalize_env_ids(env_ids)
        if selected_env_ids.numel() == 0:
            return
        valid_mask = state["neighbor_protected_root_state_valid"][selected_env_ids]
        if not torch.any(valid_mask):
            return
        valid_env_ids = selected_env_ids[valid_mask]
        object_names = list(state.get("object_names", ()))
        protected_names = ("red_anchor_block", "left_front_distractor_block")
        wrote_any = False
        for object_name in protected_names:
            if object_name not in object_names:
                continue
            object_index = object_names.index(object_name)
            if object_name not in self.scene.keys():
                continue
            asset = self.scene[object_name]
            root_state = getattr(getattr(asset, "data", None), "root_state_w", None)
            if root_state is None:
                continue
            state_cmd = state["neighbor_protected_root_state"][valid_env_ids, object_index, :].to(
                device=root_state.device,
                dtype=root_state.dtype,
            )
            state_cmd = state_cmd.clone()
            state_cmd[:, 3:7] = 0.0
            state_cmd[:, 3] = 1.0
            state_cmd[:, 7:13] = 0.0
            asset.write_root_state_to_sim(state_cmd, env_ids=valid_env_ids)
            if hasattr(asset.data, "root_link_state_w") and hasattr(asset, "write_root_link_state_to_sim"):
                link_state = asset.data.root_link_state_w[valid_env_ids].clone()
                link_state[:, :7] = state_cmd[:, :7]
                link_state[:, 7:13] = 0.0
                asset.write_root_link_state_to_sim(link_state, env_ids=valid_env_ids)
            elif hasattr(asset.data, "root_link_pose_w") and hasattr(asset.data, "root_link_vel_w"):
                link_pose = asset.data.root_link_pose_w[valid_env_ids].clone()
                link_pose[:, :7] = state_cmd[:, :7]
                link_vel = torch.zeros_like(asset.data.root_link_vel_w[valid_env_ids])
                asset.write_root_link_pose_to_sim(link_pose, env_ids=valid_env_ids)
                asset.write_root_link_velocity_to_sim(link_vel, env_ids=valid_env_ids)
            wrote_any = True
        if wrote_any:
            self.scene.write_data_to_sim()

    def _lock_reindex_neighbor_object_orientations(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        if not self._is_reindex_neighbor_task():
            return
        state = getattr(self, "_state_reindexing_runtime", None)
        if not isinstance(state, dict):
            return
        selected_env_ids = self._normalize_env_ids(env_ids)
        if selected_env_ids.numel() == 0:
            return
        object_names = list(state.get("object_names", ()))
        wrote_any = False
        for object_name in object_names:
            if object_name not in self.scene.keys():
                continue
            asset = self.scene[object_name]
            root_state = getattr(getattr(asset, "data", None), "root_state_w", None)
            if root_state is None:
                continue
            state_cmd = root_state[selected_env_ids].clone()
            state_cmd[:, 3:7] = 0.0
            state_cmd[:, 3] = 1.0
            state_cmd[:, 10:13] = 0.0
            asset.write_root_state_to_sim(state_cmd, env_ids=selected_env_ids)
            if hasattr(asset.data, "root_link_state_w") and hasattr(asset, "write_root_link_state_to_sim"):
                link_state = asset.data.root_link_state_w[selected_env_ids].clone()
                link_state[:, :7] = state_cmd[:, :7]
                link_state[:, 10:13] = 0.0
                asset.write_root_link_state_to_sim(link_state, env_ids=selected_env_ids)
            elif hasattr(asset.data, "root_link_pose_w") and hasattr(asset.data, "root_link_vel_w"):
                link_pose = asset.data.root_link_pose_w[selected_env_ids].clone()
                link_pose[:, 3:7] = state_cmd[:, 3:7]
                link_vel = asset.data.root_link_vel_w[selected_env_ids].clone()
                link_vel[:, 3:6] = 0.0
                asset.write_root_link_pose_to_sim(link_pose, env_ids=selected_env_ids)
                asset.write_root_link_velocity_to_sim(link_vel, env_ids=selected_env_ids)
            wrote_any = True
        if wrote_any:
            self.scene.write_data_to_sim()

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._apply_preloaded_gripper_target(env_ids)
        if not self._is_post_action_relocation_task():
            return

        selected_env_ids = self._normalize_env_ids(env_ids)
        self._apply_state_reindexing_intervention(selected_env_ids)
        self._lock_reindex_neighbor_protected_objects(selected_env_ids)
        self._lock_reindex_neighbor_object_orientations(selected_env_ids)
        self._settle_state_reindexing_post_intervention(selected_env_ids)
        self._lock_reindex_neighbor_protected_objects(selected_env_ids)
        self._lock_reindex_neighbor_object_orientations(selected_env_ids)
        self._refresh_state_reindexing_terminal_state(selected_env_ids)
        self._update_state_reindexing_readout(selected_env_ids)
        from .task_context_visuals import draw_task_context
        draw_task_context(self)
        self.extras.setdefault("log", {}).update(
            self._build_state_reindexing_log(selected_env_ids, prefix=self._protocol_log_prefix())
        )
        self._update_state_reindexing_extras()

    def step(self, action: torch.Tensor):
        action = self._force_preloaded_gripper_closed_action(action.to(self.device))
        if not self._is_post_action_relocation_task():
            if self._preloaded_gripper_target_object_id():
                self.action_manager.process_action(action)
                self.recorder_manager.record_pre_step()

                is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
                for _ in range(self.cfg.decimation):
                    self._sim_step_counter += 1
                    self.action_manager.apply_action()
                    self.scene.write_data_to_sim()
                    self.sim.step(render=False)
                    self.scene.update(dt=self.physics_dt)
                    self._lock_preloaded_target_to_gripper()
                    self.recorder_manager.record_post_physics_decimation_step()
                    if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                        self.sim.render()

                self.episode_length_buf += 1
                self.common_step_counter += 1
                self.reset_buf = self.termination_manager.compute()
                self.reset_terminated = self.termination_manager.terminated
                self.reset_time_outs = self.termination_manager.time_outs
                self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

                if len(self.recorder_manager.active_terms) > 0:
                    self.obs_buf = self.observation_manager.compute()
                    self.recorder_manager.record_post_step()

                reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_env_ids) > 0:
                    self.recorder_manager.record_pre_reset(reset_env_ids)
                    self._reset_idx(reset_env_ids)
                    if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                        for _ in range(self.cfg.num_rerenders_on_reset):
                            self.sim.render()
                    self.recorder_manager.record_post_reset(reset_env_ids)

                self.command_manager.compute(dt=self.step_dt)
                if "interval" in self.event_manager.available_modes:
                    self.event_manager.apply(mode="interval", dt=self.step_dt)
                self.obs_buf = self.observation_manager.compute(update_history=True)
                out = self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
            else:
                out = super().step(action)
            draw_directional_placement_success_zone_debug(self)
            return out

        self.action_manager.process_action(action)
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self._update_state_reindexing_dynamic_motion(self.physics_dt)
            self._lock_reindex_neighbor_protected_objects()
            self._lock_reindex_neighbor_object_orientations()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self._enforce_dynamic_motion_locked_orientation_after_physics()
            self._lock_reindex_neighbor_protected_objects()
            self._lock_reindex_neighbor_object_orientations()
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                from .task_context_visuals import draw_task_context
                draw_task_context(self)
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._lock_preloaded_target_to_gripper()
            self._lock_reindex_neighbor_protected_objects()
            self._lock_reindex_neighbor_object_orientations()

        state = self._ensure_state_reindexing_runtime()
        dynamic_env_ids = torch.nonzero(
            state["dynamic_motion_enabled"] | state["dynamic_motion_completed"],
            as_tuple=False,
        ).squeeze(-1)
        if dynamic_env_ids.numel() > 0:
            self._refresh_state_reindexing_terminal_state(dynamic_env_ids)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        self._update_state_reindexing_readout()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        terminal_protocol_log = self._build_state_reindexing_log(
            reset_env_ids,
            prefix=self._protocol_log_prefix(episode_level=True),
        )

        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        if terminal_protocol_log:
            self.extras.setdefault("log", {}).update(terminal_protocol_log)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        self._update_state_reindexing_readout()
        self.extras.setdefault("log", {}).update(
            self._build_state_reindexing_log(None, prefix=self._protocol_log_prefix())
        )
        self._update_state_reindexing_extras()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
