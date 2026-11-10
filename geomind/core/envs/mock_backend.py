from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ..types import EnvCfg


def _stable_token(*values: Any) -> int:
    merged = "|".join(str(value).strip() for value in values if str(value).strip())
    return sum(ord(ch) for ch in merged)


def _axis_index(name: str) -> int:
    return 0 if str(name).strip().lower() == "x" else 1


class MockBackendEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, cfg: EnvCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.scene_ref = dict(cfg.scene_ref or {})
        self.layout_ref = dict(cfg.layout_ref or {})
        self.annotation = dict(self.layout_ref.get("annotation", {}))
        self.protocol = dict(self.annotation.get("protocol", {}))
        self.scene_adaptation = dict(self.annotation.get("scene_adaptation", {}))
        self.runtime_objects = [
            dict(item)
            for item in self.scene_adaptation.get("runtime_objects", [])
            if isinstance(item, dict)
        ]
        self._objects_by_name = {
            str(item.get("name")).strip(): item for item in self.runtime_objects if str(item.get("name")).strip()
        }
        self.max_steps = int(cfg.max_episode_steps)
        self.step_count = 0
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "qpos": gym.spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32),
                "qvel": gym.spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32),
                "task_vector": gym.spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32),
                "cursor_xy": gym.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
                "goal_xy": gym.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
            }
        )
        self._scene_bias = self._compute_scene_bias()
        self._binding = self._build_task_binding()
        self._cursor_xy = np.zeros((2,), dtype=np.float32)
        self._cursor_delta = np.zeros((2,), dtype=np.float32)

    def _compute_scene_bias(self) -> float:
        scene_token = _stable_token(
            self.scene_ref.get("name", ""),
            self.scene_ref.get("usd_hint", ""),
            self.scene_ref.get("notes", ""),
        )
        return ((scene_token % 17) - 8) * 0.005

    def _xy_for(self, object_id: str) -> np.ndarray:
        payload = self._objects_by_name.get(str(object_id).strip(), {})
        xy = ((payload.get("pose") or {}).get("xy", [0.0, 0.0])) if isinstance(payload, dict) else [0.0, 0.0]
        if not isinstance(xy, list) or len(xy) < 2:
            xy = [0.0, 0.0]
        return np.asarray([float(xy[0]), float(xy[1])], dtype=np.float32)

    def _candidate_ids(self) -> list[str]:
        protocol_ids = self.protocol.get("candidate_object_ids", [])
        if isinstance(protocol_ids, list):
            normalized = [str(item).strip() for item in protocol_ids if str(item).strip()]
            if normalized:
                return normalized
        return [name for name in self._objects_by_name if name]

    def _preview_goal_radius(self) -> float:
        preview_goal = self.scene_adaptation.get("preview_proxy_goal", {})
        tolerance = (preview_goal or {}).get("tolerance", {})
        try:
            return float(tolerance.get("pos_r", 0.0))
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _build_anchor_reference_binding(self) -> dict[str, Any]:
        gt_spec = dict(self.protocol.get("gt_spec", {}))
        resolver = str(gt_spec.get("resolver", "")).strip().lower()
        anchor_object_id = str(
            gt_spec.get("anchor_object_id")
            or ((self.protocol.get("anchor_object") or {}).get("object_id", ""))
        ).strip()
        anchor_xy = self._xy_for(anchor_object_id)
        candidate_ids = self._candidate_ids()

        target_object_id = str(gt_spec.get("expected_target", "")).strip()
        if resolver == "directional_anchor_selection":
            axis_index = _axis_index(gt_spec.get("axis", "y"))
            secondary_index = 1 - axis_index
            direction_sign = -1.0 if str(gt_spec.get("direction", "positive")).strip().lower() == "negative" else 1.0
            min_primary_offset = float(gt_spec.get("min_primary_offset", 0.0) or 0.0)
            max_abs_secondary_offset = float(gt_spec.get("max_abs_secondary_offset", 10.0) or 10.0)
            eligible: list[tuple[float, str]] = []
            for candidate_id in candidate_ids:
                delta = self._xy_for(candidate_id) - anchor_xy
                signed_primary = float(delta[axis_index] * direction_sign)
                secondary = abs(float(delta[secondary_index]))
                if signed_primary >= min_primary_offset and secondary <= max_abs_secondary_offset:
                    eligible.append((signed_primary, candidate_id))
            if eligible:
                eligible.sort(reverse=True)
                target_object_id = eligible[0][1]
        elif resolver == "nearest_anchor_selection":
            axes = gt_spec.get("distance_axes", ["x", "y"])
            axis_indices = [_axis_index(axis_name) for axis_name in axes if str(axis_name).strip()]
            if not axis_indices:
                axis_indices = [0, 1]
            ranked: list[tuple[float, str]] = []
            for candidate_id in candidate_ids:
                delta = self._xy_for(candidate_id) - anchor_xy
                ranked.append((float(np.linalg.norm(delta[axis_indices])), candidate_id))
            ranked.sort(key=lambda item: item[0])
            if ranked:
                target_object_id = ranked[0][1]

        target_object_id = target_object_id or (candidate_ids[0] if candidate_ids else "")
        return {
            "goal_kind": "anchor_selection",
            "protocol_resolver": resolver,
            "anchor_object_id": anchor_object_id,
            "target_object_id": target_object_id,
            "goal_xy": self._xy_for(target_object_id),
            "start_xy": anchor_xy + np.asarray([-0.08, self._scene_bias], dtype=np.float32),
            "success_radius": 0.045,
            "candidate_count": len(candidate_ids),
            "layout_object_count": len(self.runtime_objects),
            "distractor_count": max(len(self.runtime_objects) - len(candidate_ids) - 1, 0),
            "intervention_script_name": "",
        }

    def _build_placement_binding(self) -> dict[str, Any]:
        gt_spec = dict(self.protocol.get("gt_spec", {}))
        target_zone = dict(gt_spec.get("target_zone", {}))
        target_object_id = str(
            gt_spec.get("target_object_id")
            or ((self.protocol.get("target_object") or {}).get("object_id", ""))
        ).strip()
        anchor_object_id = str(
            gt_spec.get("anchor_object_id")
            or ((self.protocol.get("anchor_object") or {}).get("object_id", ""))
        ).strip()
        anchor_xy = self._xy_for(anchor_object_id)
        center_offset = target_zone.get("center_offset_from_anchor", [0.0, 0.0])
        if not isinstance(center_offset, list) or len(center_offset) < 2:
            center_offset = [0.0, 0.0]
        goal_xy = anchor_xy + np.asarray([float(center_offset[0]), float(center_offset[1])], dtype=np.float32)
        half_extents = target_zone.get("half_extents", [0.05, 0.05])
        if not isinstance(half_extents, list) or len(half_extents) < 2:
            half_extents = [0.05, 0.05]
        preview_radius = self._preview_goal_radius()
        success_radius = preview_radius or float(min(abs(float(half_extents[0])), abs(float(half_extents[1]))))
        return {
            "goal_kind": "placement",
            "protocol_resolver": str(gt_spec.get("resolver", "")).strip().lower(),
            "anchor_object_id": anchor_object_id,
            "target_object_id": target_object_id,
            "goal_xy": goal_xy,
            "start_xy": self._xy_for(target_object_id) + np.asarray([self._scene_bias, 0.0], dtype=np.float32),
            "success_radius": max(success_radius, 0.03),
            "candidate_count": 1,
            "layout_object_count": len(self.runtime_objects),
            "distractor_count": max(len(self.runtime_objects) - 2, 0),
            "intervention_script_name": "",
        }

    def _build_move_to_object_binding(self) -> dict[str, Any]:
        gt_spec = dict(self.protocol.get("gt_spec", {}))
        target_object_id = str(
            gt_spec.get("target_object_id")
            or ((self.protocol.get("target_object") or {}).get("object_id", ""))
        ).strip()
        if not target_object_id and self._objects_by_name:
            target_object_id = next(iter(self._objects_by_name))
        success_radius = float(
            gt_spec.get("success_radius")
            or gt_spec.get("proximity_radius")
            or gt_spec.get("target_proximity_radius")
            or self._preview_goal_radius()
            or 0.085
        )
        goal_xy = self._xy_for(target_object_id)
        return {
            "goal_kind": "move_to",
            "protocol_resolver": str(gt_spec.get("resolver", "move_to_object_proximity")).strip().lower(),
            "anchor_object_id": "",
            "target_object_id": target_object_id,
            "goal_xy": goal_xy,
            "start_xy": goal_xy + np.asarray([-0.12, self._scene_bias], dtype=np.float32),
            "success_radius": max(success_radius, 0.01),
            "candidate_count": 1,
            "layout_object_count": len(self.runtime_objects),
            "distractor_count": max(len(self.runtime_objects) - 1, 0),
            "intervention_script_name": "",
        }

    def _build_square_completion_binding(self) -> dict[str, Any]:
        anchors = self.protocol.get("anchors", [])
        shared_anchor_id = ""
        adjacent_anchor_ids: list[str] = []
        if isinstance(anchors, list):
            for anchor in anchors:
                if not isinstance(anchor, dict):
                    continue
                role = str(anchor.get("role", "")).strip().lower()
                object_id = str(anchor.get("object_id", "")).strip()
                if role == "shared_corner":
                    shared_anchor_id = object_id
                elif role == "adjacent_corner" and object_id:
                    adjacent_anchor_ids.append(object_id)
        target_object_id = str(((self.protocol.get("target_object") or {}).get("object_id", ""))).strip()
        shared_xy = self._xy_for(shared_anchor_id)
        first_adjacent_xy = self._xy_for(adjacent_anchor_ids[0]) if adjacent_anchor_ids else shared_xy
        second_adjacent_xy = self._xy_for(adjacent_anchor_ids[1]) if len(adjacent_anchor_ids) > 1 else shared_xy
        goal_xy = first_adjacent_xy + second_adjacent_xy - shared_xy
        return {
            "goal_kind": "square_completion",
            "protocol_resolver": str(((self.protocol.get("gt_spec") or {}).get("rule_type", ""))).strip().lower(),
            "anchor_object_id": shared_anchor_id,
            "target_object_id": target_object_id,
            "goal_xy": goal_xy,
            "start_xy": self._xy_for(target_object_id) + np.asarray([self._scene_bias, -0.04], dtype=np.float32),
            "success_radius": float(self.protocol.get("success_tolerance_l2", 0.05) or 0.05),
            "candidate_count": len(adjacent_anchor_ids) + 1,
            "layout_object_count": len(self.runtime_objects),
            "distractor_count": max(len(self.runtime_objects) - len(adjacent_anchor_ids) - 2, 0),
            "intervention_script_name": "",
        }

    def _apply_intervention_positions(self, candidate_ids: list[str]) -> dict[str, np.ndarray]:
        positions = {candidate_id: self._xy_for(candidate_id).copy() for candidate_id in candidate_ids}
        gt_spec = dict(self.protocol.get("gt_spec", {}))
        axis_index = _axis_index(gt_spec.get("order_axis", "y"))
        ordered_ids = sorted(candidate_ids, key=lambda candidate_id: float(positions[candidate_id][axis_index]))
        intervention = dict(self.protocol.get("intervention", {}))
        script_name = str(intervention.get("script_name", "")).strip()
        if script_name == "swap_adjacent_pairs_over_4" and len(ordered_ids) >= 4:
            positions[ordered_ids[0]], positions[ordered_ids[1]] = positions[ordered_ids[1]], positions[ordered_ids[0]]
            positions[ordered_ids[2]], positions[ordered_ids[3]] = positions[ordered_ids[3]], positions[ordered_ids[2]]
        elif script_name == "swap_middle_with_seeded_neighbor_over_2s" and len(ordered_ids) >= 3:
            middle_index = len(ordered_ids) // 2
            neighbor_shift = 1 if _stable_token(self.layout_ref.get("layout_key", ""), self.scene_ref.get("name", "")) % 2 == 0 else -1
            neighbor_index = min(max(middle_index + neighbor_shift, 0), len(ordered_ids) - 1)
            if neighbor_index != middle_index:
                positions[ordered_ids[middle_index]], positions[ordered_ids[neighbor_index]] = (
                    positions[ordered_ids[neighbor_index]],
                    positions[ordered_ids[middle_index]],
                )
        return positions

    def _select_rank_target(self, ordered_ids: list[str], target_rank: str) -> str:
        normalized = str(target_rank).strip().lower()
        if not ordered_ids:
            return ""
        if normalized in {"leftmost", "frontmost", "lowest"}:
            return ordered_ids[0]
        if normalized in {"rightmost", "backmost", "highest"}:
            return ordered_ids[-1]
        if normalized == "middle":
            return ordered_ids[len(ordered_ids) // 2]
        return ordered_ids[0]

    def _build_rank_binding(self) -> dict[str, Any]:
        candidate_ids = self._candidate_ids()
        gt_spec = dict(self.protocol.get("gt_spec", {}))
        resolver = str(gt_spec.get("resolver", "")).strip().lower()
        axis_index = _axis_index(gt_spec.get("order_axis", "y"))
        initial_order = sorted(candidate_ids, key=lambda candidate_id: float(self._xy_for(candidate_id)[axis_index]))
        positions_after_intervention = self._apply_intervention_positions(candidate_ids)
        final_order = sorted(candidate_ids, key=lambda candidate_id: float(positions_after_intervention[candidate_id][axis_index]))
        target_rank = str(gt_spec.get("target_rank", "leftmost")).strip()

        if resolver == "initial_rank":
            target_object_id = self._select_rank_target(initial_order, target_rank)
        else:
            target_object_id = self._select_rank_target(final_order, target_rank)

        goal_xy = positions_after_intervention.get(target_object_id, self._xy_for(target_object_id))
        center_xy = np.mean(list(positions_after_intervention.values()), axis=0) if positions_after_intervention else np.zeros((2,))
        return {
            "goal_kind": "rank_selection",
            "protocol_resolver": resolver,
            "anchor_object_id": "",
            "target_object_id": target_object_id,
            "goal_xy": np.asarray(goal_xy, dtype=np.float32),
            "start_xy": np.asarray(center_xy, dtype=np.float32) + np.asarray([-0.08, self._scene_bias], dtype=np.float32),
            "success_radius": 0.04,
            "candidate_count": len(candidate_ids),
            "layout_object_count": len(self.runtime_objects),
            "distractor_count": max(len(self.runtime_objects) - len(candidate_ids), 0),
            "intervention_script_name": str(((self.protocol.get("intervention") or {}).get("script_name", ""))).strip(),
        }

    def _build_generic_binding(self) -> dict[str, Any]:
        object_ids = list(self._objects_by_name)
        mean_xy = (
            np.mean([self._xy_for(object_id) for object_id in object_ids], axis=0)
            if object_ids
            else np.asarray([0.3, 0.0], dtype=np.float32)
        )
        return {
            "goal_kind": "generic",
            "protocol_resolver": "",
            "anchor_object_id": "",
            "target_object_id": object_ids[0] if object_ids else "",
            "goal_xy": np.asarray(mean_xy, dtype=np.float32),
            "start_xy": np.asarray(mean_xy, dtype=np.float32) + np.asarray([-0.08, self._scene_bias], dtype=np.float32),
            "success_radius": 0.05,
            "candidate_count": len(object_ids),
            "layout_object_count": len(self.runtime_objects),
            "distractor_count": 0,
            "intervention_script_name": "",
        }

    def _build_task_binding(self) -> dict[str, Any]:
        gt_spec = dict(self.protocol.get("gt_spec", {}))
        resolver = str(gt_spec.get("resolver", "")).strip().lower()
        family = str(self.cfg.task.family).strip().lower()
        if resolver in {"move_to_object", "move_to_object_proximity", "object_proximity"}:
            binding = self._build_move_to_object_binding()
        elif family == "anchor_reference" or resolver in {"directional_anchor_selection", "nearest_anchor_selection"}:
            binding = self._build_anchor_reference_binding()
        elif family == "spatial_placement" or resolver == "directional_region_placement":
            binding = self._build_placement_binding()
        elif family == "shape_completion" or str(gt_spec.get("rule_type", "")).strip().lower() == "square_completion":
            binding = self._build_square_completion_binding()
        elif family in {"state_reindexing", "historical_identity_tracking"} or resolver in {"rank_after_intervention", "initial_rank"}:
            binding = self._build_rank_binding()
        else:
            binding = self._build_generic_binding()
        binding["scene_signature"] = str(self.scene_ref.get("name", ""))
        binding["layout_signature"] = str(self.layout_ref.get("layout_key", ""))
        binding["scene_bias"] = float(self._scene_bias)
        return binding

    def _goal_distance(self) -> float:
        goal_xy = np.asarray(self._binding["goal_xy"], dtype=np.float32)
        return float(np.linalg.norm(self._cursor_xy - goal_xy))

    def _obs(self) -> dict[str, Any]:
        goal_distance = self._goal_distance()
        progress = 1.0 - min(goal_distance / max(float(self._binding["success_radius"]) * 4.0, 1e-6), 1.0)
        qpos = np.asarray(
            [
                self._cursor_xy[0],
                self._cursor_xy[1],
                float(self._binding["goal_xy"][0]),
                float(self._binding["goal_xy"][1]),
                goal_distance,
                float(self._binding["success_radius"]),
                float(self.step_count) / max(self.max_steps, 1),
            ],
            dtype=np.float32,
        )
        qvel = np.asarray(
            [
                self._cursor_delta[0],
                self._cursor_delta[1],
                progress,
                float(self._binding["candidate_count"]),
                float(self._binding["distractor_count"]),
                float(self._binding["scene_bias"]),
                float(self.cfg.task.scene_id),
            ],
            dtype=np.float32,
        )
        task_vector = np.asarray(
            [
                float(_stable_token(self._binding["goal_kind"]) % 13),
                float(_stable_token(self.cfg.runtime_mode) % 13),
                float(self._binding["layout_object_count"]),
                float(self._binding["candidate_count"]),
                float(self._binding["distractor_count"]),
                float(_stable_token(self._binding["layout_signature"]) % 97) / 97.0,
                float(_stable_token(self._binding["scene_signature"]) % 97) / 97.0,
            ],
            dtype=np.float32,
        )
        return {
            "qpos": qpos,
            "qvel": qvel,
            "task_vector": task_vector,
            "cursor_xy": self._cursor_xy.copy(),
            "goal_xy": np.asarray(self._binding["goal_xy"], dtype=np.float32),
            "binding": {
                "goal_kind": self._binding["goal_kind"],
                "target_object_id": self._binding["target_object_id"],
                "layout_signature": self._binding["layout_signature"],
                "scene_signature": self._binding["scene_signature"],
            },
        }

    def _info(self, *, reward: float = 0.0) -> dict[str, Any]:
        return {
            "mock": True,
            "scene_id": self.cfg.task.scene_id,
            "scene_name": self.scene_ref.get("name", ""),
            "runtime_mode": self.cfg.runtime_mode,
            "layout_key": self.layout_ref.get("layout_key", ""),
            "goal_kind": self._binding["goal_kind"],
            "target_object_id": self._binding["target_object_id"],
            "anchor_object_id": self._binding["anchor_object_id"],
            "goal_xy": [float(item) for item in np.asarray(self._binding["goal_xy"], dtype=np.float32)],
            "cursor_xy": [float(item) for item in self._cursor_xy],
            "goal_distance": self._goal_distance(),
            "success_radius": float(self._binding["success_radius"]),
            "candidate_count": int(self._binding["candidate_count"]),
            "distractor_count": int(self._binding["distractor_count"]),
            "runtime_object_count": int(self._binding["layout_object_count"]),
            "protocol_resolver": self._binding["protocol_resolver"],
            "intervention_script_name": self._binding["intervention_script_name"],
            "scene_bias": float(self._scene_bias),
            "reward": float(reward),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self._cursor_delta = np.zeros((2,), dtype=np.float32)
        self._cursor_xy = np.asarray(self._binding["start_xy"], dtype=np.float32).copy()
        return self._obs(), self._info()

    def step(self, action):
        self.step_count += 1
        action_array = np.asarray(action, dtype=np.float32)
        delta = np.clip(action_array[:2], -1.0, 1.0) * 0.05
        prev_distance = self._goal_distance()
        self._cursor_delta = delta.astype(np.float32)
        self._cursor_xy = (self._cursor_xy + self._cursor_delta).astype(np.float32)
        goal_distance = self._goal_distance()
        progress = prev_distance - goal_distance
        reward = float((progress * 4.0) + max(float(self._binding["success_radius"]) - goal_distance, 0.0))
        reward -= float(self._binding["distractor_count"]) * 0.002
        terminated = goal_distance <= float(self._binding["success_radius"])
        truncated = self.step_count >= self.max_steps and not terminated
        return self._obs(), reward, bool(terminated), bool(truncated), self._info(reward=reward)
