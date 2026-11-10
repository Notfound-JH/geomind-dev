from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from geomind_lab.mimic.envs.geometric_inference_logic import (
    build_workspace_bounds_xy_from_points,
    solve_explicit_rule_gt,
)


@dataclass(frozen=True)
class GeometricInferenceAgentInput:
    query_text: str
    anchor_positions_xy: dict[str, tuple[float, float]]
    protocol_snapshot: dict[str, Any]
    gt_pos_xy: tuple[float, float] | None = None
    gt_yaw_rad: float | None = None
    workspace_bounds_xy: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass(frozen=True)
class GeometricInferenceAgentOutput:
    target_pos_xy: tuple[float, float]
    target_yaw: float | None = None


class GeometricInferenceAgent:
    name = "geometric_inference_agent"

    def act(self, agent_input: GeometricInferenceAgentInput) -> GeometricInferenceAgentOutput:
        raise NotImplementedError

    def __call__(self, agent_input: GeometricInferenceAgentInput) -> GeometricInferenceAgentOutput:
        return self.act(agent_input)


class OraclePositionAgent(GeometricInferenceAgent):
    name = "oracle"

    def act(self, agent_input: GeometricInferenceAgentInput) -> GeometricInferenceAgentOutput:
        gt_pos_xy = agent_input.gt_pos_xy
        if gt_pos_xy is None:
            snapshot_gt_pos_xy = agent_input.protocol_snapshot.get("gt_pos_xy")
            if snapshot_gt_pos_xy is not None:
                gt_pos_xy = (float(snapshot_gt_pos_xy[0]), float(snapshot_gt_pos_xy[1]))
        if gt_pos_xy is None:
            rule_type = str(agent_input.protocol_snapshot.get("rule_type", ""))
            gt_pos_xy = solve_explicit_rule_gt(
                rule_type,
                agent_input.anchor_positions_xy["A"],
                agent_input.anchor_positions_xy["B"],
            )
        gt_yaw_rad = agent_input.gt_yaw_rad
        if gt_yaw_rad is None:
            snapshot_gt_yaw = agent_input.protocol_snapshot.get("gt_yaw_rad")
            if snapshot_gt_yaw is not None:
                gt_yaw_rad = float(snapshot_gt_yaw)
        return GeometricInferenceAgentOutput(
            target_pos_xy=(float(gt_pos_xy[0]), float(gt_pos_xy[1])),
            target_yaw=None if gt_yaw_rad is None else float(gt_yaw_rad),
        )


class OracleTrendAgent(OraclePositionAgent):
    name = "implicit_trend_oracle"


class ExplicitMidpointOracleAgent(OraclePositionAgent):
    name = "explicit_midpoint_oracle"


class ExplicitExtensionOracleAgent(OraclePositionAgent):
    name = "explicit_extension_oracle"


class RandomPositionAgent(GeometricInferenceAgent):
    name = "random"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, agent_input: GeometricInferenceAgentInput) -> GeometricInferenceAgentOutput:
        workspace_bounds_xy = agent_input.workspace_bounds_xy
        if workspace_bounds_xy is None:
            workspace_bounds_xy = build_workspace_bounds_xy_from_points(
                list(agent_input.anchor_positions_xy.values())
            )
        (min_x, min_y), (max_x, max_y) = workspace_bounds_xy
        return GeometricInferenceAgentOutput(
            target_pos_xy=(
                float(self._rng.uniform(float(min_x), float(max_x))),
                float(self._rng.uniform(float(min_y), float(max_y))),
            ),
            target_yaw=(
                float(self._rng.uniform(-3.141592653589793, 3.141592653589793))
                if (
                    agent_input.gt_yaw_rad is not None
                    or agent_input.protocol_snapshot.get("gt_yaw_rad") is not None
                )
                else None
            ),
        )


class ImplicitTrendRandomAgent(RandomPositionAgent):
    name = "implicit_trend_random"


class ExplicitRandomAgent(RandomPositionAgent):
    name = "explicit_random"


def build_geometric_inference_agent(
    agent_name: str,
    *,
    seed: int | None = None,
) -> GeometricInferenceAgent:
    normalized = str(agent_name).strip().lower()
    if normalized == OraclePositionAgent.name:
        return OraclePositionAgent()
    if normalized == OracleTrendAgent.name:
        return OracleTrendAgent()
    if normalized == ExplicitMidpointOracleAgent.name:
        return ExplicitMidpointOracleAgent()
    if normalized == ExplicitExtensionOracleAgent.name:
        return ExplicitExtensionOracleAgent()
    if normalized == RandomPositionAgent.name:
        return RandomPositionAgent(seed=seed)
    if normalized == ImplicitTrendRandomAgent.name:
        return ImplicitTrendRandomAgent(seed=seed)
    if normalized == ExplicitRandomAgent.name:
        return ExplicitRandomAgent(seed=seed)
    raise ValueError(f"Unsupported geometric inference agent: {agent_name!r}")


__all__ = [
    "GeometricInferenceAgent",
    "GeometricInferenceAgentInput",
    "GeometricInferenceAgentOutput",
    "OraclePositionAgent",
    "OracleTrendAgent",
    "ExplicitMidpointOracleAgent",
    "ExplicitExtensionOracleAgent",
    "RandomPositionAgent",
    "ImplicitTrendRandomAgent",
    "ExplicitRandomAgent",
    "build_geometric_inference_agent",
]
