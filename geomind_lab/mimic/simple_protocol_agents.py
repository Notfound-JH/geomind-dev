from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any


@dataclass(frozen=True)
class StateReindexingAgentInput:
    """Minimal state-refresh agent input for choosing a target object."""

    observation: dict[str, Any]
    query_text: str
    protocol_snapshot: dict[str, Any]
    oracle_snapshot: dict[str, Any] | None = None
    control_mode: str = "full"


@dataclass(frozen=True)
class StateReindexingAgentOutput:
    """Minimal state-refresh agent output."""

    selected_object_index: int


class StateReindexingAgent:
    """Thin interface for state_reindexing target selection."""

    name = "state_reindexing_agent"

    def act(self, agent_input: StateReindexingAgentInput) -> StateReindexingAgentOutput:
        raise NotImplementedError

    def __call__(self, agent_input: StateReindexingAgentInput) -> StateReindexingAgentOutput:
        return self.act(agent_input)


class OracleAgent(StateReindexingAgent):
    name = "oracle"

    def act(self, agent_input: StateReindexingAgentInput) -> StateReindexingAgentOutput:
        oracle_snapshot = (
            agent_input.oracle_snapshot
            if isinstance(agent_input.oracle_snapshot, dict)
            else agent_input.protocol_snapshot
        )
        gt_object_indices = list(oracle_snapshot.get("gt_object_indices", []))
        selected_object_index = int(gt_object_indices[0]) if gt_object_indices else -1
        return StateReindexingAgentOutput(selected_object_index=selected_object_index)


class RandomAgent(StateReindexingAgent):
    name = "random"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, agent_input: StateReindexingAgentInput) -> StateReindexingAgentOutput:
        object_names = list(agent_input.protocol_snapshot.get("object_names", []))
        if not object_names:
            return StateReindexingAgentOutput(selected_object_index=-1)
        return StateReindexingAgentOutput(selected_object_index=self._rng.randrange(len(object_names)))


def build_state_reindexing_agent(agent_name: str, *, seed: int | None = None) -> StateReindexingAgent:
    normalized = str(agent_name).strip().lower()
    if normalized == OracleAgent.name:
        return OracleAgent()
    if normalized == RandomAgent.name:
        return RandomAgent(seed=seed)
    raise ValueError(f"Unsupported state_reindexing agent: {agent_name!r}")


__all__ = [
    "OracleAgent",
    "RandomAgent",
    "StateReindexingAgent",
    "StateReindexingAgentInput",
    "StateReindexingAgentOutput",
    "build_state_reindexing_agent",
]
