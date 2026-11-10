"""Independent v3 configuration and semantic GT API.

Simulation backends must provide measured snapshots; layout poses are not GT.
"""

from .resolver import ContractError, EpisodeResolver, ResolvedEpisode, Snapshot, solve
from .runtime import SemanticEpisode, create_episode_env, evaluate_answer

__all__ = ["ContractError", "EpisodeResolver", "ResolvedEpisode", "Snapshot", "solve",
           "SemanticEpisode", "create_episode_env", "evaluate_answer"]
