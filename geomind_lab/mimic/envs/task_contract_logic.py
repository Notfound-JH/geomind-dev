"""GT selection for structural task contracts, independent of IsaacLab."""
from __future__ import annotations

from typing import Any, Sequence
import torch


def structural_adjacency_indices(post_xy: torch.Tensor, gt_spec: dict[str, Any],
                                 object_names: Sequence[str]) -> torch.Tensor:
    names = list(object_names)
    structure = gt_spec.get("structure", {})
    nodes = list(structure.get("nodes", []))
    result = torch.full((post_xy.shape[0],), -1, dtype=torch.long, device=post_xy.device)
    reference = str(gt_spec.get("reference_object_id", ""))
    if structure.get("type") != "DirectedChain" or reference not in nodes or any(node not in names for node in nodes):
        return result
    if gt_spec.get("relation") not in {"Predecessor", "Successor"}:
        return result
    indices = torch.tensor([names.index(node) for node in nodes], device=post_xy.device)
    axis = 0 if gt_spec.get("order_axis", "y") == "x" else 1
    coords = post_xy[:, indices, axis]
    other = post_xy[:, indices, 1 - axis]
    chain = torch.argsort(coords, dim=1)
    reference_local = nodes.index(reference)
    reference_rank = torch.argmax((chain == reference_local).long(), dim=1)
    offset = -1 if gt_spec["relation"] == "Predecessor" else 1
    target_rank = reference_rank + offset
    valid = (target_rank >= 0) & (target_rank < len(nodes))
    ordered = torch.gather(coords, 1, chain)
    valid &= torch.all(ordered[:, 1:] - ordered[:, :-1] > 1e-5, dim=1)
    valid &= other.amax(dim=1) - other.amin(dim=1) <= float(structure.get("max_perpendicular_spread", float("inf")))
    target_local = torch.gather(chain, 1, target_rank.clamp(0, len(nodes)-1).unsqueeze(1)).squeeze(1)
    return torch.where(valid, indices[target_local], result)




def protocol_gt_positions(env, protocol: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Resolve per-env physical targets from the validated runtime GT snapshot.

    Invalid/incomplete interventions return a false validity mask; callers must not
    fall back to a stale fixed target ID for these tasks.
    """
    if protocol.get("runtime_target_binding") != "protocol_gt": return None
    state = getattr(env, "_state_reindexing_runtime", None)
    positions = torch.zeros((env.num_envs, 3), device=env.device)
    valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not isinstance(state, dict) or not state.get("object_names"): return positions, valid
    names = list(state["object_names"])
    indices = state.get("gt_object_indices")
    if indices is None: return positions, valid
    valid = (indices >= 0) & (indices < len(names)) & state.get("valid_episode", valid)
    all_positions = torch.stack([env.scene[name].data.root_pos_w for name in names], dim=1)
    positions = torch.gather(all_positions, 1, indices.clamp(0, len(names)-1).view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    return positions, valid
