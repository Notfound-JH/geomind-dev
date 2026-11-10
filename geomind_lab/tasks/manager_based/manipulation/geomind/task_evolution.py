"""Public task contracts applied to reusable legacy scene and motion code."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from typing import Any


DEFAULT_CONTRACTS = {
    "select_visual_spatial": {"task_id": "SelectWithVisualCue"},
    "ordinal_insert_static": {"task_id": "SelectByOrder", "parameters": {"index": 2, "from": "Left", "reference_time": "Current"}},
    "reindex_extreme": {"task_id": "SelectByOrder", "parameters": {"index": 1, "from": "Left", "reference_time": "Current"}},
    "reindex_insert": {"task_id": "UpdateObjectOrder", "parameters": {"index": 2, "from": "Left", "reference_time": "Current", "history_mode": "Deferred"}},
    "reindex_neighbor": {"task_id": "SelectByAdjacency", "parameters": {"relation": "Predecessor", "reference_object_id": "red_anchor_block", "structure": "DirectedChain"}},
    "place_inside": {"task_id": "PlaceByTopology", "variant": "Inside"},
    "place_on_top": {"task_id": "PlaceByTopology", "variant": "OnTop"},
}


def resolve_task_contract(canonical_task_id: str, explicit: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = deepcopy(DEFAULT_CONTRACTS.get(canonical_task_id, {}))
    selected = explicit
    if selected is None:
        raw = os.environ.get("GEOMIND_TASK_CONTRACT", "").strip()
        selected = json.loads(raw) if raw else {}
    if selected and selected.get("canonical_task_id", canonical_task_id) == canonical_task_id:
        parameters = {**contract.get("parameters", {}), **selected.get("parameters", {})}
        contract.update(deepcopy(selected))
        contract["parameters"] = parameters
    return contract


def _order_rank(parameters: dict[str, Any]) -> str:
    index = parameters.get("index", 1)
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ValueError("SelectByOrder index must be a positive one-based integer.")
    direction = str(parameters.get("from", "Left"))
    if direction == "Left": return f"index_{index - 1}"
    if direction == "Right" and index == 1: return "rightmost"
    if direction == "Right": return f"from_end_{index}"
    raise ValueError("Order direction must be Left or Right.")




def apply_task_contract(spec: dict[str, Any], canonical_task_id: str,
                        explicit: dict[str, Any] | None = None) -> dict[str, Any]:
    if canonical_task_id == "place_directional":
        raise ValueError("PlaceDirectional is deprecated and excluded from task registration/evaluation.")
    contract = resolve_task_contract(canonical_task_id, explicit)
    if not contract: return spec
    protocol = deepcopy(spec.get("protocol", {}))
    gt = dict(protocol.get("gt_spec", {}))
    parameters = dict(contract.get("parameters", {}))
    name = contract.get("task_id")
    is_vla = contract.get("interface") == "Vla" or str(gt.get("resolver", "")) in {"move_to_object_proximity", "move_to_object", "object_proximity"}
    verb = "Move the end effector near" if is_vla else "Pick up"
    resolver = None
    if name == "SelectByOrder":
        gt["target_rank"] = _order_rank(parameters)
        resolver = "rank_initial_layout" if protocol.get("intervention", {}).get("script_name") == "static_scene_noop" else "rank_after_intervention"
        instruction = f"{verb} block number {parameters.get('index', 1)} from the {parameters.get('from', 'Left').lower()} in the current order."
    elif name == "UpdateObjectOrder":
        if parameters.get("history_mode", "Deferred") != "Deferred":
            raise NotImplementedError("UpdateObjectOrder history semantics are deferred.")
        resolver = "rank_after_intervention"
        gt["target_rank"] = _order_rank(parameters)
        instruction = f"{verb} block number {parameters['index']} from the {parameters['from'].lower()} after the insertion."
    elif name == "SelectByAdjacency":
        resolver = "structural_adjacency"
        nodes = list(protocol.get("target_subset_spec", {}).get("object_ids", []))
        if not nodes: raise ValueError("SelectByAdjacency needs an explicit structural node set.")
        gt.update({"reference_object_id": parameters.get("reference_object_id", "red_anchor_block"),
                   "relation": parameters.get("relation", "Predecessor"),
                   "structure": {"type": "DirectedChain", "nodes": nodes, "edge_rule": "consecutive_rank", "draw_edges": True}})
        instruction = f"{verb} the node directly preceding the red node along the directed chain arrows."
    elif name == "TrackObjectOrigin":
        raise NotImplementedError("TrackObjectOrigin source-history semantics are deferred; legacy TrackClutter motion remains reusable.")
    else:
        instruction = spec.get("instruction", "")
    if resolver is not None:
        gt["selection_resolver"] = resolver
        if not is_vla: gt["resolver"] = resolver
        if is_vla:
            gt["resolver"] = "move_to_object_proximity"
            gt.pop("target_object_id", None)
            protocol.pop("target_object", None)
            protocol["runtime_target_binding"] = "protocol_gt"
            spec["goal_sequence"] = []
            spec.pop("preview_proxy_goal", None)
        protocol["gt_spec"] = gt
    spec["instruction"] = str(instruction)
    protocol["query_text"] = str(instruction)
    protocol["public_task_contract"] = deepcopy(contract)
    spec["protocol"] = protocol
    spec["public_task_contract"] = deepcopy(contract)
    return spec


def resolve_contract_target_id(spec: dict[str, Any]) -> str:
    """Predict an oracle target from the authored instance for direct ablations.

    Normal evaluation uses validated runtime GT; only the explicit direct
    ablation needs this pre-motion identity to recolor the correct object.
    """
    protocol = spec.get("protocol", {})
    gt = protocol.get("gt_spec", {})
    resolver = gt.get("selection_resolver", gt.get("resolver"))
    positions = spec.get("object_positions", {})
    names = list(positions)
    if not names: return ""
    import torch
    from geomind_lab.mimic.envs.state_reindexing_logic import build_state_reindexing_intervention_plan, _resolve_target_rank_position
    from geomind_lab.mimic.envs.task_contract_logic import structural_adjacency_indices
    initial = torch.tensor([[positions[name]["position"][:2] for name in names]], dtype=torch.float32)
    active_names = protocol.get("target_subset_spec", {}).get("object_ids", names)
    active = torch.tensor([[name in active_names for name in names]], dtype=torch.bool)
    axis = 0 if gt.get("order_axis", "y") == "x" else 1
    post = initial.clone()
    if resolver != "historical_order_after_insertion":
        plan = build_state_reindexing_intervention_plan(initial, min_reorder_gap=0.08,
            script_name=protocol.get("intervention", {}).get("script_name", "static_scene_noop"),
            order_axis=gt.get("order_axis", "y"), active_object_mask=active)
        if "per_object_target_xy" in plan:
            post = plan["per_object_target_xy"]
        elif int(plan["moved_object_indices"][0]) >= 0:
            post[0, int(plan["moved_object_indices"][0])] = plan["target_xy"][0]
    if resolver == "structural_adjacency":
        index = int(structural_adjacency_indices(post, gt, names)[0])
        return names[index] if index >= 0 else ""
    rank, supported = _resolve_target_rank_position(gt.get("target_rank", "index_0"), active.sum(dim=1))
    if not bool(supported[0]): return ""
    candidates = [index for index, name in enumerate(names) if name in active_names]
    candidates.sort(key=lambda index: float(post[0,index,axis]))
    return names[candidates[int(rank[0])]]
