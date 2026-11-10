"""Explicit render visibility for assembled rooms; never alter collision state."""
from __future__ import annotations


def apply_scene_visibility(stage, manifest):
    from pxr import UsdGeom

    policy = manifest.get("visualization", {})
    if not isinstance(policy, dict) or set(policy) - {"walls"}:
        raise ValueError("visualization must contain only the explicit walls policy")
    walls = policy.get("walls", "hidden")
    if walls not in {"hidden", "visible"}:
        raise ValueError("visualization.walls must be hidden or visible")
    hidden, visible = [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Imageable):
            continue
        name = prim.GetName().lower()
        if name.startswith("ceiling") or (name.startswith("wall") and walls == "hidden"):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(str(prim.GetPath()))
        elif name.startswith("wall"):
            UsdGeom.Imageable(prim).MakeVisible()
            visible.append(str(prim.GetPath()))
    return {"walls_policy": walls, "hidden_prims": hidden, "visible_wall_prims": visible,
            "scope": "render visibility only; collision, material and transforms unchanged"}
