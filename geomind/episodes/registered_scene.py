"""Reuse registered GeoMind scene assets and lighting for native episodes."""
from geomind_lab.utils.portable_paths import configure_material_search_path as _portable_material_paths
from pathlib import Path


def configure_material_search_path() -> None:
    """Match existing debug launchers before Kit initializes the MDL compiler."""
    _portable_material_paths()


def scene_description(scene_id: int) -> dict:
    """Read table height with the same asset metadata used by legacy tasks (no simulator)."""
    from geomind_lab.utils import asset_utils, scene_utils
    from .physical_contract import physics_profile
    root = asset_utils.resolve_assets_root()
    path = root / f"stage/{scene_id}/{scene_id}_new.usda"
    if not path.is_file():
        path = root / f"stage/{scene_id}/{scene_id}.usda"
    tree = scene_utils.extract_xform_tree(scene_utils.parse_usda_scene(path))
    def tables(nodes):
        for node in nodes:
            if node["name"].startswith("table_"):
                yield node
            yield from tables(node.get("children", []))
    table, = list(tables(tree))
    attrs = table["attrs"]
    height = asset_utils.resolve_tabletop_height(table["name"],
        position_z=attrs["translate"][2], scale_z=attrs["scale"][2])
    if height is None:
        raise ValueError("Registered scene table height is unavailable")
    physics = physics_profile()
    physics.update(scene_mode="RegisteredScene", registered_scene_id=scene_id,
                   lighting_mode="natural", hide_basket=True,
                   frame_origin_w_m=[0.5, 0.0, height], robot_base_w_m=[0.0, 0.0, height])
    # The real table mesh supplies geometry and collision; no proxy tabletop.
    del physics["table_size_m"]
    return {"scene_id": f"registered_scene_{scene_id}", "usd_path": str(path.relative_to(root)),
            "robot": "franka", "workspace": {"frame": "table",
            "axes": {"right": "+x", "forward": "+y", "up": "+z"},
            "bounds_m": {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0, 0.5]}},
            "cameras": ["table_cam"], "physics": physics}


def _surface_texture_references(material) -> list[dict]:
    """Inspect resolved asset inputs reachable from actual surface terminals."""
    from pxr import Sdf
    if not material:
        return []
    stage = material.GetPrim().GetStage()
    pending = [path for output in material.GetSurfaceOutputs()
               for path in output.GetAttr().GetConnections()]
    visited = set()
    textures = []
    while pending:
        prim = stage.GetPrimAtPath(pending.pop().GetPrimPath())
        if not prim or str(prim.GetPath()) in visited:
            continue
        visited.add(str(prim.GetPath()))
        for attr in prim.GetAttributes():
            pending.extend(attr.GetConnections())
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            asset = attr.Get()
            if not asset or not asset.path:
                continue
            textures.append({"attribute_path": str(attr.GetPath()), "asset_path": asset.path,
                             "resolved_path": asset.resolvedPath,
                             "exists": bool(asset.resolvedPath and Path(asset.resolvedPath).is_file())})
    return sorted(textures, key=lambda item: item["attribute_path"])


def _spawn_source_walls(stage, source_usd, root_path="/World/RegisteredScene/SourceWalls") -> list[dict]:
    """Reference authored walls with their transforms, collisions and materials.

    The legacy scene profile only extracts Cube walls. New scene assets use Mesh
    walls with textured materials, so rebuilding profile cuboids silently drops
    them. A filtered reference keeps the source bindings and relative texture
    paths intact, without spawning a second table, source props or lights.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    source_usd = str(Path(source_usd).resolve())
    source = Usd.Stage.Open(source_usd, load=Usd.Stage.LoadNone)
    source_root = source.GetPrimAtPath("/World")
    if not source_root:
        raise ValueError("Registered scene has no /World root for source walls")
    walls = [prim for prim in Usd.PrimRange(source_root)
             if "wall" in prim.GetName().lower()
             and (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube))]
    if not walls:
        return []
    wall_paths = [prim.GetPath() for prim in walls]
    looks = [prim.GetPath() for prim in Usd.PrimRange(source_root)
             if prim.GetName() == "Looks"]
    retained = wall_paths + looks
    root = stage.DefinePrim(root_path)
    root.GetReferences().AddReference(source_usd, source_root.GetPath())
    target_root = Sdf.Path(root_path)

    def prune(source_parent):
        for child in source_parent.GetChildren():
            path = child.GetPath()
            target = stage.GetPrimAtPath(path.ReplacePrefix(source_root.GetPath(), target_root))
            if any(path.HasPrefix(keep) for keep in retained):
                continue
            if any(keep.HasPrefix(path) for keep in retained):
                prune(child)
            else:
                target.SetActive(False)

    prune(source_root)
    records = []
    for prim in walls:
        actual = stage.GetPrimAtPath(prim.GetPath().ReplacePrefix(source_root.GetPath(), target_root))
        material, _ = UsdShade.MaterialBindingAPI(actual).ComputeBoundMaterial()
        collision = UsdPhysics.CollisionAPI(actual).GetCollisionEnabledAttr()
        records.append({"name": "source_wall_" + prim.GetName(),
                        "prim_path": str(actual.GetPath()), "usd_path": source_usd,
                        "source_prim_path": str(prim.GetPath()), "active": actual.IsActive(),
                        "geometry_type": actual.GetTypeName(),
                        "collision_enabled": bool(actual.HasAPI(UsdPhysics.CollisionAPI)
                                                  and collision and collision.Get()),
                        "bound_material_path": str(material.GetPath()) if material else None,
                        "surface_textures": _surface_texture_references(material)})
    return records


def spawn_registered_scene(scene: dict) -> dict:
    """Spawn registered task assets, retaining the source's authored walls."""
    import hashlib
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.utils.stage import get_current_stage
    from geomind_lab.tasks.manager_based.manipulation.geomind.geomind_scene_cfg import (
        GeoMindSceneCfg, get_scene_profile)
    profile = scene["physics"]
    source = get_scene_profile(profile["registered_scene_id"])
    expected = scene_description(profile["registered_scene_id"])
    if expected["usd_path"] != scene["usd_path"] or any(
            abs(a-b) > 1e-6 for a,b in zip(expected["physics"]["frame_origin_w_m"], profile["frame_origin_w_m"])):
        raise ValueError("Registered scene asset/table transform changed; regenerate scene configuration")
    config = GeoMindSceneCfg(num_envs=1, env_spacing=2.0)
    config.apply_scene(profile["registered_scene_id"], lighting_mode=profile["lighting_mode"],
                       hide_basket=profile["hide_basket"])
    spawned = []
    for name, asset in vars(config).items():
        if name.startswith("scene_wall_") or not isinstance(asset, AssetBaseCfg) or asset.spawn is None:
            continue
        path = asset.prim_path.replace("{ENV_REGEX_NS}", "/World/RegisteredScene")
        asset.spawn.func(path, asset.spawn, translation=asset.init_state.pos, orientation=asset.init_state.rot)
        spawned.append({"name": name, "prim_path": path,
                        "usd_path": getattr(asset.spawn, "usd_path", None)})
    spawned.extend(_spawn_source_walls(get_current_stage(), source.usd_path))
    return {"mode": "RegisteredScene", "scene_id": source.scene_id,
            "source_usd": source.usd_path,
            "source_sha256": hashlib.sha256(Path(source.usd_path).read_bytes()).hexdigest(),
            "loader": "GeoMindSceneCfg.apply_scene", "wall_loader": "filtered source USD reference",
            "assets": spawned}
