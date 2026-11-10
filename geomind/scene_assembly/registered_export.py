"""Export existing registered task objects into an actual cleared room.

This worker only composes USD and checks geometry. It never attaches an object
to a robot, writes an execution pose, or claims that PhysX has run. Private
legacy object names remain in the evaluator manifest; public aliases are bound
by the task pipeline after settling.
"""
from __future__ import annotations
from geomind_lab.utils.portable_paths import isaac_assets_root, resolve_asset_path

from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

from .assets import ROOT, resolve_asset, resolve_scene
from .usd import (_background_obstacles, _body_points, _bounds, _hits_obstacle,
                  _set_transform, digest, measure_table, supported_xy)


def _vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must contain {size} finite numbers")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError(f"{label} must contain {size} finite numbers")
    return [float(x) for x in value]


def _reject_containment_layout_scale(runtime_objects, protocol=None):
    protocol = protocol or {}
    gt = protocol.get("gt_spec", {})
    if (any(obj.get("role") in {"container_floor", "container_wall"} for obj in runtime_objects)
            or gt.get("resolver") == "containment_placement"
            or any(key.startswith("container_") for key in gt)
            or protocol.get("variant") == "place_inside"):
        raise ValueError("Uniform runtime XY layout scaling is unsupported for containment or container floor/wall assemblies; asset dimensions remain fixed")


def scale_runtime_xy(runtime_objects, layout_scale, safe_rect, *, reason=None, protocol=None):
    """Uniformly scale one existing sample; never sample, shrink to fit or edit assets.

    The rectangle bounds runtime pose.xy centres. Real rotated footprints and
    retained fixtures are still checked against the actual table after export.
    """
    try:
        valid = type(layout_scale) in (int, float) and math.isfinite(layout_scale) and layout_scale > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("layout_scale must be a finite positive number")
    factor = float(layout_scale)
    if factor == 1.:
        return runtime_objects, None
    bounds = _vector(safe_rect, 4, "layout scale safe rectangle")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("Layout scale safe rectangle has no positive area")
    if not isinstance(runtime_objects, list) or not runtime_objects:
        raise ValueError("Layout scale requires existing runtime objects")
    _reject_containment_layout_scale(runtime_objects, protocol)
    before = np.asarray([_vector(obj.get("pose", {}).get("xy"), 2, "layout scale runtime XY") for obj in runtime_objects])
    low, high = before.min(axis=0), before.max(axis=0)
    center = (low + high) * .5
    with np.errstate(over="ignore", invalid="ignore"):
        expanded = center + (before - center) * factor
    if not np.isfinite(expanded).all():
        raise ValueError("Scaled layout XY is not finite")
    new_low, new_high = expanded.min(axis=0), expanded.max(axis=0)
    span = new_high - new_low
    safe_low = np.asarray([bounds[0], bounds[2]])
    safe_high = np.asarray([bounds[1], bounds[3]])
    if np.any(span > safe_high - safe_low + 1e-12):
        raise ValueError(f"Scaled layout span {span.tolist()} does not fit safe rectangle; refusing compression or resampling")
    # Feasible translations form one interval per axis. Projection of zero
    # into each interval is the unique minimum Euclidean translation.
    lower_translation = safe_low - new_low
    upper_translation = safe_high - new_high
    translation = np.maximum(lower_translation, np.minimum(0., upper_translation))
    after = expanded + translation
    if np.any(after < safe_low - 1e-12) or np.any(after > safe_high + 1e-12):
        raise ValueError("Scaled layout could not be translated into the safe rectangle")
    result = deepcopy(runtime_objects)
    changes = []
    for index, obj in enumerate(result):
        obj["pose"]["xy"] = after[index].tolist()
        changes.append({"object_name": obj["name"], "before_xy_m": before[index].tolist(),
                        "scaled_xy_before_translation_m": expanded[index].tolist(),
                        "after_xy_m": after[index].tolist(),
                        "delta_xy_m": (after[index] - before[index]).tolist()})
    record = {"kind": "agent_uniform_runtime_xy_layout_scale", "applied": True,
        "layout_scale": factor, "source_xy_bbox_center_m": center.tolist(),
        "source_xy_bbox_m": [float(low[0]), float(high[0]), float(low[1]), float(high[1])],
        "scaled_xy_span_m": span.tolist(), "safe_rect_xy_m": bounds,
        "minimum_translation_xy_m": translation.tolist(), "object_transforms": changes,
        "reason": reason or "Agent explicitly selected a uniformly scaled registered layout for Robotiq clearance; task rules and assets are unchanged.",
        "reference_point": "runtime pose.xy centres; physical footprints are checked separately against the actual table",
        "randomizer_additional_calls": 0, "seed_changed": False, "compression_to_fit_used": False,
        "unchanged": ["Z policy", "asset dimensions and scale", "materials", "object yaw", "task and GT rules"],
        "geometry_support_and_runtime_gt": "unverified; original geometry checks, physical settling and measured GT solving remain required"}
    return result, record


def _primitive_points(obj):
    """Body-frame surface points; analytic primitives retain native collisions."""
    kind = obj["kind"]
    if kind == "primitive_cuboid":
        size = _vector(obj["size"], 3, "cuboid size")
        if min(size) <= 0:
            raise ValueError("Cuboid dimensions must be positive")
        return np.asarray(list(itertools.product(*[(-d / 2, d / 2) for d in size])))
    radius = float(obj["radius"])
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Primitive radius must be finite and positive")
    angles = np.arange(128) * (2 * math.pi / 128)
    ring = np.c_[np.cos(angles) * radius, np.sin(angles) * radius]
    if kind == "primitive_sphere":
        lat = np.linspace(-math.pi / 2, math.pi / 2, 65)
        return np.concatenate([np.c_[ring * math.cos(t), np.full(len(ring), radius * math.sin(t))]
                               for t in lat])
    height = float(obj["height"])
    if not math.isfinite(height) or height <= 0:
        raise ValueError("Primitive height must be finite and positive")
    bottom = np.c_[ring, np.full(len(ring), -height / 2)]
    if kind == "primitive_cylinder":
        return np.r_[bottom, np.c_[ring, np.full(len(ring), height / 2)]]
    if kind == "primitive_cone":
        return np.r_[bottom, [[0., 0., height / 2]]]
    raise ValueError(f"Unsupported runtime primitive: {kind}")


def _resolve_runtime_asset(obj):
    if obj.get("usd_path"):
        path = Path(resolve_asset_path(obj["usd_path"]))
        if not path.is_file() and "/Assets/ArchVis/" in str(path):
            path = Path(resolve_asset_path(str(path)))
        if not path.is_file():
            raise ValueError(f"Runtime USD is unavailable: {obj['usd_path']}")
        return str(path.resolve())
    return resolve_asset(str(obj.get("asset_key", "")))["usd_path"]


def _mdl_path(raw):
    raw = str(raw)
    prefix = "{NVIDIA_NUCLEUS_DIR}/"
    if raw.startswith(prefix):
        raw = str(isaac_assets_root()/"Assets/Isaac/4.5/NVIDIA" / raw[len(prefix):])
    path = Path(resolve_asset_path(raw))
    return str(path.resolve()) if path.is_file() else None


def _material(stage, prim, obj, path):
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    spec = obj.get("visual_material", {})
    override = obj.get("material_override", {})
    color = override.get("diffuse_color", spec.get("fallback_diffuse_color", obj.get("visual_color")))
    # An asset's original MDL bindings stay intact unless explicitly overridden.
    if color is None and obj["kind"] == "usd":
        return {"method": "original_usd_material"}
    color = _vector(color or [0.63, 0.48, 0.33], 3, "visual color")
    material = UsdShade.Material.Define(stage, path)
    preview = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(spec.get("fallback_roughness", .6)))
    preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    record = {"method": "preview_surface", "diffuse_color": color, "source": deepcopy(spec)}
    mdl = _mdl_path(spec.get("mdl_path", "")) if spec.get("type") == "mdl" else None
    if mdl:
        shader = UsdShade.Shader.Define(stage, path + "/MdlShader")
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath(mdl), "mdl")
        shader.SetSourceAssetSubIdentifier(Path(mdl).stem, "mdl")
        for name, typ, value in (("project_uvw", Sdf.ValueTypeNames.Bool, spec.get("project_uvw")),
                                  ("albedo_brightness", Sdf.ValueTypeNames.Float, spec.get("albedo_brightness")),
                                  ("texture_scale", Sdf.ValueTypeNames.Float2, spec.get("texture_scale"))):
            if value is not None:
                shader.CreateInput(name, typ).Set(Gf.Vec2f(*value) if name == "texture_scale" else value)
        # Preserve color predicates as well as the authored wood texture.
        shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        record.update(method="local_mdl_with_preview_fallback", mdl_path=mdl, mdl_sha256=digest(mdl))
    elif spec.get("type") == "mdl":
        record["mdl_unavailable_fallback"] = str(spec.get("mdl_path"))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    if prim.IsA(UsdGeom.Gprim):
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return record


def _physics(stage, prim, obj, material_path):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    spec = obj.get("physics", {})
    rigid = spec.get("rigid_body", {})
    collision = spec.get("collision", {})
    dynamic = rigid.get("rigid_body_enabled", True) is not False and not rigid.get("kinematic_enabled", False)
    if collision.get("type") == "static_mesh":
        dynamic = False
    for node in list(Usd.PrimRange(prim)):
        if node.IsInstance():
            node.SetInstanceable(False)
    for node in Usd.PrimRange(prim):
        if node != prim and node.HasAPI(UsdPhysics.RigidBodyAPI):
            node.RemoveAPI(UsdPhysics.RigidBodyAPI)
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(not dynamic)
    body.CreateVelocityAttr((0, 0, 0))
    body.CreateAngularVelocityAttr((0, 0, 0))
    mass = float(spec.get("mass", obj.get("mass", .1)))
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    defaults = {"linear_damping": 1.2, "angular_damping": 4.,
                "solver_position_iteration_count": 32, "solver_velocity_iteration_count": 4}
    attrs = {"linear_damping": ("linearDamping", Sdf.ValueTypeNames.Float),
             "angular_damping": ("angularDamping", Sdf.ValueTypeNames.Float),
             "solver_position_iteration_count": ("solverPositionIterationCount", Sdf.ValueTypeNames.Int),
             "solver_velocity_iteration_count": ("solverVelocityIterationCount", Sdf.ValueTypeNames.Int),
             "max_linear_velocity": ("maxLinearVelocity", Sdf.ValueTypeNames.Float),
             "max_angular_velocity": ("maxAngularVelocity", Sdf.ValueTypeNames.Float),
             "sleep_threshold": ("sleepThreshold", Sdf.ValueTypeNames.Float),
             "stabilization_threshold": ("stabilizationThreshold", Sdf.ValueTypeNames.Float),
             "disable_gravity": ("disableGravity", Sdf.ValueTypeNames.Bool)}
    for key, value in (defaults | rigid).items():
        if key in attrs:
            name, typ = attrs[key]
            prim.CreateAttribute("physxRigidBody:" + name, typ, custom=False).Set(value)
    material_spec = spec.get("material", {})
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(float(material_spec.get("static_friction", 1.15)))
    physics_material.CreateDynamicFrictionAttr(float(material_spec.get("dynamic_friction", .96)))
    physics_material.CreateRestitutionAttr(float(material_spec.get("restitution", 0.)))
    for key in ("friction_combine_mode", "restitution_combine_mode"):
        if key in material_spec:
            material.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
            name = "frictionCombineMode" if key.startswith("friction") else "restitutionCombineMode"
            material.GetPrim().CreateAttribute("physxMaterial:" + name, Sdf.ValueTypeNames.Token, custom=False).Set(material_spec[key])
    paths = []
    for node in Usd.PrimRange(prim):
        if not node.IsA(UsdGeom.Gprim):
            continue
        UsdPhysics.CollisionAPI.Apply(node).CreateCollisionEnabledAttr(collision.get("collision_enabled", True))
        if node.IsA(UsdGeom.Mesh):
            # A static bowl keeps its real concave cavity. Containers made from
            # five cuboids remain five separate colliders with an open top.
            UsdPhysics.MeshCollisionAPI.Apply(node).CreateApproximationAttr("convexHull" if dynamic else "none")
        node.AddAppliedSchema("PhysxCollisionAPI")
        for key, name in (("contact_offset", "contactOffset"), ("rest_offset", "restOffset")):
            if key in collision:
                node.CreateAttribute("physxCollision:" + name, Sdf.ValueTypeNames.Float, custom=False).Set(float(collision[key]))
        UsdShade.MaterialBindingAPI.Apply(node).Bind(material, materialPurpose="physics")
        paths.append(str(node.GetPath()))
    if not paths:
        raise ValueError(f"Runtime object {obj['name']} has no collision geometry")
    return dynamic, paths


def _draw_edges(stage, spec, objects, height):
    if spec.get('protocol',{}).get('module_name')=='canonical_extension':
        from geomind.episodes.canonical_extensions import draw_context
        return draw_context(stage,spec,objects,height)
    from pxr import Gf, UsdGeom
    gt = spec.get("protocol", {}).get("gt_spec", {})
    structure = gt.get("structure", {})
    if not structure.get("draw_edges"):
        return []
    indexed = {obj["object_id"]: obj for obj in objects}
    names = structure.get("nodes", [])
    if any(name not in indexed for name in names):
        raise ValueError("Directed chain references an absent runtime object")
    axis = 0 if gt.get("order_axis", "y") == "x" else 1
    ordered = sorted(names, key=lambda name: indexed[name]["position_w_m"][axis])
    paths = []
    for i, (a, b) in enumerate(zip(ordered, ordered[1:])):
        start, end = np.asarray(indexed[a]["position_w_m"][:2]), np.asarray(indexed[b]["position_w_m"][:2])
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            raise ValueError("Directed chain contains coincident nodes")
        direction = delta / distance
        normal = np.array([-direction[1], direction[0]])
        center = start + .65 * delta
        length, width = min(.014, distance * .25), min(.009, distance * .15)
        points = [center + direction * length, center - direction * length + normal * width,
                  center - direction * length - normal * width]
        root = f"/World/TaskContext/Arrow_{i:03d}"
        arrow = UsdGeom.Mesh.Define(stage, root)
        arrow.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), height + .002) for p in points])
        arrow.CreateFaceVertexCountsAttr([3]); arrow.CreateFaceVertexIndicesAttr([0, 1, 2])
        arrow.CreateDoubleSidedAttr(True); arrow.CreateDisplayColorAttr([Gf.Vec3f(.95, .58, .06)])
        curve = UsdGeom.BasisCurves.Define(stage, root + "_Edge")
        curve.CreateTypeAttr("linear"); curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), height + .002) for p in (start, end)])
        curve.CreateWidthsAttr([.004]); curve.CreateDisplayColorAttr([Gf.Vec3f(.95, .58, .06)])
        paths.extend([root, root + "_Edge"])
    return paths


def export_registered(plan, output, scene_id=65, seed=0, layout_scale=1.):
    """Compose heterogeneous registered objects; return a physics-unverified manifest."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from geomind_lab.tasks.manager_based.manipulation.geomind.layout_randomizer import (
        LayoutRandomizationConfig, randomize_runtime_object_layout)
    if type(scene_id) is not int or not 1 <= scene_id <= 102 or type(seed) is not int or seed < 0:
        raise ValueError("Expected scene_id in 1..102 and a nonnegative integer seed")
    try:
        scale_valid = type(layout_scale) in (int, float) and math.isfinite(layout_scale) and layout_scale > 0
    except OverflowError:
        scale_valid = False
    if not scale_valid:
        raise ValueError("layout_scale must be a finite positive number")
    spec = deepcopy(plan["task_spec"])
    registration = deepcopy(plan["registration"])
    for filename, expected in plan.get("source_hashes", {}).items():
        if digest(filename) != expected:
            raise ValueError(f"Registered source changed before export: {filename}")
    runtime = spec.get("runtime_objects")
    if not isinstance(runtime, list) or not runtime:
        raise ValueError("Registration has no runtime objects; refusing a generic replacement")
    names = [obj.get("name", "") for obj in runtime]
    if len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in names):
        raise ValueError("Runtime object names must be unique USD identifiers")
    if any(obj.get("kind") not in {"primitive_cuboid", "primitive_sphere", "primitive_cylinder", "primitive_cone", "usd"} for obj in runtime):
        raise ValueError("Unsupported runtime object kind")
    if float(layout_scale) != 1.:
        _reject_containment_layout_scale(runtime, spec.get("protocol"))
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    destination = output / "scene.usda"
    if destination.exists() or (output / "manifest.json").exists():
        raise ValueError(f"Refusing to overwrite registered assembly: {output}")
    scene = resolve_scene(scene_id)
    source = Usd.Stage.Open(scene["usd_path"])
    table = measure_table(source)
    obstacles = _background_obstacles(source, table["prim_path"])
    bounds = table["bounds_xy_m"]
    safe = (max(bounds[0] + .03, .15), min(bounds[1] - .03, .68),
            max(bounds[2] + .03, -.32), min(bounds[3] - .03, .34))
    if safe[0] >= safe[1] or safe[2] >= safe[3]:
        raise ValueError("Selected table does not overlap the registered reachable workspace")
    cfg = LayoutRandomizationConfig(True, .02, seed, True, .15, seed + 1000003, .025, safe, 256)
    authored_runtime = deepcopy(runtime)
    extension=spec.get('protocol',{}).get('module_name')=='canonical_extension'
    if extension:
        from geomind.episodes.canonical_extensions import translate_layout
        record=translate_layout(runtime,seed)
    else:
        record = randomize_runtime_object_layout(runtime, protocol=spec.get("protocol", {}),
            layout_key=registration.get("annotation_layout_key", ""), config=cfg, verbose=False)
    if record is None:
        raise ValueError("Registered layout randomization did not produce an explicit record")
    grid_constraint = None
    if spec.get("protocol", {}).get("gt_spec", {}).get("resolver") == "grid_rowcol_slot":
        from .registered_layout_constraints import repair_grid_row_quantization
        repaired, grid_constraint = repair_grid_row_quantization(authored_runtime, runtime, spec["protocol"])
        # The guard returns copies. Rebind both the compiler's private list and
        # this export loop; leaving either reference on the old sample would
        # silently export the invalid row buckets that the guard just rejected.
        spec["runtime_objects"] = deepcopy(repaired)
        runtime = spec["runtime_objects"]
        record = {**record, "post_sampling_grid_constraint": deepcopy(grid_constraint)}
    layout_transform = None
    if float(layout_scale) != 1.:
        transformed, layout_transform = scale_runtime_xy(runtime, layout_scale, safe,
            reason=plan.get("layout_scale_reason"), protocol=spec.get("protocol"))
        spec["runtime_objects"] = deepcopy(transformed)
        runtime = spec["runtime_objects"]
        layout_transform["applied_after"] = "one legacy randomizer sample and existing grid constraint"
        record = {**record, "agent_layout_transform": deepcopy(layout_transform)}
        if grid_constraint:
            from .registered_layout_constraints import validate_grid_layout
            grid_constraint["after_agent_scale_validation"] = validate_grid_layout(runtime, spec["protocol"])
    spec["applied_layout_randomization"] = record
    layer = Sdf.Layer.CreateNew(str(destination)); layer.subLayerPaths = [scene["usd_path"]]
    stage = Usd.Stage.Open(layer)
    for key in ("defaultPrim", "upAxis", "metersPerUnit", "timeCodesPerSecond", "framesPerSecond"):
        if source.GetRootLayer().pseudoRoot.HasInfo(key):
            stage.SetMetadata(key, source.GetRootLayer().pseudoRoot.GetInfo(key))
    frozen = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body = UsdPhysics.RigidBodyAPI(prim)
            body.CreateKinematicEnabledAttr(True); body.CreateVelocityAttr((0, 0, 0)); body.CreateAngularVelocityAttr((0, 0, 0))
            frozen.append(str(prim.GetPath()))
    UsdGeom.Xform.Define(stage, "/World/AssembledObjects")
    clouds, objects, assets, collisions = {}, [], [], []
    for index, obj in enumerate(runtime):
        name, kind = obj["name"], obj["kind"]
        path = f"/World/AssembledObjects/{name}"
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        pose = obj.get("pose", {})
        xy = _vector(pose.get("xy", spec.get("object_positions", {}).get(name, {}).get("position", [])[:2]), 2, "runtime XY")
        yaw = float(pose.get("yaw", 0.))
        if not math.isfinite(yaw):
            raise ValueError("Runtime yaw must be finite")
        orientation = [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
        scale = _vector(obj.get("scale", [1., 1., 1.]), 3, "runtime scale") if kind == "usd" else [1., 1., 1.]
        if min(scale) <= 0:
            raise ValueError("Runtime scale must be positive")
        source_asset = None
        if kind == "usd":
            source_asset = _resolve_runtime_asset(obj)
            prim.GetReferences().AddReference(source_asset)
            _set_transform(prim, scale, orientation, [0., 0., 0.])
            points = _body_points(stage, prim, orientation)
        else:
            shape_path = path + "/Geometry"
            if kind == "primitive_cuboid":
                shape = UsdGeom.Cube.Define(stage, shape_path); shape.CreateSizeAttr(1.)
                _set_transform(shape.GetPrim(), _vector(obj["size"], 3, "size"), [1., 0., 0., 0.], [0., 0., 0.])
            elif kind == "primitive_sphere":
                shape = UsdGeom.Sphere.Define(stage, shape_path); shape.CreateRadiusAttr(float(obj["radius"]))
            else:
                klass = UsdGeom.Cylinder if kind == "primitive_cylinder" else UsdGeom.Cone
                shape = klass.Define(stage, shape_path); shape.CreateRadiusAttr(float(obj["radius"]))
                shape.CreateHeightAttr(float(obj["height"])); shape.CreateAxisAttr("Z")
            points = _primitive_points(obj)
            _set_transform(prim, scale, orientation, [0., 0., 0.])
        lo, hi = _bounds(stage, prim)
        # Existing injection uses root XY and a tabletop-bottom policy. Explicit
        # relative Z remains relative to the measured table, never a stale room Z.
        policy = pose.get("z_policy", obj.get("z_policy", "tabletop"))
        if policy in ("tabletop", "on_table", "tabletop_bottom"):
            z = table["height_m"] - lo[2] + .001
        elif policy in ("table_relative", "tabletop_relative"):
            z = table["height_m"] + float(pose.get("z", pose.get("z_offset_m", obj.get("z_offset_m", 0.))))
        else:
            raise ValueError(f"Unsupported runtime z_policy: {policy}")
        translation = [xy[0], xy[1], z]
        _set_transform(prim, scale, orientation, translation)
        lo, hi = _bounds(stage, prim)
        center = [(a + b) / 2 for a, b in zip(lo, hi)]
        half = [(b - a) / 2 for a, b in zip(lo, hi)]
        support = supported_xy(table, center[:2], half[:2])
        hit = _hits_obstacle(center[:2], [d * 2 for d in half], lo[2], hi[2], obstacles)
        if not support or hit:
            raise ValueError(f"Runtime object {name} lacks real tabletop support or intersects a retained fixture")
        dynamic, paths = _physics(stage, prim, obj, f"/World/AssembledLooks/Physics_{index:03d}")
        material = _material(stage, prim, obj, f"/World/AssembledLooks/Visual_{index:03d}")
        key = f"points_m_{index:03d}"; clouds[key] = points
        objects.append({"object_id": name, "name": name, "prim_path": path, "kind": kind,
            "source_label": name, "asset_name": obj.get("asset_key", kind),
            "role": obj.get("role", ""), "object_category": obj.get("object_category", ""),
            "dynamic": dynamic, "scale": scale, "body_points_key": key,
            "position_w_m": translation, "orientation_wxyz": orientation,
            "bbox_center_xy_m": center[:2], "bbox_center_w_m": center,
            "bbox_min_w_m": lo, "bbox_max_w_m": hi, "dimensions_m": [d * 2 for d in half],
            "support": {"tabletop_footprint_passed": support, "fixture_aabb_clear": not hit,
                        "initial_bottom_gap_m": lo[2] - table["height_m"]},
            "z_policy": policy, "visual": material, "runtime_descriptor": deepcopy(obj)})
        assets.append({"name": name, "kind": kind, "scale": scale, "usd_path": source_asset,
                       "sha256": digest(source_asset) if source_asset else None})
        collisions.extend(paths)
    spec["object_positions"] = {obj["object_id"]: {"position": obj["position_w_m"]} for obj in objects}
    if isinstance(spec.get("scene_adaptation"), dict):
        spec["scene_adaptation"]["runtime_objects"] = deepcopy(runtime)
        spec["scene_adaptation"]["applied_layout_randomization"] = deepcopy(record)
    clouds["points_m"] = clouds["points_m_000"]
    np.savez_compressed(output / "asset_body_vertices.npz", **clouds)
    arrows = _draw_edges(stage, spec, objects, table["height_m"])
    layer.customLayerData = {"generator": "geomind.scene_assembly.registered_export", "scene_id": scene_id,
                            "registration_id": registration["task_id"], "seed": seed}
    layer.Save()
    reopened = Usd.Stage.Open(str(destination))
    for obj in objects:
        lo, hi = _bounds(reopened, reopened.GetPrimAtPath(obj["prim_path"]))
        if not np.allclose(lo + hi, obj["bbox_min_w_m"] + obj["bbox_max_w_m"], atol=1e-8, rtol=0):
            raise ValueError("Registered geometry changed after saving USD")
    public_table = {key: value for key, value in table.items() if key != "surface_triangles"}
    usd_layers = sorted({str(x.realPath) for x in reopened.GetUsedLayers() if x.realPath})
    preheld = spec.get("protocol", {}).get("pre_attached_obj")
    manifest = {"schema_version": 1, "kind": "initial_scene_assembly", "status": "assembled_physics_unverified",
        "request": {"scene_id": scene_id, "object": {"asset": "registered_runtime", "count": len(objects)},
                    "layout": {"type": "registered", "source": registration.get("annotation_layout_key")}, "seed": seed},
        "registration": registration, "registered_task_spec": spec, "scene": scene,
        "asset_preserving": deepcopy(plan.get("asset_preserving", {})),
        "asset": {"name": "registered_runtime", "heterogeneous": True, "scale": [1., 1., 1.], "objects": assets},
        "table": public_table, "objects": objects, "layout_randomization": record,
        "registered_layout_constraints": [deepcopy(grid_constraint)] if grid_constraint else [],
        "layout_frame": "original registered world XY; source tabletop-relative Z mapped to measured tabletop",
        "geometry_validation": {"saved_reopened": True, "object_count": len(objects),
            "footprint_support_samples_passed": True, "retained_fixture_aabb_checks_passed": True,
            "tabletop_method": "measured near-horizontal triangles; nine footprint samples per object",
            "collision_approximation": "analytic primitives; convex dynamic USD; concave static USD",
            "container_parts_remain_separate": True, "directed_chain_visual_paths": arrows},
        "initial_grasp": {"pre_attached": False, "source_pre_attached_object": preheld,
            "requires_real_pregrasp": bool(preheld) or bool(spec.get("protocol", {}).get("target_object")),
            "notes": "Movable objects start on the real table; execution must grasp using real finger contact"},
        "frozen_background_rigid_bodies": frozen, "collision_mesh_count": len(collisions),
        "asset_body_vertices": str(output / "asset_body_vertices.npz"), "scene_usda": str(destination),
        "physics": {"executed": False, "passed": False, "notes": "CPU USD export only; no settling or robot execution"},
        "provenance": {"cleared_scene_sha256": digest(scene["usd_path"]), "scene_usda_sha256": digest(destination),
            "asset_body_vertices_sha256": digest(output / "asset_body_vertices.npz"),
            "asset_sha256": {a["usd_path"]: a["sha256"] for a in assets if a["usd_path"]},
            "registered_sources_sha256": deepcopy(plan.get("source_hashes", {})),
            "asset_preserving": deepcopy(plan.get("asset_preserving", {})),
            "used_usd_layers": usd_layers, "used_usd_layers_sha256": {p: digest(p) for p in usd_layers}}}
    if extension:
        manifest['visualization']={'walls':'visible'}
    recorded_plan = plan
    if layout_transform:
        manifest["agent_layout_transform"] = deepcopy(layout_transform)
        manifest["request"]["layout"]["agent_scale"] = float(layout_scale)
        recorded_plan = {**deepcopy(plan), "layout_scale": float(layout_scale),
                         "agent_layout_transform": deepcopy(layout_transform)}
    (output / "registered_plan.json").write_text(json.dumps(recorded_plan, ensure_ascii=False, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
