"""Runtime convex cylinder collisions derived from immutable body geometry.

The original analytic Cylinder remains the visual.  This bounded 32-sided
approximation replaces only its collision representation, inside the same rigid
body.  It neither writes a source USD file nor adds a body or support surface.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .body_geometry import load_body_clouds


_SIDES = 32
_RING_POINTS = 128
_COLLIDER_NAME = "RuntimeCylinderCollision"


def _sha_cloud(points):
    return hashlib.sha256(np.ascontiguousarray(points, dtype="<f8").tobytes()).hexdigest()


def _root_scale(prim, expected):
    from pxr import UsdGeom

    matrix = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation(), dtype=float)
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=1)
    if not np.isfinite(matrix).all() or np.any(scale <= 0) or np.linalg.det(linear) <= 0:
        raise ValueError("Cylinder root requires a finite positive scale")
    rotation = linear / scale[:, None]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError("Cylinder root shear is unsupported")
    declared = np.asarray(expected, dtype=float)
    if declared.shape != (3,) or not np.isfinite(declared).all() or np.any(declared <= 0):
        raise ValueError("Invalid manifest cylinder scale")
    if not np.allclose(scale, declared, atol=1e-7, rtol=0):
        raise ValueError("Cylinder root scale differs from manifest")
    return scale


def _plan(stage, obj, cloud, supplied, archive_sha):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    path = obj["prim_path"]
    body = stage.GetPrimAtPath(path)
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI) or body.IsInstanceProxy():
        raise ValueError("Cylinder must be an existing editable rigid body")
    original = stage.GetPrimAtPath(path + "/Geometry")
    if not original or not original.IsA(UsdGeom.Cylinder) or original.IsInstanceProxy():
        raise ValueError("Cylinder requires its original Geometry Cylinder child")
    collision = UsdPhysics.CollisionAPI(original)
    if not collision or not collision.GetCollisionEnabledAttr().Get():
        raise ValueError("Original Cylinder collision must be enabled")
    if stage.GetPrimAtPath(path + "/" + _COLLIDER_NAME):
        raise ValueError("Runtime cylinder collider already exists")
    for prim in Usd.PrimRange(body):
        if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Nested cylinder rigid bodies are unsupported")
        if prim != original and prim.HasAPI(UsdPhysics.CollisionAPI):
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                raise ValueError("Cylinder body has another enabled collision shape")
    if not np.allclose(np.asarray(UsdGeom.Xformable(original).GetLocalTransformation()),
                       np.eye(4), atol=1e-12, rtol=0):
        raise ValueError("Transformed Cylinder Geometry is unsupported")
    shape = UsdGeom.Cylinder(original)
    radius, height = float(shape.GetRadiusAttr().Get()), float(shape.GetHeightAttr().Get())
    if shape.GetAxisAttr().Get() != "Z" or not math.isfinite(radius + height) or min(radius, height) <= 0:
        raise ValueError("Only positive Z-axis cylinders are supported")
    scale = _root_scale(body, obj.get("scale", [1., 1., 1.]))
    supplied_array = np.asarray(supplied, dtype=float)
    if supplied_array.shape != cloud.shape or not np.array_equal(supplied_array, cloud):
        raise ValueError("Supplied cylinder cloud differs from hash-bound body geometry")
    if cloud.shape != (2 * _RING_POINTS, 3) or not np.isfinite(cloud).all():
        raise ValueError("Cylinder cloud must contain two ordered 128-point rings")
    # The archive is already scaled in body axes.  Validate its full topology
    # against the saved analytic shape; divide the actual root scale only when
    # writing child-local vertices, so the root does not scale it a second time.
    angles = np.arange(_RING_POINTS) * (2 * math.pi / _RING_POINTS)
    ring = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    expected = np.r_[np.c_[ring, np.full(_RING_POINTS, -height / 2)],
                     np.c_[ring, np.full(_RING_POINTS, height / 2)]]
    if not np.allclose(cloud, expected * scale, atol=1e-8, rtol=0):
        raise ValueError("Unsupported cylinder cloud ring order, dimensions, centre or topology")
    selected = np.r_[cloud[:_RING_POINTS:4], cloud[_RING_POINTS::4]]
    vertices = selected / scale
    counts = [_SIDES, _SIDES] + [4] * _SIDES
    indices = list(range(_SIDES - 1, -1, -1)) + list(range(_SIDES, 2 * _SIDES))
    for index in range(_SIDES):
        following = (index + 1) % _SIDES
        indices.extend([index, following, following + _SIDES, index + _SIDES])
    bindings, strengths = {}, {}
    for rel in original.GetRelationships():
        name = rel.GetName()
        if not name.startswith("material:binding"):
            continue
        if ":collection:" in name:
            raise ValueError("Cylinder collection material bindings are unsupported")
        bindings[name] = [str(target) for target in rel.GetTargets()]
        strengths[name] = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(rel)
    # Inherited bindings remain inherited because both shapes have the same
    # rigid-body parent.  Record the resolved physics material for verification.
    material = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial(materialPurpose="physics")[0]
    physics_material = str(material.GetPath()) if material else None
    attributes = []
    for attr in original.GetAttributes():
        if attr.GetName().startswith("physxCollision:") and attr.HasAuthoredValueOpinion():
            if attr.GetNumTimeSamples():
                raise ValueError("Animated Cylinder collision properties are unsupported")
            attributes.append((attr.GetName(), attr.GetTypeName(), attr.Get()))
    attrs = {name: value for name, _, value in attributes}
    return {
        "original_prim_path": str(original.GetPath()), "body_prim_path": path,
        "collider_prim_path": path + "/" + _COLLIDER_NAME,
        "vertex_count": 2 * _SIDES, "face_count": len(counts),
        "polygon_sides": _SIDES, "sampled_angle_stride": 4,
        "vertices_local_m": vertices.tolist(), "face_vertex_counts": counts,
        "face_vertex_indices": indices, "root_scale": scale.tolist(),
        "source_radius_m": radius, "source_height_m": height,
        "body_scaled_dimensions_m": np.ptp(cloud, axis=0).tolist(),
        "approximation_error_bound_m": float(radius * max(scale[:2]) * (1 - math.cos(math.pi / _SIDES))),
        "approximation_bound_definition": "maximum radial polygon sagitta in scaled body axes; original analytic measurement bounds retained",
        "body_cloud_sha256": _sha_cloud(cloud), "body_cloud_hash_encoding": "contiguous little-endian float64 Nx3 bytes",
        "source_body_archive_sha256": archive_sha, "body_points_key": obj.get("body_points_key", "points_m"),
        "material_bindings": bindings, "material_binding_strengths": strengths,
        "resolved_physics_material_path": physics_material,
        "contact_offset_m": attrs.get("physxCollision:contactOffset"),
        "rest_offset_m": attrs.get("physxCollision:restOffset"),
        "_attributes": attributes,
    }


def install_cylinder_mesh_colliders(stage, manifest, body_clouds_by_prim):
    """Install 64-vertex convex hulls inside the existing cylinder bodies.

    All objects, source archive hashes, supplied clouds and analytic geometry are
    validated before any USD edits.  Only the runtime stage root layer receives
    opinions.  The caller must invoke this before physics creates its bodies.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    objects = manifest.get("objects", [])
    cylinders = [obj for obj in objects if obj.get("kind") == "primitive_cylinder"]
    result = {"applied": bool(cylinders), "colliders": [], "runtime_only": True,
              "source_usd_files_modified": False, "new_rigid_bodies": 0,
              "new_support_surfaces": 0, "model": "hash_bound_32_sided_cylinder_convex_hull"}
    if not cylinders:
        return result
    if not isinstance(body_clouds_by_prim, dict):
        raise ValueError("Cylinder body clouds must be mapped by original prim")
    paths = [obj.get("prim_path") for obj in objects]
    if any(not isinstance(path, str) or not path.startswith("/") for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("Manifest body prim paths must be unique absolute paths")
    root_layer = stage.GetRootLayer()
    source = manifest.get("scene_usda")
    if source and root_layer.realPath and Path(root_layer.realPath).resolve() == Path(source).resolve():
        raise ValueError("Cylinder collision edits require a runtime composition root, not the source USD layer")
    archive_clouds = load_body_clouds(manifest)
    plans = []
    for obj, cloud in zip(objects, archive_clouds):
        if obj.get("kind") != "primitive_cylinder":
            continue
        if obj["prim_path"] not in body_clouds_by_prim:
            raise ValueError("Missing cylinder cloud for original body prim")
        plans.append(_plan(stage, obj, cloud, body_clouds_by_prim[obj["prim_path"]],
                           manifest["provenance"]["asset_body_vertices_sha256"]))
    # A transient layer snapshot makes unexpected authoring failures reversible;
    # the validated original visual and rigid-body properties are never changed.
    previous_root = root_layer.ExportToString()
    try:
        with Usd.EditContext(stage, root_layer):
            for plan in plans:
                mesh = UsdGeom.Mesh.Define(stage, plan["collider_prim_path"])
                mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in plan["vertices_local_m"]])
                mesh.CreateFaceVertexCountsAttr(plan["face_vertex_counts"])
                mesh.CreateFaceVertexIndicesAttr(plan["face_vertex_indices"])
                mesh.CreateSubdivisionSchemeAttr("none")
                mesh.CreateOrientationAttr("rightHanded")
                points = np.asarray(plan["vertices_local_m"])
                mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
                mesh.CreateVisibilityAttr("invisible")
                mesh.CreatePurposeAttr("guide")
                prim = mesh.GetPrim()
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
                prim.AddAppliedSchema("PhysxCollisionAPI")
                for name, typ, value in plan["_attributes"]:
                    prim.CreateAttribute(name, typ, custom=False).Set(value)
                UsdShade.MaterialBindingAPI.Apply(prim)
                for name, targets in plan["material_bindings"].items():
                    rel = prim.CreateRelationship(name, custom=False)
                    rel.SetTargets(targets)
                    UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(rel, plan["material_binding_strengths"][name])
                material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")[0]
                if (str(material.GetPath()) if material else None) != plan["resolved_physics_material_path"]:
                    raise ValueError("Runtime cylinder physics material binding changed")
                original = stage.GetPrimAtPath(plan["original_prim_path"])
                UsdPhysics.CollisionAPI(original).CreateCollisionEnabledAttr(False)
                plan.pop("_attributes")
                result["colliders"].append(plan)
    except Exception:
        root_layer.ImportFromString(previous_root)
        raise
    return result
