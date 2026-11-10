"""Non-physical directed-chain edges for structural adjacency observations."""
from __future__ import annotations

import math


def draw_task_context(env) -> None:
    protocol = env._task_protocol()
    gt = protocol.get("gt_spec", {})
    structure = gt.get("structure", {})
    nodes = structure.get("nodes", []) if structure.get("draw_edges") else []
    if not nodes:
        return
    import omni.usd
    from pxr import Gf, UsdGeom
    from geomind_lab.utils import asset_utils

    stage = omni.usd.get_context().get_stage()
    profile = getattr(env.cfg, "_scene_profile", None)
    top = 0.75
    if profile is not None and profile.table_name:
        transform = profile.xforms[profile.table_name]
        top = asset_utils.resolve_tabletop_height(profile.table_name, position_z=transform["pos"][2], scale_z=transform["scale"][2]) or top
    origins = env.scene.env_origins.detach().cpu().tolist()

    def mesh(path, points, color):
        shape = UsdGeom.Mesh.Define(stage, path)
        shape.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        shape.CreateFaceVertexCountsAttr([len(points)])
        shape.CreateFaceVertexIndicesAttr(list(range(len(points))))
        shape.CreateDoubleSidedAttr(True)
        shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    for env_id, origin in enumerate(origins):
        root = f"/World/envs/env_{env_id}/TaskContext"
        z = top + origin[2] + 0.002
        if nodes and all(name in env.scene.keys() for name in nodes):
            axis = 0 if gt.get("order_axis", "y") == "x" else 1
            points = [env.scene[name].data.root_pos_w[env_id].detach().cpu().tolist() for name in nodes]
            points.sort(key=lambda point: point[axis])
            for index, (start, end) in enumerate(zip(points, points[1:])):
                dx, dy = end[0]-start[0], end[1]-start[1]
                distance = max(math.hypot(dx,dy), 1e-6)
                ux, uy = dx/distance, dy/distance
                x, y = start[0]+dx*0.65, start[1]+dy*0.65
                length, width = min(0.014,distance*0.25), 0.009
                mesh(f"{root}/Arrow_{index}", [(x+ux*length,y+uy*length,z),
                     (x-ux*length-uy*width,y-uy*length+ux*width,z),
                     (x-ux*length+uy*width,y-uy*length-ux*width,z)], (0.95,0.58,0.06))
                line = UsdGeom.BasisCurves.Define(stage, f"{root}/Edge_{index}")
                line.CreateTypeAttr("linear")
                line.CreateCurveVertexCountsAttr([2])
                line.CreatePointsAttr([Gf.Vec3f(start[0],start[1],z),Gf.Vec3f(end[0],end[1],z)])
                line.CreateWidthsAttr([0.004])
                line.CreateDisplayColorAttr([Gf.Vec3f(0.95,0.58,0.06)])
