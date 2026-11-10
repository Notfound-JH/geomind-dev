import omni
from omni.physx.scripts import utils
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

def prim_has_collision(prim: Usd.Prim) -> bool:
    # 1) Collision API already applied
    if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        return True

    # 2) Common collision attrs already authored
    a = prim.GetAttribute("physics:collisionEnabled")
    if a and a.IsAuthored():
        return True

    b = prim.GetAttribute("physxCollision:collisionApproximation")
    if b and b.IsAuthored():
        return True

    return False

def add_collision_to_stage():
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/")

    for prim in Usd.PrimRange(root):
        # Only touch geometry prims
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Gprim)):
            continue

        # IMPORTANT: don't override existing collision
        if prim_has_collision(prim):
            continue

        if prim.IsA(UsdGeom.Mesh):
            utils.setCollider(prim, approximationShape="none")
        elif (
            prim.IsA(UsdGeom.Cylinder)
            or prim.IsA(UsdGeom.Capsule)
            or prim.IsA(UsdGeom.Cone)
            or prim.IsA(UsdGeom.Sphere)
            or prim.IsA(UsdGeom.Cube)
        ):
            utils.setCollider(prim, approximationShape="convexHull")

    return
