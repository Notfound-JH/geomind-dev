import unittest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
from geomind.scene_assembly.visibility import apply_scene_visibility


class SceneVisibilityTest(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        self.wall = UsdGeom.Cube.Define(self.stage, '/World/function/Wall')
        self.wall.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
        self.wall.CreateSizeAttr(5)
        UsdPhysics.CollisionAPI.Apply(self.wall.GetPrim()).CreateCollisionEnabledAttr(True)
        material = UsdShade.Material.Define(self.stage, '/World/Looks/WallCandidate')
        UsdShade.MaterialBindingAPI.Apply(self.wall.GetPrim()).Bind(material)
        UsdGeom.Cube.Define(self.stage, '/World/Ceiling')
        UsdGeom.Cube.Define(self.stage, '/World/AssembledObjects/block')
        self.stage.SetEditTarget(self.stage.GetSessionLayer())

    def test_default_retains_previous_cutaway(self):
        record = apply_scene_visibility(self.stage, {})
        self.assertEqual(self.wall.ComputeVisibility(), 'invisible')
        self.assertIn('/World/Ceiling', record['hidden_prims'])
        self.assertFalse(record['visible_wall_prims'])

    def test_visible_walls_preserve_geometry_material_collision_and_source(self):
        source = self.stage.GetRootLayer().ExportToString()
        self.wall.MakeInvisible()
        record = apply_scene_visibility(self.stage, {'visualization': {'walls': 'visible'}})
        self.assertEqual(self.wall.ComputeVisibility(), 'inherited')
        self.assertIn('/World/function/Wall', record['visible_wall_prims'])
        self.assertEqual(self.wall.GetSizeAttr().Get(), 5)
        self.assertEqual(self.wall.GetOrderedXformOps()[0].Get(), Gf.Vec3d(1, 2, 3))
        self.assertTrue(UsdPhysics.CollisionAPI(self.wall).GetCollisionEnabledAttr().Get())
        self.assertEqual(str(UsdShade.MaterialBindingAPI(self.wall).ComputeBoundMaterial()[0].GetPath()), '/World/Looks/WallCandidate')
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), source)
        self.assertEqual(UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/AssembledObjects/block')).ComputeVisibility(), 'inherited')

    def test_invalid_policy_is_rejected_without_writes(self):
        for policy in ({'walls':'yes'}, {'wall':'visible'}, [], None):
            before = self.stage.GetSessionLayer().ExportToString()
            with self.assertRaises(ValueError):
                apply_scene_visibility(self.stage, {'visualization':policy})
            self.assertEqual(self.stage.GetSessionLayer().ExportToString(), before)


if __name__ == '__main__':
    unittest.main()
