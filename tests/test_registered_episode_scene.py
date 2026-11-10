"""Registered scene contract checks without GPU or proprietary assets."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from geomind.episodes import ContractError
from geomind.episodes.physical_contract import validate_physics
from geomind.episodes.registered_scene import scene_description


class TestRegisteredScene(unittest.TestCase):
    def scene(self):
        tree = [{"name": "table_1", "attrs": {"translate": [0.25, 0, 0.4], "scale": [0.01]*3}}]
        with patch("geomind_lab.utils.scene_utils.parse_usda_scene"), \
             patch("geomind_lab.utils.scene_utils.extract_xform_tree", return_value=tree), \
             patch("geomind_lab.utils.asset_utils.resolve_tabletop_height", return_value=0.93):
            return scene_description(65)

    def test_scene_uses_asset_table_height_without_proxy_table(self):
        scene = self.scene()
        validate_physics(scene)
        self.assertEqual(scene["physics"]["frame_origin_w_m"], [0.5, 0, 0.93])
        self.assertEqual(scene["physics"]["robot_base_w_m"][2], 0.93)
        self.assertNotIn("table_size_m", scene["physics"])

    def test_invalid_registered_fields_are_rejected(self):
        scene = self.scene()
        for key, value in (("registered_scene_id", True), ("lighting_mode", "missing"), ("hide_basket", 1)):
            invalid = deepcopy(scene)
            invalid["physics"][key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_physics(invalid)


class TestRegisteredSourceWalls(unittest.TestCase):
    def test_mesh_and_cube_walls_keep_source_bindings_without_other_assets(self):
        try:
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
        except ImportError:
            self.skipTest("USD Python bindings are unavailable")
        from geomind.episodes.registered_scene import _spawn_source_walls
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.usda"
            source = Usd.Stage.CreateNew(str(path))
            world = UsdGeom.Xform.Define(source, "/World")
            world.AddTranslateOp().Set((0.1, 0.2, 0.0))
            room = UsdGeom.Xform.Define(source, "/World/Room")
            room.AddTranslateOp().Set((1, 2, 0))
            mesh = UsdGeom.Mesh.Define(source, "/World/Room/Wall")
            mesh.CreatePointsAttr([(-1, 0, 0), (1, 0, 0), (1, 0, 2), (-1, 0, 2)])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.AddTranslateOp().Set((0, 3, 0))
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            cube = UsdGeom.Cube.Define(source, "/World/Room/Wall_1")
            cube.CreateSizeAttr(3)
            material = UsdShade.Material.Define(source, "/World/Looks/Beige")
            shader = UsdShade.Shader.Define(source, "/World/Looks/Beige/Texture")
            shader.CreateIdAttr("UsdUVTexture")
            shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("wall.jpg"))
            shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            surface = UsdShade.Shader.Define(source, "/World/Looks/Beige/Surface")
            surface.CreateIdAttr("UsdPreviewSurface")
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(shader.ConnectableAPI(), "rgb")
            surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
            unused = UsdShade.Shader.Define(source, "/World/Looks/Beige/UnusedTexture")
            unused.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("unused.jpg"))
            (Path(directory) / "wall.jpg").write_bytes(b"texture fixture")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            UsdGeom.Cube.Define(source, "/World/Room/Table")
            source.DefinePrim("/World/Light", "DomeLight")
            source.GetRootLayer().Save()

            target = Usd.Stage.CreateInMemory()
            records = _spawn_source_walls(target, path)
            self.assertEqual(len(records), 2)
            wall_record, cube_record = records
            self.assertEqual(wall_record["geometry_type"], "Mesh")
            self.assertTrue(wall_record["active"])
            self.assertTrue(wall_record["collision_enabled"])
            self.assertEqual(wall_record["bound_material_path"], "/World/RegisteredScene/SourceWalls/Looks/Beige")
            self.assertEqual(len(wall_record["surface_textures"]), 1)
            self.assertEqual(wall_record["surface_textures"][0]["resolved_path"], str(Path(directory) / "wall.jpg"))
            self.assertTrue(wall_record["surface_textures"][0]["exists"])
            self.assertEqual(cube_record["geometry_type"], "Cube")
            self.assertFalse(cube_record["collision_enabled"])
            self.assertIsNone(cube_record["bound_material_path"])
            self.assertEqual(cube_record["surface_textures"], [])
            wall_path = "/World/RegisteredScene/SourceWalls/Room/Wall"
            actual = target.GetPrimAtPath(wall_path)
            self.assertEqual(UsdGeom.Mesh(actual).GetPointsAttr().Get(), mesh.GetPointsAttr().Get())
            self.assertEqual(UsdGeom.Xformable(actual).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                             mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            self.assertTrue(UsdPhysics.CollisionAPI(actual).GetCollisionEnabledAttr().Get())
            bound, _ = UsdShade.MaterialBindingAPI(actual).ComputeBoundMaterial()
            self.assertEqual(str(bound.GetPath()), "/World/RegisteredScene/SourceWalls/Looks/Beige")
            texture = target.GetPrimAtPath(str(bound.GetPath()) + "/Texture")
            self.assertEqual(UsdShade.Shader(texture).GetInput("file").Get().resolvedPath,
                             str(Path(directory) / "wall.jpg"))
            active = {str(prim.GetPath()) for prim in target.Traverse()}
            self.assertNotIn("/World/RegisteredScene/SourceWalls/Room/Table", active)
            self.assertNotIn("/World/RegisteredScene/SourceWalls/Light", active)
            self.assertEqual(UsdGeom.Cube(target.GetPrimAtPath(wall_path + "_1")).GetSizeAttr().Get(), 3)
