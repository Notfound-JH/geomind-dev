"""Synthetic USD geometry regressions; skip without a local pxr installation.

These tests open in-memory USD stages only. They do not start Isaac Sim, access
the real room/asset libraries, or execute PhysX.
"""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Gf = Usd = UsdGeom = None

if Usd is not None:
    from geomind.scene_assembly.usd import _body_points, _bounds, _positions_have_clearance, _set_transform, _world_yaw_orientation, measure_table, supported_xy


@unittest.skipIf(Usd is None, "pxr is not installed in this CPU environment")
class TestAssemblyUSDGeometry(unittest.TestCase):
    def test_diagonal_ring_uses_its_circular_clearance_contract(self):
        from geomind.scene_assembly.layouts import generate_positions
        positions=generate_positions({'type':'ring'},4,(-.5,.5,-.5,.5),(.05,.05))
        self.assertTrue(_positions_have_clearance(positions,(.05,.05),circular_envelope=True))
        # A circle-safe diagonal gap need not have 2.5cm separation on X or Y.
        self.assertFalse(_positions_have_clearance(positions,(.05,.05)))
        compressed=[[x*.8,y*.8] for x,y in positions]
        self.assertFalse(_positions_have_clearance(compressed,(.05,.05),circular_envelope=True))

    def test_yaw_composes_about_world_z_without_tipping_catalog_pose(self):
        # Start from a noncommuting Y rotation: multiplying in the wrong order
        # would tip the upright axis instead of preserving world-Z heights.
        base=[math.sqrt(.5),0,math.sqrt(.5),0]
        points=np.asarray([[.01,.03,.07],[-.1,.2,.4]])
        base_points=points@self.rotation(base)
        result=_world_yaw_orientation(base,37)
        actual=points@self.rotation(result)
        yaw=[math.cos(math.radians(37)/2),0,0,math.sin(math.radians(37)/2)]
        np.testing.assert_allclose(actual,base_points@self.rotation(yaw),atol=1e-12)
        np.testing.assert_allclose(actual[:,2],base_points[:,2],atol=1e-12)

    def stage(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/table_1")
        return stage

    def planes(self, stage, path, heights, bounds=(-1.0, 1.0, -0.5, 0.5)):
        xmin, xmax, ymin, ymax = bounds
        points = []
        for height in heights:
            points.extend(((xmin, ymin, height), (xmax, ymin, height), (xmax, ymax, height), (xmin, ymax, height)))
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * len(heights))
        mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
        return mesh.GetPrim()

    def box(self, stage, path, low, high):
        points = [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
        # All six faces are represented; winding does not affect measured AABBs.
        faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
        return mesh.GetPrim(), np.asarray(points, dtype=float)

    def rotation(self, quat):
        return np.asarray(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(float(quat[0]), Gf.Vec3d(*[float(v) for v in quat[1:]])))))

    def test_top_survives_merged_bottom_planes_with_double_area(self):
        stage = self.stage()
        # Two very close underside planes fall into the same 3 mm height group.
        # Their accumulated area must not cause the smaller true top to vanish.
        self.planes(stage, "/World/table_1/table_top", (0.560, 0.561, 0.595))
        table = measure_table(stage)
        self.assertAlmostEqual(table["height_m"], 0.595, places=6)
        self.assertAlmostEqual(table["surface_area_m2"], 2.0)
        np.testing.assert_allclose(table["bounds_xy_m"], [-1, 1, -0.5, 0.5])

    def test_monitor_aabb_top_does_not_become_tabletop(self):
        stage = self.stage()
        # No tabletop naming hint: a much smaller monitor surface remains above
        # the table, making table-root AABB height unusable as a support height.
        self.planes(stage, "/World/table_1/board", (0.595,))
        self.box(stage, "/World/table_1/monitor", (-0.2, -0.05, 0.60), (0.2, 0.05, 1.50))
        _, overall_high = _bounds(stage, stage.GetPrimAtPath("/World/table_1"))
        self.assertAlmostEqual(overall_high[2], 1.5)
        table = measure_table(stage)
        self.assertAlmostEqual(table["height_m"], 0.595, places=6)
        self.assertEqual(table["mesh_paths"], ["/World/table_1/board"])

    def test_explicit_native_table_path_measures_renamed_reference(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, '/World/RegisteredScene/furniture_table')
        self.planes(stage, '/World/RegisteredScene/furniture_table/table_top', (.93285,))
        table = measure_table(stage, '/World/RegisteredScene/furniture_table')
        self.assertAlmostEqual(table['height_m'], .93285, places=6)
        self.assertEqual(table['prim_path'], '/World/RegisteredScene/furniture_table')
        with self.assertRaisesRegex(ValueError, 'Invalid explicit table'):
            measure_table(stage, '/World/Missing')

    def test_named_tabletop_is_preferred_over_other_horizontal_furniture(self):
        stage = self.stage()
        self.planes(stage, "/World/table_1/table_top", (0.595,))
        self.planes(stage, "/World/table_1/overhead_panel", (1.5,))
        table = measure_table(stage)
        self.assertAlmostEqual(table["height_m"], 0.595, places=6)
        self.assertEqual(table["mesh_paths"], ["/World/table_1/table_top"])

    def test_support_samples_reject_corner_past_real_edge(self):
        stage = self.stage()
        self.planes(stage, "/World/table_1/table_top", (0.6,))
        table = measure_table(stage)
        self.assertTrue(supported_xy(table, (0.8, 0.0), (0.1, 0.2)))
        self.assertFalse(supported_xy(table, (0.95, 0.0), (0.1, 0.2)))
        self.assertFalse(supported_xy(table, (0.0, 0.45), (0.1, 0.1)))

    def test_support_samples_use_mesh_instead_of_its_rectangle(self):
        stage = self.stage()
        mesh = UsdGeom.Mesh.Define(stage, "/World/table_1/table_top")
        mesh.CreatePointsAttr([(0, 0, 0.6), (1, 0, 0.6), (0, 1, 0.6)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        table = measure_table(stage)
        self.assertTrue(supported_xy(table, (0.2, 0.2), (0.1, 0.1)))
        # Every footprint corner is within the mesh AABB, but some lie beyond
        # the actual diagonal table edge x+y=1.
        self.assertFalse(supported_xy(table, (0.6, 0.3), (0.15, 0.15)))

    def test_body_points_preserve_scale_and_update_bottom_after_rotation(self):
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
        _, local_points = self.box(stage, "/Object/mesh", (0.02, -0.03, 0.01), (0.12, 0.17, 0.41))
        scale = [1.0, 2.0, 0.5]
        angle = math.radians(27)
        initial_quat = [math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]
        _set_transform(root, scale, initial_quat, [0, 0, 0])
        frame_points = _body_points(stage, root, initial_quat)
        np.testing.assert_allclose(frame_points, local_points * scale, atol=1e-7)
        initial_low, _ = _bounds(stage, root)

        # Rotate 90 degrees about Y. Actual bottom now depends on the object's
        # X extent, and cannot be computed using initial min-z + delta-z.
        current_quat = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]
        current_pos = np.asarray([0.3, -0.4, 0.8])
        measured_points = frame_points @ self.rotation(current_quat) + current_pos
        _set_transform(root, scale, current_quat, current_pos.tolist())
        actual_low, actual_high = _bounds(stage, root)
        np.testing.assert_allclose(measured_points.min(axis=0), actual_low, atol=1e-7)
        np.testing.assert_allclose(measured_points.max(axis=0), actual_high, atol=1e-7)
        self.assertAlmostEqual(float(measured_points[:, 2].min()), 0.68, places=6)
        self.assertGreater(abs(initial_low[2] + current_pos[2] - actual_low[2]), 0.05)

    def test_canonical_trs_retains_half_scale_when_saved_and_reopened(self):
        with tempfile.TemporaryDirectory() as directory:
            asset_path = Path(directory) / "asset.usda"
            output_path = Path(directory) / "assembled.usda"
            asset = Usd.Stage.CreateNew(str(asset_path))
            asset_root = UsdGeom.Xform.Define(asset, "/Root").GetPrim()
            asset.SetDefaultPrim(asset_root)
            self.box(asset, "/Root/mesh", (-0.1, -0.2, -0.3), (0.1, 0.2, 0.3))
            # A referenced root may already carry a scale. Catalog transforms
            # must override it with an explicit reusable canonical scale op.
            UsdGeom.Xformable(asset_root).AddScaleOp().Set(Gf.Vec3f(2, 2, 2))
            asset.GetRootLayer().Save()

            stage = Usd.Stage.CreateNew(str(output_path))
            root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
            self.assertTrue(root.GetReferences().AddReference(str(asset_path)))
            _set_transform(root, [0.5] * 3, [1, 0, 0, 0], [0.2, 0.3, 0.6])
            # Updating a pose a second time must reuse the scale op and keep
            # its value, rather than hiding scale inside a matrix transform.
            _set_transform(root, [0.5] * 3, [1, 0, 0, 0], [0.2, 0.3, 0.6])
            stage.GetRootLayer().Save()

            reopened = Usd.Stage.Open(str(output_path))
            saved = reopened.GetPrimAtPath("/Object")
            self.assertEqual(list(saved.GetAttribute("xformOpOrder").Get()), ["xformOp:translate", "xformOp:orient", "xformOp:scale"])
            np.testing.assert_allclose(saved.GetAttribute("xformOp:scale").Get(), [0.5] * 3)
            low, high = _bounds(reopened, saved)
            np.testing.assert_allclose(low, [0.15, 0.2, 0.45], atol=1e-7)
            np.testing.assert_allclose(high, [0.25, 0.4, 0.75], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
