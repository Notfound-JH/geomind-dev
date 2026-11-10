"""Geometry contract checks for registered heterogeneous USD exports (CPU only)."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    HAS_USD = True
except ImportError:
    HAS_USD = False

from geomind.scene_assembly.registered_export import _primitive_points, export_registered, scale_runtime_xy
from geomind.scene_assembly.usd import digest


def _cube(name, xy, *, role="candidate", size=(.045, .045, .045), static=False, color=(.8, .1, .1)):
    return {"name": name, "kind": "primitive_cuboid", "size": list(size), "role": role,
            "visual_color": list(color), "pose": {"xy": list(xy), "yaw": 0.},
            "physics": {"rigid_body": {"rigid_body_enabled": not static}}}


@unittest.skipUnless(HAS_USD, "OpenUSD CPU libraries are unavailable")
class RegisteredExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        source = self.root / "source.usda"
        stage = Usd.Stage.CreateNew(str(source))
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.Xform.Define(stage, "/World/table_1")
        top = UsdGeom.Mesh.Define(stage, "/World/table_1/table_top")
        top.CreatePointsAttr([Gf.Vec3f(x, y, .7) for x, y in ((0., -1.), (1., -1.), (1., 1.), (0., 1.))])
        top.CreateFaceVertexCountsAttr([4]); top.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        stage.GetRootLayer().Save()
        self.scene = {"scene_id": 65, "scene_name": "test_table", "usd_path": str(source)}
        self.scene_patch = patch("geomind.scene_assembly.registered_export.resolve_scene", return_value=self.scene)
        self.scene_patch.start()

    def tearDown(self):
        self.scene_patch.stop()
        self.tmp.cleanup()

    def plan(self, objects, protocol=None):
        return {"schema_version": 1, "registration": {"task_id": "Example_v0", "annotation_layout_key": "test"},
                "task_spec": {"runtime_objects": objects, "protocol": protocol or {}}, "source_hashes": {}, "seed": 5,
                "asset_preserving": {"authored_runtime_assets_preserved": True, "object_mode_conversion_applied": False}}

    def test_heterogeneous_shapes_keep_separate_body_clouds_and_materials(self):
        objects = [_cube("red_cube", [.3, -.1], static=True),
                   {"name": "ball", "kind": "primitive_sphere", "radius": .03,
                    "visual_color": [.1, .1, .8], "pose": {"xy": [.5, .1], "yaw": 0.}}]
        plan = self.plan(objects); before = deepcopy(plan)
        manifest = export_registered(plan, self.root / "out", seed=5)
        self.assertEqual(plan, before)
        self.assertEqual(manifest["asset_preserving"], plan["asset_preserving"])
        self.assertEqual(manifest["provenance"]["asset_preserving"], plan["asset_preserving"])
        self.assertFalse(manifest["physics"]["executed"])
        self.assertFalse(manifest["physics"]["passed"])
        self.assertEqual(manifest["status"], "assembled_physics_unverified")
        self.assertFalse(manifest["objects"][0]["dynamic"])
        self.assertTrue(manifest["objects"][1]["dynamic"])
        self.assertEqual([obj["source_label"] for obj in manifest["objects"]], ["red_cube", "ball"])
        clouds = np.load(manifest["asset_body_vertices"], allow_pickle=False)
        self.assertNotEqual(clouds["points_m_000"].shape, clouds["points_m_001"].shape)
        np.testing.assert_array_equal(clouds["points_m"], clouds["points_m_000"])
        for obj in manifest["objects"]:
            self.assertAlmostEqual(obj["support"]["initial_bottom_gap_m"], .001, places=6)
            self.assertTrue(obj["support"]["tabletop_footprint_passed"])
            self.assertEqual(obj["scale"], [1., 1., 1.])
        reopened = Usd.Stage.Open(manifest["scene_usda"])
        self.assertTrue(UsdPhysics.RigidBodyAPI(reopened.GetPrimAtPath("/World/AssembledObjects/red_cube")).GetKinematicEnabledAttr().Get())
        self.assertTrue(manifest["provenance"]["used_usd_layers_sha256"])
        self.assertEqual(json.loads((self.root / "out" / "manifest.json").read_text())["request"]["object"]["count"], 2)

    def test_open_container_parts_are_not_convexified_together_or_attached(self):
        objects = [_cube("floor", [.43, .04], role="container_floor", size=(.17, .15, .012), static=True),
                   _cube("wall_left", [.351, .04], role="container_wall", size=(.012, .15, .07), static=True),
                   _cube("wall_right", [.509, .04], role="container_wall", size=(.012, .15, .07), static=True),
                   _cube("wall_front", [.43, .109], role="container_wall", size=(.146, .012, .07), static=True),
                   _cube("wall_back", [.43, -.029], role="container_wall", size=(.146, .012, .07), static=True),
                   _cube("movable", [.24, -.11], role="target")]
        protocol = {"variant": "place_inside", "pre_attached_obj": "movable", "target_object": {"object_id": "movable"},
                    "gt_spec": {"container_floor_object_id": "floor", "container_wall_object_ids": [obj["name"] for obj in objects[1:5]]}}
        manifest = export_registered(self.plan(objects, protocol), self.root / "box")
        self.assertEqual(len(manifest["objects"]), 6)
        self.assertFalse(manifest["initial_grasp"]["pre_attached"])
        self.assertTrue(manifest["initial_grasp"]["requires_real_pregrasp"])
        stage = Usd.Stage.Open(manifest["scene_usda"])
        self.assertEqual(len(stage.GetPrimAtPath("/World/AssembledObjects").GetChildren()), 6)
        self.assertFalse(any(prim.IsA(UsdPhysics.Joint) for prim in stage.Traverse()))
        self.assertTrue(all(not obj["dynamic"] for obj in manifest["objects"][:5]))

    def test_directed_chain_arrows_have_no_collision_or_private_name_text(self):
        objects = [_cube("left", [.4, -.12]), _cube("red", [.4, 0.]), _cube("right", [.4, .12])]
        protocol = {"gt_spec": {"resolver": "structural_adjacency", "order_axis": "y", "reference_object_id": "red",
                    "relation": "Predecessor", "structure": {"type": "DirectedChain", "nodes": ["left", "red", "right"], "draw_edges": True}}}
        manifest = export_registered(self.plan(objects, protocol), self.root / "chain")
        paths = manifest["geometry_validation"]["directed_chain_visual_paths"]
        self.assertEqual(len(paths), 4)
        stage = Usd.Stage.Open(manifest["scene_usda"])
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            self.assertTrue(prim)
            self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))

    def test_original_usd_scale_and_static_concave_collision_are_preserved(self):
        source = self.root / "asset.usda"
        stage = Usd.Stage.CreateNew(str(source))
        UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Asset"))
        mesh = UsdGeom.Mesh.Define(stage, "/Asset/Surface")
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in ((-1., -1., 0.), (1., -1., 0.), (1., 1., 0.), (-1., 1., 0.), (0., 0., 2.))])
        mesh.CreateFaceVertexCountsAttr([4, 3, 3, 3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4])
        stage.GetRootLayer().Save()
        before = digest(source)
        obj = {"name": "bowl", "kind": "usd", "usd_path": str(source), "asset_key": "bowl_1",
               "scale": [.03, .02, .015], "pose": {"xy": [.4, 0.], "yaw": 0.},
               "physics": {"rigid_body": {"rigid_body_enabled": False}, "collision": {"type": "static_mesh"}}}
        manifest = export_registered(self.plan([obj]), self.root / "usd")
        self.assertEqual(digest(source), before)
        exported = manifest["objects"][0]
        self.assertEqual(exported["scale"], [.03, .02, .015])
        self.assertEqual(exported["visual"]["method"], "original_usd_material")
        np.testing.assert_allclose(exported["dimensions_m"], [.06, .04, .03], atol=1e-8)
        stage = Usd.Stage.Open(manifest["scene_usda"])
        collider = stage.GetPrimAtPath("/World/AssembledObjects/bowl/Surface")
        self.assertEqual(UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get(), "none")
        self.assertEqual(manifest["provenance"]["asset_sha256"][str(source)], before)

    def test_registered_grid_uses_one_seeded_sample_and_exports_the_guarded_copies(self):
        from geomind_lab.tasks.manager_based.manipulation.geomind import layout_randomizer
        from geomind.scene_assembly.registered_layout_constraints import validate_grid_layout
        runtime = [_cube(f"cell_{i}", [x, y], size=(.0276, .0276, .027))
                   for i, (x, y) in enumerate(( (x,y) for x in (.37,.5) for y in (-.15,-.03,.09) ))]
        protocol = {"intervention": {"script_name": "static_scene_noop"}, "gt_spec": {
            "resolver": "grid_rowcol_slot", "grid_num_rows": 2, "grid_num_cols": 3,
            "row_axis": "x", "col_axis": "y", "grid_quant_step": .028,
            "target_row": 1, "target_col": 2, "min_center_separation": .024}}
        plan = self.plan(runtime, protocol)
        plan["task_spec"]["scene_adaptation"] = {"runtime_objects": deepcopy(runtime)}
        original = deepcopy(plan)
        with patch.object(layout_randomizer, "randomize_runtime_object_layout",
                          wraps=layout_randomizer.randomize_runtime_object_layout) as randomizer:
            manifest = export_registered(plan, self.root / "grid", seed=0)
        self.assertEqual(randomizer.call_count, 1)
        self.assertEqual(plan, original)
        spec = manifest["registered_task_spec"]
        checked = validate_grid_layout(spec["runtime_objects"], spec["protocol"])
        self.assertTrue(checked["passed"])
        self.assertEqual(checked["slots"]["cell_1"], {"row": 1, "column": 2})
        constraint = manifest["registered_layout_constraints"][0]
        self.assertEqual(constraint["randomizer_additional_calls"], 0)
        self.assertFalse(constraint["seed_changed"])
        self.assertFalse(constraint["gt_rule_changed"])
        self.assertEqual(manifest["request"]["seed"], 0)
        self.assertEqual(manifest["layout_randomization"]["post_sampling_grid_constraint"], constraint)
        descriptors = {obj["name"]: obj for obj in spec["runtime_objects"]}
        for obj in manifest["objects"]:
            np.testing.assert_allclose(obj["position_w_m"][:2], descriptors[obj["name"]]["pose"]["xy"], atol=1e-12)
            self.assertEqual(obj["runtime_descriptor"], descriptors[obj["name"]])
        self.assertEqual(spec["scene_adaptation"]["runtime_objects"], spec["runtime_objects"])

    def test_explicit_layout_scale_exports_fitted_runtime_and_records_one_rng_call(self):
        from geomind_lab.tasks.manager_based.manipulation.geomind import layout_randomizer
        plan = self.plan([_cube("left", [.3, -.1]), _cube("right", [.47, .1])])
        plan["task_spec"]["scene_adaptation"] = {"runtime_objects": deepcopy(plan["task_spec"]["runtime_objects"])}
        before = deepcopy(plan)
        with patch.object(layout_randomizer, "randomize_runtime_object_layout",
                          wraps=layout_randomizer.randomize_runtime_object_layout) as randomizer:
            manifest = export_registered(plan, self.root / "scaled", seed=0, layout_scale=2.5)
        self.assertEqual(randomizer.call_count, 1)
        self.assertEqual(plan, before)
        record = manifest["agent_layout_transform"]
        self.assertEqual(record["layout_scale"], 2.5)
        self.assertEqual(record["randomizer_additional_calls"], 0)
        self.assertFalse(record["seed_changed"])
        self.assertFalse(record["compression_to_fit_used"])
        self.assertTrue(record["reason"])
        self.assertEqual(manifest["request"]["layout"]["agent_scale"], 2.5)
        saved = json.loads((self.root / "scaled/registered_plan.json").read_text())
        self.assertEqual(saved["agent_layout_transform"], record)
        runtime = manifest["registered_task_spec"]["runtime_objects"]
        self.assertEqual(manifest["registered_task_spec"]["scene_adaptation"]["runtime_objects"], runtime)
        for obj, desc in zip(manifest["objects"], runtime):
            np.testing.assert_allclose(obj["position_w_m"][:2], desc["pose"]["xy"], atol=1e-12)
            self.assertTrue(obj["support"]["tabletop_footprint_passed"])
            self.assertAlmostEqual(obj["support"]["initial_bottom_gap_m"], .001, places=6)
            self.assertEqual(desc["size"], [.045, .045, .045])
        self.assertFalse(manifest["physics"]["executed"])

    def test_default_and_explicit_scale_one_keep_the_existing_export_path(self):
        plan = self.plan([_cube("left", [.3, -.1]), _cube("right", [.47, .1])])
        default = export_registered(plan, self.root / "default", seed=11)
        explicit = export_registered(plan, self.root / "one", seed=11, layout_scale=1.)
        self.assertEqual(default["objects"], explicit["objects"])
        self.assertEqual(default["layout_randomization"], explicit["layout_randomization"])
        self.assertNotIn("agent_layout_transform", default)
        self.assertNotIn("agent_layout_transform", explicit)
        self.assertNotIn("agent_scale", default["request"]["layout"])

    def test_containment_scaling_is_rejected_before_randomization_or_output(self):
        from geomind_lab.tasks.manager_based.manipulation.geomind import layout_randomizer
        for index, (role, protocol) in enumerate([
                ("container_floor", {}), ("container_wall", {}),
                ("reference", {"gt_spec": {"resolver": "containment_placement"}})]):
            destination = self.root / f"container_scale_{index}"
            with patch.object(layout_randomizer, "randomize_runtime_object_layout",
                              side_effect=AssertionError("must not sample")) as randomizer:
                with self.assertRaisesRegex(ValueError, "unsupported for containment"):
                    export_registered(self.plan([_cube("reference", [.4, 0.], role=role)], protocol),
                                      destination, layout_scale=2.5)
            self.assertEqual(randomizer.call_count, 0)
            self.assertFalse(destination.exists())

    def test_no_silent_asset_substitution_overwrite_or_bad_geometry(self):
        plan = self.plan([_cube("cube", [.4, 0.])])
        export_registered(plan, self.root / "once")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            export_registered(plan, self.root / "once")
        with self.assertRaisesRegex(ValueError, "Unsupported runtime object kind"):
            export_registered(self.plan([{"name": "unknown", "kind": "mesh_guess"}]), self.root / "bad")
        with self.assertRaisesRegex(ValueError, "unique USD identifiers"):
            export_registered(self.plan([_cube("same", [.3, 0.]), _cube("same", [.5, 0.])]), self.root / "duplicate")
        with self.assertRaisesRegex(ValueError, "dimensions must be positive"):
            export_registered(self.plan([_cube("cube", [.4, 0.], size=(.1, .1, -.1))]), self.root / "negative")


class UniformLayoutScaleTest(unittest.TestCase):
    def test_scale_one_is_a_true_noop_without_input_conversion(self):
        objects = [_cube("a", [.25, -.15]), _cube("b", [.5, .1])]
        actual, record = scale_runtime_xy(objects, 1., [.15, .68, -.32, .34])
        self.assertIs(actual, objects)
        self.assertIsNone(record)

    def test_uniform_positive_scale_preserves_order_z_assets_and_minimally_translates(self):
        objects = [_cube("a", [.25, -.15]), _cube("b", [.4, 0.]), _cube("c", [.5, .1])]
        for obj in objects:
            obj["pose"]["z_policy"] = "table_relative"
            obj["pose"]["z"] = .03
        baseline = deepcopy(objects)
        actual, record = scale_runtime_xy(objects, 2., [.15, .68, -.32, .34])
        self.assertEqual(objects, baseline)
        np.testing.assert_allclose(record["source_xy_bbox_center_m"], [.375, -.025])
        np.testing.assert_allclose(record["minimum_translation_xy_m"], [.025, 0.], atol=1e-12)
        before = np.asarray([obj["pose"]["xy"] for obj in objects])
        after = np.asarray([obj["pose"]["xy"] for obj in actual])
        np.testing.assert_allclose(np.diff(after, axis=0), 2 * np.diff(before, axis=0), atol=1e-12)
        self.assertTrue(np.all(np.diff(after, axis=0) > 0))
        for old, new in zip(objects, actual):
            old = deepcopy(old); new = deepcopy(new)
            old["pose"].pop("xy"); new["pose"].pop("xy")
            self.assertEqual(old, new)

    def test_container_scale_one_is_noop_and_nondefault_scale_is_rejected(self):
        objects = [_cube("floor", [.4, 0.], role="container_floor")]
        unchanged, record = scale_runtime_xy(objects, 1., [.15, .68, -.32, .34])
        self.assertIs(unchanged, objects)
        self.assertIsNone(record)
        with self.assertRaisesRegex(ValueError, "unsupported for containment"):
            scale_runtime_xy(objects, 2.5, [.15, .68, -.32, .34])
        with self.assertRaisesRegex(ValueError, "unsupported for containment"):
            scale_runtime_xy([_cube("bowl", [.4, 0.])], 2.5, [.15, .68, -.32, .34],
                             protocol={"gt_spec": {"resolver": "containment_placement"}})

    def test_span_that_cannot_fit_is_rejected_without_shrinking_or_resampling(self):
        objects = [_cube("a", [.2, 0.]), _cube("b", [.6, .1])]
        with patch("geomind_lab.tasks.manager_based.manipulation.geomind.layout_randomizer.randomize_runtime_object_layout",
                   side_effect=AssertionError("must not sample")):
            with self.assertRaisesRegex(ValueError, "does not fit.*refusing compression or resampling"):
                scale_runtime_xy(objects, 2.5, [.15, .68, -.32, .34])
        for factor in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite positive"):
                scale_runtime_xy(objects, factor, [.15, .68, -.32, .34])

class PrimitivePointsTest(unittest.TestCase):
    def test_cylinder_and_cone_keep_authored_vertical_dimensions(self):
        for kind in ("primitive_cylinder", "primitive_cone"):
            points = _primitive_points({"kind": kind, "radius": .03, "height": .12})
            np.testing.assert_allclose(points.min(axis=0), [-.03, -.03, -.06])
            np.testing.assert_allclose(points.max(axis=0), [.03, .03, .06])


if __name__ == "__main__":
    unittest.main()
