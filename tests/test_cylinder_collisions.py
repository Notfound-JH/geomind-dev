"""Runtime cylinder colliders keep hash-bound visual geometry and material rules."""
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from geomind.scene_assembly.cylinder_collisions import install_cylinder_mesh_colliders


class CylinderCollisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='geomind-cylinder-test-')
        self.addCleanup(self.temp.cleanup)

    def fixture(self, scale=(1., 1., 1.), purpose='physics', cloud_change=None):
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, '/World/Obj')
        root.AddTranslateOp().Set(Gf.Vec3d(.31, -.12, .97))
        root.AddOrientOp().Set(Gf.Quatf(math.cos(.23), Gf.Vec3f(0., 0., math.sin(.23))))
        root.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        cylinder = UsdGeom.Cylinder.Define(stage, '/World/Obj/Geometry')
        cylinder.CreateAxisAttr('Z')
        cylinder.CreateRadiusAttr(.026)
        cylinder.CreateHeightAttr(.07)
        cylinder.CreateExtentAttr([Gf.Vec3f(-.026, -.026, -.035), Gf.Vec3f(.026, .026, .035)])
        cylinder.CreateDisplayColorAttr([Gf.Vec3f(.1, .3, .8)])
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim()).CreateCollisionEnabledAttr(True)
        cylinder.GetPrim().CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.002)
        cylinder.GetPrim().CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(.0001)
        material = UsdShade.Material.Define(stage, '/World/Materials/Surface')
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(.9)
        UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim()).Bind(
            material, materialPurpose='physics' if purpose == 'physics' else '')
        angle = np.arange(128) * (2 * math.pi / 128)
        xy = np.column_stack((.026 * np.cos(angle), .026 * np.sin(angle)))
        cloud = np.vstack((np.column_stack((xy, np.full(128, -.035))),
                           np.column_stack((xy, np.full(128, .035))))) * np.asarray(scale)
        if cloud_change is not None:
            cloud = cloud_change(cloud.copy())
        path = Path(self.temp.name) / ('cloud_%d.npz' % len(list(Path(self.temp.name).iterdir())))
        np.savez(path, points_m_000=cloud)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {'asset_body_vertices': str(path),
                    'provenance': {'asset_body_vertices_sha256': digest},
                    'objects': [{'prim_path': '/World/Obj', 'kind': 'primitive_cylinder',
                                 'body_points_key': 'points_m_000', 'scale': list(scale)}]}
        return stage, cylinder, manifest, {'/World/Obj': cloud}

    @staticmethod
    def visual_state(prim):
        return ({a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                 if a.GetName() != 'physics:collisionEnabled'},
                {r.GetName(): [str(v) for v in r.GetTargets()] for r in prim.GetRelationships()})

    def assert_mesh_geometry(self, stage, manifest, clouds, item):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(item['collider_prim_path']))
        self.assertTrue(mesh)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = list(mesh.GetFaceVertexCountsAttr().Get())
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        self.assertEqual(vertices.shape, (64, 3))
        self.assertEqual(counts, [32, 32] + [4] * 32)
        self.assertEqual(len(indices), 192)
        self.assertEqual((item['vertex_count'], item['face_count']), (64, 34))
        self.assertEqual(item['polygon_sides'], 32)
        self.assertEqual(item['sampled_angle_stride'], 4)
        np.testing.assert_allclose(item['vertices_local_m'], vertices, atol=1e-8, rtol=0)
        self.assertEqual(item['face_vertex_counts'], counts)
        self.assertEqual(item['face_vertex_indices'], indices)
        scale = np.asarray(manifest['objects'][0]['scale'])
        source = clouds['/World/Obj'][::4]
        np.testing.assert_allclose(vertices * scale, source, atol=1e-8, rtol=0)
        # Compare the composed world geometry as well as the local numbers.
        cache = UsdGeom.XformCache()
        transform = cache.GetLocalToWorldTransform(mesh.GetPrim())
        actual_world = np.asarray([transform.Transform(Gf.Vec3d(*v)) for v in vertices])
        root_transform = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Obj'))
        expected_world = np.asarray([root_transform.Transform(Gf.Vec3d(*(v / scale))) for v in source])
        np.testing.assert_allclose(actual_world, expected_world, atol=1e-8, rtol=0)
        # A closed manifold has each undirected edge twice, once each direction.
        edges = {}
        cursor = 0
        for count in counts:
            face = indices[cursor:cursor + count]
            cursor += count
            face_vertices = vertices[face]
            normal = np.cross(face_vertices, np.roll(face_vertices, -1, axis=0)).sum(axis=0)
            self.assertGreater(float(normal @ face_vertices.mean(axis=0)), 0.)
            for first, second in zip(face, face[1:] + face[:1]):
                edges.setdefault(tuple(sorted((first, second))), []).append((first, second))
        self.assertTrue(edges)
        self.assertTrue(all(len(v) == 2 and v[0] == v[1][::-1] for v in edges.values()))
        self.assertEqual(mesh.GetVisibilityAttr().Get(), 'invisible')
        self.assertEqual(mesh.GetPurposeAttr().Get(), 'guide')
        self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get(), 'convexHull')
        self.assertTrue(UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get())
        self.assertAlmostEqual(item['approximation_error_bound_m'],
                               .026 * max(scale[:2]) * (1 - math.cos(math.pi / 32)), places=12)
        self.assertEqual(item['source_body_archive_sha256'], manifest['provenance']['asset_body_vertices_sha256'])
        self.assertEqual(item['body_cloud_sha256'],
                         hashlib.sha256(np.asarray(clouds['/World/Obj'], dtype=np.float64).tobytes()).hexdigest())

    def test_closed_outward_collision_mesh_preserves_source_visuals_material_and_offsets(self):
        for scale, purpose in [((1., 1., 1.), 'physics'), ((2., 1.5, .8), 'allPurpose')]:
            with self.subTest(scale=scale, purpose=purpose):
                stage, cylinder, manifest, clouds = self.fixture(scale, purpose)
                original = self.visual_state(cylinder.GetPrim())
                result = install_cylinder_mesh_colliders(stage, manifest, clouds)
                self.assertTrue(result['applied'])
                self.assertTrue(result['runtime_only'])
                self.assertFalse(result['source_usd_files_modified'])
                self.assertEqual(len(result['colliders']), 1)
                item = result['colliders'][0]
                self.assertEqual(item['body_prim_path'], '/World/Obj')
                self.assertEqual(item['original_prim_path'], '/World/Obj/Geometry')
                self.assertEqual(item['collider_prim_path'], '/World/Obj/RuntimeCylinderCollision')
                self.assertFalse(UsdPhysics.CollisionAPI(cylinder.GetPrim()).GetCollisionEnabledAttr().Get())
                self.assertEqual(self.visual_state(cylinder.GetPrim()), original)
                self.assert_mesh_geometry(stage, manifest, clouds, item)
                mesh_prim = stage.GetPrimAtPath(item['collider_prim_path'])
                relationship = 'material:binding' + (':physics' if purpose == 'physics' else '')
                self.assertEqual(mesh_prim.GetRelationship(relationship).GetTargets(), [Sdf.Path('/World/Materials/Surface')])
                self.assertEqual(item['material_bindings'][relationship], ['/World/Materials/Surface'])
                for key, value in [('contactOffset', .002), ('restOffset', .0001)]:
                    self.assertAlmostEqual(mesh_prim.GetAttribute('physxCollision:' + key).Get(), value, places=8)
                self.assertAlmostEqual(item['contact_offset_m'], .002, places=8)
                self.assertAlmostEqual(item['rest_offset_m'], .0001, places=8)
                # Cylinder and polygon have the same extremal X/Y/Z samples.
                points = np.asarray(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()) * np.asarray(scale)
                source_bounds = np.asarray([[-.026, -.026, -.035], [.026, .026, .035]]) * np.asarray(scale)
                np.testing.assert_allclose([points.min(axis=0), points.max(axis=0)], source_bounds, atol=1e-8, rtol=0)

    def test_malformed_or_reordered_archive_cloud_is_rejected_without_stage_mutation(self):
        changes = [lambda p: p[:-1], lambda p: p[::-1],
                   lambda p: np.concatenate((p[128:], p[:128])),
                   lambda p: p + np.where(np.arange(len(p))[:, None] == 7, [0., .001, 0.], 0.)]
        for change in changes:
            with self.subTest(change=change):
                stage, _, manifest, clouds = self.fixture(cloud_change=change)
                before = stage.GetRootLayer().ExportToString()
                with self.assertRaises(ValueError):
                    install_cylinder_mesh_colliders(stage, manifest, clouds)
                self.assertEqual(stage.GetRootLayer().ExportToString(), before)

    def test_archive_hash_or_supplied_cloud_mismatch_is_rejected_without_mutation(self):
        for failure in ('archive_hash', 'supplied_cloud'):
            with self.subTest(failure=failure):
                stage, _, manifest, clouds = self.fixture()
                if failure == 'archive_hash':
                    with open(manifest['asset_body_vertices'], 'ab') as stream:
                        stream.write(b'changed archive provenance')
                else:
                    clouds['/World/Obj'] = clouds['/World/Obj'].copy()
                    clouds['/World/Obj'][1, 0] += .00001
                before = stage.GetRootLayer().ExportToString()
                with self.assertRaises(ValueError):
                    install_cylinder_mesh_colliders(stage, manifest, clouds)
                self.assertEqual(stage.GetRootLayer().ExportToString(), before)

    def test_all_objects_are_validated_before_first_collider_is_applied(self):
        stage, _, manifest, clouds = self.fixture()
        # A valid first object must not be changed when a later object is invalid.
        second = UsdGeom.Xform.Define(stage, '/World/Other')
        UsdPhysics.RigidBodyAPI.Apply(second.GetPrim())
        cylinder = UsdGeom.Cylinder.Define(stage, '/World/Other/Geometry')
        cylinder.CreateAxisAttr('Z')
        cylinder.CreateRadiusAttr(.026)
        cylinder.CreateHeightAttr(.07)
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim()).CreateCollisionEnabledAttr(True)
        manifest['objects'].append(dict(manifest['objects'][0], prim_path='/World/Other'))
        clouds['/World/Other'] = clouds['/World/Obj'].copy()
        clouds['/World/Other'][0, 0] += .01
        before = stage.GetRootLayer().ExportToString()
        with self.assertRaises(ValueError):
            install_cylinder_mesh_colliders(stage, manifest, clouds)
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)

    def test_missing_offsets_remain_unauthored_and_non_cylinders_are_untouched(self):
        stage, cylinder, manifest, clouds = self.fixture()
        cylinder.GetPrim().RemoveProperty('physxCollision:contactOffset')
        cylinder.GetPrim().RemoveProperty('physxCollision:restOffset')
        result = install_cylinder_mesh_colliders(stage, manifest, clouds)
        item = result['colliders'][0]
        mesh_prim = stage.GetPrimAtPath(item['collider_prim_path'])
        self.assertIsNone(item['contact_offset_m'])
        self.assertIsNone(item['rest_offset_m'])
        self.assertFalse(mesh_prim.GetAttribute('physxCollision:contactOffset').HasAuthoredValueOpinion())
        self.assertFalse(mesh_prim.GetAttribute('physxCollision:restOffset').HasAuthoredValueOpinion())
        stage, _, manifest, clouds = self.fixture()
        manifest['objects'][0]['kind'] = 'primitive_cuboid'
        before = stage.GetRootLayer().ExportToString()
        result = install_cylinder_mesh_colliders(stage, manifest, clouds)
        self.assertFalse(result['applied'])
        self.assertEqual(result['colliders'], [])
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)


if __name__ == '__main__':
    unittest.main()
