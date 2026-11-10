"""Static target-support regressions; no physics is executed by these fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pxr import Usd,UsdGeom

from geomind.scene_assembly.task_pipeline import check_goal_geometry


class AssembledTaskGeometryTests(unittest.TestCase):
    def fixture(self, root, fixture=False):
        path=root/'scene.usda';stage=Usd.Stage.CreateNew(str(path))
        UsdGeom.Xform.Define(stage,'/World/function/table_1')
        mesh=UsdGeom.Mesh.Define(stage,'/World/function/table_1/table_top')
        mesh.CreatePointsAttr([[-.5,-.5,.6],[.5,-.5,.6],[.5,.5,.6],[-.5,.5,.6]])
        mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
        if fixture:
            cube=UsdGeom.Cube.Define(stage,'/World/decoration/fixture_1');cube.CreateSizeAttr(.1)
            UsdGeom.Xformable(cube).AddTranslateOp().Set((0,0,.65))
        stage.GetRootLayer().Save()
        cloud=np.asarray([[x,y,z] for x in [-.02,.02] for y in [-.02,.02] for z in [-.03,.03]])
        np.savez_compressed(root/'asset_body_vertices.npz',points_m=cloud)
        states=np.zeros((600,4,13));states[:,:,3]=1
        states[:,:,:3]=[[-.2,-.2,.63],[-.2,.2,.63],[.2,-.2,.63],[.2,.2,.63]]
        np.savez_compressed(root/'object_states.npz',data=states,prim_paths=[f'/World/AssembledObjects/blue_can_{i:03d}' for i in range(1,5)])
        manifest={'scene_usda':str(path),'asset_body_vertices':str(root/'asset_body_vertices.npz'),'table':{'height_m':.6}}
        (root/'manifest.json').write_text(json.dumps(manifest))
        task={'operation':{'type':'place','moving_object':'blue_can_002'}}
        return root/'manifest.json',task

    def test_empty_supported_target_is_valid_but_not_robot_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path,task=self.fixture(Path(directory))
            result=check_goal_geometry(path,task,{'ground_truth':{'type':'Point','position_m':[0,0,.63]}})
            self.assertTrue(result['passed'])
            self.assertFalse(result['task_action_physics_executed'])
            self.assertFalse(result['robot_reachability_verified'])

    def test_height_edge_and_occupied_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,task=self.fixture(Path(directory))
            for point,message in [([0,0,.8],'standing height'),([.49,0,.63],'actual tabletop'),([-.2,-.2,.63],'bbox gap')]:
                with self.subTest(point=point),self.assertRaisesRegex(ValueError,message):
                    check_goal_geometry(path,task,{'ground_truth':{'type':'Point','position_m':point}})

    def test_retained_furniture_at_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,task=self.fixture(Path(directory),fixture=True)
            with self.assertRaisesRegex(ValueError,'retained furniture'):
                check_goal_geometry(path,task,{'ground_truth':{'type':'Point','position_m':[0,0,.63]}})


if __name__=='__main__':unittest.main()
