"""Independent measured-state scoring regressions, with no physics runtime."""
import unittest
import numpy as np

from geomind.scene_assembly.audit import grid_dimensions_match,measured_bounds,score_window


class TestAssemblyAudit(unittest.TestCase):
    def test_randomized_grid_bands_preserve_row_column_order(self):
        points=[[-.103,-.105],[-.097,.101],[.100,-.099],[.104,.104]]
        self.assertTrue(grid_dimensions_match(points,2,2,band_span_m=.01))
        swapped=[points[0],points[2],points[1],points[3]]
        self.assertFalse(grid_dimensions_match(swapped,2,2,band_span_m=.01))
        broken=np.asarray(points);broken[1,0]+=.03
        self.assertFalse(grid_dimensions_match(broken,2,2,band_span_m=.01))

    def fixture(self):
        points=np.asarray([[x,y,z] for x in [-.02,.02] for y in [-.02,.02] for z in [-.05,.05]])
        manifest={'table':{'height_m':.6,'bounds_xy_m':[-.5,.5,-.5,.5]},'objects':[
            {'position_w_m':[0,0,.65],'orientation_wxyz':[1,0,0,0]}]}
        states=np.zeros((100,1,13));states[:,:,2]=.65;states[:,:,3]=1
        return points,manifest,states

    def test_stable_layout_and_shifted_layout(self):
        points,manifest,states=self.fixture()
        self.assertEqual(score_window(states,manifest,points),100)
        states[:,0,0]=.05
        self.assertEqual(score_window(states,manifest,points),0)

    def test_bottom_recomputed_from_measured_rotation(self):
        points,manifest,states=self.fixture()
        angle=.10
        q=np.array([np.cos(angle/2),0,np.sin(angle/2),0])
        low,high=measured_bounds(points,np.array([0,0,.65]),q)
        self.assertLess(low[2],.6)
        self.assertAlmostEqual(low[2],.65-(.05*np.cos(angle)+.02*np.sin(angle)),places=12)
        # Large horizontal drift cannot be hidden by a long preceding stable run.
        states[-1,0,0]=.02
        self.assertEqual(score_window(states,manifest,points),0)

    def test_velocity_and_support_failures_are_rejected(self):
        points,manifest,states=self.fixture()
        states[-10:,0,7]=.02
        self.assertEqual(score_window(states,manifest,points),0)
        states[:,:,7]=0;states[:,:,2]=.68
        self.assertEqual(score_window(states,manifest,points),0)


if __name__=='__main__':unittest.main()
