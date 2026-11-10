import unittest
import numpy as np
from geomind.episodes.assembled_actions import object_geometry
from geomind.episodes.assembled_relations import measure_relation


class RelationTests(unittest.TestCase):
    def geometry(self, target):
        cloud=lambda a,b,c:np.array([[x,y,z] for x in [-a,a] for y in [-b,b] for z in [-c,c]])
        points=[cloud(.08,.07,.006),cloud(.02,.02,.02)]
        states=np.zeros((2,13));states[:,3]=1;states[:,:3]=[[.4,0,.606],target]
        return points,states,object_geometry(points,states)

    def score(self, contract, target, released=True):
        points,states,geometry=self.geometry(target)
        return measure_relation(contract,0,1,points,states,*geometry,released=released)

    def test_on_top_requires_actual_contact_overlap_and_release(self):
        spec=dict(resolver='support_top_placement',contact_required=True,support_required=True,must_be_released=True)
        self.assertTrue(self.score(spec,[.4,0,.632])['passed'])
        for target in ([.4,0,.7],[.6,0,.632],[.4,0,.62]):
            self.assertFalse(self.score(spec,target)['passed'])
        self.assertFalse(self.score(spec,[.4,0,.632],False)['passed'])

    def test_inside_requires_floor_opening_margin_and_release(self):
        spec=dict(resolver='containment_placement',interior_half_extents_xy=[.073,.063],
                  interior_floor_top_offset=.006,opening_plane_z_offset=.064,xy_inner_margin=.0015,
                  max_floor_gap=.018,max_floor_penetration=.004,min_top_clearance_below_opening=.004,
                  must_be_released=True)
        self.assertTrue(self.score(spec,[.4,0,.632])['passed'])
        for target in ([.46,0,.632],[.4,0,.66],[.4,0,.61]):
            self.assertFalse(self.score(spec,target)['passed'])
        self.assertFalse(self.score(spec,[.4,0,.632],False)['passed'])

if __name__=='__main__':unittest.main()
