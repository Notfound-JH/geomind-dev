"""Uniform taller candidates preserve the two-hop task and legal pad clearance."""
import math
import unittest
import numpy as np
from geomind.episodes.extended_static_tasks import TASK_IDS,build_spec,solve_spec,check_spec
from geomind.episodes.gripper_profiles import gripper_profile
from geomind.scene_assembly.grasp_orientation import plan_grasp_orientation

class IndirectHeightGeometryTests(unittest.TestCase):
    def test_all_three_candidates_share_geometry_and_references_are_unchanged(self):
        for task in TASK_IDS:
            s=build_spec(task)
            cand=s['protocol']['candidate_object_ids']
            for o in s['runtime_objects']:
                expected=[.04,.04,.05] if task=='SelectByIndirectReference_v0' and o['name'] in cand else [.04,.04,.04]
                self.assertEqual(o['size'],expected)
                self.assertEqual(o['physics']['mass'],.05)
            self.assertNotIn('grip_hold_policy',s['protocol'])
        s=build_spec('SelectByIndirectReference_v0')
        self.assertEqual(len(s['protocol']['candidate_object_ids']),3)
        self.assertEqual({tuple(o['visual_color']) for o in s['runtime_objects'] if o['name'] in s['protocol']['candidate_object_ids']},{(.63,.48,.33)})

    def test_two_hops_and_both_shortcut_distractors_are_necessary_and_gt_unchanged(self):
        s=build_spec('SelectByIndirectReference_v0',seed=17)
        p={o['name']:np.array([*o['pose']['xy'],.95 if o['name'] not in s['protocol']['candidate_object_ids'] else .955]) for o in s['runtime_objects']}
        near=lambda ref,items:min(items,key=lambda i:np.linalg.norm(p[i][:2]-p[ref][:2]))
        self.assertEqual(near('n0',['n1','n2']),'n1')
        self.assertEqual(near('n1',['n3','n4','n5']),'n3')
        self.assertEqual(near('n0',['n3','n4','n5']),'n4')
        self.assertEqual(near('n2',['n3','n4','n5']),'n5')
        self.assertEqual(solve_spec(s,{k:v.tolist() for k,v in p.items()})['selected'],'n3')
        self.assertTrue(check_spec(s,{k:v.tolist() for k,v in p.items()},{'selected':'n3'})['success'])
        self.assertEqual(s['instruction'],'先找离红色标记最近的蓝色标记，再拿起离那个蓝色标记最近的棕色方块。')
        self.assertEqual({o['name']:o['pose']['xy'] for o in s['runtime_objects']},{'n0':[.24,-.06],'n1':[.36,-.03],'n2':[.52,.20],'n3':[.39,-.12],'n4':[.22,-.19],'n5':[.55,.10]})

    def test_asset_pad_centering_and_four_mm_clearance_have_independent_solution(self):
        # Source USD pad span is38.0001888mm; rotation translates both endpoints equally.
        low,high=.11041999608278298,.148420184850693
        span=high-low;profile=gripper_profile('franka_robotiq')
        h=.05;offset=profile.grasp_tcp_height_offset_m('PickLiftSelectedObject',h,width_m=.04,asset_kind='primitive_cuboid',contact_angle_geometry=True)
        # Independently invert width with trigonometry-derived linkage translation.
        a,b=0.,math.radians(47.)
        for _ in range(70):
            q=(a+b)/2;inward=-.03716*(math.cos(q)-1)+.04343*math.sin(q)
            if .0870968475937844-2*inward>.04:a=q
            else:b=q
        q=(a+b)/2;dz=.03716*math.sin(q)+.04343*(math.cos(q)-1)
        ideal=(low+high)/2+dz-.14
        self.assertAlmostEqual(offset,ideal,places=12)
        self.assertGreaterEqual(h/2-span/2,.004)
        self.assertAlmostEqual(h/2-span/2,.005999905616044994,places=12)
        self.assertLess(.04/2-span/2,.004)
        self.assertGreater(span+.008,.046)

    def test_height_change_does_not_change_xy_orientation_screen(self):
        s=build_spec('SelectByIndirectReference_v0');objects=sorted(s['runtime_objects'],key=lambda o:o['name'])
        centres=np.array([[*o['pose']['xy'],.95] for o in objects]);half=np.array([o['size'] for o in objects])/2
        low=centres-half;high=centres+half
        first=plan_grasp_orientation(centres,low,high,3,open_gripper_half_xy_m=[.015263,.076449])
        half[:,2]=.02
        second=plan_grasp_orientation(centres,centres-half,centres+half,3,open_gripper_half_xy_m=[.015263,.076449])
        self.assertEqual(first['yaw_degrees'],second['yaw_degrees'])
        for a,b in zip(first['candidate_orientations'],second['candidate_orientations']):
            self.assertEqual(a['minimum_reference_clearance_m'],b['minimum_reference_clearance_m'])
            self.assertEqual(a['grip_width_m'],b['grip_width_m'])

if __name__=='__main__':unittest.main()
