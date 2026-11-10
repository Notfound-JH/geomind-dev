"""Independent CPU audit checks; synthetic states are not PhysX evidence."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geomind.episodes import EpisodeResolver, SemanticEpisode
from geomind.tools.generate_spatial_suite import generate
from geomind.tools.audit_assembled_robot import (
    _finger_binding, _finger_condition, _geometry, _load_body_geometry,
    _relation_sample, _replay_query_gt, _gt_equivalent, _place_sample_valid,
    _command_sample_valid, _valid_topology_negative, _place_chain_complete,
    _grasp_width_axis,
    _closing_control_width,
)


def box(size,offset=(0.,0.,0.)):
    return np.asarray([[x,y,z] for x in [-size[0]/2,size[0]/2]
                       for y in [-size[1]/2,size[1]/2] for z in [-size[2]/2,size[2]/2]])+offset


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AssembledRobotAuditTests(unittest.TestCase):
    def closing_fixture(self):
        cloud=box([.045,.045,.045]);world_width=.061
        manifest={'objects':[{'prim_path':'/World/unrelated','kind':'usd_asset'},
                             {'prim_path':'/World/square','kind':'primitive_cuboid'}]}
        bindings=[{'object_id':'selected','prim_path':'/World/square'},
                  {'object_id':'other','prim_path':'/World/unrelated'}]
        physical={'finger_close_target_rad':.4492487906869165,'closing_control_width':{
            'width_m':.045,'measured_world_axis_width_m':world_width,'local_dimensions_m':[.045,.045,.045],
            'asset_kind':'primitive_cuboid','policy':'square_cuboid_local_face_for_closing'}}
        return physical,world_width,cloud,manifest,bindings

    def test_closing_width_is_recomputed_from_source_prim_local_face_and_angle(self):
        p,width,cloud,manifest,bindings=self.closing_fixture()
        checked=_closing_control_width(p,'franka_robotiq',width,cloud,'selected',manifest,bindings)
        self.assertTrue(checked['verified']);self.assertEqual(checked['source_prim_path'],'/World/square')
        self.assertAlmostEqual(checked['control_width_m'],.045)
        self.assertAlmostEqual(checked['initial_close_gap_m'],.042,places=12)
        for key,value in [('width_m',width),('measured_world_axis_width_m',.045),
                          ('local_dimensions_m',[.04,.04,.045]),('asset_kind','usd_asset'),
                          ('policy','measured_world_axis_width')]:
            altered=deepcopy(p);altered['closing_control_width'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                _closing_control_width(altered,'franka_robotiq',width,cloud,'selected',manifest,bindings)
        altered=deepcopy(p);altered['finger_close_target_rad']+=.01
        with self.assertRaisesRegex(ValueError,'close angle'):
            _closing_control_width(altered,'franka_robotiq',width,cloud,'selected',manifest,bindings)

    def test_rectangular_box_cannot_claim_square_face_closing_policy(self):
        p,width,_,manifest,bindings=self.closing_fixture();cloud=box([.04,.07,.045])
        p['closing_control_width'].update(width_m=.04,local_dimensions_m=[.04,.07,.045])
        with self.assertRaisesRegex(ValueError,'policy'):
            _closing_control_width(p,'franka_robotiq',width,cloud,'selected',manifest,bindings)

    def test_legacy_and_panda_closing_width_paths_remain_unchanged(self):
        p,width,cloud,manifest,bindings=self.closing_fixture()
        old=_closing_control_width({},'franka_robotiq',width,cloud,'selected',manifest,bindings)
        self.assertFalse(old['verified']);self.assertEqual(old['control_width_m'],width)
        panda=_closing_control_width(p,'franka',width,cloud,'selected',manifest,bindings)
        self.assertFalse(panda['verified']);self.assertEqual(panda['control_width_m'],width)

    def test_grasp_jaw_axis_and_width_follow_tool_rotation_instead_of_plan_claim(self):
        low=np.asarray([[-.02,-.035,.6]]);high=np.asarray([[.02,.035,.64]])
        zero=[0.,1.,0.,0.];quarter=[0.,math.sqrt(.5),math.sqrt(.5),0.]
        self.assertEqual(_grasp_width_axis({'tool_quaternion_wxyz':zero},{},low,high,0),('y',.07))
        physical={'grasp_orientation_plan':{'tool_quat_wxyz':quarter,'grip_width_axis':0,'grip_width_m':.04}}
        self.assertEqual(_grasp_width_axis({'tool_quaternion_wxyz':quarter},physical,low,high,0),('x',.04))
        tensor_precision=[float(np.float32(value)) for value in quarter]
        self.assertEqual(_grasp_width_axis({'tool_quaternion_wxyz':tensor_precision},physical,low,high,0),('x',.04))
        # Quaternion sign is physically immaterial; asymmetric X/Y sizes make
        # a false axis or reused Y-width claim observable.
        self.assertEqual(_grasp_width_axis({'tool_quaternion_wxyz':[-v for v in quarter]},physical,low,high,0),('x',.04))
        for key,value in [('grip_width_axis',1),('grip_width_axis',False),('grip_width_m',.07),
                          ('grip_width_m',.04+1e-11),('tool_quat_wxyz',zero)]:
            altered=deepcopy(physical);altered['grasp_orientation_plan'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                _grasp_width_axis({'tool_quaternion_wxyz':quarter},altered,low,high,0)
        # A generic 45-degree yaw and an unreported yaw90 do not silently fall
        # back to axis-aligned Y width.
        yaw45=[0.,math.cos(math.pi/8),math.sin(math.pi/8),0.]
        with self.assertRaisesRegex(ValueError,'yaw0/yaw90'):
            _grasp_width_axis({'tool_quaternion_wxyz':yaw45},physical,low,high,0)
        with self.assertRaisesRegex(ValueError,'Legacy grasp'):
            _grasp_width_axis({'tool_quaternion_wxyz':quarter},{},low,high,0)

    def test_gt_numeric_tolerance_preserves_strict_schema_frames_identity_and_relation_contract(self):
        point={'type':'Point','position_m':[.5256578347235923,.1941252936625268,.9554389073171518],'frame':'world'}
        replay=deepcopy(point);replay['position_m'][1]=.19412529366252673
        self.assertTrue(_gt_equivalent(point,replay))
        for change in ['coordinate','frame','extra','bool','nan']:
            altered=deepcopy(point)
            if change=='coordinate':altered['position_m'][0]+=1e-11
            if change=='frame':altered['frame']='table'
            if change=='extra':altered['extra']='not_allowed'
            if change=='bool':altered['position_m'][0]=True
            if change=='nan':altered['position_m'][0]=float('nan')
            with self.subTest(change=change):self.assertFalse(_gt_equivalent(point,altered))
        self.assertFalse(_gt_equivalent({'type':'Object','object_id':'a'},{'type':'Object','object_id':'b'}))
        relation=dict(point,type='Relation',relation='containment_placement',target_object_id='a',reference_object_id='b',
                      contract={'resolver':'containment_placement','max_floor_gap':.02},point_role='representative_action_goal')
        replay=deepcopy(relation);replay['position_m'][1]=.19412529366252673
        self.assertTrue(_gt_equivalent(relation,replay))
        for key,value in [('target_object_id','different'),('reference_object_id','different'),
                          ('contract',{'resolver':'containment_placement','max_floor_gap':.0200000000000001}),
                          ('point_role','answer'),('relation','support_top_placement')]:
            altered=deepcopy(relation);altered[key]=value
            with self.subTest(key=key):self.assertFalse(_gt_equivalent(relation,altered))

    def test_independent_gripper_units_and_both_robotiq_outer_states(self):
        names=['panda_joint1','right_outer_knuckle_joint','finger_joint']
        self.assertEqual(_finger_binding('franka_robotiq',names),[2,1])
        self.assertTrue(_finger_condition('franka_robotiq',[0.,0.],closed=False,width_m=.0512))
        self.assertFalse(_finger_condition('franka_robotiq',[0.,.2],closed=False,width_m=.0512))
        self.assertTrue(_finger_condition('franka_robotiq',[math.radians(20)]*2,closed=True,width_m=.0512))
        self.assertFalse(_finger_condition('franka_robotiq',[.39,0.],closed=True,width_m=.0512))
        self.assertFalse(_finger_condition('franka_robotiq',[.1,.1],closed=True,width_m=.0512))
        self.assertTrue(_finger_condition('franka',[.04,.04],closed=False,width_m=.0512))
        self.assertFalse(_finger_condition('franka',[.04,.04],closed=True,width_m=.0512))

    def test_mixed_geometry_is_bound_by_original_prim_before_semantic_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);small=box([.04,.06,.08],offset=[.03,0,0]);large=box([.10,.12,.14])
            np.savez_compressed(root/'body.npz',small=small,large=large)
            (root/'scene.usda').write_text('#usda 1.0\n# CPU fixture only\n')
            manifest={'asset_body_vertices':str(root/'body.npz'),'scene_usda':str(root/'scene.usda'),
                'objects':[{'prim_path':'/World/large','body_points_key':'large'},
                           {'prim_path':'/World/small','body_points_key':'small'}],
                'provenance':{'asset_body_vertices_sha256':digest(root/'body.npz'),
                              'scene_usda_sha256':digest(root/'scene.usda')}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            profile={'assembly_manifest_path':str(root/'manifest.json'),
                     'assembly_manifest_sha256':digest(root/'manifest.json'),
                     'asset_body_vertices_path':str(root/'body.npz'),
                     'asset_body_vertices_sha256':digest(root/'body.npz'),
                     'assembly_object_bindings':[{'object_id':'opaque2','prim_path':'/World/large'},
                                                 {'object_id':'opaque1','prim_path':'/World/small'}]}
            _,points=_load_body_geometry(profile,['opaque1','opaque2'])
            states=np.zeros((2,13));states[:,3]=1.;states[1,3:7]=[math.sqrt(.5),0,math.sqrt(.5),0]
            centres,lo,hi=_geometry(points,states)
            np.testing.assert_allclose(centres,[[.03,0,0],[0,0,0]],atol=1e-12)
            np.testing.assert_allclose(hi-lo,[[.04,.06,.08],[.14,.12,.10]],atol=1e-12)
            (root/'body.npz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'provenance changed'):
                _load_body_geometry(profile,['opaque1','opaque2'])

    def test_native_history_replay_uses_raw_initial_state_and_rejects_forged_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generate(root,[11]);resolved=EpisodeResolver(root).resolve('DetectSceneChange_v0')
            env=SemanticEpisode(resolved);observation,_=env.reset()
            while not observation['allow_response']:observation,*_=env.step()
            evidence=env.export_evidence();ids=list(evidence['current']['positions_m'])
            query_tick=120+100*evidence['current']['step']
            evidence['physical_clock']={'current_tick':query_tick,'history_ticks':{
                key:120+100*value['step'] for key,value in evidence['history'].items()}}
            ticks=np.arange(1,query_tick+1);states=np.zeros((len(ticks),len(ids),13));states[:,:,3]=1.
            states[:,:,:3]=np.asarray([evidence['current']['positions_m'][identity] for identity in ids])
            for key,snapshot in evidence['history'].items():
                row=evidence['physical_clock']['history_ticks'][key]-1
                states[row,:,:3]=[snapshot['positions_m'][identity] for identity in ids]
            replay=_replay_query_gt(resolved,evidence,ticks,states,ids,box([.02,.02,.02]))
            self.assertEqual(replay['ground_truth'],env.result['ground_truth'])
            forged=deepcopy(evidence)
            next(iter(forged['history'].values()))['positions_m'][ids[0]][0]+=.05
            with self.assertRaisesRegex(ValueError,'differs from .* evidence'):
                _replay_query_gt(resolved,forged,ticks,states,ids,box([.02,.02,.02]))

    def topology_fixture(self):
        points=[box([.04,.04,.04]),box([.20,.20,.02])]
        states=np.zeros((2,13));states[:,3]=1.;states[:,:3]=[[0,0,.64],[0,0,.61]]
        return points,states

    def test_on_top_requires_actual_support_gap_overlap_and_release(self):
        points,states=self.topology_fixture()
        contract={'resolver':'support_top_placement','must_be_released':True,'contact_required':True,
                  'support_required':True,'max_support_gap':.006,'min_support_projection_overlap_ratio':.72}
        def score(released=True):
            centre,lo,hi=_geometry(points,states)
            return _relation_sample(contract,1,0,states,points,centre,lo,hi,released=released)
        self.assertTrue(score()['passed'])
        self.assertAlmostEqual(score()['support_height_w_m'],.62)
        self.assertFalse(score(False)['passed'])
        # The moved object is stably on the table, but not on the elevated support.
        states[0,2]=.62
        self.assertFalse(score()['passed'])
        states[0,2]=.64;states[0,0]=.115
        self.assertFalse(score()['passed'])

    def test_inside_uses_container_frame_floor_opening_and_full_footprint(self):
        points,states=self.topology_fixture()
        # Floor centre at .61, floor top .62. Target bottom .62 and top .66.
        contract={'resolver':'containment_placement','must_be_released':True,
            'interior_half_extents_xy':[.07,.05],'interior_floor_top_offset':.01,
            'opening_plane_z_offset':.09,'xy_inner_margin':.005,
            'max_floor_gap':.006,'max_floor_penetration':.004,'min_top_clearance_below_opening':.005}
        def score(released=True):
            centre,lo,hi=_geometry(points,states)
            return _relation_sample(contract,1,0,states,points,centre,lo,hi,released=released)
        self.assertTrue(score()['passed'])
        self.assertAlmostEqual(score()['support_height_w_m'],.62)
        states[:,3:7]=[math.sqrt(.5),0,0,math.sqrt(.5)]
        states[0,1]=.04  # In reference-local X after a 90-degree world yaw.
        self.assertTrue(score()['passed'])
        states[0,0]=.04  # Reference-local Y room is only .025m.
        self.assertFalse(score()['passed'])
        states[0,:2]=0;states[0,2]=.63  # 10mm through the floor.
        self.assertFalse(score()['passed'])
        states[0,2]=.69  # Above opening, despite fitting the XY footprint.
        self.assertFalse(score()['passed'])
        states[0,2]=.64
        self.assertFalse(score(False)['passed'])

    def test_legal_relation_region_is_not_narrowed_to_point_or_extra_six_mm_gap(self):
        points,states=self.topology_fixture();states[0,:3]=[.04,0,.65]
        centres,lo,hi=_geometry(points,states)
        contract={'resolver':'support_top_placement','must_be_released':True,'contact_required':True,
                  'support_required':True,'max_support_gap':.015,'min_support_projection_overlap_ratio':.72}
        relation=_relation_sample(contract,1,0,states,points,centres,lo,hi,released=True)
        self.assertTrue(relation['passed']);self.assertAlmostEqual(relation['support_gap_m'],.01)
        error=float(np.linalg.norm(centres[0]-[0,0,.64]));self.assertGreater(error,.01)
        sample=dict(relation=relation,point_error=error,point_tolerance=.01,speed=0.,spin=0.,drift=0.,angle=0.,
                    fingers_open=True,table_valid=False)
        self.assertTrue(_place_sample_valid(**sample))
        self.assertFalse(_place_sample_valid(**dict(sample,relation=None)))
        self.assertFalse(_place_sample_valid(**dict(sample,speed=.02)))

    def test_topology_wrong_command_must_really_execute_and_end_outside_relation(self):
        self.assertTrue(_command_sample_valid(error=.002,speed=0.,spin=0.,drift=0.,angle=0.,
                                             fingers_open=True,bottom=.6,support_height=.6))
        criteria=dict(grasp=True,chain_complete=True,command_dwell=60,table_dwell=60,required=60,
                      final_relation={'passed':False})
        self.assertTrue(_valid_topology_negative(**criteria))
        for key,value in [('grasp',False),('chain_complete',False),('command_dwell',59),('table_dwell',59),
                          ('final_relation',{'passed':True})]:
            with self.subTest(key=key):self.assertFalse(_valid_topology_negative(**dict(criteria,**{key:value})))

    def test_release_and_retreat_must_match_real_stage_clock_coverage(self):
        stages=[];rows=[];tick=0
        for label in range(2,11):
            start=tick
            for _ in range(2):tick+=1;rows.append([tick,label])
            stages.append({'stage':label,'start_tick':start,'end_tick':tick,'steps':2,'reached':True,
                           'tcp_error_m':.001,'orientation_error_rad':.01})
        physical={'stages':stages,'release_command_tick':14,'release_tick':16,'retreat_end_tick':18,'hold_start_tick':18}
        raw=np.asarray(rows)
        self.assertTrue(_place_chain_complete(physical,raw))
        corrupted=raw.copy();corrupted[corrupted[:,1]==9,1]=8
        self.assertFalse(_place_chain_complete(physical,corrupted))
        changed=deepcopy(physical);changed['release_tick']=15
        self.assertFalse(_place_chain_complete(changed,raw))


if __name__=='__main__':
    unittest.main()
