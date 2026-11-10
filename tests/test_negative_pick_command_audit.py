from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from geomind.tools.audit_negative_pick import audit_negative_picks, evaluate_selected_wrong_pick


class SelectedWrongPickAuditTests(unittest.TestCase):
    def evidence(self):
        ticks=np.arange(1,72,dtype=float);count=len(ticks);dt=.01
        states=np.zeros((count,2,13));states[:,:,3]=1.
        states[:,0,:3]=[0.,0.,.02];states[:,1,:3]=[.2,0.,.02]
        states[:,1,2]+=.08*np.clip((ticks-8)/3,0.,1.)
        states[:,1,9]=np.gradient(states[:,1,2],dt)
        labels=np.full(count,12.)
        labels[0]=0.;stages=[]
        for label in range(2,7):
            start=1+(label-2)*2;end=start+2
            labels[(ticks>start)&(ticks<=end)]=label
            stages.append({'stage':label,'start_tick':start,'end_tick':end,'steps':2,
                           'reached':True,'tcp_error_m':0.,'orientation_error_rad':0.})
        obj=np.column_stack([ticks,labels,states.reshape(count,-1)])
        control=np.zeros((count,10));control[:,2:4]=.01
        cloud=np.asarray([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (-.02,.02)])
        physical={'action':'PickLiftSelectedObject','selected_object_id':'wrong','expected_object_id':'right',
                  'grasp_verified':True,'stages':stages,'action_start_tick':1,'hold_start_tick':11,'action_end_tick':71,
                  'required_dwell_steps':50,'finger_joint_ids':[0,1],
                  'finger_joint_names':['panda_finger_joint1','panda_finger_joint2'],
                  '_recorded_tool_quaternion_wxyz':[0.,1.,0.,0.],
                  'object_pose_writes_after_reset':0,'joint_state_writes_during_motion':0,
                  'attachments_used':False,'object_external_forces_used':False,
                  'object_external_forces_during_robot_action':False}
        return {'object_data':obj,'object_ids':['right','wrong'],'body_points':cloud,'dt_s':dt,'query_tick':1,
                'controls':control,'joint_names':['panda_finger_joint1','panda_finger_joint2'],
                'tcp_w':states[:,1,:3]+[0.,0.,.01],'trace_ticks':ticks,'trace_stages':labels,
                'physical':physical,'ground_truth':{'type':'Object','object_id':'right'},'robot_name':'franka'}

    def test_actual_stable_wrong_body_grasp_is_valid_despite_correct_body_not_moving(self):
        result=evaluate_selected_wrong_pick(**self.evidence())
        self.assertTrue(result['command_execution_valid'])
        self.assertAlmostEqual(result['selected_body_lift_at_stage6_end_m'],.08)
        self.assertEqual(result['trailing_hold_steps'],60)
        self.assertAlmostEqual(result['trailing_hold_duration_s'],.6)

    def test_semantic_rejection_and_reported_flags_cannot_hide_an_unlifted_wrong_body(self):
        args=self.evidence();args['object_data'][:,2+13+2]=.02;args['object_data'][:,2+13+9]=0.
        result=evaluate_selected_wrong_pick(**args)
        self.assertFalse(result['command_execution_valid'])
        self.assertIn('selected_body_actual_lift_over_6cm',result['failed_checks'])
        self.assertEqual(result['trailing_hold_steps'],0)

    def test_failed_or_unexecuted_grasp_stage_cannot_be_accepted_as_control(self):
        args=self.evidence();args['physical']['stages'][1]['reached']=False
        self.assertFalse(evaluate_selected_wrong_pick(**args)['command_execution_valid'])
        args=self.evidence();args['object_data'][(args['trace_ticks']>3)&(args['trace_ticks']<=5),1]=12
        args['trace_stages']=args['object_data'][:,1].copy()
        self.assertFalse(evaluate_selected_wrong_pick(**args)['command_execution_valid'])

    def test_hold_must_be_trailing_and_other_bodies_must_remain_unchanged(self):
        for bad in ('tail_speed','tail_spin','reference_drift','fingers_open','tcp_far'):
            args=self.evidence()
            if bad=='tail_speed':args['object_data'][-1,2+13+7]=.02
            if bad=='tail_spin':args['object_data'][-1,2+13+10]=.2
            if bad=='reference_drift':args['object_data'][-60:,2]=.006
            if bad=='fingers_open':args['controls'][-60:,2:4]=.04
            if bad=='tcp_far':args['tcp_w'][-60:,0]+=1.
            with self.subTest(bad=bad):
                self.assertFalse(evaluate_selected_wrong_pick(**args)['command_execution_valid'])

    def test_insufficient_actual_hold_and_wrong_identity_are_rejected(self):
        args=self.evidence();args['physical']['required_dwell_steps']=100
        self.assertFalse(evaluate_selected_wrong_pick(**args)['command_execution_valid'])
        args=self.evidence();args['physical']['selected_object_id']='right'
        result=evaluate_selected_wrong_pick(**args)
        self.assertFalse(result['checks']['selected_identity_is_wrong'])
        self.assertFalse(result['command_execution_valid'])

    def saved_evidence(self,case,*,lifted=True):
        """Save the same synthetic raw states through the public case reader."""
        args=self.evidence();identity='assembled_task';instance='synthetic_instance'
        for directory in ('robot_configs/episodes','robot_configs/scenes','robot_physics/traces','robot_physics/evidence'):
            (case/directory).mkdir(parents=True)
        (case/'robot_configs/episodes'/f'{identity}.json').write_text(json.dumps({'scene_id':'synthetic_scene'}))
        (case/'robot_configs/scenes/synthetic_scene.json').write_text(json.dumps({'physics':{},'robot':'franka'}))
        trace=case/'robot_physics/traces'/f'{identity}_negative.npz'
        np.savez_compressed(trace,data=np.column_stack([args['trace_ticks'],np.zeros(len(args['trace_ticks'])),
            args['trace_stages'],args['tcp_w']]),
            columns=np.array(['physics_step','protocol_step','stage','tcp_x_w','tcp_y_w','tcp_z_w']))
        if not lifted:
            args['object_data'][:,2+13+2]=.02;args['object_data'][:,2+13+9]=0.
        np.savez_compressed(trace.with_name(trace.stem+'_objects.npz'),data=args['object_data'],
            object_ids=np.array(args['object_ids']),dt_s=args['dt_s'])
        controls=trace.with_name(trace.stem+'_controls.npz')
        np.savez_compressed(controls,data=args['controls'],joint_names=np.array(args['joint_names']),
            dt_s=args['dt_s'],instance_id=instance,contract_hash='synthetic_contract')
        (case/'robot_physics/evidence'/f'{trace.stem}.json').write_text(json.dumps({
            'current':{'instance_id':instance},'physical_clock':{'current_tick':args['query_tick'],'dt_s':args['dt_s']}}))
        physical=deepcopy(args['physical']);physical['success']=False
        row={'episode_id':identity,'negative_control':True,'success':False,'verification_passed':False,
             'semantic':{'success':False},'physical':physical,'ground_truth':args['ground_truth'],
             'instance_id':instance,'trace':str(trace),'control_trace':str(controls),
             'tool_quaternion_wxyz':[0.,1.,0.,0.],'provenance':{'contract_hash':'synthetic_contract'}}
        return row,args['body_points']

    def test_semantic_rejection_is_independent_of_total_verification_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            case=Path(directory);row,cloud=self.saved_evidence(case)
            for overall in (False,True):
                row['verification_passed']=overall
                with self.subTest(overall=overall),patch('geomind.tools.audit_negative_pick._load_body_geometry',return_value=({},cloud)):
                    result=audit_negative_picks(case,{'results':[row]})
                    self.assertTrue(result['cases'][0]['semantic_rejection_correct'])
                    self.assertTrue(result['cases'][0]['command_execution_valid'])
                    self.assertTrue(result['cases'][0]['actual_selected_command_valid'])
                    self.assertTrue(result['passed'])

    def test_failed_wrong_body_execution_stays_invalid_despite_correct_semantic_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            case=Path(directory);row,cloud=self.saved_evidence(case,lifted=False)
            with patch('geomind.tools.audit_negative_pick._load_body_geometry',return_value=({},cloud)):
                result=audit_negative_picks(case,{'results':[row]})
            self.assertTrue(result['cases'][0]['semantic_rejection_correct'])
            self.assertFalse(result['cases'][0]['command_execution_valid'])
            self.assertFalse(result['cases'][0]['actual_selected_command_valid'])
            self.assertFalse(result['passed'])

    def test_semantic_rejection_requires_both_actual_semantic_and_physical_false(self):
        with tempfile.TemporaryDirectory() as directory:
            case=Path(directory);row,cloud=self.saved_evidence(case)
            for semantic,physical in ((True,False),(False,True),(True,True),(None,False),(False,None)):
                row['semantic']['success']=semantic;row['physical']['success']=physical;row['verification_passed']=True
                with self.subTest(semantic=semantic,physical=physical),patch('geomind.tools.audit_negative_pick._load_body_geometry',return_value=({},cloud)):
                    result=audit_negative_picks(case,{'results':[row]})
                    self.assertFalse(result['cases'][0]['semantic_rejection_correct'])
                    self.assertTrue(result['cases'][0]['command_execution_valid'])
                    self.assertFalse(result['cases'][0]['actual_selected_command_valid'])
                    self.assertFalse(result['passed'])

