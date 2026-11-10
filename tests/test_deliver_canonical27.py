from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from geomind.core.registry import GeoMindRegistry
from geomind.episodes.registered_tasks import REGISTERED_TASK_IDS
from geomind.episodes.canonical_extensions import task_ids
from geomind.tools.deliver_canonical27 import (
    geometry_shortcut_row_quality, new_completion_gates, select_new_versions, verify_raw_hashes,
)


class CanonicalDeliveryGates(unittest.TestCase):
    def good(self):
        positive={'negative_control':False,'success':True,'independent_gt_check':{'success':True},
                  'physical':{'success':True,'grasp_verified':True,'frozen_ground_truth':{'type':'Object','object_id':'x'}}}
        negative={'negative_control':True,'success':False,'verification_passed':True,
                  'physical':{'action':'PickLiftSelectedObject','grasp_verified':True,
                              'selected_object_id':'wrong','expected_object_id':'right',
                              'stages':[{'stage':label,'reached':True} for label in range(2,7)]}}
        return ({'task_completed':True,'initial_layout_physics_verified':True,'task_action_physics_verified':True,
                 'gt_verified':True,'provenance_check':{'passed':True}},
                {'passed':True},{'passed':True},{'physical_execution_verified':True,'results':[positive,negative]},
                {'passed':True},{'saved_source_snapshot_complete':True,'saved_config_hashes_complete':True},
                {'passed':True},{'passed':True},{'passed':True,'applicable_negative_pick_cases':1})

    def test_completed_flag_cannot_replace_actual_controls_gt_or_raw_audits(self):
        good=self.good();self.assertTrue(all(new_completion_gates(*good).values()))
        for change in ('negative','gt','raw','wrist','initial'):
            args=deepcopy(good)
            if change=='negative':args[3]['results'].pop()
            if change=='gt':args[3]['results'][0]['independent_gt_check']['success']=False
            if change=='raw':args[6]['passed']=False
            if change=='wrist':args[7]['passed']=False
            if change=='initial':args[2]['passed']=False
            with self.subTest(change=change):self.assertFalse(all(new_completion_gates(*args).values()))

    def test_later_incomplete_directory_wins_over_earlier_success(self):
        task=task_ids()[0]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old';new=root/'retry';old.mkdir();new.mkdir()
            (old/task).mkdir();(old/task/'result.json').write_text('{"task_completed":true}')
            (new/task).mkdir()
            _,selection=select_new_versions(old,[new])
            chosen=next(row for row in selection if row['task_id']==task)
            self.assertEqual(chosen['selected']['case'],str((new/task).resolve()))
            self.assertEqual(len(chosen['versions']),2)

    def test_semantically_rejected_but_failed_wrong_pick_is_not_valid_negative(self):
        args=deepcopy(self.good())
        args[3]['results'][1]['physical']['grasp_verified']=False
        args[3]['results'][1]['physical']['stages']=args[3]['results'][1]['physical']['stages'][:2]
        args[3]['results'][1]['physical']['stages'][-1]['reached']=False
        args[8]['passed']=False
        gates=new_completion_gates(*args)
        self.assertTrue(gates['actual_negative_controls_correctly_rejected'])
        self.assertFalse(gates['negative_command_stages_complete'])
        self.assertFalse(gates['negative_command_execution_valid'])

    def test_tampered_and_out_of_case_raw_paths_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);case=root/'case';case.mkdir();p=case/'states.npz';p.write_bytes(b'original')
            expected=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertTrue(verify_raw_hashes(case,{str(p):expected})['passed'])
            p.write_bytes(b'changed');self.assertFalse(verify_raw_hashes(case,{str(p):expected})['passed'])
            outside=root/'outside';outside.write_bytes(b'original')
            self.assertFalse(verify_raw_hashes(case,{str(outside):expected})['passed'])

    def test_twenty_old_and_ten_new_instances_group_to_twenty_seven_types(self):
        registry=GeoMindRegistry();registry.load()
        self.assertEqual(len({registry.get(task).public_task_id for task in REGISTERED_TASK_IDS}),17)
        self.assertEqual(len(REGISTERED_TASK_IDS)+len(task_ids()),30)
        self.assertEqual(len({registry.get(task).public_task_id for task in (*REGISTERED_TASK_IDS,*task_ids())}),27)

    def geometry(self,name,fraction=.65):
        a=[.32,-.12];b=[.48,.12];d=[b[i]-a[i] for i in range(2)]
        foot=[a[i]+fraction*d[i] for i in range(2)];p=[foot[0]-.09,foot[1]+.06]
        q=foot if name=='ProjectOntoLine' else [2*foot[i]-p[i] for i in range(2)]
        row={'negative_control':False,'success':True,'instance_id':'physical_1',
             'ground_truth':{'type':'Point','position_m':[*q,.95]},
             'physical':{'position_tolerance_m':.01,'frozen_ground_truth':{'position_m':[*q,.95]}},
             'provenance':{'parameters':{'object_id_map':{'a':'red','b':'blue','p':'green'},
                 'legacy_task_spec':{'protocol':{'gt_spec':{'line_object_ids':['red','blue'],'point_object_id':'green'}}}},
                 'calculation':{'resolver':name,'line_parameter':fraction,'projection_xy_m':foot,
                    'line_endpoints_xy_m':[a,b],'source_point_xy_m':p,'planned_positions_used_as_observations':False}}}
        evidence={'current':{'instance_id':'physical_1','positions_m':{'a':a,'b':b,'p':p}}}
        return row,evidence

    def test_midpoint_degenerate_projection_and_point_symmetry_are_not_quality_passes(self):
        for name in ('ProjectOntoLine','ReflectAcrossLine'):
            row,evidence=self.geometry(name,.500045)
            quality=geometry_shortcut_row_quality(row,evidence)
            with self.subTest(task=name):
                self.assertFalse(quality['passed'])
                self.assertLess(quality['target_midpoint_shortcut_separation_m'],.00003)
                self.assertIn('projection_distinct_from_endpoint_midpoint',quality['failed_checks'])

    def test_oblique_non_midpoint_geometry_passes_with_twice_tolerance_margin(self):
        for name in ('ProjectOntoLine','ReflectAcrossLine'):
            row,evidence=self.geometry(name)
            quality=geometry_shortcut_row_quality(row,evidence)
            with self.subTest(task=name):
                self.assertTrue(quality['passed'])
                self.assertGreater(quality['projection_midpoint_separation_m'],.043)
                self.assertEqual(quality['required_shortcut_separation_m'],.02)
                row['physical']['position_tolerance_m']=.03
                self.assertFalse(geometry_shortcut_row_quality(row,evidence)['passed'])

    def test_reflection_midpoint_shortcut_cannot_replace_actual_line_reflection_gt(self):
        row,evidence=self.geometry('ReflectAcrossLine')
        p=evidence['current']['positions_m']['p'];a=evidence['current']['positions_m']['a'];b=evidence['current']['positions_m']['b']
        row['ground_truth']['position_m'][:2]=[a[i]+b[i]-p[i] for i in range(2)]
        row['physical']['frozen_ground_truth']=deepcopy(row['ground_truth'])
        quality=geometry_shortcut_row_quality(row,evidence)
        self.assertFalse(quality['passed'])
        self.assertFalse(quality['checks']['frozen_gt_matches_independent_line_geometry'])
        self.assertEqual(quality['target_midpoint_shortcut_separation_m'],0.)

    def test_saved_calculation_must_match_original_query_instance_and_positions(self):
        for change in ('instance','positions','calculation','planned'):
            row,evidence=self.geometry('ProjectOntoLine')
            if change=='instance':evidence['current']['instance_id']='other_instance'
            if change=='positions':evidence['current']['positions_m']['p']=[.1,.1]
            if change=='calculation':row['provenance']['calculation']['line_parameter']=.8
            if change=='planned':row['provenance']['calculation']['planned_positions_used_as_observations']=True
            with self.subTest(change=change):self.assertFalse(geometry_shortcut_row_quality(row,evidence)['passed'])

