"""Strict agent choices and native task contracts, without simulator startup."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from geomind.scene_assembly.request import ASSET_IDS, RequestError
from geomind.scene_assembly.task_request import assembly_options, validate_task_request


class AssembledTaskRequestTests(unittest.TestCase):
    def request(self):
        return {"output":"outputs/research/assembled_test", "assembly_manifest":"outputs/assemblies/example/manifest.json",
            "task":{"task_id":"InferMidpoint","parameters":{"endpoints":["blue_can_001","blue_can_004"]},
                    "operation":{"type":"place","moving_object":"blue_can_002"}},
            "selection":{"actor":"agent","reason":"Use the opposite grid corners to define a midpoint with free space.",
                         "considered_scene_ids":[60,65],"considered_layouts":["grid","random"]}}

    def test_midpoint_contract_is_explicit_and_canonical_without_mutation(self):
        request=self.request();before=deepcopy(request)
        canonical=validate_task_request(request)
        self.assertEqual(request,before)
        self.assertEqual(canonical["schema_version"],1)
        self.assertEqual(canonical["stage"],"assemble_and_solve_gt")
        self.assertEqual(canonical["timeout_s"],420)
        self.assertEqual(canonical["task"]["parameters"]["frame"],"world")
        self.assertEqual(canonical["task"]["operation"],request["task"]["operation"])
        self.assertEqual(validate_task_request(canonical),canonical)

    def test_new_assembly_reuses_existing_canonical_request_and_records_choice(self):
        request=self.request();request.pop("assembly_manifest")
        request["assembly"]={"scene_id":60,"object":{"asset":"blue_can","count":4},"layout":{"type":"grid"},"randomization":{}}
        with patch("geomind.scene_assembly.task_request.resolve_asset",return_value={}),patch("geomind.scene_assembly.task_request.resolve_scene",return_value={}):
            result=validate_task_request(request)
            self.assertEqual(result["assembly"]["layout"],{"type":"grid","rows":2,"cols":2})
            self.assertEqual(result["assembly"]["randomization"]["enabled"],True)
            request["selection"]["considered_scene_ids"]=[65]
            with self.assertRaisesRegex(RequestError,"recorded considered"):
                validate_task_request(request)

    def test_geometric_defaults_match_native_resolver(self):
        for task_id in ("ProjectOntoLine","ReflectAcrossLine"):
            request=self.request()
            request["task"]["task_id"]=task_id
            request["task"]["parameters"]={"point":"blue_can_003","line":["blue_can_001","blue_can_004"]}
            result=validate_task_request(request)["task"]["parameters"]
            self.assertEqual(result["geometry"],"InfiniteLine3D")
            self.assertEqual(result["degeneracy_tolerance_m"],.0001)

    def test_order_requires_pick_and_one_based_index(self):
        request=self.request();request["task"]={"task_id":"SelectByOrder","parameters":{
            "index":2,"from":"Right","candidates":["blue_can_001","blue_can_002"]},"operation":{"type":"pick"}}
        result=validate_task_request(request)
        self.assertEqual(result["task"]["parameters"]["tie_tolerance_m"],.01)
        for index in (0,3,True,1.0):
            request["task"]["parameters"]["index"]=index
            with self.assertRaises(RequestError):validate_task_request(request)

    def test_operation_cannot_change_reference_objects(self):
        request=self.request()
        for moving in (None,"blue_can_001"):
            request["task"]["operation"]={"type":"place"}
            if moving is not None:request["task"]["operation"]["moving_object"]=moving
            with self.assertRaises(RequestError):validate_task_request(request)
        request=self.request();request["task"]["operation"]["type"]="pick"
        with self.assertRaises(RequestError):validate_task_request(request)

    def test_unknown_fields_dynamic_tasks_and_alternate_frames_are_rejected(self):
        mutations=[lambda r:r.update(backend="mock"),lambda r:r["task"].update(task_id="DetectSceneChange"),
            lambda r:r["task"].update(task_id="SelectByGridIndex"),lambda r:r["task"]["parameters"].update(frame="table"),
            lambda r:r["task"]["parameters"].update(expected_answer=[0,0,0]),
            lambda r:r["task"]["parameters"].update(endpoints=["blue_can_001","blue_can_001"]),
            lambda r:r["task"]["operation"].update(force_teleport=True),lambda r:r.update(stage="execute_robot")]
        for mutate in mutations:
            request=self.request();mutate(request)
            with self.subTest(request=request),self.assertRaises(RequestError):validate_task_request(request)

    def test_selection_is_required_and_not_a_hidden_agent(self):
        mutations=[lambda r:r.pop("selection"),lambda r:r["selection"].update(actor="llm_auto"),
            lambda r:r["selection"].update(reason="  "),lambda r:r["selection"].update(considered_scene_ids=[True]),
            lambda r:r["selection"].update(considered_scene_ids=[60,60]),
            lambda r:r["selection"].update(considered_layouts=["invalid"])]
        for mutate in mutations:
            request=self.request();mutate(request)
            with self.assertRaises(RequestError):validate_task_request(request)

    def test_bounds_and_reference_choice_types_are_strict(self):
        for timeout in (True,float("nan"),math.inf,0,.5,421,"420",10**400):
            request=self.request();request["timeout_s"]=timeout
            with self.assertRaises(RequestError):validate_task_request(request)
        request=self.request();request["assembly"]={}
        with self.assertRaisesRegex(RequestError,"exactly one"):validate_task_request(request)
        request=self.request();request.pop("assembly_manifest")
        with self.assertRaisesRegex(RequestError,"exactly one"):validate_task_request(request)

    def test_inventory_reflects_existing_tasks_and_excludes_unavailable_assets(self):
        def descriptor(name):
            if name not in {"blue_can","red_can"}:raise ValueError("Not a trusted local descriptor")
            return {"name":name,"usd_path":f"/models/{name}.usd"}
        with patch("geomind.scene_assembly.task_request.resolve_asset",side_effect=descriptor):
            result=assembly_options()
        self.assertEqual(result["scene_count"],102)
        self.assertEqual(result["layout_count"],5)
        self.assertEqual(result["asset_count"],2)
        self.assertEqual(len(result["excluded_assets"]),len(ASSET_IDS)-2)
        self.assertEqual({t["task_id"] for t in result["tasks"]},{"SelectByOrder","InferMidpoint","ProjectOntoLine","ReflectAcrossLine"})
        self.assertTrue(all(t["solver"]=="geomind.episodes.resolver.solve" for t in result["tasks"]))
        self.assertEqual(len(result["robot_options"]),5)
        self.assertEqual([r['embodiment'] for r in result['robot_options'] if r['assembled_manipulation_actions']],['franka','franka_robotiq'])

    def robot_request(self):
        request=self.request()
        request.update(stage='assemble_solve_execute',robot={'embodiment':'franka','base_position_w_m':[-.1,0,.6]},
                       execution={'action':'PickPlaceObject'})
        return request

    def test_explicit_robot_stage_has_bounded_action_defaults(self):
        request=self.robot_request();before=deepcopy(request)
        result=validate_task_request(request)
        self.assertEqual(request,before)
        self.assertEqual(result['stage'],'assemble_solve_execute')
        self.assertEqual(result['robot']['embodiment'],'franka')
        self.assertEqual(result['execution']['max_steps'],3000)
        self.assertEqual(result['execution']['dwell_steps'],60)
        self.assertTrue(result['execution']['negative_controls'])
        self.assertEqual(validate_task_request(result),result)

    def test_robotiq_finer_clock_has_an_action_budget_and_preserves_explicit_limits(self):
        request=self.robot_request();request['robot']['embodiment']='franka_robotiq'
        result=validate_task_request(request)
        self.assertEqual(result['execution']['max_steps'],12000)
        self.assertEqual(result['execution']['timeout_s'],420.)
        request['execution']['max_steps']=6000
        self.assertEqual(validate_task_request(request)['execution']['max_steps'],6000)

    def test_selection_uses_lift_and_never_a_reach_fallback(self):
        request=self.robot_request();request['task']={'task_id':'SelectByOrder','parameters':{
            'index':2,'from':'Left','candidates':['blue_can_001','blue_can_002']},'operation':{'type':'pick'}}
        request['execution']={'action':'PickLiftSelectedObject'}
        self.assertEqual(validate_task_request(request)['execution']['action'],'PickLiftSelectedObject')
        for action in ['ReachAboveTarget','PickPlaceObject','UnknownAction']:
            request['execution']['action']=action
            with self.assertRaisesRegex(RequestError,'Natural operation'):
                validate_task_request(request)

    def test_robot_capability_and_ambiguous_configuration_fail_explicitly(self):
        robotiq=self.robot_request();robotiq['robot']['embodiment']='franka_robotiq'
        self.assertEqual(validate_task_request(robotiq)['robot']['embodiment'],'franka_robotiq')
        for name in ['fr3','piper','piper_dual']:
            request=self.robot_request();request['robot']['embodiment']=name
            if name=='piper_dual':request['robot']['active_arm']='right'
            with self.assertRaisesRegex(RequestError,'not implemented or verified'):
                validate_task_request(request)
        for mutate in [lambda r:r.pop('robot'),lambda r:r.pop('execution'),
                       lambda r:r.update(stage='assemble_and_solve_gt'),
                       lambda r:r['robot'].update(active_arm='left'),
                       lambda r:r['robot'].update(embodiment='imaginary')]:
            request=self.robot_request();mutate(request)
            with self.assertRaises(RequestError):validate_task_request(request)

    def test_robot_execution_numeric_and_field_limits(self):
        mutations=[lambda r:r['robot'].update(base_position_w_m=[0,float('nan'),.6]),
            lambda r:r['robot'].update(base_position_w_m=[True,0,.6]),
            lambda r:r['robot'].update(base_position_w_m=[10**400,0,.6]),
            lambda r:r['robot'].update(snap_object_to_hand=True),
            lambda r:r['execution'].update(timeout_s=421),
            lambda r:r['execution'].update(max_steps=True),
            lambda r:r['execution'].update(dwell_steps=59),
            lambda r:r['execution'].update(negative_controls='true'),
            lambda r:r['execution'].update(position_tolerance_m=.2),
            lambda r:r['execution'].update(teleport=True)]
        for mutate in mutations:
            request=self.robot_request();mutate(request)
            with self.assertRaises(RequestError):validate_task_request(request)


if __name__=="__main__":unittest.main()
