from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from geomind.core.registry import GeoMindRegistry
from geomind_lab.tasks.manager_based.manipulation.geomind.task_evolution import apply_task_contract
from geomind_lab.mimic.envs.state_reindexing_logic import evaluate_state_reindexing_terminal_state
from geomind_lab.mimic.envs.task_contract_logic import structural_adjacency_indices, protocol_gt_positions


class TestTaskEvolution(unittest.TestCase):
    def test_merged_tasks_require_parameters_or_variant(self):
        registry = GeoMindRegistry(); registry.load()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            registry.resolve_task_id("SelectByOrder")
        self.assertEqual(registry.resolve_task_id("SelectByOrder", parameters={"index": 1, "from": "Left"}), "SelectByOrderFirst_v0")
        self.assertEqual(registry.resolve_task_id("SelectByOrder", parameters={"index": 1, "from": "Right"}), "SelectByOrderLast_v0")
        self.assertEqual(registry.resolve_task_id("SelectByOrder", parameters={"index": 2, "from": "Left"}), "SelectByOrderSecond_v0")
        self.assertEqual(registry.resolve_task_id("PlaceByTopology", task_variant="Inside"), "PlaceByTopologyInside_v0")
        self.assertEqual(registry.resolve_task_id("PlaceByTopology", task_variant="OnTop"), "PlaceByTopologyOnTop_v0")
        for retired in ("PlaceDirectional", "PlaceDirectional_v0", "PlaceDirectionalVla_v0", "RelationBaseBase_PlaceDirectional_v0", "place_directional"):
            with self.subTest(retired=retired), self.assertRaises(KeyError): registry.resolve_task_id(retired)

    def test_insertion_event_is_preserved_while_history_requirement_is_deferred(self):
        initial = torch.tensor([[[.43,-.15],[.43,-.03],[.43,.08],[.43,.18]]])
        post = initial.clone(); post[0,3,1] = -.09
        spec = apply_task_contract({"protocol":{"intervention":{"script_name":"rightmost_to_between_first_and_second"},"gt_spec":{"resolver":"rank_after_intervention","order_axis":"y"}}}, "reindex_insert", {"parameters":{"from":"Left"}})
        gt = spec["protocol"]["gt_spec"]
        result = evaluate_state_reindexing_terminal_state(
            initial_xy=initial,post_xy=post,initial_rank_indices=torch.tensor([[0,1,2,3]]),moved_object_indices=torch.tensor([3]),
            displacement_tol=.02,script_name="rightmost_to_between_first_and_second",target_rank=gt["target_rank"],order_axis="y",
            planned_moved_mask=torch.tensor([[False,False,False,True]]),gt_spec=gt)
        self.assertEqual(result["gt_object_indices"].tolist(),[3])
        self.assertEqual(result["post_rank_indices"][0,1].item(),3)
        self.assertTrue(result["valid_episode"][0])
        self.assertFalse(spec["protocol"].get("requires_history", False))
        self.assertNotIn("initial_order_cue", spec["protocol"])

    def test_origin_uses_native_history_recipe_and_is_not_a_fake_legacy_alias(self):
        registry=GeoMindRegistry();registry.load()
        self.assertEqual(registry.get("TrackObjectOrigin").implementation_status,"implemented")
        self.assertIn("TrackObjectOriginClutter_v0",registry.list_task_ids(include_placeholders=False))
        self.assertFalse(registry.capabilities('TrackObjectOrigin')['isaaclab_legacy'])
        with self.assertRaisesRegex(NotImplementedError,"deferred"):
            apply_task_contract({},"track_clutter",{"task_id":"TrackObjectOrigin"})

    def test_generated_vla_definitions_preserve_public_variant_and_parameters(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from geomind.tools import generate_vla_task_suite as generator
        registry=GeoMindRegistry();registry.load()
        for task_id in ["PlaceByTopologyInside_v0","SelectByOrderLast_v0"]:
            source=registry.get_definition(task_id)
            with tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);src=folder/'source.json';src.write_text(json.dumps(source))
                out=folder/'vla';out.mkdir()
                with patch.object(generator,'OUT_DIR',out):generator.emit_task_json(src,{})
                generated=json.loads(next(out.glob('*.json')).read_text())
            self.assertEqual(generated['task_id'],source['task_id'])
            self.assertEqual(generated.get('variant'),source.get('variant'))
            self.assertEqual(generated.get('parameters'),source.get('parameters'))


    def test_adjacency_uses_declared_chain_not_nearest_distractor(self):
        names=["left","red","inserted","off_chain"]
        post=torch.tensor([[[.4,-.1],[.4,0.],[.4,-.05],[.401,-.001]]])
        gt={"order_axis":"y","reference_object_id":"red","relation":"Predecessor", "structure":{"type":"DirectedChain","nodes":["left","red","inserted"]}}
        self.assertEqual(structural_adjacency_indices(post,gt,names).tolist(),[2])
        self.assertLess(torch.linalg.vector_norm(post[0,3]-post[0,1]),torch.linalg.vector_norm(post[0,2]-post[0,1]))
        gt["relation"]="Successor"
        self.assertEqual(structural_adjacency_indices(post,gt,names).tolist(),[-1])


    def test_dynamic_target_binding_is_per_env_and_masks_invalid_histories(self):
        scene={name:SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor(values))) for name,values in {
            "a":[[1.,0.,0.],[2.,0.,0.]],"b":[[3.,0.,0.],[4.,0.,0.]]}.items()}
        env=SimpleNamespace(num_envs=2,device="cpu",scene=scene,_state_reindexing_runtime={"object_names":["a","b"],"gt_object_indices":torch.tensor([1,0]),"valid_episode":torch.tensor([True,False])})
        positions,valid=protocol_gt_positions(env,{"runtime_target_binding":"protocol_gt"})
        self.assertEqual(positions[:,0].tolist(),[3.,2.]);self.assertEqual(valid.tolist(),[True,False])


if __name__=="__main__":unittest.main()
