"""CPU contract fixtures only; these synthetic reports are not physical evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from geomind.scene_assembly.audit import measured_bounds
from geomind.scene_assembly.task_binding import bind_and_solve
from geomind.episodes.resolver import EpisodeResolver
from geomind.episodes.runtime import evaluate_answer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestAssembledTaskBinding(unittest.TestCase):
    def fixture(self, root):
        source = root / "assembly"; source.mkdir()
        paths = [f"/World/AssembledObjects/blue_can_{i:03d}" for i in range(1, 5)]
        # Asymmetric geometry: body origin X and geometric centre X differ 3cm.
        points = np.asarray([[x,y,z] for x in (.01,.05) for y in (-.02,.02) for z in (-.03,.03)])
        raw = np.zeros((600,4,13));raw[:,:,3] = 1
        raw[:,:,:3] = [[-.30,-.10,.63],[-.10,.15,.63],[.10,-.1,.63],[.30,.1,.63]]
        columns = ["x","y","z","qw","qx","qy","qz","vx","vy","vz","wx","wy","wz"]
        np.savez_compressed(source / "object_states.npz", data=raw, dt_s=1/120, columns=columns, prim_paths=paths)
        np.savez_compressed(source / "asset_body_vertices.npz", points_m=points)
        (source / "scene.usda").write_text("#usda 1.0\n# Synthetic CPU fixture; not a physics execution.\n")
        for name in ("preview.png","detail_preview.png","settling.mp4"):
            (source / name).write_bytes(b"CPU fixture artifact, not rendered media")
        objects = []; metrics = []
        for path,state in zip(paths,raw[-1]):
            low,high = measured_bounds(points,state[:3],state[3:7])
            objects.append({"prim_path":path,"position_w_m":state[:3].tolist(),"orientation_wxyz":state[3:7].tolist(),
                            "bbox_center_xy_m":((low+high)*.5)[:2].tolist()})
            metrics.append({"prim_path":path,"measured_bbox_min_w_m":low.tolist(),"measured_bbox_max_w_m":high.tolist()})
        hashes = {name:digest(source/name) for name in ("preview.png","detail_preview.png","settling.mp4","object_states.npz","asset_body_vertices.npz")}
        report = {"executed":True,"passed":True,"backend":"isaacsim_physx","kind":"initial_layout_settling",
                  "physics_steps":600,"warmup_steps":20,"stable_steps":600,"dt_s":1/120,"object_count":4,"objects":metrics,
                  "state_trace":str(source/"object_states.npz"),"sha256":hashes}
        offline = {"passed":True,"stable_steps_recomputed":600,"offline_wrong_layout_control_rejected":True}
        manifest = {"kind":"initial_scene_assembly","status":"assembled_physics_passed","physics":report,"offline_audit":offline,
                    "request":{"object":{"count":4},"seed":0},"asset":{"name":"blue_can"},"objects":objects,
                    "table":{"height_m":.6,"bounds_xy_m":[-.6,.6,-.5,.5]},"scene_usda":str(source/"scene.usda"),
                    "asset_body_vertices":str(source/"asset_body_vertices.npz"),
                    "provenance":{"scene_usda_sha256":digest(source/"scene.usda"),"asset_body_vertices_sha256":hashes["asset_body_vertices.npz"]}}
        write_json(source/"physics_report.json",report);write_json(source/"offline_audit.json",offline)
        write_json(source/"manifest.json",manifest)
        return source/"manifest.json",manifest

    def specs(self):
        ids = [f"blue_can_{i:03d}" for i in range(1,5)]
        return [
            {"task_id":"SelectByOrder","parameters":{"index":2,"from":"Left","candidates":ids,"tie_tolerance_m":.001},"operation":{"type":"pick"}},
            {"task_id":"InferMidpoint","parameters":{"endpoints":[ids[0],ids[3]]},"operation":{"type":"place","moving_object":ids[1]}},
            {"task_id":"ProjectOntoLine","parameters":{"point":ids[2],"line":ids[:2],"geometry":"InfiniteLine3D","degeneracy_tolerance_m":1e-5},"operation":{"type":"place","moving_object":ids[3]}},
            {"task_id":"ReflectAcrossLine","parameters":{"point":ids[2],"line":ids[:2],"geometry":"InfiniteLine3D","degeneracy_tolerance_m":1e-5},"operation":{"type":"place","moving_object":ids[3]}},
        ]

    def test_four_static_tasks_use_measured_centres_and_existing_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path,_ = self.fixture(root)
            initial_hash = digest(path)
            for spec in self.specs():
                result = bind_and_solve(path,spec,root/spec["task_id"])
                self.assertEqual(result["status"],"gt_solved_task_action_unverified")
                self.assertTrue(result["checks"]["native_resolver_used"])
                self.assertTrue(result["checks"]["independent_gt_agrees"])
                self.assertTrue(result["checks"]["wrong_answer_rejected"])
                self.assertFalse(result["action_physics_executed"])
                self.assertFalse(result["task_completed"])
                self.assertEqual(result["source"]["query_physics_tick"],620)
                self.assertEqual(result["source"]["state_row"],599)
                self.assertAlmostEqual(result["objects"][0]["position_m"][0],-.27)
                self.assertAlmostEqual(result["objects"][0]["rigid_origin_world_m"][0],-.30)
                self.assertTrue(Path(result["native_config_root"]).name=="configs")
                public = json.loads(Path(result["public_observation_path"]).read_text())
                self.assertNotIn("ground_truth",public);self.assertNotIn("parameters",public)
                self.assertNotIn("operation",public);self.assertNotIn("provenance",public)
                self.assertEqual(public["axes"],{"right":"+x","forward":"+y","up":"+z"})
                self.assertEqual(public["units"],"m")
                self.assertFalse(public["input_contract"]["visual_only"])
                self.assertEqual(public["reference_point_definition"],"WorldAxisAlignedBoundingBoxCenter")
                self.assertIn("抓起" if spec["operation"]["type"]=="pick" else "放到",public["instruction"])
                self.assertNotIn("世界X",public["instruction"])
                scene = json.loads((Path(result["native_config_root"])/"scenes/assembled_scene.json").read_text())
                self.assertNotIn("physics",scene);self.assertEqual(scene["robot"],"not_configured")
                layout = json.loads((Path(result["native_config_root"])/"layouts/assembled_layout.json").read_text())
                self.assertEqual(layout["randomization"],{"position_jitter_m":0})
            self.assertEqual(digest(path),initial_hash)

    def test_semantic_identity_and_presentation_do_not_encode_source_or_task_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_=self.fixture(root)
            pick=bind_and_solve(path,self.specs()[0],root/"pick")
            place=bind_and_solve(path,self.specs()[1],root/"place")
            mapping=pick["source"]["identity_mapping"]
            self.assertEqual(mapping,place["source"]["identity_mapping"])
            self.assertEqual(set(mapping["physical_to_semantic"].values()),{f"obj_{i:03d}" for i in range(1,5)})
            self.assertEqual(mapping["semantic_to_physical"][pick["ground_truth"]["object_id"]],"blue_can_002")
            self.assertEqual(mapping["semantic_to_physical"][place["operation"]["moving_object"]],"blue_can_002")
            self.assertNotEqual(mapping["identity_seed_sha256"],mapping["presentation_seed_sha256"])
            self.assertNotEqual(list(mapping["physical_to_semantic"].values()),[f"obj_{i:03d}" for i in range(1,5)])
            public=json.loads(Path(pick["public_observation_path"]).read_text())
            public_order=[obj["object_id"] for obj in public["objects"]]
            self.assertNotEqual(public_order,[obj["id"] for obj in pick["objects"]])
            self.assertEqual(public_order,mapping["public_presentation_ids"])
            self.assertNotIn("identity_mapping",public)
            self.assertNotIn("source_object_id",json.dumps(public))
            self.assertNotIn("prim_path",json.dumps(public))
            self.assertNotIn("blue_can_001",json.dumps(public))
            self.assertTrue(all(obj["source_object_id"].startswith("blue_can_") for obj in pick["objects"]))
            self.assertTrue(all(obj["id"].startswith("obj_") for obj in pick["objects"]))
            instruction=public["instruction"]
            self.assertLess(instruction.index("1号蓝色罐"),instruction.index("2号蓝色罐"))
            self.assertLess(instruction.index("2号蓝色罐"),instruction.index("3号蓝色罐"))
            self.assertLess(instruction.index("3号蓝色罐"),instruction.index("4号蓝色罐"))
            different_target=self.specs()[0]
            different_target["parameters"]["candidates"].reverse()
            different_target["parameters"].update(index=1,**{"from":"Right"})
            other=bind_and_solve(path,different_target,root/"other_target")
            self.assertEqual(mapping,other["source"]["identity_mapping"])
            self.assertEqual(other["source"]["identity_mapping"]["semantic_to_physical"][other["ground_truth"]["object_id"]],"blue_can_004")

    def test_point_answers_use_one_millimetre_semantic_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_=self.fixture(root)
            result=bind_and_solve(path,self.specs()[1],root/"result")
            self.assertEqual(result["semantic_position_tolerance_m"],.001)
            resolved=EpisodeResolver(result["native_config_root"]).resolve(result["episode_id"])
            solved=json.loads(Path(result["ground_truth_path"]).read_text())
            point=list(result["ground_truth"]["position_m"])
            near=point.copy();near[0]+=.0009
            far=point.copy();far[0]+=.0011
            self.assertTrue(evaluate_answer(resolved,solved,{"position_m":near,"frame":"world"})["success"])
            self.assertFalse(evaluate_answer(resolved,solved,{"position_m":far,"frame":"world"})["success"])
            self.assertIn("not a mass centroid",result["reference_point_definition"])

    def test_different_measured_position_changes_gt_without_planned_pose_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);path,manifest = self.fixture(root)
            first = bind_and_solve(path,self.specs()[1],root/"first")
            # Move every final measured state by 2mm, within settling tolerance;
            # keep manifest planned origins unchanged and rebind fixture hashes.
            state_path=path.parent/"object_states.npz"
            with np.load(state_path,allow_pickle=False) as archive:
                data={key:archive[key] for key in archive.files}
            data["data"][:,:,0]+=.002
            np.savez_compressed(state_path,**data)
            manifest["physics"]["sha256"]["object_states.npz"]=digest(state_path)
            for metric in manifest["physics"]["objects"]:
                metric["measured_bbox_min_w_m"][0]+=.002;metric["measured_bbox_max_w_m"][0]+=.002
            write_json(path.parent/"physics_report.json",manifest["physics"]);write_json(path,manifest)
            second = bind_and_solve(path,self.specs()[1],root/"second")
            self.assertAlmostEqual(second["ground_truth"]["position_m"][0]-first["ground_truth"]["position_m"][0],.002)

    def test_missing_original_body_hash_and_tampered_artifact_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest=self.fixture(root)
            saved=deepcopy(manifest)
            del manifest["provenance"]["asset_body_vertices_sha256"]
            write_json(path,manifest)
            with self.assertRaisesRegex(ValueError,"Original body-point hash"):
                bind_and_solve(path,self.specs()[0],root/"missing")
            write_json(path,saved)
            (path.parent/"settling.mp4").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"Artifact hash changed"):
                bind_and_solve(path,self.specs()[0],root/"tampered")

    def test_mixed_body_clouds_and_asset_labels_are_bound_per_source_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest=self.fixture(root)
            body=path.parent/'asset_body_vertices.npz'
            with np.load(body,allow_pickle=False) as archive:shared=archive['points_m']
            other=shared.copy();other[:,0]=np.where(other[:,0]<.03,-.01,.09)
            np.savez_compressed(body,points_m=shared,other=other)
            manifest['objects'][0].update(body_points_key='other',asset_name='box')
            manifest['physics']['objects'][0]['measured_bbox_min_w_m'][0]=-.31
            manifest['physics']['objects'][0]['measured_bbox_max_w_m'][0]=-.21
            manifest['physics']['sha256']['asset_body_vertices.npz']=digest(body)
            manifest['provenance']['asset_body_vertices_sha256']=digest(body)
            write_json(path.parent/'physics_report.json',manifest['physics']);write_json(path,manifest)
            result=bind_and_solve(path,self.specs()[0],root/'mixed')
            by_source={obj['source_object_id']:obj for obj in result['objects']}
            self.assertAlmostEqual(by_source['blue_can_001']['position_m'][0],-.26)
            self.assertEqual(by_source['blue_can_001']['asset'],'box')
            self.assertEqual(by_source['blue_can_002']['asset'],'blue_can')

    def test_state_identity_order_and_dt_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest=self.fixture(root)
            state_path=path.parent/"object_states.npz"
            with np.load(state_path,allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
            data["prim_paths"]=data["prim_paths"][::-1]
            np.savez_compressed(state_path,**data)
            manifest["physics"]["sha256"]["object_states.npz"]=digest(state_path)
            write_json(path.parent/"physics_report.json",manifest["physics"]);write_json(path,manifest)
            with self.assertRaisesRegex(ValueError,"State prim paths"):
                bind_and_solve(path,self.specs()[0],root/"wrong_order")
            data["prim_paths"]=data["prim_paths"][::-1];data["dt_s"]=1/60
            np.savez_compressed(state_path,**data)
            manifest["physics"]["sha256"]["object_states.npz"]=digest(state_path)
            write_json(path.parent/"physics_report.json",manifest["physics"]);write_json(path,manifest)
            with self.assertRaisesRegex(ValueError,"Unexpected physics timestep"):
                bind_and_solve(path,self.specs()[0],root/"wrong_dt")

    def test_reference_moving_conflict_unknown_tasks_and_wrong_selected_id_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_=self.fixture(root)
            invalid=self.specs()[1];invalid["operation"]["moving_object"]="blue_can_001"
            with self.assertRaisesRegex(ValueError,"differ from all task references"):
                bind_and_solve(path,invalid,root/"conflict")
            invalid=self.specs()[0];invalid["operation"]["moving_object"]="blue_can_004"
            with self.assertRaisesRegex(ValueError,"differs from selected GT"):
                bind_and_solve(path,invalid,root/"wrong_selected")
            invalid=self.specs()[0];invalid["task_id"]="DetectSceneChange"
            with self.assertRaisesRegex(ValueError,"Unsupported assembled static task"):
                bind_and_solve(path,invalid,root/"dynamic")

    def test_existing_output_and_independent_disagreement_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_=self.fixture(root)
            result=bind_and_solve(path,self.specs()[1],root/"result")
            with self.assertRaisesRegex(ValueError,"overwrite"):
                bind_and_solve(path,self.specs()[1],root/"result")
            with patch("geomind.scene_assembly.task_binding._independent_gt",return_value={"type":"Point","position_m":[9,9,9],"frame":"world"}):
                with self.assertRaisesRegex(ValueError,"independent GT disagree"):
                    bind_and_solve(path,self.specs()[1],root/"disagree")


if __name__=="__main__":unittest.main()
