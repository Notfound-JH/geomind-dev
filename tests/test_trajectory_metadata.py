from __future__ import annotations

import unittest

from geomind_lab.data.trajectory import build_trajectory_metadata_contract


class _ArmAction:
    scale = 0.5


class _Actions:
    arm_action = _ArmAction()


class _EnvCfg:
    actions = _Actions()


class TestTrajectoryMetadata(unittest.TestCase):
    def test_contract_contains_sampling_action_objects_and_ground_truth(self) -> None:
        objects = {
            "cube": {
                "object_id": "cube",
                "position": [0.1, 0.2, 0.3],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        }
        metadata = build_trajectory_metadata_contract(
            env_cfg=_EnvCfg(),
            task_spec={
                "protocol": {"gt_spec": {"resolver": "directional_region_placement"}},
                "goal_sequence": [{"goals": [{"object_id": "cube"}]}],
            },
            sample_hz=10.0,
            initial_object_poses=objects,
        )
        self.assertEqual(metadata["sample_frequency_hz"], 10.0)
        self.assertEqual(metadata["action_scale"], 0.5)
        self.assertEqual(metadata["object_info"], objects)
        self.assertEqual(metadata["ground_truth"]["gt_spec"]["resolver"], "directional_region_placement")
        self.assertEqual(metadata["ground_truth"]["goal_sequence"][0]["goals"][0]["object_id"], "cube")


if __name__ == "__main__":
    unittest.main()