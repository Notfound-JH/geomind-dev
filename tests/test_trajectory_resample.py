from __future__ import annotations

import unittest

from geomind_lab.data.trajectory import EpisodeCollector, resample_episode


class TestTrajectoryResample(unittest.TestCase):
    def test_preserves_actual_pose_and_explicit_sample_clock(self) -> None:
        collector = EpisodeCollector(metadata={"task_id": "test"}, sample_hz=10.0)
        for sim_step, timestamp_s in enumerate((1.0, 1.04, 1.10, 1.14, 1.20), start=24):
            collector.record_step(
                timestamp_s=timestamp_s,
                sim_step_index=sim_step,
                stage="approach",
                action_eef_abs=[timestamp_s, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                eef_pose_world=[timestamp_s - 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                gripper_command=1.0,
                joint_pos=[0.0],
                joint_vel=[0.0],
                gripper_state=[0.04],
            )
        sampled = resample_episode(collector.as_dict())
        self.assertEqual(sampled["steps"]["source_index"], [0, 2, 4])
        self.assertEqual(sampled["steps"]["sim_step_index"], [24, 26, 28])
        self.assertEqual(sampled["steps"]["eef_pose_world"][0][0], 0.99)
        self.assertNotEqual(sampled["steps"]["eef_abs"][0], sampled["steps"]["eef_pose_world"][0])
        self.assertEqual(sampled["steps"]["eef_delta_rel"][0], [0.0] * 7)
        self.assertEqual(sampled["metadata"]["segmenter_version"], "stage_v1")
        self.assertEqual(sampled["metadata"]["segment_source"], "stage_v1")
        self.assertIn("start_reason", sampled["segments"][0])
        self.assertIn("end_reason", sampled["segments"][0])

    def test_rejects_non_monotonic_timestamps(self) -> None:
        collector = EpisodeCollector(metadata={}, sample_hz=10.0)
        collector.record_step(
            timestamp_s=1.0,
            sim_step_index=1,
            stage="approach",
            action_eef_abs=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            eef_pose_world=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            gripper_command=1.0,
            joint_pos=[0.0],
            joint_vel=[0.0],
            gripper_state=[0.0],
        )
        with self.assertRaises(ValueError):
            collector.record_step(
                timestamp_s=0.5,
                sim_step_index=2,
                stage="approach",
                action_eef_abs=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                eef_pose_world=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                gripper_command=1.0,
                joint_pos=[0.0],
                joint_vel=[0.0],
                gripper_state=[0.0],
            )

    def test_event_kinematics_segment_source(self) -> None:
        collector = EpisodeCollector(metadata={"task_id": "test"}, sample_hz=10.0)
        for sim_step, (timestamp_s, x, gripper, grasped) in enumerate(
            ((0.0, 0.0, 1.0, False), (0.1, 0.1, 1.0, False), (0.2, 0.1, -1.0, True)),
            start=1,
        ):
            collector.record_step(
                timestamp_s=timestamp_s,
                sim_step_index=sim_step,
                stage="",
                action_eef_abs=[x, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
                eef_pose_world=[x, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
                gripper_command=gripper,
                joint_pos=[0.0],
                joint_vel=[0.0],
                gripper_state=[0.04],
                object_grasped=grasped,
            )
        sampled = resample_episode(collector.as_dict(), segment_source="event_kinematics_v2")
        self.assertEqual(sampled["metadata"]["segmenter_version"], "event_kinematics_v2")
        self.assertEqual(sampled["steps"]["atomic_id"], [0, 5, 3])
        self.assertEqual(sampled["segments"][2]["start_reason"], "grasp_event")
        self.assertEqual(sampled["steps"]["object_grasped"], [False, False, True])

    def test_raw_grasp_events_survive_between_10hz_samples(self) -> None:
        collector = EpisodeCollector(metadata={"task_id": "test"}, sample_hz=10.0)
        for sim_step, (timestamp_s, grasped) in enumerate(
            ((0.0, False), (0.04, True), (0.06, False), (0.10, False), (0.20, False)),
            start=1,
        ):
            collector.record_step(
                timestamp_s=timestamp_s,
                sim_step_index=sim_step,
                stage="",
                action_eef_abs=[0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
                eef_pose_world=[0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
                gripper_command=1.0,
                joint_pos=[0.0],
                joint_vel=[0.0],
                gripper_state=[0.04],
                object_grasped=grasped,
            )
        sampled = resample_episode(collector.as_dict(), segment_source="event_kinematics_v2")
        self.assertEqual(sampled["steps"]["object_grasped"], [False, False, False])
        self.assertEqual(sampled["steps"]["grasp_transition"], [0, 1, -1])
        self.assertEqual(sampled["steps"]["atomic_id"], [0, 3, 8])
        self.assertEqual(sampled["events"]["raw_grasp_transition_kind"], [1, -1])
        self.assertEqual(sampled["events"]["raw_grasp_transition_source_index"], [1, 2])
        projection_diag = sampled["metadata"]["segmenter_diagnostics"]["event_projection"]
        self.assertEqual(projection_diag["collision_count"], 1)
        self.assertEqual(projection_diag["spill_count"], 1)
        self.assertEqual(projection_diag["dropped_event_count"], 0)


if __name__ == "__main__":
    unittest.main()
