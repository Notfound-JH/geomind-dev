from __future__ import annotations

import unittest

from geomind_lab.data.trajectory.segmenter import (
    SEGMENTER_VERSION_EVENT_KINEMATICS_V2,
    SEGMENTER_VERSION_STAGE_V1,
    SegmentationConfig,
    build_segments,
    segment_records,
    segment_ids,
    stage_to_atomic_id,
)


class TestTrajectorySegmenter(unittest.TestCase):
    def test_stage_mapping_and_unknown_fallback(self) -> None:
        self.assertEqual(stage_to_atomic_id("approach"), 1)
        self.assertEqual(stage_to_atomic_id("close"), 3)
        self.assertEqual(stage_to_atomic_id("unknown"), 0)

    def test_contiguous_segments_cover_every_step(self) -> None:
        atomic_ids = [1, 1, 2, 2, 2, 4]
        self.assertEqual(segment_ids(atomic_ids), [0, 0, 1, 1, 1, 2])
        segments = build_segments(atomic_ids)
        self.assertEqual(
            [(segment.start_step, segment.end_step, segment.atomic_id) for segment in segments],
            [(0, 1, 1), (2, 4, 2), (5, 5, 4)],
        )
        self.assertEqual(segments[0].start_reason, "episode_start")
        self.assertEqual(segments[0].end_reason, "stage_change")
        self.assertEqual(segments[-1].end_reason, "episode_end")

    def test_segmenter_version_constant(self) -> None:
        self.assertEqual(SEGMENTER_VERSION_STAGE_V1, "stage_v1")
        self.assertEqual(SEGMENTER_VERSION_EVENT_KINEMATICS_V2, "event_kinematics_v2")

    def test_event_kinematics_v2_detects_motion_and_gripper_boundary(self) -> None:
        records = [
            {"timestamp_s": 0.0, "stage": "", "eef_pose_world": [0.0, 0.0, 0.2], "gripper_command": 1.0},
            {"timestamp_s": 0.1, "stage": "", "eef_pose_world": [0.1, 0.0, 0.2], "gripper_command": 1.0},
            {"timestamp_s": 0.2, "stage": "", "eef_pose_world": [0.1, 0.0, 0.2], "gripper_command": -1.0},
            {"timestamp_s": 0.3, "stage": "", "eef_pose_world": [0.1, 0.0, 0.25], "gripper_command": -1.0},
        ]
        result = segment_records(records, SegmentationConfig(source="event_kinematics_v2"))
        self.assertEqual(result.version, "event_kinematics_v2")
        self.assertEqual(result.atomic_ids, [0, 5, 3, 4])
        self.assertIn("gripper_command_change", [segment.start_reason for segment in result.segments])

    def test_event_kinematics_v2_uses_gripper_state_thresholds(self) -> None:
        records = [
            {
                "timestamp_s": 0.0,
                "stage": "",
                "eef_pose_world": [0.0, 0.0, 0.2],
                "gripper_command": 0.0,
                "gripper_state": [0.04],
            },
            {
                "timestamp_s": 0.1,
                "stage": "",
                "eef_pose_world": [0.0, 0.0, 0.2],
                "gripper_command": 0.0,
                "gripper_state": [0.005],
            },
            {
                "timestamp_s": 0.2,
                "stage": "",
                "eef_pose_world": [0.0, 0.0, 0.2],
                "gripper_command": 0.0,
                "gripper_state": [0.05],
            },
        ]
        result = segment_records(records, SegmentationConfig(source="event_kinematics_v2"))
        self.assertEqual(result.atomic_ids, [0, 3, 8])
        self.assertIn("gripper_state_change", [segment.start_reason for segment in result.segments])

    def test_event_kinematics_v2_prioritizes_grasp_and_release_events(self) -> None:
        records = [
            {"timestamp_s": 0.0, "eef_pose_world": [0.0, 0.0, 0.2], "gripper_command": -1.0, "object_grasped": False},
            {"timestamp_s": 0.1, "eef_pose_world": [0.0, 0.0, 0.2], "gripper_command": -1.0, "object_grasped": True},
            {"timestamp_s": 0.2, "eef_pose_world": [0.0, 0.0, 0.2], "gripper_command": -1.0, "object_grasped": False},
        ]
        result = segment_records(records, SegmentationConfig(source="event_kinematics_v2"))
        self.assertEqual(result.atomic_ids, [0, 3, 8])
        self.assertEqual(result.segments[1].start_reason, "grasp_event")
        self.assertEqual(result.segments[2].start_reason, "release_event")

    def test_event_kinematics_v2_merges_short_semantic_noise_only(self) -> None:
        records = [
            {"timestamp_s": 0.0, "stage": "transit_xy"},
            {"timestamp_s": 0.1, "stage": "hold"},
            {"timestamp_s": 0.2, "stage": "transit_xy"},
        ]
        result = segment_records(records, SegmentationConfig(source="event_kinematics_v2", min_segment_steps=2))
        self.assertEqual(result.atomic_ids, [5, 5, 5])
        self.assertEqual(result.diagnostics["merged_fragment_count"], 1)

        hard_records = [
            {"timestamp_s": 0.0, "stage": "transit_xy"},
            {"timestamp_s": 0.1, "stage": "close"},
            {"timestamp_s": 0.2, "stage": "transit_xy"},
        ]
        hard_result = segment_records(hard_records, SegmentationConfig(source="event_kinematics_v2", min_segment_steps=2))
        self.assertEqual(hard_result.atomic_ids, [5, 3, 5])
        self.assertEqual(hard_result.diagnostics["merged_fragment_count"], 0)

    def test_event_kinematics_v2_prefers_kinematics_over_stage_motion_defaults(self) -> None:
        records = [
            {
                "timestamp_s": 0.0,
                "stage": "transit_xy",
                "eef_pose_world": [0.0, 0.0, 0.2],
                "gripper_command": 0.0,
            },
            {
                "timestamp_s": 0.1,
                "stage": "transit_xy",
                "eef_pose_world": [0.0, 0.0, 0.26],
                "gripper_command": 0.0,
            },
        ]
        result = segment_records(records, SegmentationConfig(source="event_kinematics_v2"))
        self.assertEqual(result.atomic_ids, [5, 4])


if __name__ == "__main__":
    unittest.main()
