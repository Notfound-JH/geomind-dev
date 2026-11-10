from __future__ import annotations

import importlib.util
import unittest
from argparse import Namespace
from pathlib import Path
import tempfile


_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_anchor_reference_trajectory_dataset.py"
_SPEC = importlib.util.spec_from_file_location("collect_anchor_reference_trajectory_dataset", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


class TestCollectAnchorReferenceTrajectoryDataset(unittest.TestCase):
    def test_protocol_adapters_are_present(self) -> None:
        self.assertEqual(
            set(_RUNNER.PROTOCOL_RUNNERS),
            {"anchor_reference", "geometric_inference", "spatial_placement"},
        )
        self.assertTrue(all(path.is_file() for path in _RUNNER.PROTOCOL_RUNNERS.values()))

    def test_phase2_retains_fixed_trajectory_seed(self) -> None:
        args = Namespace(vary_trajectory_seed=False, trajectory_augmentation_enable=False)
        self.assertFalse(_RUNNER._should_vary_trajectory_seed(args))
        seeds = _RUNNER._trajectory_seeds(7, 0, 5, vary=_RUNNER._should_vary_trajectory_seed(args))
        self.assertEqual(len(set(seeds)), 1)

    def test_phase5_augmentation_uses_independent_trajectory_seeds(self) -> None:
        args = Namespace(vary_trajectory_seed=False, trajectory_augmentation_enable=True)
        self.assertTrue(_RUNNER._should_vary_trajectory_seed(args))
        seeds = _RUNNER._trajectory_seeds(7, 0, 5, vary=_RUNNER._should_vary_trajectory_seed(args))
        self.assertEqual(len(seeds), 5)
        self.assertEqual(len(set(seeds)), 5)

    def test_resume_requires_completed_layout_and_all_h5_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "trajectory.h5"
            h5_path.write_bytes(b"fixture")
            layout = {
                "layout_index": 0,
                "jitter_seed": 11,
                "shift_seed": 12,
                "returncode": 0,
                "collection_complete": True,
                "episodes": [{"h5_path": str(h5_path)}],
            }
            self.assertTrue(_RUNNER._layout_is_complete(layout, layout_index=0, jitter_seed=11, shift_seed=12, rollout_count=1))
            layout["collection_complete"] = False
            self.assertFalse(_RUNNER._layout_is_complete(layout, layout_index=0, jitter_seed=11, shift_seed=12, rollout_count=1))

    def test_episode_indexes_report_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layouts = [
                {
                    "layout_index": 0,
                    "jitter_seed": 1,
                    "shift_seed": 2,
                    "returncode": 0,
                    "collection_complete": True,
                    "episodes": [
                        {"episode_uid": "duplicate", "rollout_index": 0, "trajectory_seed": 3, "h5_path": "a.h5", "success": True},
                        {"episode_uid": "duplicate", "rollout_index": 1, "trajectory_seed": 4, "h5_path": "b.h5", "success": False},
                    ],
                }
            ]
            report = _RUNNER._write_episode_indexes(Path(directory), layouts)
            self.assertEqual(report["episode_count"], 2)
            self.assertEqual(report["duplicate_episode_keys"], ["duplicate"])
            quality = _RUNNER._quality_report(layouts, report, None)
            self.assertEqual(quality["success_count"], 1)
            self.assertFalse(quality["passed"])

    def test_resume_rejects_changed_collection_configuration(self) -> None:
        args = Namespace(
            layouts=1,
            rollouts_per_layout=5,
            meta_seed=0,
            layout_key="layout",
            scene_id=65,
            trajectory_augmentation_enable=True,
            trajectory_augmentation_candidates=12,
            resume=True,
            dry_run=False,
            verify_phase2=False,
            verify_phase3=False,
            verify_phase5=False,
            verify_phase6=False,
            keep_going=False,
            out_dir=Path("new-output"),
        )
        existing = {"config": {key: value for key, value in vars(args).items() if key not in {"resume", "out_dir"}}}
        existing["config"]["trajectory_augmentation_candidates"] = 8
        self.assertFalse(_RUNNER._resume_config_matches(existing, args))


if __name__ == "__main__":
    unittest.main()