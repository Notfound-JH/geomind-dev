from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from geomind.episodes import ContractError, EpisodeResolver, Snapshot, solve
from geomind.episodes.resolver import load_json


EXAMPLES = Path(__file__).resolve().parents[1] / "geomind/configs/episode_v3_examples"


class TestEpisodeContractV3(unittest.TestCase):
    def setUp(self):
        self.resolver = EpisodeResolver(EXAMPLES)

    def state(self, resolved, *, step=0, instance="run_1", positions=None):
        return Snapshot.from_dict({
            "instance_id": instance, "contract_hash": resolved.contract_hash,
            "frame": "table", "step": step,
            "positions_m": positions if positions is not None else {
                "block_a": [-0.12, 0, 0.02], "block_b": [-0.04, 0, 0.02],
                "block_c": [0.04, 0, 0.02], "block_d": [0.12, 0, 0.02],
            },
        })

    def test_selection_uses_measured_state_not_layout_order(self):
        resolved = self.resolver.resolve("select_second")
        positions = self.state(resolved).to_dict()["positions_m"]
        positions["block_b"], positions["block_d"] = positions["block_d"], positions["block_b"]
        result = solve(resolved, self.state(resolved, positions=positions))
        self.assertEqual(result["ground_truth"], {"type": "Object", "object_id": "block_d"})
        self.assertEqual(result["provenance"]["calculation"]["ordered_candidates"],
                         ["block_a", "block_d", "block_c", "block_b"])

    def test_ties_rejected_even_with_distinct_object_ids(self):
        resolved = self.resolver.resolve("select_second")
        positions = self.state(resolved).to_dict()["positions_m"]
        positions["block_b"][0] = positions["block_a"][0] + 0.00001
        with self.assertRaisesRegex(ContractError, "Ambiguous order"):
            solve(resolved, self.state(resolved, positions=positions))

    def test_midpoint_is_explicitly_three_dimensional(self):
        resolved = self.resolver.resolve("infer_midpoint")
        positions = self.state(resolved).to_dict()["positions_m"]
        positions["block_d"] = [0.08, 0.2, 0.08]
        result = solve(resolved, self.state(resolved, positions=positions))
        for actual, expected in zip(result["ground_truth"]["position_m"], [-0.02, 0.1, 0.05]):
            self.assertAlmostEqual(actual, expected)

    def test_memory_uses_historical_location_not_current_location(self):
        resolved = self.resolver.resolve("recall_blue_location")
        reference = self.state(resolved, step=60)
        positions = reference.to_dict()["positions_m"]
        positions["block_b"] = [0.2, 0.1, 0.02]
        result = solve(resolved, self.state(resolved, step=180, positions=positions),
                       history={"observe.End": reference})
        self.assertEqual(result["ground_truth"]["position_m"], [-0.04, 0.0, 0.02])
        self.assertEqual(result["provenance"]["reference"], "observe.End")

    def test_memory_missing_wrong_run_wrong_boundary_and_early_recall_fail(self):
        resolved = self.resolver.resolve("recall_blue_location")
        for history, step, error in (
            ({}, 180, "unavailable"),
            ({"observe.End": self.state(resolved, step=60, instance="run_2")}, 180, "different run"),
            ({"observe.End": self.state(resolved, step=59)}, 180, "Observe.End"),
            ({"observe.End": self.state(resolved, step=60)}, 179, "delay completed"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ContractError, error):
                solve(resolved, self.state(resolved, step=step), history=history)

    def test_changed_seed_invalidates_snapshot(self):
        resolved = self.resolver.resolve("select_second")
        state = self.state(resolved).to_dict()
        state["contract_hash"] = "0" * 64
        with self.assertRaisesRegex(ContractError, "different configuration"):
            solve(resolved, Snapshot.from_dict(state))

    def test_nonfinite_state_and_missing_objects_fail(self):
        resolved = self.resolver.resolve("select_second")
        state = self.state(resolved).to_dict()
        state["positions_m"]["block_a"][0] = float("nan")
        with self.assertRaisesRegex(ContractError, "finite"):
            Snapshot.from_dict(state)
        positions = self.state(resolved).to_dict()["positions_m"]
        positions.pop("block_a")
        with self.assertRaisesRegex(ContractError, "identities"):
            solve(resolved, self.state(resolved, positions=positions))

    def test_mutation_after_resolution_is_rejected(self):
        resolved = self.resolver.resolve("select_second")
        resolved.episode["parameters"]["index"] = 1
        with self.assertRaisesRegex(ContractError, "mutated"):
            solve(resolved, self.state(resolved))

    def test_generated_bad_parameters_are_rejected_at_load_time(self):
        original = self.resolver.resolve("select_second").episode
        mutations = [
            lambda d: d["parameters"].update(index=True),
            lambda d: d["parameters"].update(index=0),
            lambda d: d["parameters"].update(candidates=["missing"]),
            lambda d: d["parameters"].update(frame="camera"),
            lambda d: d["parameters"].update(tie_tolerance_m=0),
            lambda d: d["parameters"].update(typo=1),
            lambda d: d.update(protocol_id="observe_delay_recall"),
            lambda d: d.pop("seed"),
        ]
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "configs"
            shutil.copytree(EXAMPLES, root)
            resolver = EpisodeResolver(root)
            for index, mutate in enumerate(mutations):
                data = deepcopy(original)
                mutate(data)
                (root / "episodes/select_second.json").write_text(json.dumps(data))
                with self.subTest(index=index), self.assertRaises(ContractError):
                    resolver.resolve("select_second")

    def test_all_examples_validate_and_resolution_is_reproducible(self):
        for episode in ("select_second", "infer_midpoint", "recall_blue_location"):
            first, second = self.resolver.resolve(episode), self.resolver.resolve(episode)
            self.assertEqual(first.contract_hash, second.contract_hash)

    def test_duplicate_json_keys_and_nonfinite_constants_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for contents in ('{"index": 1, "index": 2}', '{"value": NaN}'):
                path.write_text(contents)
                with self.subTest(contents=contents), self.assertRaises(ContractError):
                    load_json(path)


if __name__ == "__main__":
    unittest.main()
