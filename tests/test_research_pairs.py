"""CPU-only evidence checks; no tested model or physics execution."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from geomind.episodes import EpisodeResolver
from geomind.tools import research_pairs as rp


class TestResearchPairs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.root = self.run / "configs"
        self.index = rp.generate(self.root, 6, 17)
        self.pair = self.index["pairs"][0]

    def test_six_balanced_pairs_and_scoring(self):
        report = rp.validate(self.root, self.index, self.run / "reports")
        self.assertTrue(report["passed"])
        self.assertEqual(report["pairs_passed"], 6)
        self.assertEqual(report["correct_answer_checks"], 12)
        self.assertEqual(report["wrong_answer_checks"], 24)
        self.assertEqual(set(report["targets_a"].values()), {2})
        self.assertEqual(set(report["targets_b"].values()), {2})
        self.assertFalse(report["model_evaluated"])

    def test_history_differs_and_final_payload_is_identical(self):
        result = rp.validate_pair(self.root, self.pair)
        a, b = result["a"]["stream"], result["b"]["stream"]
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(rp.encoded(rp.condition_payload(a, "final-only")),
                         rp.encoded(rp.condition_payload(b, "final-only")))
        self.assertEqual(len(rp.condition_payload(a, "full-history")["observations"]), 6)
        self.assertLessEqual(result["final_only_rule"]["correct_count"], 1)

    def test_seed_replays_exact_configs_and_order(self):
        other = Path(self.temp.name) / "other" / "configs"
        self.assertEqual(self.index, rp.generate(other, 6, 17))
        for path in self.root.rglob("*.json"):
            self.assertEqual(path.read_bytes(), (other / path.relative_to(self.root)).read_bytes())

    def test_invalid_pairs_are_saved_and_rejected(self):
        result = rp.failure_examples(self.root, self.pair, self.run / "invalid")
        self.assertTrue(result["passed"], result)
        self.assertEqual({c["case"] for c in result["cases"]},
                         {"same_answer", "different_final", "multiple_changes", "threshold_boundary",
                          "outside_workspace", "protocol_conflict"})
        for case in result["cases"]:
            self.assertTrue((self.run / "invalid" / case["case"] / "rejection.json").is_file())

    def test_final_object_order_is_part_of_equality(self):
        path = self.root / "layouts" / f"{self.pair['b']}.json"
        layout = rp.load_json(path)
        layout["objects"].reverse()
        rp.write(path, layout)
        with self.assertRaises(rp.ValidationError) as caught:
            rp.validate_pair(self.root, self.pair)
        self.assertEqual(caught.exception.code, "different_final_observation")

    def test_oracle_does_not_call_production_solve(self):
        resolved = EpisodeResolver(self.root).resolve(self.pair["a"])
        with patch("geomind.episodes.runtime.solve", side_effect=AssertionError("production oracle forbidden")):
            expected = rp.independent_case(resolved)
        self.assertIn(expected["answer"], expected["final"])

    def test_corrupt_production_gt_is_detected(self):
        import geomind.episodes.runtime as runtime
        original = runtime.solve
        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            result["ground_truth"]["object_id"] = "wrong-id"
            return result
        with patch.object(runtime, "solve", side_effect=corrupt):
            with self.assertRaises(rp.ValidationError) as caught:
                rp.validate_pair(self.root, self.pair)
        self.assertEqual(caught.exception.code, "production_gt_mismatch")

    def test_payload_rejects_privileged_extra_fields(self):
        stream = rp.validate_pair(self.root, self.pair)["a"]["stream"]
        outer = deepcopy(stream)
        outer[-1]["pair_id"] = "private-index"
        with self.assertRaises(rp.ValidationError):
            rp.condition_payload(outer, "final-only")
        nested = deepcopy(stream)
        nested[-1]["objects"][0]["ground_truth"] = True
        with self.assertRaises(rp.ValidationError):
            rp.condition_payload(nested, "full-history")

    def test_existing_configs_are_not_overwritten(self):
        before = (self.root / "episodes" / f"{self.pair['a']}.json").read_bytes()
        with self.assertRaises(rp.ValidationError):
            rp.generate(self.root, 6, 99)
        self.assertEqual(before, (self.root / "episodes" / f"{self.pair['a']}.json").read_bytes())

    def test_snapshot_detects_config_and_source_changes(self):
        rp.snapshot(self.run, ["unittest"])
        self.assertTrue(rp.check_provenance(self.run)["passed"])
        config = self.root / "episodes" / f"{self.pair['a']}.json"
        config.write_text(config.read_text() + "\n")
        self.assertFalse(rp.check_provenance(self.run)["passed"])
        saved = self.run / "source_snapshot/geomind/tools/research_pairs.py"
        saved.write_text(saved.read_text() + "\n")
        self.assertTrue(any(s.startswith("saved_source:") for s in rp.check_provenance(self.run)["mismatches"]))


if __name__ == "__main__":
    unittest.main()
