from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from geomind.tools import research_projection as rp
from geomind.tools.research_pairs import ValidationError, check_provenance


class TestProjectionResearch(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)
        self.root=self.run/"configs"
        self.manifest=rp.generate(self.root,7)

    def test_current_config_and_three_diagnostic_pairs(self):
        result=rp.verify(self.root,self.manifest,self.run/"reports")
        self.assertEqual(result["answer_checks"],77)
        self.assertFalse(result["model_evaluated"])

    def test_bad_independent_target_is_rejected(self):
        case=deepcopy(self.manifest["cases"][1]); case["expected"][0]+=0.01
        with self.assertRaises(ValidationError):
            rp.verify_case(self.root,case,self.run/"bad")

    def test_generation_never_calls_production_solve(self):
        with patch("geomind.episodes.runtime.solve",side_effect=AssertionError("forbidden")):
            manifest=rp.generate(self.run/"independent",7)
        self.assertEqual(manifest,self.manifest)

    def test_static_index_uses_shared_provenance_checks(self):
        rp.snapshot(self.run,["test"],index_filename="case_index.json",extra_paths=["tests/test_research_projection.py"])
        self.assertTrue(check_provenance(self.run)["passed"])
        index=self.run/"case_index.json"
        index.write_text(index.read_text()+"\n")
        self.assertFalse(check_provenance(self.run)["passed"])


if __name__ == "__main__":
    unittest.main()
