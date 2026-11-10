"""CPU layout constraints only: no physics and no production pipeline change."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from geomind.scene_assembly.registered_layout_constraints import repair_grid_row_quantization, validate_grid_layout


def objects(points):
    return [{"name": f"cell_{index}", "kind": "primitive_cuboid", "size": [.03,.03,.03],
             "pose": {"xy": list(point), "yaw": .1}, "visual_color": [.8,.1,.1]}
            for index, point in enumerate(points)]


class GridLayoutConstraintTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"intervention": {"script_name": "static_scene_noop"}, "gt_spec": {
            "resolver": "grid_rowcol_slot", "grid_num_rows": 2, "grid_num_cols": 3,
            "row_axis": "x", "col_axis": "y", "grid_quant_step": .028,
            "target_row": 1, "target_col": 2, "min_center_separation": .024}}
        self.authored = objects([[x,y] for x in (.37,.5) for y in (-.15,-.03,.09)])
        self.actual_sample = objects([
            [.28253218816254955,-.1694278415423615],
            [.2863475223098758,-.038835546945126365],
            [.282046699421522,.09710041779857505],
            [.4348136198865484,-.16803130348522383],
            [.42604576931540955,-.032500666252030214],
            [.4307572054472683,.09244015802527877]])

    def test_real_seed_zero_cross_bucket_case_preserves_authored_cell_slots(self):
        original_authored = deepcopy(self.authored); original_sample = deepcopy(self.actual_sample)
        with self.assertRaisesRegex(ValueError, "quantized row/column"):
            validate_grid_layout(self.actual_sample, self.protocol)
        repaired, record = repair_grid_row_quantization(self.authored,self.actual_sample,self.protocol)
        checked = validate_grid_layout(repaired, self.protocol)
        self.assertEqual(checked["slots"], {f"cell_{i}": {"row": i//3+1, "column": i%3+1} for i in range(6)})
        self.assertEqual(self.authored, original_authored); self.assertEqual(self.actual_sample, original_sample)
        self.assertAlmostEqual(record["rows"][0]["nearest_bucket_center_m"], .28)
        self.assertAlmostEqual(record["rows"][1]["nearest_bucket_center_m"], .42)
        for index, correction in enumerate(record["object_corrections"]):
            self.assertEqual(correction["before_xy_m"][1], correction["after_xy_m"][1])
            self.assertEqual(correction["delta_xy_m"][1], 0.)
            self.assertLessEqual(abs(correction["delta_xy_m"][0]), .014+1e-12)
            self.assertEqual(repaired[index]["visual_color"], original_sample[index]["visual_color"])
        self.assertTrue(all(row["within_row_compression_factor"]==1 for row in record["rows"]))
        self.assertEqual(record["randomizer_additional_calls"], 0)
        self.assertFalse(record["seed_changed"]); self.assertFalse(record["gt_rule_changed"])

    def test_existing_randomizer_is_never_called_and_jitter_is_retained(self):
        with patch("geomind_lab.tasks.manager_based.manipulation.geomind.layout_randomizer.randomize_runtime_object_layout",
                   side_effect=AssertionError("must not resample")):
            repaired, _ = repair_grid_row_quantization(self.authored,self.actual_sample,self.protocol)
        for row in range(2):
            for col in range(1,3):
                before = self.actual_sample[row*3+col]["pose"]["xy"][0]-self.actual_sample[row*3]["pose"]["xy"][0]
                after = repaired[row*3+col]["pose"]["xy"][0]-repaired[row*3]["pose"]["xy"][0]
                self.assertAlmostEqual(before, after, places=14)

    def test_excessive_within_row_offsets_are_compressed_within_movement_bound(self):
        sampled = objects([[x,y] for mean in (.28,.42) for x,y in zip((mean-.018,mean,mean+.018),(-.15,-.03,.09))])
        repaired, record = repair_grid_row_quantization(self.authored,sampled,self.protocol)
        self.assertTrue(all(0<row["within_row_compression_factor"]<1 for row in record["rows"]))
        for row in range(2):
            mean = sum(obj["pose"]["xy"][0] for obj in repaired[row*3:(row+1)*3])/3
            self.assertAlmostEqual(mean, (.28,.42)[row])
            self.assertTrue(all(abs(obj["pose"]["xy"][0]-mean)<=.007+1e-12 for obj in repaired[row*3:(row+1)*3]))
        self.assertTrue(validate_grid_layout(repaired,self.protocol)["passed"])

    def test_y_row_axis_and_x_columns_use_same_original_slot_contract(self):
        protocol = deepcopy(self.protocol); protocol["gt_spec"].update(row_axis="y",col_axis="x")
        authored=deepcopy(self.authored);sample=deepcopy(self.actual_sample)
        for obj in authored+sample:obj["pose"]["xy"].reverse()
        repaired,record=repair_grid_row_quantization(authored,sample,protocol)
        self.assertEqual(record["row_axis"], "y")
        self.assertEqual(validate_grid_layout(repaired,protocol)["slots"]["cell_4"], {"row":2,"column":2})
        self.assertEqual([obj["pose"]["xy"][0] for obj in repaired],[obj["pose"]["xy"][0] for obj in sample])

    def test_cannot_group_invalid_authored_structure(self):
        bad=deepcopy(self.authored)
        for obj in bad:obj["pose"]["xy"][0]=.37
        with self.assertRaisesRegex(ValueError,"grouped|distinctly"):
            repair_grid_row_quantization(bad,self.actual_sample,self.protocol)
        bad=deepcopy(self.authored);bad[0]["pose"]["xy"][1]=.3
        with self.assertRaisesRegex(ValueError,"grouped|bands"):
            repair_grid_row_quantization(bad,self.actual_sample,self.protocol)

    def test_no_column_swap_shared_bucket_or_unbounded_repair_is_allowed(self):
        swapped=deepcopy(self.actual_sample)
        swapped[0]["pose"]["xy"][1],swapped[1]["pose"]["xy"][1]=swapped[1]["pose"]["xy"][1],swapped[0]["pose"]["xy"][1]
        with self.assertRaisesRegex(ValueError,"column order"):
            repair_grid_row_quantization(self.authored,swapped,self.protocol)
        overlap=objects([[x,y] for x in (.28,.29) for y in (-.15,-.03,.09)])
        with self.assertRaisesRegex(ValueError,"share their nearest"):
            repair_grid_row_quantization(self.authored,overlap,self.protocol)
        too_wide=objects([[x,y] for mean in (.28,.42) for x,y in zip((mean-.05,mean,mean+.05),(-.15,-.03,.09))])
        with self.assertRaisesRegex(ValueError,"movement bound"):
            repair_grid_row_quantization(self.authored,too_wide,self.protocol)

    def test_non_grid_and_mismatched_identities_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"only available for grid_rowcol_slot"):
            repair_grid_row_quantization(self.authored,self.actual_sample,{"gt_spec":{"resolver":"rank_initial_layout"}})
        with self.assertRaisesRegex(ValueError,"identities differ"):
            repair_grid_row_quantization(self.authored,self.actual_sample[:-1],self.protocol)
        bad=deepcopy(self.actual_sample);bad[0]["pose"]["xy"][0]=float("nan")
        with self.assertRaisesRegex(ValueError,"finite"):
            repair_grid_row_quantization(self.authored,bad,self.protocol)


if __name__=="__main__":unittest.main()
