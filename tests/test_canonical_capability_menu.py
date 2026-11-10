import json
from pathlib import Path
import tempfile
import unittest

from geomind.tools.research_capabilities import build, _new_evidence
from geomind.scene_assembly.task_request import assembly_options


class CanonicalCapabilityInventoryTests(unittest.TestCase):
    def test_implemented_recipes_are_not_automatically_physical_passes(self):
        rows=build(new_batches=[])
        self.assertEqual(len(rows),27)
        new=[row for row in rows if row['registration_kind']=='AssembledNative']
        self.assertEqual(len(new),10)
        for row in new:
            self.assertTrue(row['native_execution_implemented'])
            self.assertFalse(any(case['task_completed'] for case in row['execution_evidence']['new_instances']))
        self.assertEqual(sum(len(row['execution_evidence']['historical_saved_instances']) for row in rows),20)

    def test_incomplete_newer_capability_evidence_does_not_select_old_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            task='SelectByRegion_v0';(old/task).mkdir();(new/task).mkdir()
            (old/task/'result.json').write_text(json.dumps({'task_completed':True,'status':'task_action_physics_passed'}))
            selected=_new_evidence(task,[old,new])
            self.assertEqual(selected['case_directory'],str(new/task))
            self.assertFalse(selected['task_completed'])

    def test_registered_manipulation_menu_is_separate_from_the_four_static_schemas(self):
        menu=assembly_options()
        self.assertEqual(menu['task_count'],4)
        self.assertEqual(menu['registered_manipulation_task_count'],30)
        self.assertEqual(menu['registered_manipulation_canonical_type_count'],27)
        entries=menu['registered_manipulation_tasks']
        self.assertEqual(sum(row['registration_kind']=='AssembledNative' for row in entries),10)
        self.assertTrue(all(row['runner_entrypoint']=='geomind.tools.regenerate_registered_tasks' for row in entries))

