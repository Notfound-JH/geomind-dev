"""The mount exclusion must be narrow and fail on unknown asset topology."""
import unittest
from pxr import Usd, UsdPhysics
from geomind.episodes.robot_profiles import filter_robotiq_mount

class TestRobotiqMountFilter(unittest.TestCase):
    def test_only_mount_and_housing_pairs_are_filtered(self):
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim('/Robot', 'Xform')
        UsdPhysics.ArticulationRootAPI.Apply(root)
        mount = stage.DefinePrim('/Robot/Robotiq_2F_85_edit/Robotiq_2F_85/base_link', 'Xform')
        wrist = stage.DefinePrim('/Robot/panda_link7', 'Xform')
        other = stage.DefinePrim('/Robot/panda_link6', 'Xform')
        links = [stage.DefinePrim('/Robot/Robotiq_2F_85_edit/Robotiq_2F_85/' + name, 'Xform') for name in ('left_inner_knuckle', 'right_inner_knuckle')]
        filter_robotiq_mount(stage, '/Robot')
        self.assertEqual(UsdPhysics.FilteredPairsAPI(mount).GetFilteredPairsRel().GetTargets(), [wrist.GetPath(), *[link.GetPath() for link in links]])
        self.assertFalse(other.HasAPI(UsdPhysics.FilteredPairsAPI))
        self.assertFalse(wrist.HasAPI(UsdPhysics.FilteredPairsAPI))

    def test_unknown_mount_topology_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'topology changed'):
            filter_robotiq_mount(Usd.Stage.CreateInMemory(), '/Robot')
