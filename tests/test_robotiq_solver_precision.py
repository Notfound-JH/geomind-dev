"""Solver precision changes may not alter actuator or other profile settings."""
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from geomind.episodes.robot_profiles import robot_profile

class FakeConfig:
    def __init__(self, **values): self.__dict__.update(values)
    def copy(self): return copy.deepcopy(self)

class RobotiqPrecisionTests(unittest.TestCase):
    def test_robotiq_iteration_precision_and_all_other_fields_preserved(self):
        robotiq=FakeConfig(spawn=FakeConfig(usd_path='original_asset',func='original_spawn',
            articulation_props=FakeConfig(solver_position_iteration_count=32,solver_velocity_iteration_count=32,
                enabled_self_collisions=True)),actuators={'panda_shoulder':{'K':1},'panda_forearm':{'K':2},
                'hand':{'K':17,'D':.3}},unchanged={'mass':'asset-default','dt':'caller','materials':'asset-default'})
        panda=FakeConfig(actuators={'panda_shoulder':{'K':400},'panda_forearm':{'K':400}})
        module=types.ModuleType('isaaclab_assets.robots.franka')
        module.FRANKA_ROBOTIQ_GRIPPER_CFG=robotiq;module.FRANKA_PANDA_HIGH_PD_CFG=panda
        with patch.dict(sys.modules,{'isaaclab_assets.robots.franka':module}):
            result,arm,hand,finger=robot_profile('franka_robotiq')
        self.assertEqual(result.spawn.articulation_props.solver_position_iteration_count,32)
        self.assertEqual(result.spawn.articulation_props.solver_velocity_iteration_count,32)
        self.assertTrue(result.spawn.articulation_props.enabled_self_collisions)
        self.assertEqual(result.actuators,{'panda_shoulder':{'K':400},'panda_forearm':{'K':400},'hand':{'K':17,'D':.3}})
        self.assertEqual(result.unchanged,robotiq.unchanged)
        self.assertEqual(result.spawn.usd_path,'/home/elm/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd')
        self.assertEqual((arm,hand,finger),('panda_joint[1-7]','panda_hand','finger_joint'))
        self.assertEqual(robotiq.spawn.articulation_props.solver_position_iteration_count,32)
        self.assertEqual(robotiq.actuators['panda_shoulder'],{'K':1})

    def test_franka_and_fr3_do_not_get_robotiq_precision_override(self):
        panda=FakeConfig(spawn=FakeConfig(usd_path='panda_default',articulation_props=FakeConfig(
            solver_position_iteration_count=12,solver_velocity_iteration_count=1)),init_state=FakeConfig(joint_pos={'panda_joint1':.1}))
        fr3=FakeConfig(spawn=FakeConfig(usd_path='fr3_default',articulation_props=FakeConfig(
            solver_position_iteration_count=16,solver_velocity_iteration_count=2)),init_state=FakeConfig(joint_pos={}))
        assets=types.ModuleType('isaaclab_assets');assets.FRANKA_PANDA_HIGH_PD_CFG=panda
        module=types.ModuleType('isaaclab_assets.robots.fr3');module.FRANKA_FR3_HIGH_PD_CFG=fr3
        with patch.dict(sys.modules,{'isaaclab_assets':assets,'isaaclab_assets.robots.fr3':module}):
            p,*_=robot_profile('franka');f,*_=robot_profile('fr3')
        self.assertEqual((p.spawn.articulation_props.solver_position_iteration_count,p.spawn.articulation_props.solver_velocity_iteration_count),(12,1))
        self.assertEqual((f.spawn.articulation_props.solver_position_iteration_count,f.spawn.articulation_props.solver_velocity_iteration_count),(16,2))

if __name__=='__main__': unittest.main()
