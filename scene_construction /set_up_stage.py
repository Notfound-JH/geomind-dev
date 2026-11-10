from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.world import World
import numpy as np
import random
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import add_reference_to_stage
from scene_construction.configs.asset_config import *
from scene_construction.configs.stage_config import *
#from scene_construction.configs.robot_config import robots_config
from scene_construction.scene_utils.add_light import add_dome_light
from scene_construction.scene_utils.add_planes import add_default_plane
from scene_construction.scene_utils.add_collision import add_collision_to_stage
from scene_construction.scene_utils.add_walls import add_walls
from scene_construction.arrange_object import arrange_objects
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_euler_angles

def add_camera(scene: Scene):
    camera_prim_path = find_unique_string_name(
        initial_name="/World/Camera", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
    from isaacsim.sensors.camera import Camera
    camera = Camera(
        name="left_camera",
        prim_path=camera_prim_path,
        position=[0.35, 3.2, 4],
        frequency=20,
        resolution=(224, 224),
        orientation=euler_angles_to_quats([0, 45, -90], degrees=True) ,
    )
    scene.add(camera)
    camera.initialize()

    camera_prim_path = find_unique_string_name(
        initial_name="/World/Camera", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
    camera = Camera(
        name="front_camera",
        prim_path=camera_prim_path,
        position=[1.4, 0.13, 1.8],
        frequency=20,
        resolution=(224, 224),
        orientation=euler_angles_to_quats([0, 45, 180], degrees=True) ,
    )
    scene.add(camera)
    camera.initialize()

    camera_prim_path = find_unique_string_name(
        initial_name="/World/Camera", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
    camera = Camera(
        name="ego_view_camera",
        prim_path=camera_prim_path,
        position=[-2, 0.0, 5],
        frequency=20,
        resolution=(224, 224),
        orientation=euler_angles_to_quats([0, 60, 0], degrees=True) ,
    )
    scene.add(camera)
    camera.initialize()

    camera_prim_path = find_unique_string_name(
        initial_name="/World/Camera", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
    camera = Camera(
        name="right_camera",
        prim_path=camera_prim_path,
        position=[0.35, -3.2, 4],
        frequency=20,
        resolution=(224, 224),
        orientation=euler_angles_to_quats([0, 45, 90], degrees=True) ,
    )
    scene.add(camera)
    camera.initialize()

def add_thing(scene, asset, position):
    obj_name = find_unique_string_name(
            initial_name=asset.name, is_unique_fn=lambda x: not scene.object_exists(x)
            )
    obj_prim_path = find_unique_string_name(
            initial_name=asset.prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
    add_reference_to_stage(asset.usd_path, prim_path=obj_prim_path)
    obj = SingleRigidPrim(prim_path=obj_prim_path, 
            name=obj_name,
            position=position,
            orientation=asset.orientation[0],
            scale=asset.scale
        )
    scene.add(obj)

def set_up_stage(scene: Scene, stage_asset: Optional[STAGEAsset] = None):
    
    if stage_asset is None:
        # stage_asset = random.choice([STAGE_1, STAGE_2, STAGE_3])
        stage_asset = STAGE_1


    for asset, position in zip(stage_asset.items, stage_asset.positions):   
        add_thing(scene, asset, position)

    
world = World()

add_camera(world.scene)

add_dome_light()

plane = random.choice(["wood_ground", "wood_ground_2", "wood_ground_3", "wood_ground_4"])
add_default_plane(plane)

add_walls(world.scene)


set_up_stage(world.scene, STAGE_3)


arrange_objects(task="Line", tabletop=STAGE_3.height+0.05, object_num=[3])

add_collision_to_stage()


world.reset()

for i in range(50000):
    world.step(render=True)

from pxr import Usd
import omni.usd
# get current stage
stage = omni.usd.get_context().get_stage()
# export to usda
# stage.Export("/home/elm/projects/geomind_assets/stage/stage_1/stage_1_empty.usda")


#simulation_app.close()


   