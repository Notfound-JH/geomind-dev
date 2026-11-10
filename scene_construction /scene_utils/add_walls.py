from isaacsim.core.api.world import World
import numpy as np
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.rotations import euler_angles_to_quat
from scene_construction.configs.asset_config import *
from scene_construction.configs.stage_config import *


def add_walls(scene: Scene, size: float = 5.0):
    
    x_position = size / 2
    z_postion = x_position / 4
    stage_units = get_stage_units()
    positions = np.array([[-x_position, 0, z_postion],[x_position,0,z_postion],[0,x_position,z_postion],[0,-x_position,z_postion]]) / stage_units
    orientation_rz = euler_angles_to_quat(np.array([0, 0, np.pi / 2]))
    orientations = [np.array([1,0,0,0]),np.array([1,0,0,0]),orientation_rz,orientation_rz]
    walls = []

    for pos, orient in zip(positions, orientations):
        cube_prim_path = find_unique_string_name(
            initial_name="/World/Wall", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        cube_name = find_unique_string_name(initial_name="wall", is_unique_fn=lambda x: not scene.object_exists(x))
        walls.append(scene.add(
            VisualCuboid(
                name=cube_name,
                position=pos,
                orientation=orient,
                prim_path=cube_prim_path,
                size=size,
                scale=np.array([0.01, 1, 0.5]) / stage_units,
                color=np.array([139, 69, 19]),
        )
    ))
    return cube_name, walls