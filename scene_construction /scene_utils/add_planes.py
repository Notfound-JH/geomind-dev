# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd

texture_path = "/Isaac/Materials/Textures/Patterns/nv_wood_laminate_tile_light.jpg"

def add_plane_with_texture(scene, texture_path=None):
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from isaacsim.core.api.materials.omni_pbr import OmniPBR
    from isaacsim.core.api.materials.physics_material import PhysicsMaterial
    from isaacsim.core.api.objects.ground_plane import GroundPlane

    assets_root_path = get_assets_root_path()
    if texture_path is None:
        raise ValueError("Please provide a valid texture path.")
    asset_path = assets_root_path + texture_path

    textured_material = OmniPBR(
        prim_path="/World/visual_cube_material",
        name="omni_pbr",
        color=np.array([1, 0, 0]),
        texture_path=asset_path,
        texture_scale=[0.2, 0.2],
        texture_translate=[0.5, 0],
    )
    physics_material_path = find_unique_string_name(
        initial_name="/World/Physics_Materials/physics_material", is_unique_fn=lambda x: not is_prim_path_valid(x)
    )

    physics_material = PhysicsMaterial(
        prim_path=physics_material_path,
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8,
    )

    plane = GroundPlane(
                size = 2,
                prim_path="/World/GroundPlane",
                name="ground_plane",
                z_position=0,
                physics_material=physics_material,
                visual_material = textured_material,
            )
    
    scene.add(plane)

geomind_path = "/home/elm/projects/geomind_assets"
from pathlib import Path
def add_default_plane(plane_name):
    stage_context = omni.usd.get_context()
    current_stage = stage_context.get_stage()
    if current_stage is None:
        print("Error: No current stage found.")
        return False
    usd_path = f"{geomind_path}/ground/{plane_name}.usda"
    if not Path(usd_path).exists():
        usd_path = f"{geomind_path}/ground/{plane_name}.usd"
    prim_path = f"/World/{plane_name}"
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    return True

