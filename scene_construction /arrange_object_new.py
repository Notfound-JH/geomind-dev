from typing import Optional
from typing import Union
from scene_construction.configs.asset_config import *
try:
    # object_config may not exist in all versions; try to import it if present
    from scene_construction.configs.object_config import *  # type: ignore
except Exception:
    pass
import random
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import get_stage_units
from scene_construction.scene_utils.poisson_disk_disks import poisson_disk_disks

class ObjectLayout:
    def __init__(self,
                 tabletop: float,
                 mid: np.ndarray,
                 interval_x: float = 0.1,
                 interval_y: float = 0.1,
                 offset: Optional[np.ndarray] = None,)-> None:
        self._midx = mid[0]
        self._midy = mid[1]
        self._tabletop = tabletop
        self._interval_x = interval_x
        self._interval_y = interval_y
        self._offset = offset
        if self._offset is None:
            self._offset = np.array([0.0, 0.0, 0.0])

    def build_line(self,
                   object_num: int = 3,
                   slope: float = 0,):
        positions = []
        for i in range(object_num):
            gap_x = self._interval_x * (i - (object_num - 1.0) / 2.0)
            gap_y = self._interval_y * (i - (object_num - 1.0) / 2.0)
            pos = np.array([self._midx + gap_x * slope, 
                            self._midy + gap_y * (1 - slope), 
                            self._tabletop]) + self._offset
            positions.append(pos)
        return positions
    
    def build_grid(self,
                   object_num_x: int = 2,
                   object_num_y: int = 2,):
        positions = []
        for i in range(object_num_x):
            gap_x = self._interval_x * (i - (object_num_x - 1.0) / 2.0)
            for j in range(object_num_y):
                gap_y = self._interval_y * (j - (object_num_y - 1.0) / 2.0)
                pos = np.array([self._midx + gap_x, 
                                     self._midy + gap_y, 
                                     self._tabletop]) + self._offset
                positions.append(pos)
        return positions
    
    def build_Tshaped(self,
                      type: str = "D",):
        x = self._midx + self._offset[0]
        y = self._midy + self._offset[1]
        z = self._tabletop + + self._offset[2]
        interval_x = self._interval_x
        interval_y = self._interval_y
        positions = []
        if type == "U":
            positions.append(np.array([x + interval_x, y, z]))
            positions.append(np.array([x + interval_x, y + interval_y, z]))
            positions.append(np.array([x + interval_x, y - interval_y, z]))
            positions.append(np.array([x, y, z]))
            positions.append(np.array([x - interval_x, y, z]))
        elif type == "D":
            positions.append(np.array([x - interval_x, y, z]))
            positions.append(np.array([x - interval_x, y + interval_y, z]))
            positions.append(np.array([x - interval_x, y - interval_y, z]))
            positions.append(np.array([x, y, z]))
            positions.append(np.array([x + interval_x, y, z]))
        elif type == "L":
            positions.append(np.array([x, y + interval_y, z]))
            positions.append(np.array([x + interval_x, y + interval_y, z]))
            positions.append(np.array([x - interval_x, y + interval_y, z]))
            positions.append(np.array([x, y, z]))
            positions.append(np.array([x, y - interval_y, z]))
        else:
            positions.append(np.array([x, y - interval_y, z]))
            positions.append(np.array([x + interval_x, y - interval_y, z]))
            positions.append(np.array([x - interval_x, y - interval_y, z]))
            positions.append(np.array([x, y, z]))
            positions.append(np.array([x, y + interval_y, z]))
        for i in range(len(positions)):
            positions[i] = positions[i] + self._offset
        return positions

    def random_arrange(self,
                       object_num: int = 3,
                       dx: float = 0,
                       dy: float = 0,):
        lx = self._midx - dx
        rx = self._midx + dx
        ly = self._midy - dy
        ry = self._midy + dy
        pos = poisson_disk_disks(rx - lx, ry - ly, self._interval_x, object_num)
        pos.sort()
        print("POS=======", object_num, pos)
        positions = []
        for i in range(object_num):
            positions.append(np.array([pos[i][0] + lx, 
                                       pos[i][1] + ly, 
                                       self._tabletop]) + self._offset)
        return positions
    
    def two_type(self,
                 object_num_1: int = 2,
                 object_num_2: int = 3,):
        line1 = self.build_line(object_num_1, 0)
        for i in range(len(line1)):
            line1[i][0] -= self._interval_x / 2.0

        line2 = self.build_line(object_num_2, 0)
        for i in range(len(line2)):
            line2[i][0] += self._interval_x / 2.0

        return line1 + line2

    
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)


class ObjectPlacing:
    def __init__(self):
        world = World.instance()
        self._scene = world.scene

    def add_object(self, object_asset, position, noise, znoise):
        obj_name = find_unique_string_name(
                initial_name=object_asset.name, is_unique_fn=lambda x: not self._scene.object_exists(x)
                )
        obj_prim_path = find_unique_string_name(
                initial_name=object_asset.prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
                )
        add_reference_to_stage(object_asset.usd_path, prim_path=obj_prim_path)


        prim = self._scene.stage.GetPrimAtPath(obj_prim_path)
        imageable = UsdGeom.Imageable(prim)
    
        local_bbox = imageable.ComputeLocalBound(Usd.TimeCode.Default(), "default")
        res_range = local_bbox.GetRange()
        z_min = res_range.GetMin()[2]  
    
        scale_z = object_asset.scale[2]
        bottom_offset = z_min * scale_z
        corrected_position = np.array(position)
        corrected_position[2] = position[2] - bottom_offset

        if noise is not None:
            pos_noise = np.array([random.uniform(-noise, noise),
                                  random.uniform(-noise, noise),
                                  0])
            corrected_position = corrected_position + pos_noise
        
    
        if znoise is not None:
            alpha = random.uniform(-znoise, znoise)
            rot_z = np.array([np.sin(alpha / 2.0),
                              0.0,
                              0.0,
                              np.cos(alpha / 2.0)], 
                              dtype=np.float64)
            rot = random.choice(object_asset.orientation)
            orientation = quat_mul(rot, rot_z)
        else:
            orientation = random.choice(object_asset.orientation)

        obj = SingleRigidPrim(prim_path=obj_prim_path,
                            name=obj_name,
                            position=corrected_position,
                            orientation=orientation,
                            scale=object_asset.scale,
                            )
        self._scene.add(obj)
        return obj_name

    def placing_object(self,
                       positions: list[np.ndarray],
                       objects: list[str] = ["BLUE_CAN"],
                       object_nums: list[int] = [0],
                       shuffle: bool = False,
                       noise: Optional[float] = None,
                       znoise: Optional[float] = None,):
        object_assets = []
        for i in range(len(objects)):
            object_asset = globals()[objects[i]]
            for j in range(object_nums[i]):
                object_assets.append(object_asset)

        # print("OBJECT ASSETS:", [obj.name for obj in object_assets])
        
        if shuffle is True:
            random.shuffle(object_assets)
        
        object_names = []
        for i in range(len(object_assets)):
            obj_name = self.add_object(object_assets[i], positions[i], noise, znoise)
            object_names.append(obj_name)
        return object_names
        
def arrange_objects(task: str = "Line",
                    tabletop: float = 0.8,
                    mid: np.ndarray = np.array([0.4, 0.1]),
                    interval: Union[float, tuple[float, float], None] = None,
                    objects: Optional[list[str]] = None,
                    object_num: list[int] = [4],
                    offset: Optional[np.ndarray] = None,):
    
    if isinstance(interval, float):
        interval_x = interval
        interval_y = interval
    elif isinstance(interval, tuple):
        interval_x = interval[0]
        interval_y = interval[1]
    else:
        if task == "Random":
            interval_x = 0.05
            interval_y = 0.05
        else:
            interval_x = random.uniform(0.11, 0.125)
            interval_y = random.uniform(0.11, 0.125)

    total_num = sum(object_num)

    layout = ObjectLayout(tabletop, mid, interval_x, interval_y, offset)
    placer = ObjectPlacing()

    if task == "Line":
        slope = random.choice([0, 0.5])
        if total_num <= 4:
            slope = random.choice([0, 1, 0.5])
        positions = layout.build_line(object_num=total_num, slope=slope)

        if objects is None:
            objects_pool = OBJECT_ALL
            if slope != 0:
                objects_pool += OBJECT_X
            if slope != 1:
                objects_pool += OBJECT_Y
            objects = random.choices(objects_pool, k=len(object_num))

        return placer.placing_object(positions, objects, object_num, shuffle=True, noise=0.02, znoise=np.pi/24)
    
    elif task == "Grid":
        object_num_x = 2
        while total_num % object_num_x != 0:
            object_num_x += 1
        object_num_x = random.choice([object_num_x, total_num // object_num_x])
        object_num_y = total_num // object_num_x
        positions = layout.build_grid(object_num_x, object_num_y)

        if objects is None:
            objects = random.choices(OBJECT_ALL + OBJECT_X, k=len(object_num))

        objects = ["WHITE_BOTTLE"]

        return placer.placing_object(positions, objects, object_num, shuffle=True, noise=0.02, znoise=np.pi)
    
    elif task == "Tshaped":
        type = random.choice(["U", "D", "L", "R"])
        positions = layout.build_Tshaped(type)

        if objects is None:
            objects = random.choices(OBJECT_ALL + OBJECT_X, k=len(object_num))

        return placer.placing_object(positions, objects, object_num, shuffle=True, noise=0.02, znoise=np.pi/24)
    
    elif task == "Random":
        positions = layout.random_arrange(object_num=total_num, dx=0.2, dy=0.2)

        if objects is None:
            objects = random.choices(OBJECT_ALL, k=len(object_num))

        return placer.placing_object(positions, objects, object_num, shuffle=True, noise=0, znoise=np.pi/24)
    
    elif task == "TwoType":
        positions = layout.two_type(object_num[0], object_num[1])

        if objects is None:
            objects = random.choices(OBJECT_ALL + OBJECT_X, k=len(object_num))
        print("OBJECTS:", objects)

        return placer.placing_object(positions, objects, object_num, shuffle=False, noise=0.02, znoise=np.pi/24)