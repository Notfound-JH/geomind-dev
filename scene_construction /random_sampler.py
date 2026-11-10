
try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    OpenAI = None
import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.core.utils.stage import add_reference_to_stage


from scene_construction.configs.asset_config import *
from scene_construction.configs.stage_config import *
from scene_construction.scene_utils.add_light import add_dome_light
from scene_construction.scene_utils.add_planes import add_default_plane
from scene_construction.scene_utils.add_collision import add_collision_to_stage
from scene_construction.scene_utils.add_walls import add_walls
from scene_construction.arrange_object import arrange_objects




DEFAULT_SYSTEM_PROMPT="""
        You are a 3D Scene Architect. 
        
        ### MANDATORY ASSET POOL (STRICT CLOSED SET):
        {knowledge_str}

        ### ABSOLUTE RULES:
        1. **NO HALLUCINATIONS**: You are FORBIDDEN to use any item name that is not in the list above. 
        2. **STRICT MATCHING**: You must use the EXACT string from the "name" field. Do not shorten or modify it.
        3. **EXCLUSION**: Do not select any items from 'catch_object' category.
        4. **STRUCTURE**:
           - ONE 'table' at [0,0,0].
           - ONE 'container' (if appropriate) on the table.
           - ONE 'decoration' on the table or in the container.
           - ONE 'furniture' on the floor around the table.
        
        ### SELECTION LIMIT:
        Select ONLY ONE unique item per category to keep the scene clean.

        ### OUTPUT FORMAT (JSON):
        {{
            "items": [
                {{
                    "name": "EXACT_NAME_FROM_LIST",
                    "pos_xy": [x, y],
                    "yaw_deg": 0,
                    "type": "table/furniture/decoration/container"
                }}
            ]
        }}
        """
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
   

def add_thing(scene, asset_dict, position):
  
    z_min = asset_dict.get("z_min_offset", 0.0)
    
    
    scale = np.array(asset_dict.get("scale", [1.0, 1.0, 1.0]))
    bottom_offset = z_min * scale[2]

    corrected_position = np.array(position, dtype=float)
    corrected_position[2] = position[2] - bottom_offset


    obj_name = find_unique_string_name(
        initial_name=asset_dict["name"], 
        is_unique_fn=lambda x: not scene.object_exists(x)
    )
    obj_prim_path = find_unique_string_name(
        initial_name=asset_dict["prim_path"], 
        is_unique_fn=lambda x: not is_prim_path_valid(x)
    )

    
    add_reference_to_stage(asset_dict["usd_path"], prim_path=obj_prim_path)
    
    
    orientation = np.array(asset_dict["orientations"][0])

    obj = SingleRigidPrim(
        prim_path=obj_prim_path, 
        name=obj_name,
        translation=corrected_position,
        orientation=orientation,
        scale=scale
    )
    scene.add(obj)

class SceneSampler:
    def __init__(
            self, 
            catalog_path,
            api_key=None,
            model= "gpt-4o",
            base_url = None,
            system_prompt= None
            ):
        with open(catalog_path, 'r', encoding='utf-8') as f:
            self.catalog = json.load(f)
        self.assets = self.catalog["assets"]
        self.registry = {item["name"].lower(): item for cat in self.assets for item in self.assets[cat]}
        self.table_heights = self.catalog.get("metadata", {}).get("table_heights", {})

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        self.model = model
        asset_info = self._build_asset_knowledge()
        system_prompt=f"""
        You are a 3D Scene Architect. Your task is to design a complete room layout including furniture placement and the arrangement of small manipulatable objects.

### MANDATORY ASSET POOL (STRICT CLOSED SET):
{asset_info}

### ABSOLUTE RULES:
1. **NO HALLUCINATIONS**: Use ONLY exact names from the pool.
2. **ROOM BOUNDARY**: Room 5m x 5m (X/Y: [-2.5, 2.5]).
3. **STRUCTURE & PLACEMENT**:
   - **Table**: EXACTLY ONE 'table' at [0, 0, 0].
   - **Basket/Container**: Must be placed ON the table. 
   - **[NEW] ZERO OVERHANG**: The container's entire footprint must stay within the table's dimensions. No part may hang off the edge.
   - **Furniture/Decor**: On FLOOR (Z=0). No overlap with table footprint.
4. **MANIPULATABLE OBJECTS**: 
   - Small objects (catch_objects) on the table.
   - Choose ONE pattern: [List, Grid, Tshaped, Random, TwoType, Ring]. Quantity: 1-5.
5. **[NEW] NO OVERLAP**: All items (furniture, containers, and small objects) must have distinct bounding boxes. Overlapping is strictly FORBIDDEN.

### SELECTION LIMIT:
Select ONLY ONE unique item per category (one table, one chair, one basket, etc.).

### OUTPUT FORMAT (JSON):
{{
    "items": [
        {{
            "name": "EXACT_NAME_FROM_LIST",
            "pos_xy": [x, y],
            "yaw_deg": 0,
            "type": "table/furniture/decoration/container"
        }}
    ],
    "arrangement_config": {{
        "pattern": "List/Grid/Tshaped/Random/TwoType/Ring",
        "object_count": 3
    }}
}}
        """
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        
       
        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = OpenAI(**client_kwargs)
    
    def _build_asset_knowledge(self):
        knowledge = {}
        for cat, items in self.assets.items():
            if cat == "catch_object": 
                continue  
            
            knowledge[cat] = []
            for i in items:
           
                scale = i.get("scale", [1.0, 1.0, 1.0])
          
                raw_dims = i.get("dimensions", {"x": 0.5, "y": 0.5, "z": 0.5})
            
            
                real_size = {
                    "x": round(raw_dims["x"] * scale[0], 3),
                    "y": round(raw_dims["y"] * scale[1], 3),
                    "z": round(raw_dims["z"] * scale[2], 3)
                }
            
                knowledge[cat].append({
                    "name": i["name"], 
                    "real_size_meters": real_size  # 明确告诉 LLM 这是米为单位
                })
            
        return json.dumps(knowledge, indent=2)
    def generate_config_by_llm(self, user_prompt):
        """调用 LLM 生成场景配置"""
        if not self.client:
            print("Error: OpenAI client not initialized. Please provide api_key.")
            return None
        
        

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"} 
            )
            config = json.loads(response.choices[0].message.content)
            
            # 简单清洗：确保 LLM 选的资产确实在 registry 里
            config["furniture"] = [f for f in config.get("furniture", []) if f.lower() in self.registry]
            config["decoration"] = [d for d in config.get("decoration", []) if d.lower() in self.registry]
            
            return config
        except Exception as e:
            print(f"LLM Generation Failed: {e}")
            return None
    
    def get_scene_layout(self, llm_config):
        layout = []
        table_z_top = 0.0

        for item in llm_config.get("items", []):
            name_raw = item["name"]
            name_lower = name_raw.lower()
            
            if name_lower in self.registry and item["type"] == "table":
                asset = self.registry[name_lower]
             
                table_key = name_raw.upper() 
                table_z_top = self.table_heights.get(table_key, 0.75) 
                
                print(f"检测到桌子: {table_key}, 桌面高度设置为: {table_z_top}")
                
                layout.append({
                    "asset": asset, 
                    "pos": [0.0, 0.0, 0.0], 
                    "yaw": item.get("yaw_deg", 0)
                })
                break

        for item in llm_config.get("items", []):
            name_lower = item["name"].lower()
            if name_lower in self.registry and item["type"] != "table":
                asset = self.registry[name_lower]
                pos_xy = item.get("pos_xy", [0.0, 0.0])
                
           
                pos_z = 0.0 
                if item["type"] == "container": 
                    pos_z = table_z_top
                    
                
                layout.append({
                    "asset": asset,
                    "pos": [pos_xy[0], pos_xy[1], pos_z],
                    "yaw": item.get("yaw_deg", 0)
                })
        
        return layout,table_z_top

world = World()
add_dome_light()
plane = random.choice(["wood_ground", "wood_ground_2"])
add_default_plane(plane)
add_walls(world.scene)

sampler = SceneSampler(
    catalog_path="scene_construction/assets_catalog_with_metadata.json",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"))
manual_config = {
    "table": "table_5", 
    "furniture": ["chair_2", "chair_2", "chair_2"], 
    "decoration": ["laptop", "mug"]}
user_query = "Generate a 3D indoor scene: Place a rectangular table in the center of the room. Place a basket (container) on the north edge of the table(on the table). On the tabletop, arrange 5 bottles in a 'Ring' pattern. Position a chair next to the table with the backrest facing the table. Finally, place some green plants (decorations) near the walls."
llm_config = sampler.generate_config_by_llm(user_query)
if llm_config:
    save_dir = "scene_construction/generated_config"
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用时间戳命名，防止覆盖
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_filename = f"{save_dir}/scene_config_{timestamp}.json"
    
    # 3. 将 LLM 生成的内容写入文件
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(llm_config, f, indent=4, ensure_ascii=False)
    
    print(f"✅ LLM 场景配置已保存至: {json_filename}")
    scene_data,table_height = sampler.get_scene_layout(llm_config)
    for entry in scene_data:
        add_thing(world.scene, entry["asset"], entry["pos"])
    
    table_entry = next((item for item in scene_data if "table" in item["asset"]["name"].lower()), None)
    if table_entry:
        
        asset = table_entry["asset"]
        table_pos_world = np.array(table_entry.get("pos", [0.0, 0.0, 0.0]))[:2] 
        t_dims = np.array([
            asset.get("dimensions", {}).get("x", 1.0) * asset.get("scale", [1,1,1])[0],
            asset.get("dimensions", {}).get("y", 1.0) * asset.get("scale", [1,1,1])[1]
        ])
        if np.any(t_dims > 5.0): t_dims /= 100.0 # 单位纠正
        desired_mid = np.array([0.2, 0.1]) 
    
   
        limit_x = (t_dims[0] / 2.0) - 0.15
        limit_y = (t_dims[1] / 2.0) - 0.15
    
        final_mid = np.array([
            np.clip(desired_mid[0], table_pos_world[0] - limit_x, table_pos_world[0] + limit_x),
            np.clip(desired_mid[1], table_pos_world[1] - limit_y, table_pos_world[1] + limit_y)
    ])
        # 排列操作物体
        arrange_config = llm_config.get("arrangement_config", {})
        arrange_objects(
            task=arrange_config.get("pattern", "Random"),
            tabletop=table_height+0.1,
            mid=np.array(final_mid),
            table_dims=t_dims,
            object_num=[arrange_config.get("object_count", 3)]
        )
add_collision_to_stage()


world.reset()
  

for i in range(500):
    world.step(render=True)


from pxr import Usd
import omni.usd

stage = omni.usd.get_context().get_stage()
stage.Export("/home/elm/projects/geomind_assets/stage2_scenes/stage_1_fixed.usda")
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
