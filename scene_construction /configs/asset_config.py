from geomind_lab.utils.portable_paths import geomind_assets_root, isaac_assets_root
from dataclasses import dataclass
from isaacsim.storage.native import get_assets_root_path
import numpy as np

def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)

QUAT_RX90 = np.array([0.0, 0.0, np.cos(3*np.pi/4), np.sin(3*np.pi/4)])
QUAT_RX90_1 = np.array([0.0, 0.0, np.cos(np.pi/4), np.sin(np.pi/4)])
QUAT_RY90 = np.array([0.0, np.sin(3*np.pi/4), 0.0, np.cos(3*np.pi/4)])
QUAT_RY180 =  np.array([0.0, 1.0, 0.0, 0.0])
QUAT_RZ90 = np.array([np.sin(3*np.pi/4), 0.0, 0.0, np.cos(3*np.pi/4)])
QUAT_RZ90_1 = np.array([np.sin(np.pi/4), 0.0, 0.0, np.cos(np.pi/4)])
QUAT_RZ180 = np.array([1.0, 0.0, 0.0, 0.0])
QUAT_RX90_RZ90 = quat_mul(QUAT_RX90, QUAT_RZ90)
QUAT_RX90_1_RZ90 = quat_mul(QUAT_RX90_1, QUAT_RZ90)
QUAT_RX90_RY90 = quat_mul(QUAT_RX90, QUAT_RY90)
QUAT_RX90_1_RY90 = quat_mul(QUAT_RX90_1, QUAT_RY90)
QUAT_DEFAULT = np.array([0.0, 0.0, 0.0, 1.0])


TABLE_HEIGHT={
    "TABLE_1": 0.915,
    "TABLE_2": 0.95,
    "TABLE_3": 0.75,
    "TABLE_4": 0.75,
    "TABLE_5": 0.59,
    "TABLE_6": 0.59,
    "TABLE_7": 0.7,
    "TABLE_8": 0.71,
    "TABLE_9": 0.55,
}

@dataclass
class USDAsset:
    name: str
    prim_path: str
    usd_path: str
    scale: np.array
    orientation: list[np.array]

geomind_asset_path = str(geomind_assets_root()) + "/"
# TABLE ASSETS
TABLE_1 = USDAsset(
    name = "table_1",
    prim_path="/World/table_1",
    usd_path=geomind_asset_path + "table/table_1.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
TABLE_2 = USDAsset(
    name = "table_2",
    prim_path="/World/table_2",
    usd_path=geomind_asset_path + "table/table_2.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_RZ90]
)
TABLE_3 = USDAsset(
    name = "table_3",
    prim_path="/World/table_3",
    usd_path=geomind_asset_path + "table/table_3.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
TABLE_4 = USDAsset(
    name = "table_4",
    prim_path="/World/table_4",
    usd_path=geomind_asset_path + "table/table_4.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
TABLE_5 = USDAsset(
    name = "table_5",
    prim_path="/World/table_5",
    usd_path=geomind_asset_path + "table/table_5.usda",
    scale=np.array([0.013, 0.013, 0.013]),
    orientation=[QUAT_RZ90]
)
TABLE_6 = USDAsset(
    name = "table_6",
    prim_path="/World/table_6",
    usd_path=geomind_asset_path + "table/table_6.usda",
    scale=np.array([0.013, 0.013, 0.013]),
    orientation=[QUAT_DEFAULT]
)
TABLE_7 = USDAsset(
    name = "table_7",
    prim_path="/World/table_7",
    usd_path=geomind_asset_path + "table/table_7.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_RZ90]
)
TABLE_8 = USDAsset(
    name = "table_8",
    prim_path="/World/table_8",
    usd_path=geomind_asset_path + "table/table_8.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
TABLE_9 = USDAsset(
    name = "table_9",
    prim_path="/World/table_9",
    usd_path=geomind_asset_path + "table/table_9.usda",
    scale=np.array([0.015, 0.015, 0.015]),
    orientation=[QUAT_DEFAULT]
)

#SHELF ASSETS
SHELF_1 = USDAsset(
    name = "shelf_1",
    prim_path="/World/shelf_1",
    usd_path=geomind_asset_path + "layout/shelf_1.usda",
    scale=np.array([0.0025, 0.0025, 0.0025]),
    orientation=[QUAT_RZ90]
)
SHELF_2 = USDAsset(
    name = "shelf_2",
    prim_path="/World/shelf_2",
    usd_path=geomind_asset_path + "layout/shelf_2.usda",
    scale=np.array([0.0025, 0.0025, 0.0025]),
    orientation=[QUAT_RZ90]
)

#PLANT ASSETS   
PLANT_1 = USDAsset(
    name = "plant_1",
    prim_path="/World/plant_1",
    usd_path=geomind_asset_path + "layout/plant_1.usda",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_DEFAULT]
)
PLANT_2 = USDAsset(
    name = "plant_2",
    prim_path="/World/plant_2",
    usd_path=geomind_asset_path + "layout/plant_2.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
PLANT_3 = USDAsset(
    name = "plant_3",
    prim_path="/World/plant_3",
    usd_path=geomind_asset_path + "layout/plant_3.usda",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_DEFAULT]
)
PLANT_4 = USDAsset(
    name = "plant_4",
    prim_path="/World/plant_4",
    usd_path=geomind_asset_path + "layout/plant_4.usda",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_DEFAULT]
)
PLANT_5 = USDAsset(
    name = "plant_5",
    prim_path="/World/plant_5",
    usd_path=geomind_asset_path + "layout/plant_5.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_RX90_1]
)
PLANT_6 = USDAsset(
    name = "plant_6",
    prim_path="/World/plant_6",
    usd_path=geomind_asset_path + "layout/plant_6.usda",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_RZ180]
)

#SOFA ASSETS
SOFA_1 = USDAsset(
    name = "sofa_1",
    prim_path="/World/sofa_1",
    usd_path=geomind_asset_path + "layout/sofa_1.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_RZ90]
)
SOFA_2 = USDAsset(
    name = "sofa_2",
    prim_path="/World/sofa_2",
    usd_path=geomind_asset_path + "layout/sofa_2.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
SOFA_3 = USDAsset(
    name = "sofa_3",
    prim_path="/World/sofa_3",
    usd_path=geomind_asset_path + "layout/sofa_3.usda",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_DEFAULT]
)

#TRASHCAN ASSETS
TRASHCAN_1 = USDAsset(
    name = "trashcan_1",
    prim_path="/World/trashcan_1",
    usd_path=geomind_asset_path + "layout/trashcan_1.usda",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_DEFAULT]
)


#CHAIR ASSETS
CHAIR_1 = USDAsset(
    name = "chair_1",
    prim_path="/World/chair_1",
    usd_path=geomind_asset_path + "layout/chair_1.usda",
    scale=np.array([0.012, 0.012, 0.012]),
    orientation=[QUAT_RX90_1]
)
CHAIR_2 = USDAsset(
    name = "chair_2",
    prim_path="/World/chair_2",
    usd_path=geomind_asset_path + "layout/chair_2.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
CHAIR_3 = USDAsset(
    name = "chair_3",
    prim_path="/World/chair_3",
    usd_path=geomind_asset_path + "layout/chair_3.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_RZ90_1]
)
CHAIR_4 = USDAsset(
    name = "chair_4",
    prim_path="/World/chair_4",
    usd_path=geomind_asset_path + "layout/chair_4.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
CHAIR_5 = USDAsset(
    name = "chair_5",
    prim_path="/World/chair_5",
    usd_path=geomind_asset_path + "layout/chair_5.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_RZ90_1]
)
CHAIR_6 = USDAsset(
    name = "chair_6",
    prim_path="/World/chair_6",
    usd_path=geomind_asset_path + "layout/chair_6.usda",
    scale=np.array([0.013, 0.013, 0.013]),
    orientation=[QUAT_RZ90_1]
)

#CABINET ASSETS
CABINET_1 = USDAsset(
    name = "cabinet_1",
    prim_path="/World/cabinet_1",
    usd_path=geomind_asset_path + "layout/cabinet_1.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_RZ90_1]
)
CABINET_2 = USDAsset(
    name = "cabinet_2",
    prim_path="/World/cabinet_2",
    usd_path=geomind_asset_path + "layout/cabinet_2.usda",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_RZ180]
)
CABINET_3 = USDAsset(
    name = "cabinet_3",
    prim_path="/World/cabinet_3",
    usd_path=geomind_asset_path + "layout/cabinet_3.usda",
    scale=np.array([0.015, 0.015, 0.015]),
    orientation=[QUAT_RZ180]
)
CABINET_4 = USDAsset(
    name = "cabinet_4",
    prim_path="/World/cabinet_4",
    usd_path=geomind_asset_path + "layout/cabinet_4.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_RZ180]
)
CABINET_5 = USDAsset(
    name = "cabinet_5",
    prim_path="/World/cabinet_5",
    usd_path=geomind_asset_path + "layout/cabinet_5.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_RZ90_1]
)
CABINET_6 = USDAsset(
    name = "cabinet_6",
    prim_path="/World/cabinet_6",
    usd_path=geomind_asset_path + "layout/cabinet_6.usda",
    scale=np.array([0.007, 0.007, 0.007]),
    orientation=[QUAT_RZ90]
)

#LAMP ASSETS
LAMP_1 = USDAsset(
    name = "lamp_1",
    prim_path="/World/lamp_1",
    usd_path=geomind_asset_path + "layout/lamp_1.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)

#PLATE ASSETS   
PLATE_1 = USDAsset(
    name = "plate_1",
    prim_path="/World/plate_1",
    usd_path=geomind_asset_path + "objects/plate_1.usda",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_DEFAULT]
)
PLATE_2 = USDAsset(
    name = "plate_2",
    prim_path="/World/plate_2",
    usd_path=geomind_asset_path + "objects/plate_2.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)

#BOWL ASSETS   
BOWL_1 = USDAsset(
    name = "bowl_1",
    prim_path="/World/bowl_1",
    usd_path=geomind_asset_path + "objects/bowl_1.usda",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_DEFAULT]
)
BOWL_2 = USDAsset(
    name = "bowl_2",
    prim_path="/World/bowl_2",
    usd_path=geomind_asset_path + "objects/bowl_2.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)

# BASKET_ASSET
BASKET_1 = USDAsset(
    name = "basket_1",
    prim_path="/World/basket_1",
    usd_path=geomind_asset_path + "objects/basket_1.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
BASKET_2 = USDAsset(
    name = "basket_2",
    prim_path="/World/basket_2",
    usd_path=geomind_asset_path + "objects/basket_2.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
BASKET_3 = USDAsset(
    name = "basket_3",
    prim_path="/World/basket_3",
    usd_path=geomind_asset_path + "objects/basket_3.usda",
    scale=np.array([0.007, 0.007, 0.007]),
    orientation=[QUAT_RZ90]
)

#POT ASSETS
POT_1 = USDAsset(
    name = "pot_1",
    prim_path="/World/pot_1",
    usd_path=geomind_asset_path + "layout/pot_1.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
POT_2 = USDAsset(
    name = "pot_2",
    prim_path="/World/pot_2",
    usd_path=geomind_asset_path + "layout/pot_2.usda",
    scale=np.array([0.012, 0.012, 0.012]),
    orientation=[QUAT_DEFAULT]
)

#CUP ASSETS
CUP_1 = USDAsset(
    name = "cup_1",
    prim_path="/World/cup_1",
    usd_path=geomind_asset_path + "layout/cup_1.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[np.array([0.0019, 0.9643, 0.271, -0.2635])]
)
CUP_2 = USDAsset(
    name = "cup_2",
    prim_path="/World/cup_2",
    usd_path=geomind_asset_path + "layout/cup_2.usda",
    scale=np.array([0.012, 0.012, 0.012]),
    orientation=[QUAT_RY180]
)

#BLANKET ASSETS
BLANKET_1 = USDAsset(
    name = "blanket_1",
    prim_path="/World/blanket_1",
    usd_path=geomind_asset_path + "layout/blanket_1.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
BLANKET_2 = USDAsset(
    name = "blanket_2",
    prim_path="/World/blanket_2",
    usd_path=geomind_asset_path + "layout/blanket_2.usda",
    scale=np.array([0.012, 0.012, 0.012]),
    orientation=[QUAT_DEFAULT]
)
BLANKET_3 = USDAsset(
    name = "blanket_3",
    prim_path="/World/blanket_3",
    usd_path=geomind_asset_path + "layout/blanket_3.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
BLANKET_4 = USDAsset(
    name = "blanket_4",
    prim_path="/World/blanket_4",
    usd_path=geomind_asset_path + "layout/blanket_4.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
BLANKET_5 = USDAsset(
    name = "blanket_5",
    prim_path="/World/blanket_5",
    usd_path=geomind_asset_path + "layout/blanket_5.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
BLANKET_6 = USDAsset(
    name = "blanket_6",
    prim_path="/World/blanket_6",
    usd_path=geomind_asset_path + "layout/blanket_6.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)

asset_path = str(isaac_assets_root()/"Assets/Isaac/4.5")
#CAN ASSETS

BLUE_CAN = USDAsset(
    name = "blue_can",
    prim_path="/World/blue_can",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/002_master_chef_can.usd",
    scale=np.array([0.5, 0.5, 0.5]),
    orientation=[QUAT_RX90, QUAT_RX90_1]
)
RED_CAN = USDAsset(
    name = "red_can",
    prim_path="/World/red_can",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
    scale=np.array([0.8, 0.8, 0.8]),
    orientation=[QUAT_RX90, QUAT_RX90_1]
)
CUBE_CAN = USDAsset(
    name = "cube_can",
    prim_path="/World/cube_can",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
    scale=np.array([0.8, 0.8, 0.8]),
    orientation=[QUAT_RX90, QUAT_RX90_1]
)
CUBE_CAN_RZ90 = USDAsset(
    name = "cube_can",
    prim_path="/World/cube_can",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
    scale=np.array([0.8, 0.8, 0.8]),
    orientation=[QUAT_RX90_RZ90, QUAT_RX90_1_RZ90]
)
LOW_CAN = USDAsset(
    name = "low_can",
    prim_path="/World/low_can",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
    scale=np.array([0.8, 0.8, 0.8]),
    orientation=[QUAT_RX90, QUAT_RX90_1]
)
GLASS_CAN = USDAsset(
    name = "glass_can",
    prim_path="/World/glass_can",
    usd_path=geomind_asset_path + "objects/glass_can.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)
GREEN_TIN = USDAsset(
    name = "green_tin",
    prim_path="/World/green_tin",
    usd_path=geomind_asset_path + "objects/green_tin.usda",
    scale=np.array([0.01, 0.01, 0.01]),
    orientation=[QUAT_DEFAULT]
)

RUBIKS_CUBE = USDAsset(
    name = "rubiks_cube",
    prim_path="/World/rubiks_cube",
    usd_path=asset_path + "/Isaac/Props/Rubiks_Cube/rubiks_cube.usd",
    scale=np.array([0.7, 0.7, 0.7]),
    orientation=[QUAT_RX90, QUAT_RX90_1, QUAT_RY90, QUAT_RZ90, QUAT_DEFAULT]
)

#BOX ASSETS
RED_BOX = USDAsset(
    name = "red_box",
    prim_path="/World/red_box",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
    scale=np.array([0.5, 0.5, 0.5]),
    orientation=[QUAT_RX90, QUAT_RX90_1, QUAT_RX90_RY90, QUAT_RX90_1_RY90]
)
BROWN_BOX = USDAsset(
    name = "brown_box",
    prim_path="/World/brown_box",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
    scale=np.array([0.8, 0.8, 0.8]),
    orientation=[QUAT_RX90, QUAT_RX90_1, QUAT_RX90_RY90, QUAT_RX90_1_RY90]
)

RED_BOX_RZ90 = USDAsset(
    name = "red_box",
    prim_path="/World/red_box",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
    scale=np.array([0.5, 0.5, 0.5]),
    orientation=[QUAT_RX90_RZ90, QUAT_RX90_1_RZ90]
)
BROWN_BOX_RZ90 = USDAsset(
    name = "brown_box",
    prim_path="/World/brown_box",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
    scale=np.array([0.8, 0.8, 0.8]),
    orientation=[QUAT_RX90_RZ90, QUAT_RX90_1_RZ90]
)

#BOTTLE ASSETS
YELLOW_BOTTLE = USDAsset(
    name = "yellow_bottle",
    prim_path="/World/yellow_bottle",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
    scale=np.array([0.8, 0.8, 0.8]),
    orientation=[QUAT_RX90]
)
WHITE_BOTTLE = USDAsset(
    name = "white_bottle",
    prim_path="/World/white_bottle",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/021_bleach_cleanser.usd",
    scale=np.array([0.8, 0.8, 0.8]),
    orientation=[QUAT_RX90]
)
YELLOW_BOTTLE_RZ90 = USDAsset(
    name = "yellow_bottle",
    prim_path="/World/yellow_bottle",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
    scale=np.array([0.8, 0.8, 0.8]),
    orientation=[QUAT_RX90_RZ90]
)
WHITE_BOTTLE_RZ90 = USDAsset(
    name = "white_bottle",
    prim_path="/World/white_bottle",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/021_bleach_cleanser.usd",
    scale=np.array([0.8, 0.8, 0.8]),
    orientation=[QUAT_RX90_RZ90]
)

#CUP ASSETS
GLASS_CUP = USDAsset(
    name = "glass_cup",
    prim_path="/World/glass_cup",
    usd_path="/media/elm/files/nvidia_assets/Residential_NVD@10012/Assets/ArchVis/Residential/Kitchen/Kitchenware/Dinnerware/P_Glassware_Short.usd",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_DEFAULT]
)

#FRUIT ASSETS
BANANA = USDAsset(
    name = "banana",
    prim_path="/World/banana",
    usd_path=asset_path + "/Isaac/Props/YCB/Axis_Aligned/011_banana.usd",
    scale=np.array([1.0, 1.0, 1.0]),
    orientation=[QUAT_DEFAULT]
)
APPLE = USDAsset(
    name = "apple",
    prim_path="/World/apple",
    usd_path="/media/elm/files/nvidia_assets/Residential_NVD@10012/Assets/ArchVis/Residential/Food/Fruit/Apple.usd",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_DEFAULT]
)
LIME = USDAsset(
    name = "lime",
    prim_path="/World/lime",
    usd_path="/media/elm/files/nvidia_assets/Residential_NVD@10012/Assets/ArchVis/Residential/Food/Fruit/Lime01.usd",
    scale=np.array([0.009, 0.009, 0.009]),
    orientation=[QUAT_DEFAULT]
)
POMEGRANATE = USDAsset(
    name = "pomegranate",
    prim_path="/World/pomegranate",
    usd_path="/media/elm/files/nvidia_assets/Residential_NVD@10012/Assets/ArchVis/Residential/Food/Fruit/Pomegranate01.usd",
    scale=np.array([0.0075, 0.0075, 0.0075]),
    orientation=[QUAT_DEFAULT]
)
AVOCADO = USDAsset(
    name = "avocado",
    prim_path="/World/avocado",
    usd_path="/media/elm/files/nvidia_assets/Residential_NVD@10012/Assets/ArchVis/Residential/Food/Fruit/Avocado01.usd",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_RY90]
)

SIMPLE_ROOM = USDAsset(
    name = "simple_room",
    prim_path="/World/simple_room",
    usd_path=asset_path + "/Isaac/Environments/Simple_Room/Simple_Room.usd",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_RZ90]
)
TABLE = USDAsset(
    name = "table",
    prim_path="/World/table",
    usd_path=asset_path + "/Isaac/Props/Mounts/thor_table.usd",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_RZ90]
)
ROUGH_GROUND = USDAsset(
    name = "rough_ground",
    prim_path="/World/rough_ground",
    usd_path=asset_path + "/Isaac/Environments/Terrains/flat_plane.usd",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_RZ90]
)
BIN = USDAsset(
    name = "bin",
    prim_path="/World/bin",
    usd_path=asset_path + "/Isaac/Props/KLT_Bin/small_KLT_visual.usd",
    scale=np.array([0.008, 0.008, 0.008]),
    orientation=[QUAT_RZ90]
)

