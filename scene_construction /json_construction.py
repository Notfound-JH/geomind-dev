from pathlib import Path
from geomind_lab.utils.portable_paths import geomind_assets_root



import json
import os
from pxr import Usd, UsdGeom

# ================= 配置区 =================
JSON_PATH = str(Path(__file__).resolve().parent/"assets_catalog_full.json") 
WALL_FOLDER = str(geomind_assets_root()/"wall")
# =========================================

def get_asset_info_fixed(usd_path):
    stage = Usd.Stage.Open(usd_path)
    if not stage: return None

    # --- 修复后的获取 Prim 逻辑 ---
    prim = stage.GetDefaultPrim()
    if not prim.IsValid():
        # 如果没有默认根节点，取遍历到的第一个有效 Prim
        for p in stage.Traverse():
            if p:
                prim = p
                break
    
    if not prim or not prim.IsValid():
        print(f"警告: {usd_path} 中没有找到有效的 Prim")
        return None
    # ---------------------------

    # 使用文件名（去除后缀）作为唯一 ID
    file_name = os.path.splitext(os.path.basename(usd_path))[0]

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeLocalBound(prim)
    range_box = bbox.GetRange()
    
    if range_box.IsEmpty(): return None

    min_pt = range_box.GetMin()
    max_pt = range_box.GetMax()
    dims = max_pt - min_pt

    return {
        "name": file_name,
        "prim_path": f"/World/{file_name}",
        "usd_path": usd_path,
        "scale": [1.0, 1.0, 1.0],
        "orientations": [[0.0, 0.0, 0.0, 1.0]],
        "z_min_offset": float(min_pt[2]),
        "dimensions": {
            "x": float(dims[0]),
            "y": float(dims[1]),
            "z": float(dims[2])
        }
    }

    return {
        "name": file_name,  # 【关键修改】改用文件名
        "prim_path": f"/World/{file_name}",
        "usd_path": usd_path,
        "scale": [1.0, 1.0, 1.0], # 墙体建议先设为1，或手动读取
        "orientations": [[0.0, 0.0, 0.0, 1.0]],
        "z_min_offset": float(min_pt[2]),
        "dimensions": {
            "x": float(dims[0]),
            "y": float(dims[1]),
            "z": float(dims[2])
        }
    }

def main():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "wall" not in data["assets"]:
        data["assets"]["wall"] = []

    # 记录本次处理的名称，用于对比
    new_wall_list = data["assets"]["wall"]

    for file in os.listdir(WALL_FOLDER):
        if file.endswith((".usd", ".usda")):
            full_path = os.path.join(WALL_FOLDER, file)
            info = get_asset_info_fixed(full_path)
            
            if info:
                # 更新或追加
                for i, existing in enumerate(new_wall_list):
                    if existing["name"] == info["name"]:
                        new_wall_list[i] = info
                        break
                else:
                    new_wall_list.append(info)
                print(f"已处理资产: {info['name']}")

    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("JSON 更新完成！")

main()