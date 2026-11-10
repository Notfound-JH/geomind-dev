# utils/lighting.py
from typing import Optional, Tuple
from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf
import omni.usd


def add_dome_light():
    stage = omni.usd.get_context().get_stage()
    dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)
