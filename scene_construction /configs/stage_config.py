from dataclasses import dataclass
import numpy as np
from .asset_config import *


@dataclass
class STAGEAsset:
    height: float
    items: list
    positions: list

STAGE_1 = STAGEAsset(
    height = 0.55,
    items = [
        BLANKET_2,
        TABLE_9,
        SOFA_1,
        PLANT_2
    ],
    positions = [
        np.array([0.25, 0.0, 0.004]),
        np.array([0.25, 0.0, 0.28]),
        np.array([-1.2, 0.1, 0.43]),
        np.array([-1.2, -1.6, 0.74])
    ]
)

STAGE_2 = STAGEAsset(
    height = 0.59,
    items = [
        TABLE_5,
        BLANKET_1,
        CHAIR_3,
        CHAIR_3,
        CABINET_1,
        PLANT_6
    ],
    positions = [
        np.array([0.25, 0.0, 0.31]),
        np.array([0.25, 0.0, 0.051]),
        np.array([-1, 0.5, 0.3]),
        np.array([-1, -0.5, 0.3]),
        np.array([0, -1.5, 0.45]),
        np.array([-0.2, -1.5, 1.12])
    ]
)

STAGE_3 = STAGEAsset(
    height = 0.76,
    items = [
        TABLE_3,
        BLANKET_4,
        LAMP_1,
        CHAIR_1,
    ],
    positions = [
        np.array([0.25, 0.0, 0.38]),
        np.array([0.25, 0.0, 0.003]),
        np.array([-0.03, -0.65, 1.01]),
        np.array([-1.2, 0.1, 0.47]),
    ]
)

STAGE_4 = STAGEAsset(
    height = 0.915,
    items = [
        BLANKET_3,
        TABLE_1,
        CHAIR_5,
        CHAIR_5,
        CABINET_6,
        CABINET_5,
    ],
    positions = [
        np.array([0.25, 0.0, 0.01]),
        np.array([0.25, 0.0, 0.48]),
        np.array([-1, -0.1, 0.4]),
        np.array([0.1, 1.5, 0.4]),
        np.array([-2, -1.7, 1]),
        np.array([-2, -0.5, 1])
    ]
)

STAGE_5 = STAGEAsset(
    height = 0.59,
    items = [
        BLANKET_5,
        TABLE_6,
        SOFA_2,
        TRASHCAN_1,
        POT_2,
        CUP_2,
    ],
    positions = [
        np.array([0.25, 0.0, 0.012]),
        np.array([0.25, 0.0, 0.3]),
        np.array([-1, -0.5, 0.4]),
        np.array([-0.8, 1.5, 0.5]),
        np.array([-0.05, -0.7, 0.69]),
        np.array([0.2, -0.7, 0.63])
    ]
)