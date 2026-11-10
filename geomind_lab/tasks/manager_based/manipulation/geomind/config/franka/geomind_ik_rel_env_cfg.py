from __future__ import annotations

from isaaclab.utils import configclass

from .franka_env_cfg import FrankaGeoMindEnvCfg


@configclass
class FrankaGeoMindIKRelEnvCfg(FrankaGeoMindEnvCfg):
    """Compatibility alias for the Franka IK-relative GeoMind env config."""

    pass
