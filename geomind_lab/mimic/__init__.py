"""GeoMind mimic wrappers and configs."""

__all__ = ["GeoMindFrankaIKRelMimicEnv", "GeoMindFrankaIKRelMimicEnvCfg"]


def __getattr__(name: str):
    if name == "GeoMindFrankaIKRelMimicEnvCfg":
        from .configs.geomind_franka_ik_rel_mimic_env_cfg import GeoMindFrankaIKRelMimicEnvCfg

        return GeoMindFrankaIKRelMimicEnvCfg
    if name == "GeoMindFrankaIKRelMimicEnv":
        from .envs.geomind_franka_ik_rel_mimic_env import GeoMindFrankaIKRelMimicEnv

        return GeoMindFrankaIKRelMimicEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
