"""Mimic environment wrappers."""

__all__ = ["GeoMindFrankaIKRelMimicEnv"]


def __getattr__(name: str):
    if name == "GeoMindFrankaIKRelMimicEnv":
        from .geomind_franka_ik_rel_mimic_env import GeoMindFrankaIKRelMimicEnv

        return GeoMindFrankaIKRelMimicEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
