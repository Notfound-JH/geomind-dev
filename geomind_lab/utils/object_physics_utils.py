from __future__ import annotations

from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg


DEFAULT_OBJECT_MASS = 0.25
DEFAULT_OBJECT_LINEAR_DAMPING = 0.2
DEFAULT_OBJECT_ANGULAR_DAMPING = 0.5


def build_geomind_object_physics_cfg(
    *,
    mass: float = DEFAULT_OBJECT_MASS,
    linear_damping: float = DEFAULT_OBJECT_LINEAR_DAMPING,
    angular_damping: float = DEFAULT_OBJECT_ANGULAR_DAMPING,
    solver_position_iteration_count: int = 20,
    solver_velocity_iteration_count: int = 20,
    max_linear_velocity: float = 1000.0,
    max_angular_velocity: float = 1000.0,
    max_depenetration_velocity: float = 5.0,
    disable_gravity: bool = False,
) -> tuple[RigidBodyPropertiesCfg, MassPropertiesCfg]:
    """Build the default rigid-body and mass cfg used by GeoMind task objects."""
    if mass <= 0.0:
        raise ValueError(f"Object mass must be > 0. Received: {mass}")
    if linear_damping < 0.0:
        raise ValueError(f"Object linear_damping must be >= 0. Received: {linear_damping}")
    if angular_damping < 0.0:
        raise ValueError(f"Object angular_damping must be >= 0. Received: {angular_damping}")

    rigid_props = RigidBodyPropertiesCfg(
        solver_position_iteration_count=solver_position_iteration_count,
        solver_velocity_iteration_count=solver_velocity_iteration_count,
        linear_damping=linear_damping,
        angular_damping=angular_damping,
        max_angular_velocity=max_angular_velocity,
        max_linear_velocity=max_linear_velocity,
        max_depenetration_velocity=max_depenetration_velocity,
        disable_gravity=disable_gravity,
    )
    mass_props = MassPropertiesCfg(mass=mass)
    return rigid_props, mass_props
