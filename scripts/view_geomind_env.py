#!/usr/bin/env python3
from __future__ import annotations

import argparse
import faulthandler
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass

from geomind_lab.bootstrap.isaaclab_path import ensure_isaaclab_source_on_path


ensure_isaaclab_source_on_path(REPO_ROOT)

# Import pinocchio before AppLauncher so IsaacLab uses the conda-installed version.
if sys.platform != "win32":
    try:
        import pinocchio  # noqa: F401
    except Exception:
        pass

from isaaclab.app import AppLauncher


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a GeoMind IsaacLab viewer using a runnable Franka env config derived from "
            "geomind_env_cfg.py."
        )
    )
    parser.add_argument("--scene-id", type=int, default=1, help="Scene id under GEOMIND_ASSETS_ROOT/stage/.")
    parser.add_argument(
        "--layout-key",
        type=str,
        default="line_3",
        help="Annotated task layout key, for example: line_3, grid_2_3, cluster.",
    )
    parser.add_argument("--task-index", type=int, default=0, help="Zero-based task index inside the layout.")
    parser.add_argument(
        "--cfg-kind",
        type=str,
        default="ik-rel",
        choices=("ik-rel", "visuomotor", "visuomotor-skillgen"),
        help="Concrete Franka cfg to use. Base GeoMindEnvCfg is not directly runnable.",
    )
    parser.add_argument(
        "--lighting-mode",
        type=str,
        default="natural",
        choices=("natural", "bright"),
        help="Scene lighting preset: natural for normal daylight, bright for higher brightness.",
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        default=None,
        help="Override task_mode. If omitted, vla is used except for visuomotor-skillgen.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized environments to spawn.")
    parser.add_argument(
        "--action",
        type=str,
        default="zero",
        choices=("zero", "random"),
        help="Action policy to use while the viewer is running.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum simulation steps before exit. Use 0 to run until the app window is closed.",
    )
    parser.add_argument(
        "--disable-fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O instead.",
    )
    parser.add_argument(
        "--viewer-eye",
        type=float,
        nargs=3,
        default=(2.2, 2.2, 1.8),
        metavar=("X", "Y", "Z"),
        help="Viewer camera eye position.",
    )
    parser.add_argument(
        "--viewer-lookat",
        type=float,
        nargs=3,
        default=(0.3, 0.0, 0.8),
        metavar=("X", "Y", "Z"),
        help="Viewer camera look-at position.",
    )
    parser.add_argument(
        "--high-quality-render",
        action="store_true",
        default=False,
        help="Enable a higher-quality render preset through SimulationCfg.render.",
    )
    parser.add_argument(
        "--ui-viewport-camera-map",
        type=str,
        default=None,
        help=(
            "Optional JSON dict mapping 1-based viewport indices to camera names or camera prim paths. "
            'Example: {"1":"table_cam","2":"wrist_cam"}.'
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def print_failure_context(
    stage: str,
    *,
    step_count: int | None = None,
    env_cfg=None,
) -> None:
    print("\n" + "=" * 72, file=sys.stderr, flush=True)
    print("GeoMind IsaacLab Viewer Failure", file=sys.stderr, flush=True)
    print("=" * 72, file=sys.stderr, flush=True)
    print(f"stage:          {stage}", file=sys.stderr, flush=True)

    args_obj = globals().get("args_cli")
    if args_obj is not None:
        print(f"cfg_kind:       {getattr(args_obj, 'cfg_kind', None)}", file=sys.stderr, flush=True)
        print(f"scene_id:       {getattr(args_obj, 'scene_id', None)}", file=sys.stderr, flush=True)
        print(f"layout_key:     {getattr(args_obj, 'layout_key', None)}", file=sys.stderr, flush=True)
        print(f"task_index:     {getattr(args_obj, 'task_index', None)}", file=sys.stderr, flush=True)
        print(f"task_mode_arg:  {getattr(args_obj, 'task_mode', None)}", file=sys.stderr, flush=True)
        print(f"lighting_mode:  {getattr(args_obj, 'lighting_mode', None)}", file=sys.stderr, flush=True)
        print(f"ui_viewports:   {getattr(args_obj, 'ui_viewport_camera_map', None)}", file=sys.stderr, flush=True)
        print(f"device:         {getattr(args_obj, 'device', None)}", file=sys.stderr, flush=True)
        print(f"headless:       {getattr(args_obj, 'headless', None)}", file=sys.stderr, flush=True)
        print(f"hq_render:      {getattr(args_obj, 'high_quality_render', None)}", file=sys.stderr, flush=True)

    if env_cfg is not None:
        print(f"resolved_mode:  {getattr(env_cfg, 'task_mode', None)}", file=sys.stderr, flush=True)
        print(f"num_envs:       {getattr(getattr(env_cfg, 'scene', None), 'num_envs', None)}", file=sys.stderr, flush=True)
        scene_profile = getattr(env_cfg, "_scene_profile", None)
        if scene_profile is not None:
            print(f"scene_usd:      {getattr(scene_profile, 'usd_path', None)}", file=sys.stderr, flush=True)
        print(f"instruction:    {getattr(env_cfg, 'task_instruction', '')}", file=sys.stderr, flush=True)

    if step_count is not None:
        print(f"step_count:     {step_count}", file=sys.stderr, flush=True)

    print("-" * 72, file=sys.stderr, flush=True)
    traceback.print_exc()
    print("=" * 72, file=sys.stderr, flush=True)


args_cli = build_argparser().parse_args()
if args_cli.cfg_kind != "ik-rel":
    setattr(args_cli, "enable_cameras", True)

simulation_app = None
try:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
except Exception:
    print_failure_context("app_launcher")
    raise SystemExit(1)


import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import Camera
from isaaclab.utils import configclass

from geomind_lab.utils import asset_utils
from geomind_lab.utils import render_utils
from geomind_lab.utils.ui_viewport_utils import configure_env_camera_viewports, parse_ui_viewport_camera_map
from geomind_lab.tasks.manager_based.manipulation.geomind.config.franka import (
    FrankaGeoMindIKRelEnvCfg,
    FrankaGeoMindIKRelVisuomotorEnvCfg,
    FrankaGeoMindIKRelVisuomotorSkillgenEnvCfg,
)


CFG_KIND_TO_CLASS = {
    "ik-rel": FrankaGeoMindIKRelEnvCfg,
    "visuomotor": FrankaGeoMindIKRelVisuomotorEnvCfg,
    "visuomotor-skillgen": FrankaGeoMindIKRelVisuomotorSkillgenEnvCfg,
}
RESET_SETTLE_STEPS = 60


def _resolved_task_mode() -> str:
    if args_cli.task_mode:
        return args_cli.task_mode
    if args_cli.cfg_kind == "visuomotor-skillgen":
        return "skillgen"
    return "vla"


def build_runtime_env_cfg():
    base_cfg_cls = CFG_KIND_TO_CLASS[args_cli.cfg_kind]
    task_mode = _resolved_task_mode()
    ui_viewport_camera_map = parse_ui_viewport_camera_map(args_cli.ui_viewport_camera_map)

    def __post_init__(self):
        self.scene_id = args_cli.scene_id
        self.task_layout_key = args_cli.layout_key
        self.task_index = args_cli.task_index
        self.task_mode = task_mode
        self.lighting_mode = args_cli.lighting_mode
        self.ui_viewport_camera_map = ui_viewport_camera_map
        self.scene.num_envs = args_cli.num_envs
        self.sim.device = args_cli.device
        self.sim.use_fabric = not args_cli.disable_fabric

        base_cfg_cls.__post_init__(self)
        if args_cli.high_quality_render:
            render_utils.apply_high_quality_render_settings(self.sim)

        self.viewer.eye = tuple(float(v) for v in args_cli.viewer_eye)
        self.viewer.lookat = tuple(float(v) for v in args_cli.viewer_lookat)

    runtime_cfg_cls = configclass(
        type(
            "RuntimeGeoMindEnvCfg",
            (base_cfg_cls,),
            {
                "__doc__": "Runtime GeoMind viewer cfg with scene/task selectors set from CLI.",
                "scene_id": args_cli.scene_id,
                "task_layout_key": args_cli.layout_key,
                "task_index": args_cli.task_index,
                "task_mode": task_mode,
                "lighting_mode": args_cli.lighting_mode,
                "ui_viewport_camera_map": ui_viewport_camera_map,
                "__post_init__": __post_init__,
            },
        )
    )
    return runtime_cfg_cls()


def make_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    if args_cli.action == "random":
        return 2.0 * torch.rand(env.action_space.shape, device=env.device) - 1.0
    return torch.zeros(env.action_space.shape, device=env.device)


def should_reset(terminated, truncated) -> bool:
    if terminated is not None and torch.any(terminated):
        return True
    if truncated is not None and torch.any(truncated):
        return True
    return False


def settle_env_after_reset(env: ManagerBasedRLEnv, settle_steps: int = RESET_SETTLE_STEPS) -> None:
    if settle_steps <= 0:
        return

    zero_actions = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(settle_steps):
        _, _, terminated, truncated, _ = env.step(zero_actions)
        if should_reset(terminated, truncated):
            env.reset()


def resolve_table_top_height(env_cfg) -> float | None:
    scene_profile = getattr(env_cfg, "_scene_profile", None)
    if scene_profile is None or not scene_profile.table_name:
        return None

    table_xform = scene_profile.xforms.get(scene_profile.table_name)
    if table_xform is None:
        return asset_utils.resolve_tabletop_height(scene_profile.table_name)

    return asset_utils.resolve_tabletop_height(
        scene_profile.table_name,
        position_z=table_xform["pos"][2],
        scale_z=table_xform["scale"][2],
    )


def print_camera_height_offsets(env: ManagerBasedRLEnv, env_cfg) -> None:
    table_top_height = resolve_table_top_height(env_cfg)
    print(f"table_top_z:    {table_top_height if table_top_height is not None else '<unknown>'}")
    if table_top_height is None:
        return

    print("camera_height_offsets:")
    found_camera = False
    for sensor_name, sensor in sorted(env.scene.sensors.items()):
        if not isinstance(sensor, Camera):
            continue
        sensor.update(dt=0.0, force_recompute=True)
        camera_pos_w = sensor.data.pos_w
        if camera_pos_w is None or camera_pos_w.numel() == 0:
            continue

        camera_height = float(camera_pos_w[0, 2].item())
        height_delta = camera_height - table_top_height
        print(f"  {sensor_name:<16} camera_z={camera_height:.4f} delta_vs_table={height_delta:+.4f}")
        found_camera = True

    if not found_camera:
        print("  <no camera sensors found>")


def _format_tensor_values(values: torch.Tensor, precision: int = 4) -> str:
    flat_values = values.detach().cpu().reshape(-1).tolist()
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in flat_values) + "]"


def _format_inertia_diag(inertia_values: torch.Tensor) -> str:
    flat_values = inertia_values.detach().cpu().reshape(-1)
    if flat_values.numel() >= 9:
        diag_values = torch.stack((flat_values[0], flat_values[4], flat_values[8]))
    else:
        diag_values = flat_values
    return _format_tensor_values(diag_values)


def _format_optional_scalar(value, precision: int = 4) -> str:
    if value is None:
        return "<unset>"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_unique_values(values: list[object], precision: int = 4) -> str:
    unique_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            display_value = value
        else:
            try:
                display_value = f"{float(value):.{precision}f}"
            except (TypeError, ValueError):
                display_value = str(value)
        if display_value in seen_values:
            continue
        seen_values.add(display_value)
        unique_values.append(display_value)

    if not unique_values:
        return "<unset>"
    if len(unique_values) == 1:
        return unique_values[0]
    return "[" + ", ".join(unique_values) + "]"


def _find_first_api_prim(root_prim, api_schema):
    from pxr import Usd

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid() or prim.IsInstanceProxy():
            continue
        if prim.HasAPI(api_schema):
            return prim
    return None


def _collect_collider_prims(root_prim) -> list:
    from pxr import Usd, UsdPhysics

    collider_prims = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid() or prim.IsInstanceProxy():
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collider_prims.append(prim)
    return collider_prims


def _resolve_physics_material_path(collider_prim, default_material_path: str) -> str:
    from pxr import UsdShade

    current_prim = collider_prim
    while current_prim and current_prim.IsValid():
        material_binding_api = UsdShade.MaterialBindingAPI(current_prim)
        material_binding = material_binding_api.GetDirectBinding("physics")
        material_path = str(material_binding.GetMaterialPath())
        if material_path:
            return material_path
        current_prim = current_prim.GetParent()
    return default_material_path


def _read_material_properties(stage, material_path: str) -> dict[str, object]:
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        return {
            "material_path": material_path,
            "static_friction": None,
            "dynamic_friction": None,
            "restitution": None,
        }

    return {
        "material_path": material_path,
        "static_friction": material_prim.GetAttribute("physics:staticFriction").Get(),
        "dynamic_friction": material_prim.GetAttribute("physics:dynamicFriction").Get(),
        "restitution": material_prim.GetAttribute("physics:restitution").Get(),
    }


def summarize_entity_physics(stage, root_prim_path: str, *, default_material_path: str) -> dict[str, object] | None:
    from pxr import PhysxSchema, UsdPhysics

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        return None

    collider_prims = _collect_collider_prims(root_prim)
    rigid_prim = _find_first_api_prim(root_prim, UsdPhysics.RigidBodyAPI)

    collider_types: list[object] = []
    material_paths: list[object] = []
    static_frictions: list[object] = []
    dynamic_frictions: list[object] = []
    restitutions: list[object] = []
    contact_offsets: list[object] = []
    rest_offsets: list[object] = []

    for collider_prim in collider_prims:
        material_props = _read_material_properties(
            stage,
            _resolve_physics_material_path(collider_prim, default_material_path),
        )
        material_paths.append(material_props["material_path"])
        static_frictions.append(material_props["static_friction"])
        dynamic_frictions.append(material_props["dynamic_friction"])
        restitutions.append(material_props["restitution"])

        physx_collision_api = PhysxSchema.PhysxCollisionAPI(collider_prim)
        contact_offsets.append(physx_collision_api.GetContactOffsetAttr().Get())
        rest_offsets.append(physx_collision_api.GetRestOffsetAttr().Get())

        mesh_collision_api = UsdPhysics.MeshCollisionAPI(collider_prim)
        if mesh_collision_api:
            collider_type = mesh_collision_api.GetApproximationAttr().Get()
        else:
            collider_type = None
        collider_types.append(str(collider_type) if collider_type else collider_prim.GetTypeName())

    linear_damping = None
    angular_damping = None
    solver_position_iteration_count = None
    solver_velocity_iteration_count = None
    explicit_mass = None

    if rigid_prim is not None:
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(rigid_prim)
        linear_damping = physx_rigid_body_api.GetLinearDampingAttr().Get()
        angular_damping = physx_rigid_body_api.GetAngularDampingAttr().Get()
        solver_position_iteration_count = physx_rigid_body_api.GetSolverPositionIterationCountAttr().Get()
        solver_velocity_iteration_count = physx_rigid_body_api.GetSolverVelocityIterationCountAttr().Get()

        mass_api = UsdPhysics.MassAPI(rigid_prim)
        if mass_api:
            explicit_mass = mass_api.GetMassAttr().Get()

    return {
        "root_prim_path": root_prim_path,
        "collider_count": len(collider_prims),
        "collider_types": collider_types,
        "material_paths": material_paths,
        "static_friction": static_frictions,
        "dynamic_friction": dynamic_frictions,
        "restitution": restitutions,
        "linear_damping": linear_damping,
        "angular_damping": angular_damping,
        "contact_offset": contact_offsets,
        "rest_offset": rest_offsets,
        "solver_position_iteration_count": solver_position_iteration_count,
        "solver_velocity_iteration_count": solver_velocity_iteration_count,
        "explicit_mass": explicit_mass,
    }


def print_entity_physics_block(
    label: str,
    entity_summary: dict[str, object] | None,
    *,
    mass: float | None = None,
    inertia_diag: str | None = None,
    root_pos_w: str | None = None,
    root_lin_vel_w: str | None = None,
    root_ang_vel_w: str | None = None,
) -> None:
    print(f"  {label}:")
    if entity_summary is None:
        print("    <prim not found>")
        return

    print(f"    root_prim={entity_summary['root_prim_path']}")
    print(f"    collider_count={entity_summary['collider_count']}")
    print(f"    collider_type={_format_unique_values(entity_summary['collider_types'])}")
    print(f"    material_path={_format_unique_values(entity_summary['material_paths'])}")
    print(f"    static_friction={_format_unique_values(entity_summary['static_friction'])}")
    print(f"    dynamic_friction={_format_unique_values(entity_summary['dynamic_friction'])}")
    print(f"    restitution={_format_unique_values(entity_summary['restitution'])}")
    print(f"    linear_damping={_format_optional_scalar(entity_summary['linear_damping'])}")
    print(f"    angular_damping={_format_optional_scalar(entity_summary['angular_damping'])}")
    print(f"    contact_offset={_format_unique_values(entity_summary['contact_offset'])}")
    print(f"    rest_offset={_format_unique_values(entity_summary['rest_offset'])}")
    print(
        "    "
        f"solver_position_iteration_count={_format_optional_scalar(entity_summary['solver_position_iteration_count'], 0)} "
        f"solver_velocity_iteration_count={_format_optional_scalar(entity_summary['solver_velocity_iteration_count'], 0)}"
    )

    resolved_mass = mass if mass is not None else entity_summary.get("explicit_mass")
    print(f"    mass={_format_optional_scalar(resolved_mass)}")
    if inertia_diag is not None:
        print(f"    inertia_diag={inertia_diag}")
    if root_pos_w is not None:
        print(f"    root_pos_w={root_pos_w}")
    if root_lin_vel_w is not None:
        print(f"    root_lin_vel_w={root_lin_vel_w}")
    if root_ang_vel_w is not None:
        print(f"    root_ang_vel_w={root_ang_vel_w}")


def print_robot_and_object_physics(env: ManagerBasedRLEnv, env_cfg, title: str = "physics_summary") -> None:
    print(f"{title}:")
    print(f"  dt={env.physics_dt:.6f}")

    robot = getattr(env.scene, "articulations", {}).get("robot")
    if isinstance(robot, Articulation):
        robot.update(env.physics_dt)
        robot_data = robot.data
        total_mass = float(robot_data.default_mass[0].sum().item()) if robot_data.default_mass is not None else float("nan")
        print("robot:")
        print(f"  bodies={robot.num_bodies} joints={robot.num_joints} total_mass={total_mass:.4f}")
        print(f"  root_pos_w={_format_tensor_values(robot_data.root_pos_w[0])}")
        print(f"  root_lin_vel_w={_format_tensor_values(robot_data.root_lin_vel_w[0])}")
        print(f"  root_ang_vel_w={_format_tensor_values(robot_data.root_ang_vel_w[0])}")
        print("  joints:")
        for joint_index, joint_name in enumerate(robot_data.joint_names):
            joint_pos = float(robot_data.joint_pos[0, joint_index].item())
            joint_vel = float(robot_data.joint_vel[0, joint_index].item())
            joint_stiffness = float(robot_data.joint_stiffness[0, joint_index].item())
            joint_damping = float(robot_data.joint_damping[0, joint_index].item())
            joint_static_friction = float(robot_data.joint_friction_coeff[0, joint_index].item())
            joint_dynamic_friction = float(robot_data.joint_dynamic_friction_coeff[0, joint_index].item())
            joint_limits = _format_tensor_values(robot_data.joint_pos_limits[0, joint_index])
            print(
                f"    {joint_name:<24} pos={joint_pos:+.4f} vel={joint_vel:+.4f} "
                f"stiffness={joint_stiffness:.4f} damping={joint_damping:.4f} "
                f"static_friction={joint_static_friction:.4f} dynamic_friction={joint_dynamic_friction:.4f} "
                f"limits={joint_limits}"
            )
    else:
        print("robot:")
        print("  <robot articulation not found>")

    stage = env.scene.stage
    env_root_path = env.scene.env_prim_paths[0]
    default_material_path = f"{getattr(env_cfg.sim, 'physics_prim_path', '/physicsScene')}/defaultMaterial"
    object_names = list(getattr(getattr(env_cfg, "_scene_profile", None), "object_names", []))
    print("objects:")
    if not object_names:
        print("  <no scene objects>")
    else:
        rigid_objects = getattr(env.scene, "rigid_objects", {})
        found_object = False
        for object_name in object_names:
            rigid_object = rigid_objects.get(object_name)
            if not isinstance(rigid_object, RigidObject):
                print(f"  {object_name}: <rigid object not found>")
                continue

            rigid_object.update(env.physics_dt)
            rigid_data = rigid_object.data
            object_mass = float(rigid_data.default_mass[0, 0].item()) if rigid_data.default_mass is not None else None
            print_entity_physics_block(
                object_name,
                summarize_entity_physics(
                    stage,
                    f"{env_root_path}/{object_name}",
                    default_material_path=default_material_path,
                ),
                mass=object_mass,
                inertia_diag=_format_inertia_diag(rigid_data.default_inertia[0]),
                root_pos_w=_format_tensor_values(rigid_data.root_pos_w[0]),
                root_lin_vel_w=_format_tensor_values(rigid_data.root_lin_vel_w[0]),
                root_ang_vel_w=_format_tensor_values(rigid_data.root_ang_vel_w[0]),
            )
            found_object = True

        if not found_object:
            print("  <no rigid scene objects found>")

    print("table:")
    print_entity_physics_block(
        "furniture_table",
        summarize_entity_physics(
            stage,
            f"{env_root_path}/furniture_table",
            default_material_path=default_material_path,
        ),
    )


def main() -> None:
    env = None
    env_cfg = None
    step_count = 0
    stage = "build_runtime_env_cfg"

    try:
        env_cfg = build_runtime_env_cfg()

        print("=" * 72)
        print("GeoMind IsaacLab Viewer")
        print("=" * 72)
        print(f"cfg_kind:       {args_cli.cfg_kind}")
        print(f"scene_id:       {env_cfg.scene_id}")
        print(f"layout_key:     {env_cfg.task_layout_key}")
        print(f"task_index:     {env_cfg.task_index}")
        print(f"task_mode:      {env_cfg.task_mode}")
        print(f"lighting_mode:  {env_cfg.lighting_mode}")
        print(f"ui_viewports:   {getattr(env_cfg, 'ui_viewport_camera_map', None)}")
        print(f"num_envs:       {env_cfg.scene.num_envs}")
        print(f"device:         {env_cfg.sim.device}")
        print(f"use_fabric:     {env_cfg.sim.use_fabric}")
        print(f"hq_render:      {args_cli.high_quality_render}")
        print(f"aa_mode:        {env_cfg.sim.render.antialiasing_mode}")
        print(f"spp:            {env_cfg.sim.render.samples_per_pixel}")
        print(f"render_mode:    {env_cfg.sim.render.rendering_mode}")
        print(f"viewer_eye:     {tuple(env_cfg.viewer.eye)}")
        print(f"viewer_lookat:  {tuple(env_cfg.viewer.lookat)}")

        stage = "create_env"
        env = ManagerBasedRLEnv(cfg=env_cfg)
        if env.sim.has_gui():
            configure_env_camera_viewports(env, getattr(env_cfg, "ui_viewport_camera_map", None))
        print(f"scene_usd:      {env_cfg._scene_profile.usd_path if env_cfg._scene_profile is not None else '<unknown>'}")
        print(f"instruction:    {getattr(env_cfg, 'task_instruction', '')}")
        print(f"action_space:   {env.action_space}")
        print(f"observation:    {env.observation_space}")
        print("=" * 72)

        stage = "env_reset"
        env.reset()
        # print_camera_height_offsets(env, env_cfg)
        # print_robot_and_object_physics(env, env_cfg, title="physics_summary_after_reset")
        with torch.inference_mode():
            settle_env_after_reset(env, RESET_SETTLE_STEPS)
        # print_robot_and_object_physics(
        #     env,
        #     env_cfg,
        #     title=f"physics_summary_after_reset_settle_{RESET_SETTLE_STEPS}",
        # )

        while simulation_app.is_running():
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break

            with torch.inference_mode():
                stage = f"env_step[{step_count}]"
                actions = make_actions(env)
                _, _, terminated, truncated, _ = env.step(actions)
                step_count += 1

                if should_reset(terminated, truncated):
                    stage = f"env_reset_after_step[{step_count}]"
                    env.reset()

    except Exception:
        print_failure_context(stage, step_count=step_count, env_cfg=env_cfg)
        raise SystemExit(1)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
