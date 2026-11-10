from __future__ import annotations

import os
from dataclasses import MISSING
from functools import lru_cache
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

from geomind_lab.utils import asset_utils
from geomind_lab.utils.object_physics_utils import build_geomind_object_physics_cfg

from . import mdp as geomind_mdp
from .geomind_scene_cfg import DEFAULT_LIGHTING_MODE, GeoMindSceneCfg, SceneProfile, get_scene_profile, normalize_lighting_mode
from .task_specs import load_task, resolve_task_instruction, resolve_task_protocol, resolve_task_spec_object_names


@lru_cache(maxsize=64)
def _usd_declares_rigid_body(usd_path: str) -> bool:
    try:
        return 'PhysicsRigidBodyAPI' in Path(usd_path).read_text(encoding="utf-8")
    except Exception:
        return False


@configclass
class ActionsCfg:
    """Action terms configured by robot-specific environment cfgs."""

    arm_action: Any = MISSING
    gripper_action: Any = MISSING


@configclass
class ObservationsCfg:
    """Observation terms shared by VLA and SkillGen variants."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=stack_mdp.last_action)
        joint_pos = ObsTerm(func=stack_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=stack_mdp.joint_vel_rel)
        object = ObsTerm(
            func=geomind_mdp.object_obs,
            params={"object_cfgs": [], "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        object_positions = ObsTerm(
            func=geomind_mdp.cube_positions_in_world_frame,
            params={"object_cfgs": []},
        )
        object_orientations = ObsTerm(
            func=geomind_mdp.cube_orientations_in_world_frame,
            params={"object_cfgs": []},
        )
        eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=stack_mdp.gripper_pos)
        goal_error = ObsTerm(func=geomind_mdp.task_goal_xy_error, params={"goal_sequence": [], "task_protocol": {}})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        success = ObsTerm(func=geomind_mdp.goal_sequence_reached, params={"goal_sequence": [], "task_protocol": {}})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Task and safety-based terminations."""

    time_out = DoneTerm(func=geomind_mdp.time_out, time_out=True)
    success = DoneTerm(func=geomind_mdp.goal_sequence_reached_for_done_term, params={"goal_sequence": [], "task_protocol": {}})


@configclass
class GeoMindEnvCfg(ManagerBasedRLEnvCfg):
    """Base GeoMind environment with scene/task configured from annotation."""

    scene: GeoMindSceneCfg = GeoMindSceneCfg(num_envs=128, env_spacing=3.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = {}
    rewards = {}
    events = {}
    curriculum = {}

    task_id: str = "reindex_extreme"
    task_variant: str = ""
    scene_name: str = ""
    object_mode: str = ""
    task_layout_key: str = ""
    task_index: int = 0
    scene_id: int = 0
    task_category: str = "generic"
    task_mode: str = "vla"
    lighting_mode: str = DEFAULT_LIGHTING_MODE
    # Optional 1-based extra UI viewport mapping. The default Viewport window is always kept.
    # Example: {1: "table_cam", 2: "wrist_cam"} will open two additional camera windows.
    # Available GeoMind camera names: table_cam, wrist_cam, ego_view_camera, front_camera, left_camera, right_camera.
    ui_viewport_camera_map: dict[int, str] | None = None

    task_spec: dict[str, Any] | None = None
    task_protocol: dict[str, Any] | None = None
    task_metadata: dict[str, Any] | None = None
    task_instruction: str = ""
    _scene_profile: SceneProfile | None = None
    protocol_displacement_tol: float = 0.02
    protocol_min_reorder_gap: float = 0.08
    reset_stabilization_steps: int = 60
    post_intervention_settle_steps: int = 60
    # When true and the viewer has a GUI, draw ``gt_spec.target_zone`` as a green debug-line rectangle
    # for ``directional_region_placement`` (e.g. place_directional) each simulation step.
    viz_directional_placement_zone: bool = True

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = str(os.environ.get(name, "")).strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = str(os.environ.get(name, "")).strip()
        if not raw:
            return int(default)
        try:
            return int(raw)
        except Exception:
            return int(default)

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.sim.dt = 1/120
        policy_hz = self._env_float("REMOTE_POLICY_HZ", 0.0)
        if policy_hz > 0.0:
            self.decimation = max(int(round((1.0 / float(self.sim.dt)) / policy_hz)), 1)
        else:
            self.decimation = max(self._env_int("GEOMIND_ACTION_REPEAT", 5), 1)
        self.reset_stabilization_steps = self._env_int(
            "GEOMIND_RESET_STABILIZATION_STEPS",
            int(self.reset_stabilization_steps),
        )
        self.post_intervention_settle_steps = self._env_int(
            "GEOMIND_POST_INTERVENTION_SETTLE_STEPS",
            int(self.post_intervention_settle_steps),
        )
        # Keep render interval aligned with control decimation to avoid extra
        # renders per env-step that can cause visible stutter.
        self.sim.render_interval = self.decimation
        # Make direct lighting explicit instead of inheriting a renderer-dependent default.
        self.sim.render.enable_direct_lighting = True
        # Disable DLSS frame generation explicitly for more predictable rendering behavior.
        self.sim.render.enable_dlssg = False

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.lighting_mode = normalize_lighting_mode(self.lighting_mode)
        for attr_name in ("lighting_mode", "_lighting_mode"):
            if attr_name in self.scene.__dict__:
                delattr(self.scene, attr_name)

        if self.task_spec is None:
            loaded_task = load_task(
                task_id=self.task_id or None,
                variant=self.task_variant or None,
                scene=self.scene_name or None,
                scene_id=self.scene_id,
                layout_key=self.task_layout_key or None,
                task_index=self.task_index,
                object_mode=self.object_mode or None,
            )
            self.task_spec = loaded_task.task_spec
            self.task_id = loaded_task.request.task_id or self.task_id
            self.task_variant = loaded_task.request.variant or self.task_variant
            self.scene_name = loaded_task.request.scene or self.scene_name
            self.object_mode = loaded_task.request.object_mode or self.object_mode
            self.scene_id = int(loaded_task.request.scene_id)
            self.task_layout_key = str(loaded_task.request.layout_key)
            self.task_index = int(loaded_task.request.task_index)
        resolved_task_request = (
            self.task_spec.get("resolved_task_request", {})
            if isinstance(self.task_spec, dict)
            else {}
        )
        if isinstance(resolved_task_request, dict):
            self.task_id = str(resolved_task_request.get("task_id") or self.task_id)
            self.task_variant = str(resolved_task_request.get("variant") or self.task_variant)
            self.scene_name = str(
                resolved_task_request.get("scene_name")
                or resolved_task_request.get("scene")
                or self.scene_name
            )
            self.object_mode = str(resolved_task_request.get("object_mode") or self.object_mode)
            self.task_layout_key = str(resolved_task_request.get("layout_key") or self.task_layout_key)
            if resolved_task_request.get("task_index") is not None:
                self.task_index = int(resolved_task_request["task_index"])
            if resolved_task_request.get("scene_id") is not None:
                self.scene_id = int(resolved_task_request["scene_id"])

        self._scene_profile = get_scene_profile(self.scene_id)
        scene_adaptation = self.task_spec.get("scene_adaptation", {}) if isinstance(self.task_spec, dict) else {}
        background_overrides = (
            scene_adaptation.get("background_overrides", {}) if isinstance(scene_adaptation, dict) else {}
        )
        hide_basket = bool(background_overrides.get("hide_basket", False)) if isinstance(background_overrides, dict) else False
        self.scene.apply_scene(self.scene_id, lighting_mode=self.lighting_mode, hide_basket=hide_basket)
        self.task_protocol = resolve_task_protocol(self.task_spec)
        raw_task_metadata = self.task_spec.get("task_metadata", {}) if isinstance(self.task_spec, dict) else {}
        self.task_metadata = dict(raw_task_metadata) if isinstance(raw_task_metadata, dict) else {}
        self.task_instruction = resolve_task_instruction(self.task_spec, self.task_protocol)

        object_names = resolve_task_spec_object_names(self.task_spec, self._scene_profile)
        self.print_object_runtime_dimensions(object_names)
        self._configure_dynamic_terms(object_names)
        self._configure_object_layer(object_names)

        self.observations.policy.goal_error.params["goal_sequence"] = self.task_spec["goal_sequence"]
        self.observations.policy.goal_error.params["task_protocol"] = self.task_protocol
        if hasattr(self.observations, "subtask_terms") and hasattr(self.observations.subtask_terms, "success"):
            self.observations.subtask_terms.success.params["goal_sequence"] = self.task_spec["goal_sequence"]
            self.observations.subtask_terms.success.params["task_protocol"] = self.task_protocol
        self.terminations.success.params["goal_sequence"] = self.task_spec["goal_sequence"]
        self.terminations.success.params["task_protocol"] = self.task_protocol

    def _configure_dynamic_terms(self, object_names: list[str]) -> None:
        object_cfgs = [SceneEntityCfg(name) for name in object_names]
        policy = self.observations.policy
        if hasattr(policy, "object"):
            policy.object.params["object_cfgs"] = object_cfgs
        if hasattr(policy, "object_positions"):
            policy.object_positions.params["object_cfgs"] = object_cfgs
        if hasattr(policy, "object_orientations"):
            policy.object_orientations.params["object_cfgs"] = object_cfgs

        if hasattr(self.observations, "subtask_terms"):
            for key in list(self.observations.subtask_terms.__dict__.keys()):
                if key.startswith("grasp_"):
                    delattr(self.observations.subtask_terms, key)
            for idx, object_name in enumerate(object_names):
                setattr(
                    self.observations.subtask_terms,
                    f"grasp_{idx + 1}",
                    ObsTerm(
                        func=geomind_mdp.object_grasped,
                        params={
                            "robot_cfg": SceneEntityCfg("robot"),
                            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                            "object_cfg": SceneEntityCfg(object_name),
                        },
                    ),
                )

        for key in list(self.terminations.__dict__.keys()):
            if key.endswith("_dropping"):
                delattr(self.terminations, key)
        for object_name in object_names:
            setattr(
                self.terminations,
                f"{object_name}_dropping",
                DoneTerm(
                    func=geomind_mdp.asset_height_below_minimum,
                    params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg(object_name)},
                ),
            )

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _object_asset_name(object_name: str) -> str:
        prefix, separator, suffix = object_name.rpartition("_")
        if separator and suffix.isdigit():
            return prefix
        return object_name

    def _resolve_object_runtime_dimensions(
        self,
        object_name: str,
        ref_pose: dict[str, tuple[float, ...]] | None = None,
    ) -> tuple[float, float, float] | None:
        runtime_object_specs = self.task_spec.get("runtime_objects", []) if isinstance(self.task_spec, dict) else []
        if isinstance(runtime_object_specs, list):
            for runtime_object in runtime_object_specs:
                if not isinstance(runtime_object, dict):
                    continue
                if str(runtime_object.get("name", "")).strip() != object_name:
                    continue
                kind = str(runtime_object.get("kind", "")).strip()
                if kind == "primitive_cuboid":
                    size = runtime_object.get("size", ())
                    if isinstance(size, (list, tuple)) and len(size) >= 3:
                        dim_x = self._safe_float(size[0])
                        dim_y = self._safe_float(size[1])
                        dim_z = self._safe_float(size[2])
                        if dim_x is not None and dim_y is not None and dim_z is not None:
                            return (abs(dim_x), abs(dim_y), abs(dim_z))
                if kind == "primitive_sphere":
                    radius = self._safe_float(runtime_object.get("radius"))
                    if radius is not None:
                        diameter = 2.0 * abs(radius)
                        return (diameter, diameter, diameter)
                if kind == "primitive_cylinder":
                    radius = self._safe_float(runtime_object.get("radius"))
                    height = self._safe_float(runtime_object.get("height"))
                    if radius is not None and height is not None:
                        diameter = 2.0 * abs(radius)
                        return (diameter, diameter, abs(height))
                if kind == "primitive_cone":
                    radius = self._safe_float(runtime_object.get("radius"))
                    height = self._safe_float(runtime_object.get("height"))
                    if radius is not None and height is not None:
                        diameter = 2.0 * abs(radius)
                        return (diameter, diameter, abs(height))
                if kind == "usd":
                    asset_category = str(runtime_object.get("asset_category", "")).strip()
                    asset_key = str(runtime_object.get("asset_key", "")).strip()
                    asset_info = (
                        asset_utils.get_asset_info_by_name(asset_category, asset_key)
                        if asset_category and asset_key
                        else None
                    )
                    if isinstance(asset_info, dict):
                        dims = asset_info.get("dimensions", {})
                        if isinstance(dims, dict):
                            dim_x = self._safe_float(dims.get("x"))
                            dim_y = self._safe_float(dims.get("y"))
                            dim_z = self._safe_float(dims.get("z"))
                            if dim_x is not None and dim_y is not None and dim_z is not None:
                                scale = runtime_object.get("scale", ())
                                scale_x = 1.0
                                scale_y = 1.0
                                scale_z = 1.0
                                if isinstance(scale, (list, tuple)) and len(scale) >= 3:
                                    maybe_scale_x = self._safe_float(scale[0])
                                    maybe_scale_y = self._safe_float(scale[1])
                                    maybe_scale_z = self._safe_float(scale[2])
                                    if maybe_scale_x is not None:
                                        scale_x = maybe_scale_x
                                    if maybe_scale_y is not None:
                                        scale_y = maybe_scale_y
                                    if maybe_scale_z is not None:
                                        scale_z = maybe_scale_z
                                return (abs(dim_x * scale_x), abs(dim_y * scale_y), abs(dim_z * scale_z))
                dimensions = runtime_object.get("dimensions", ())
                if isinstance(dimensions, (list, tuple)) and len(dimensions) >= 3:
                    dim_x = self._safe_float(dimensions[0])
                    dim_y = self._safe_float(dimensions[1])
                    dim_z = self._safe_float(dimensions[2])
                    if dim_x is not None and dim_y is not None and dim_z is not None:
                        return (abs(dim_x), abs(dim_y), abs(dim_z))

        if ref_pose is None:
            if self._scene_profile is None:
                return None
            ref_pose = self._scene_profile.xforms.get(object_name)
        if not isinstance(ref_pose, dict):
            return None

        asset_info = asset_utils.get_asset_info_by_name("catch_object", self._object_asset_name(object_name))
        if not isinstance(asset_info, dict):
            return None

        dims = asset_info.get("dimensions", {})
        if not isinstance(dims, dict):
            return None

        dim_x = self._safe_float(dims.get("x"))
        dim_y = self._safe_float(dims.get("y"))
        dim_z = self._safe_float(dims.get("z"))
        if dim_x is None or dim_y is None or dim_z is None:
            return None

        scale = ref_pose.get("scale", ())
        scale_x = 1.0
        scale_y = 1.0
        scale_z = 1.0
        if isinstance(scale, (list, tuple)) and len(scale) >= 3:
            maybe_scale_x = self._safe_float(scale[0])
            maybe_scale_y = self._safe_float(scale[1])
            maybe_scale_z = self._safe_float(scale[2])
            if maybe_scale_x is not None:
                scale_x = maybe_scale_x
            if maybe_scale_y is not None:
                scale_y = maybe_scale_y
            if maybe_scale_z is not None:
                scale_z = maybe_scale_z

        return (abs(dim_x * scale_x), abs(dim_y * scale_y), abs(dim_z * scale_z))

    def _resolve_object_runtime_height(
        self,
        object_name: str,
        ref_pose: dict[str, tuple[float, ...]] | None = None,
    ) -> float | None:
        dimensions = self._resolve_object_runtime_dimensions(object_name, ref_pose)
        if dimensions is None:
            return None
        return dimensions[2]

    def print_object_runtime_dimensions(
        self,
        object_names: list[str] | None = None,
    ) -> dict[str, tuple[float, float, float] | None]:
        if object_names is None:
            if self._scene_profile is None:
                return {}
            object_names = list(self._scene_profile.object_names)

        print(
            f"[object_dims] scene_id={self.scene_id}, "
            f"task_layout_key={self.task_layout_key}, task_index={self.task_index}"
        )
        dimensions_by_object: dict[str, tuple[float, float, float] | None] = {}
        for object_name in object_names:
            ref_pose = self._scene_profile.xforms.get(object_name) if self._scene_profile is not None else None
            dimensions = self._resolve_object_runtime_dimensions(object_name, ref_pose)
            dimensions_by_object[object_name] = dimensions
            if dimensions is None:
                print(f"[object_dims] {object_name}: unavailable")
                continue
            print(
                f"[object_dims] {object_name}: "
                f"length={dimensions[0]:.6f}, width={dimensions[1]:.6f}, height={dimensions[2]:.6f}"
            )
        return dimensions_by_object

    def _configure_object_layer(self, object_names: list[str]) -> None:
        object_positions = self.task_spec.get("object_positions", {})
        runtime_object_specs = self.task_spec.get("runtime_objects", []) if isinstance(self.task_spec, dict) else []
        runtime_object_by_name = {
            str(runtime_object.get("name", "")).strip(): runtime_object
            for runtime_object in runtime_object_specs
            if isinstance(runtime_object, dict) and str(runtime_object.get("name", "")).strip()
        }

        object_rigid_props, object_mass_props = build_geomind_object_physics_cfg()

        try:
            from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg

            object_collision_props = CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            )
        except Exception:
            object_collision_props = None

        try:
            from isaaclab.sim.schemas.schemas_cfg import ConvexDecompositionPropertiesCfg

            object_mesh_collision_props = ConvexDecompositionPropertiesCfg()
        except Exception:
            object_mesh_collision_props = None

        def _spawn_usd_with_physics_schemas(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
            from isaaclab.sim import schemas
            from isaaclab.sim.spawners.from_files import from_files
            from pxr import Usd, UsdGeom, UsdPhysics

            prim = from_files.spawn_from_usd(
                prim_path, cfg, translation=translation, orientation=orientation, **kwargs
            )
            real_path = prim.GetPath().pathString

            stage = prim.GetStage()
            root_prim = stage.GetPrimAtPath(real_path)
            existing_rigid_body_path = None
            physics_path = real_path
            for p in Usd.PrimRange(root_prim):
                if not p.IsValid() or p.IsInstanceProxy():
                    continue
                if existing_rigid_body_path is None and p.HasAPI(UsdPhysics.RigidBodyAPI):
                    existing_rigid_body_path = p.GetPath().pathString
                if not p.IsA(UsdGeom.Mesh):
                    continue
                if existing_rigid_body_path is None:
                    parent = p.GetParent()
                    if parent and parent.IsValid():
                        physics_path = parent.GetPath().pathString
                        break
                else:
                    physics_path = existing_rigid_body_path
                    break

            rigid_props = getattr(cfg, "_geomind_rigid_props", None)
            mass_props = getattr(cfg, "_geomind_mass_props", None)
            collision_props = getattr(cfg, "_geomind_collision_props", None)
            mesh_collision_props = getattr(cfg, "_geomind_mesh_collision_props", None)
            mesh_collision_props = _resolve_runtime_mesh_collision_cfg(mesh_collision_props)

            if rigid_props is not None:
                if existing_rigid_body_path is not None:
                    schemas.modify_rigid_body_properties(existing_rigid_body_path, rigid_props)
                else:
                    schemas.define_rigid_body_properties(physics_path, rigid_props)
            if mass_props is not None:
                if existing_rigid_body_path is not None:
                    schemas.modify_mass_properties(existing_rigid_body_path, mass_props)
                else:
                    schemas.define_mass_properties(physics_path, mass_props)

            for p in Usd.PrimRange(root_prim):
                if not p.IsA(UsdGeom.Mesh):
                    continue
                mesh_path = p.GetPath().pathString
                if collision_props is not None:
                    if p.HasAPI(UsdPhysics.CollisionAPI):
                        schemas.modify_collision_properties(mesh_path, collision_props)
                    else:
                        schemas.define_collision_properties(mesh_path, collision_props)
                if mesh_collision_props is not None:
                    schemas.define_mesh_collision_properties(mesh_path, mesh_collision_props)

            return prim

        def _spawn_usd_with_static_collision(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
            from isaaclab.sim import schemas
            from isaaclab.sim.spawners.from_files import from_files
            from pxr import Usd, UsdGeom, UsdPhysics

            prim = from_files.spawn_from_usd(
                prim_path, cfg, translation=translation, orientation=orientation, **kwargs
            )
            real_path = prim.GetPath().pathString

            stage = prim.GetStage()
            root_prim = stage.GetPrimAtPath(real_path)
            collision_props = getattr(cfg, "_geomind_collision_props", None)
            mesh_collision_props = _resolve_runtime_mesh_collision_cfg(getattr(cfg, "_geomind_mesh_collision_props", None))

            try:
                from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

                disabled_rigid_props = RigidBodyPropertiesCfg(rigid_body_enabled=False)
            except Exception:
                disabled_rigid_props = None

            for p in Usd.PrimRange(root_prim):
                if not p.IsValid() or p.IsInstanceProxy():
                    continue
                if disabled_rigid_props is not None and p.HasAPI(UsdPhysics.RigidBodyAPI):
                    schemas.modify_rigid_body_properties(p.GetPath().pathString, disabled_rigid_props)
                if not p.IsA(UsdGeom.Mesh):
                    continue
                mesh_path = p.GetPath().pathString
                if collision_props is not None:
                    if p.HasAPI(UsdPhysics.CollisionAPI):
                        schemas.modify_collision_properties(mesh_path, collision_props)
                    else:
                        schemas.define_collision_properties(mesh_path, collision_props)
                if mesh_collision_props is not None:
                    schemas.define_mesh_collision_properties(mesh_path, mesh_collision_props)

            return prim

        def _resolve_scene_table_top_height() -> float:
            table_name = self._scene_profile.table_name if self._scene_profile is not None else None
            table_xform = self._scene_profile.xforms.get(table_name) if self._scene_profile and table_name else None
            if table_name and table_xform:
                resolved = asset_utils.resolve_tabletop_height(
                    table_name,
                    position_z=table_xform["pos"][2],
                    scale_z=table_xform["scale"][2],
                )
                if resolved is not None:
                    return float(resolved)
            if table_name:
                resolved = asset_utils.resolve_tabletop_height(table_name)
                if resolved is not None:
                    return float(resolved)
            return 0.75

        def _coerce_vector(value: Any, length: int, default: tuple[float, ...]) -> tuple[float, ...]:
            if isinstance(value, (list, tuple)) and len(value) >= length:
                resolved: list[float] = []
                for idx in range(length):
                    maybe_value = self._safe_float(value[idx])
                    resolved.append(default[idx] if maybe_value is None else maybe_value)
                return tuple(resolved)
            return default

        def _runtime_object_xy(runtime_object: dict[str, Any], object_name: str) -> tuple[float, float]:
            configured = object_positions.get(object_name, {}).get("position", [])
            if isinstance(configured, list) and len(configured) >= 2:
                return (float(configured[0]), float(configured[1]))
            pose = runtime_object.get("pose", {})
            xy = pose.get("xy", []) if isinstance(pose, dict) else []
            if isinstance(xy, list) and len(xy) >= 2:
                return (float(xy[0]), float(xy[1]))
            return (0.0, 0.0)

        def _runtime_object_yaw(runtime_object: dict[str, Any]) -> float:
            pose = runtime_object.get("pose", {})
            if not isinstance(pose, dict):
                return 0.0
            maybe_yaw = self._safe_float(pose.get("yaw", 0.0))
            return 0.0 if maybe_yaw is None else maybe_yaw

        def _yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
            import math

            half_yaw = 0.5 * float(yaw)
            return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))

        def _runtime_object_z(
            runtime_object: dict[str, Any],
            table_top_height: float,
        ) -> float:
            kind = str(runtime_object.get("kind", "")).strip()
            if kind == "primitive_cuboid":
                size = runtime_object.get("size", ())
                if isinstance(size, (list, tuple)) and len(size) >= 3:
                    half_height = self._safe_float(size[2])
                    if half_height is not None:
                        return float(table_top_height) + 0.5 * float(half_height)
                return float(table_top_height) + 0.02
            if kind == "primitive_sphere":
                radius = self._safe_float(runtime_object.get("radius"))
                if radius is not None:
                    return float(table_top_height) + abs(float(radius))
                return float(table_top_height) + 0.02
            if kind == "primitive_cylinder":
                height = self._safe_float(runtime_object.get("height"))
                if height is not None:
                    return float(table_top_height) + 0.5 * abs(float(height))
                return float(table_top_height) + 0.03
            if kind == "primitive_cone":
                height = self._safe_float(runtime_object.get("height"))
                if height is not None:
                    return float(table_top_height) + 0.5 * abs(float(height))
                return float(table_top_height) + 0.03

            asset_category = str(runtime_object.get("asset_category", "")).strip()
            asset_key = str(runtime_object.get("asset_key", "")).strip()
            asset_info = asset_utils.get_asset_info_by_name(asset_category, asset_key) if asset_category and asset_key else None
            scale = runtime_object.get("scale", ())
            scale_z = 1.0
            if isinstance(scale, (list, tuple)) and len(scale) >= 3:
                maybe_scale_z = self._safe_float(scale[2])
                if maybe_scale_z is not None:
                    scale_z = maybe_scale_z
            if isinstance(asset_info, dict):
                vertical_extent = asset_utils.get_asset_vertical_extent(asset_info)
                if vertical_extent is not None:
                    z_min = float(vertical_extent[0]) * float(scale_z)
                    return float(table_top_height) - z_min
            return float(table_top_height) + 0.03

        def _runtime_visual_material(runtime_object: dict[str, Any]):
            material_spec = runtime_object.get("visual_material", {})
            if not isinstance(material_spec, dict):
                material_spec = {}

            material_type = str(material_spec.get("type", "")).strip().lower()
            fallback_color = _coerce_vector(
                material_spec.get("fallback_diffuse_color", runtime_object.get("visual_color", (0.63, 0.48, 0.33))),
                3,
                (0.63, 0.48, 0.33),
            )
            fallback_roughness = self._safe_float(material_spec.get("fallback_roughness", 0.6))
            if fallback_roughness is None:
                fallback_roughness = 0.6

            if material_type == "mdl":
                mdl_path = str(material_spec.get("mdl_path", "")).strip()
                if mdl_path:
                    try:
                        from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR, check_file_path

                        resolved_mdl_path = mdl_path.format(NVIDIA_NUCLEUS_DIR=NVIDIA_NUCLEUS_DIR)
                        if check_file_path(resolved_mdl_path) != 0:
                            texture_scale = material_spec.get("texture_scale")
                            texture_scale_cfg = None
                            if isinstance(texture_scale, (list, tuple)) and len(texture_scale) >= 2:
                                texture_scale_cfg = (
                                    float(texture_scale[0]),
                                    float(texture_scale[1]),
                                )
                            project_uvw = material_spec.get("project_uvw")
                            albedo_brightness = self._safe_float(material_spec.get("albedo_brightness"))
                            return sim_utils.MdlFileCfg(
                                mdl_path=mdl_path,
                                project_uvw=bool(project_uvw) if isinstance(project_uvw, bool) else None,
                                texture_scale=texture_scale_cfg,
                                albedo_brightness=albedo_brightness,
                            )
                    except Exception:
                        pass

            return sim_utils.PreviewSurfaceCfg(
                diffuse_color=fallback_color,
                roughness=float(fallback_roughness),
            )

        def _runtime_physics_spec(runtime_object: dict[str, Any]) -> dict[str, Any]:
            physics_spec = runtime_object.get("physics", {})
            return physics_spec if isinstance(physics_spec, dict) else {}

        def _runtime_collision_props(runtime_object: dict[str, Any], default_cfg):
            if default_cfg is None:
                return None
            physics_spec = _runtime_physics_spec(runtime_object)
            collision_spec = physics_spec.get("collision", {})
            if not isinstance(collision_spec, dict):
                return default_cfg

            resolved = dict(default_cfg.to_dict()) if hasattr(default_cfg, "to_dict") else {}
            for attr_name in (
                "collision_enabled",
                "contact_offset",
                "rest_offset",
                "torsional_patch_radius",
                "min_torsional_patch_radius",
            ):
                if attr_name in collision_spec:
                    resolved[attr_name] = collision_spec.get(attr_name)
            return type(default_cfg)(**resolved)

        def _runtime_physics_material(runtime_object: dict[str, Any]):
            physics_spec = _runtime_physics_spec(runtime_object)
            material_spec = physics_spec.get("material", {})
            if not isinstance(material_spec, dict) or not material_spec:
                return None

            resolved: dict[str, Any] = {}
            for attr_name in (
                "static_friction",
                "dynamic_friction",
                "restitution",
                "friction_combine_mode",
                "restitution_combine_mode",
                "compliant_contact_stiffness",
                "compliant_contact_damping",
            ):
                if attr_name in material_spec:
                    resolved[attr_name] = material_spec.get(attr_name)
            return sim_utils.RigidBodyMaterialCfg(**resolved)

        def _runtime_mesh_collision_props(runtime_object: dict[str, Any]):
            physics_spec = _runtime_physics_spec(runtime_object)
            collision_spec = physics_spec.get("collision", {})
            if not isinstance(collision_spec, dict):
                return None

            collision_type = str(collision_spec.get("type", "")).strip().lower()
            if not collision_type:
                return None

            if collision_type in {"sdf", "sdf_mesh", "static_mesh"}:
                resolved: dict[str, Any] = {}
                for attr_name in (
                    "sdf_margin",
                    "sdf_narrow_band_thickness",
                    "sdf_resolution",
                    "sdf_subgrid_resolution",
                ):
                    if attr_name in collision_spec:
                        resolved[attr_name] = collision_spec.get(attr_name)
                return {"type": "sdf", "kwargs": resolved}

            if collision_type in {"convex", "convex_decomposition"}:
                resolved: dict[str, Any] = {}
                for attr_name in (
                    "hull_vertex_limit",
                    "max_convex_hulls",
                    "min_thickness",
                    "voxel_resolution",
                    "error_percentage",
                    "shrink_wrap",
                ):
                    if attr_name in collision_spec:
                        resolved[attr_name] = collision_spec.get(attr_name)
                return {"type": "convex_decomposition", "kwargs": resolved}

            return None

        def _resolve_runtime_mesh_collision_cfg(mesh_collision_props):
            if isinstance(mesh_collision_props, dict):
                mesh_collision_type = str(mesh_collision_props.get("type", "")).strip().lower()
                mesh_collision_kwargs = mesh_collision_props.get("kwargs", {})
                if not isinstance(mesh_collision_kwargs, dict):
                    mesh_collision_kwargs = {}
                if mesh_collision_type == "sdf":
                    from isaaclab.sim.schemas.schemas_cfg import SDFMeshPropertiesCfg

                    return SDFMeshPropertiesCfg(**mesh_collision_kwargs)
                if mesh_collision_type in {"convex", "convex_decomposition"}:
                    from isaaclab.sim.schemas.schemas_cfg import ConvexDecompositionPropertiesCfg

                    return ConvexDecompositionPropertiesCfg(**mesh_collision_kwargs)
                return None
            return mesh_collision_props

        def _runtime_should_spawn_static_asset(runtime_object: dict[str, Any]) -> bool:
            physics_spec = _runtime_physics_spec(runtime_object)
            rigid_spec = physics_spec.get("rigid_body", {})
            collision_spec = physics_spec.get("collision", {})
            rigid_body_enabled = None
            if isinstance(rigid_spec, dict) and "rigid_body_enabled" in rigid_spec:
                rigid_body_enabled = rigid_spec.get("rigid_body_enabled")
            collision_type = (
                str(collision_spec.get("type", "")).strip().lower() if isinstance(collision_spec, dict) else ""
            )
            return rigid_body_enabled is False or collision_type == "static_mesh"

        def _runtime_rigid_props(runtime_object: dict[str, Any], default_cfg):
            physics_spec = _runtime_physics_spec(runtime_object)
            rigid_spec = physics_spec.get("rigid_body", {})
            if not isinstance(rigid_spec, dict) or not rigid_spec:
                return default_cfg

            resolved = dict(default_cfg.to_dict()) if default_cfg is not None and hasattr(default_cfg, "to_dict") else {}
            for attr_name in (
                "rigid_body_enabled",
                "kinematic_enabled",
                "disable_gravity",
                "linear_damping",
                "angular_damping",
                "max_linear_velocity",
                "max_angular_velocity",
                "max_depenetration_velocity",
                "max_contact_impulse",
                "enable_gyroscopic_forces",
                "retain_accelerations",
                "solver_position_iteration_count",
                "solver_velocity_iteration_count",
                "sleep_threshold",
                "stabilization_threshold",
            ):
                if attr_name in rigid_spec:
                    resolved[attr_name] = rigid_spec.get(attr_name)
            from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

            return RigidBodyPropertiesCfg(**resolved)

        if runtime_object_by_name:
            table_top_height = _resolve_scene_table_top_height()
            for object_name in object_names:
                runtime_object = runtime_object_by_name.get(object_name)
                if runtime_object is None:
                    continue

                object_xy = _runtime_object_xy(runtime_object, object_name)
                object_pos = [
                    float(object_xy[0]),
                    float(object_xy[1]),
                    _runtime_object_z(runtime_object, table_top_height),
                ]
                object_rot = list(_yaw_to_quat_wxyz(_runtime_object_yaw(runtime_object)))

                kind = str(runtime_object.get("kind", "")).strip()
                if kind == "primitive_cuboid":
                    object_size = _coerce_vector(runtime_object.get("size", (0.045, 0.045, 0.045)), 3, (0.045, 0.045, 0.045))
                    cuboid_collision_props = _runtime_collision_props(runtime_object, object_collision_props)
                    cuboid_rigid_props = _runtime_rigid_props(runtime_object, object_rigid_props)
                    cuboid_physics_material = _runtime_physics_material(runtime_object)
                    spawn_cfg = sim_utils.CuboidCfg(
                        size=object_size,
                        rigid_props=cuboid_rigid_props,
                        mass_props=object_mass_props,
                        collision_props=cuboid_collision_props,
                        physics_material=cuboid_physics_material,
                        visual_material=_runtime_visual_material(runtime_object),
                    )
                elif kind == "primitive_sphere":
                    sphere_collision_props = _runtime_collision_props(runtime_object, object_collision_props)
                    sphere_rigid_props = _runtime_rigid_props(runtime_object, object_rigid_props)
                    sphere_physics_material = _runtime_physics_material(runtime_object)
                    sphere_radius = self._safe_float(runtime_object.get("radius"))
                    if sphere_radius is None:
                        sphere_radius = 0.03
                    spawn_cfg = sim_utils.SphereCfg(
                        radius=abs(float(sphere_radius)),
                        rigid_props=sphere_rigid_props,
                        mass_props=object_mass_props,
                        collision_props=sphere_collision_props,
                        physics_material=sphere_physics_material,
                        visual_material=_runtime_visual_material(runtime_object),
                    )
                elif kind == "primitive_cylinder":
                    cylinder_collision_props = _runtime_collision_props(runtime_object, object_collision_props)
                    cylinder_rigid_props = _runtime_rigid_props(runtime_object, object_rigid_props)
                    cylinder_physics_material = _runtime_physics_material(runtime_object)
                    cylinder_radius = self._safe_float(runtime_object.get("radius"))
                    if cylinder_radius is None:
                        cylinder_radius = 0.026
                    cylinder_height = self._safe_float(runtime_object.get("height"))
                    if cylinder_height is None:
                        cylinder_height = 0.12
                    spawn_cfg = sim_utils.CylinderCfg(
                        radius=abs(float(cylinder_radius)),
                        height=abs(float(cylinder_height)),
                        axis="Z",
                        rigid_props=cylinder_rigid_props,
                        mass_props=object_mass_props,
                        collision_props=cylinder_collision_props,
                        physics_material=cylinder_physics_material,
                        visual_material=_runtime_visual_material(runtime_object),
                    )
                elif kind == "primitive_cone":
                    cone_collision_props = _runtime_collision_props(runtime_object, object_collision_props)
                    cone_rigid_props = _runtime_rigid_props(runtime_object, object_rigid_props)
                    cone_physics_material = _runtime_physics_material(runtime_object)
                    cone_radius = self._safe_float(runtime_object.get("radius"))
                    if cone_radius is None:
                        cone_radius = 0.02
                    cone_height = self._safe_float(runtime_object.get("height"))
                    if cone_height is None:
                        cone_height = 0.06
                    spawn_cfg = sim_utils.ConeCfg(
                        radius=abs(float(cone_radius)),
                        height=abs(float(cone_height)),
                        axis="Z",
                        rigid_props=cone_rigid_props,
                        mass_props=object_mass_props,
                        collision_props=cone_collision_props,
                        physics_material=cone_physics_material,
                        visual_material=_runtime_visual_material(runtime_object),
                    )
                else:
                    asset_category = str(runtime_object.get("asset_category", "")).strip()
                    asset_key = str(runtime_object.get("asset_key", "")).strip()
                    asset_info = asset_utils.get_asset_info_by_name(asset_category, asset_key) if asset_category and asset_key else None
                    if not isinstance(asset_info, dict) or not asset_info.get("usd_path"):
                        raise ValueError(f"Runtime object '{object_name}' is missing a valid USD asset reference.")
                    usd_path = str(asset_info["usd_path"])
                    object_scale = _coerce_vector(runtime_object.get("scale", (1.0, 1.0, 1.0)), 3, (1.0, 1.0, 1.0))
                    # Only override USD looks when annotation explicitly sets diffuse_color.
                    # Using .get("material_override", {}) made missing keys look like {} and
                    # _coerce_vector(None, default=blue) always applied a flat PreviewSurface,
                    # which replaces USDA/MDL materials and hides textures (e.g. viz clutter).
                    material_override = runtime_object.get("material_override")
                    diffuse_color = None
                    if isinstance(material_override, dict) and "diffuse_color" in material_override:
                        raw_diffuse = material_override.get("diffuse_color")
                        if raw_diffuse is not None:
                            diffuse_color = _coerce_vector(raw_diffuse, 3, (0.12, 0.28, 0.82))
                    spawn_cfg = UsdFileCfg(
                        usd_path=usd_path,
                        scale=object_scale,
                        rigid_props=None,
                        mass_props=None,
                        collision_props=None,
                        visual_material=(
                            sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse_color, roughness=0.35)
                            if diffuse_color is not None
                            else None
                        ),
                    )
                    usd_rigid_props = _runtime_rigid_props(runtime_object, object_rigid_props)
                    usd_collision_props = _runtime_collision_props(runtime_object, object_collision_props)
                    usd_mesh_collision_props = _runtime_mesh_collision_props(runtime_object)
                    static_asset = _runtime_should_spawn_static_asset(runtime_object)
                    if static_asset:
                        spawn_cfg._geomind_collision_props = usd_collision_props
                        spawn_cfg._geomind_mesh_collision_props = (
                            usd_mesh_collision_props if usd_mesh_collision_props is not None else object_mesh_collision_props
                        )
                        spawn_cfg.func = _spawn_usd_with_static_collision
                        setattr(
                            self.scene,
                            object_name,
                            AssetBaseCfg(
                                prim_path=f"{{ENV_REGEX_NS}}/{object_name}",
                                init_state=AssetBaseCfg.InitialStateCfg(pos=object_pos, rot=object_rot),
                                spawn=spawn_cfg,
                            ),
                        )
                        continue

                    usd_has_physics_override = bool(_runtime_physics_spec(runtime_object))
                    if usd_has_physics_override or not _usd_declares_rigid_body(usd_path):
                        spawn_cfg._geomind_rigid_props = usd_rigid_props
                        spawn_cfg._geomind_mass_props = object_mass_props
                        spawn_cfg._geomind_collision_props = usd_collision_props
                        spawn_cfg._geomind_mesh_collision_props = (
                            usd_mesh_collision_props if usd_mesh_collision_props is not None else object_mesh_collision_props
                        )
                        spawn_cfg.func = _spawn_usd_with_physics_schemas

                if kind in {"primitive_cuboid", "primitive_sphere", "primitive_cylinder", "primitive_cone"} and _runtime_should_spawn_static_asset(runtime_object):
                    setattr(
                        self.scene,
                        object_name,
                        AssetBaseCfg(
                            prim_path=f"{{ENV_REGEX_NS}}/{object_name}",
                            init_state=AssetBaseCfg.InitialStateCfg(pos=object_pos, rot=object_rot),
                            spawn=spawn_cfg,
                        ),
                    )
                    continue

                setattr(
                    self.scene,
                    object_name,
                    RigidObjectCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/{object_name}",
                        init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=object_rot),
                        spawn=spawn_cfg,
                    ),
                )
            return

        for object_name in object_names:
            ref_pose = self._scene_profile.xforms[object_name]
            object_xy = object_positions.get(object_name, {}).get("position", list(ref_pose["pos"][:2]))
            object_pos = [float(object_xy[0]), float(object_xy[1]), float(ref_pose["pos"][2])]
            object_rot = list(ref_pose["rot"])
            object_scale = tuple(ref_pose["scale"])
            usd_path = self._scene_profile.object_assets[object_name]["usd_path"]

            spawn_cfg = UsdFileCfg(
                usd_path=usd_path,
                scale=object_scale,
                rigid_props=None,
                mass_props=None,
                collision_props=None,
            )
            spawn_cfg._geomind_rigid_props = object_rigid_props
            spawn_cfg._geomind_mass_props = object_mass_props
            spawn_cfg._geomind_collision_props = object_collision_props
            spawn_cfg._geomind_mesh_collision_props = object_mesh_collision_props
            spawn_cfg.func = _spawn_usd_with_physics_schemas

            setattr(
                self.scene,
                object_name,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{object_name}",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=object_rot),
                    spawn=spawn_cfg,
                ),
            )
