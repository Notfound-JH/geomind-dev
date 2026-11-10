"""Headless PhysX backend for native v3 spatial episodes.

Simulator imports are lazy. Agent observations, historical evidence and semantic
GT follow the same contract as the CPU backend; all poses here come from PhysX.
"""
from __future__ import annotations
from geomind_lab.utils.portable_paths import configure_material_search_path as _portable_material_paths

from copy import deepcopy
import math
import random
import uuid

from .physical_contract import supports_physics_task, validate_execution, validate_physics
from .resolver import ResolvedEpisode, Snapshot, _check, _fingerprint, _position, solve
from .runtime import SemanticEpisode, evaluate_answer


ASSEMBLED_CONTACT_DYNAMICS = {
    "linear_damping": 1.2,
    "angular_damping": 4.0,
    "solver_position_iteration_count": 32,
    "solver_velocity_iteration_count": 4,
}


def load_assembled_source(resolved):
    """Validate immutable scene/state bindings before any simulator is started."""
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    from geomind.scene_assembly.audit import measured_bounds
    from geomind.scene_assembly.body_geometry import load_body_clouds, points_for

    profile = resolved.scene["physics"]
    manifest_path = Path(profile["assembly_manifest_path"]).resolve()
    _check(manifest_path.is_file(), "Assembled scene manifest is unavailable")
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    _check(digest(manifest_path) == profile["assembly_manifest_sha256"], "Assembly manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    _check(manifest.get("status") == "assembled_physics_passed" and manifest.get("offline_audit", {}).get("passed") is True,
           "Assembled scene requires previously verified initial settling")
    scene_path = Path(resolved.scene["usd_path"]).resolve()
    _check(scene_path == Path(manifest["scene_usda"]).resolve(), "Scene USD differs from its assembly manifest")
    _check(digest(scene_path) == manifest["provenance"]["scene_usda_sha256"], "Assembled scene USD hash changed")
    for path, expected in manifest["provenance"].get("used_usd_layers_sha256", {}).items():
        _check(digest(path) == expected, f"Assembly USD layer changed: {path}")
    body_path = Path(profile["asset_body_vertices_path"]).resolve()
    _check(body_path == Path(manifest["asset_body_vertices"]).resolve(), "Body point cloud differs from assembly")
    _check(digest(body_path) == profile["asset_body_vertices_sha256"] == manifest["provenance"]["asset_body_vertices_sha256"],
           "Assembled body point cloud hash changed")
    state_path = manifest_path.parent / "object_states.npz"
    _check(digest(state_path) == manifest["physics"]["sha256"]["object_states.npz"], "Original settled state trace changed")
    with np.load(state_path, allow_pickle=False) as archive:
        states = np.asarray(archive["data"], dtype=float)
        paths = archive["prim_paths"].tolist()
    _check(states.shape == (600, len(paths), 13) and np.isfinite(states).all(), "Invalid assembled state trace")
    _check(len(set(paths)) == len(paths), "Duplicate assembled state prim paths")
    # Legacy homogeneous manifests did not repeat object metadata here. Real
    # mixed assemblies bind each archive key to its manifest prim, then reorder
    # geometry into the runtime's semantic body order below.
    source_objects = manifest.get("objects", [{"prim_path": path} for path in paths])
    _check(isinstance(source_objects, list) and len(source_objects) == len(paths)
           and {obj["prim_path"] for obj in source_objects} == set(paths),
           "Manifest body-cloud prim bindings differ from settled states")
    try:
        source_clouds = load_body_clouds(dict(manifest, objects=source_objects, asset_body_vertices=str(body_path)))
    except ValueError as exc:
        _check(False, str(exc))
    clouds_by_path = {obj["prim_path"]: cloud for obj, cloud in zip(source_objects, source_clouds)}
    by_path = dict(zip(paths, states[-1]))
    binding_by_id = {item["object_id"]: item for item in profile["assembly_object_bindings"]}
    ids = [obj["id"] for obj in resolved.layout["objects"]]
    _check(set(binding_by_id) == set(ids) and len(ids) == len(paths), "Assembly object identities/count differ")
    _check({item["prim_path"] for item in binding_by_id.values()} == set(paths), "Assembly prim binding differs from actual scene")
    _check(resolved.layout["frame"] == "world" and resolved.layout["randomization"]["position_jitter_m"] == 0,
           "Assembled object layouts cannot be transformed or randomized a second time")
    _check(not any(obj.get("physical_asset") for obj in resolved.layout["objects"]),
           "Assembled GT uses measured AABB centres, not a second physical-asset origin descriptor")
    bindings = [binding_by_id[identity] for identity in ids]
    body_clouds = [clouds_by_path[binding["prim_path"]] for binding in bindings]
    # Preserve the public return type of the older shared-cloud path.
    points = source_clouds[0] if all(obj.get("body_points_key", "points_m") == "points_m" for obj in source_objects) else body_clouds
    for index, (obj, binding) in enumerate(zip(resolved.layout["objects"], bindings)):
        state = by_path[binding["prim_path"]]
        _check(np.allclose(binding["root_position_w_m"], state[:3], atol=1e-8, rtol=0)
               and np.allclose(binding["root_orientation_wxyz"], state[3:7], atol=1e-8, rtol=0),
               "Assembled reset pose differs from the measured source state")
        low, high = measured_bounds(points_for(points, index), state[:3], state[3:7])
        _check(np.allclose(obj["position_m"], (low + high) * .5, atol=1e-8, rtol=0),
               "Semantic object centre differs from its bound physical body")
    _check(abs(profile["tabletop_height_m"] - manifest["table"]["height_m"]) < 1e-8,
           "Assembled tabletop height differs from the verified source")
    return manifest, points, bindings


class IsaacLabEpisode(SemanticEpisode):
    backend = "isaaclab_physx"

    def __init__(self, resolved: ResolvedEpisode, *, device: str = "cuda:0", headless: bool = True,
                 recording: dict | None = None):
        _check(headless is True, "This physical backend is explicitly headless")
        _check(supports_physics_task(resolved.task["task_id"]), "No native physical implementation for task")
        _check("physical_execution" in resolved.episode and "physics" in resolved.scene,
               "Episode needs explicit physical_execution and scene.physics profiles")
        validate_physics(resolved.scene)
        validate_execution(resolved.episode, resolved.scene)
        super().__init__(resolved)
        self.profile = deepcopy(resolved.scene["physics"])
        self.scene_hash = _fingerprint(resolved.scene)
        self.device = device
        self.headless = headless
        self.recording_config = deepcopy(recording)
        self.recorder = None
        self.wrist_recorders = {}
        self._video_deferred_tick = None
        if recording is not None:
            from .video import validate_recording
            validate_recording(recording, self.profile["dt_s"])
        self.object_count = len(resolved.layout["objects"])
        self.asset_order = [o["asset"] for o in resolved.layout["objects"]]
        self.asset_specs = [o.get("physical_asset") for o in resolved.layout["objects"]]
        self.asset_signature = _fingerprint(self.asset_specs)
        self.is_assembled = self.profile["scene_mode"] == "AssembledScene"
        if self.is_assembled:
            self.assembled_manifest, self.assembled_body_points, self.assembled_bindings = load_assembled_source(resolved)
            source_by_prim={obj['prim_path']:obj for obj in self.assembled_manifest.get('objects',[])}
            self.assembled_dynamic={binding['object_id']:source_by_prim.get(binding['prim_path'],{}).get('dynamic',True)
                                    for binding in self.assembled_bindings}
            import hashlib
            presentation_seed = hashlib.sha256(("assembly_public_presentation_v1:" +
                self.assembled_manifest["physics"]["sha256"]["object_states.npz"]).encode()).hexdigest()
            self.assembled_public_order = [obj["id"] for obj in resolved.layout["objects"]]
            random.Random(int(presentation_seed, 16)).shuffle(self.assembled_public_order)
        for asset, spec in zip(self.asset_order, self.asset_specs):
            if self.is_assembled:
                continue
            if spec:
                import hashlib
                from pathlib import Path
                _check(hashlib.sha256(Path(spec["usd_path"]).read_bytes()).hexdigest() == spec["sha256"],
                       "Object USD changed; regenerate its descriptor")
            else:
                _check(asset in {"red_block", "blue_block", "green_block", "yellow_block"}, "Missing physical asset descriptor")
        if self.profile["scene_mode"] in {"RegisteredScene", "AssembledScene"}:
            from .registered_scene import configure_material_search_path
            configure_material_search_path()
        if self.is_assembled:
            import os
            from pathlib import Path
            _portable_material_paths()
        from geomind.core.envs.isaaclab_native_delegate import _IsaacLabAppRuntime
        _IsaacLabAppRuntime.ensure_started(headless=True, enable_cameras=recording is not None)
        self.app_runtime = _IsaacLabAppRuntime
        self._build_world()

    def _build_world(self):
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from .robot_profiles import robot_profile
        self.torch = torch
        self.dt = float(self.profile["dt_s"])
        self.protocol_decimation = round(self.profile["protocol_step_s"] / self.dt)
        solver=sim_utils.PhysxCfg()
        self.assembled_solver_selection=None
        if self.is_assembled and self.resolved.scene['robot']=='franka_robotiq':
            kinds=sorted({obj.get('kind','unknown') for obj in self.assembled_manifest['objects']})
            curved='primitive_sphere' in kinds
            # Spherical point contacts use TGS. Cylinders below have explicit
            # convex mesh colliders and use the same PGS linkage regime as boxes.
            solver.solver_type=1 if curved else 0
            self.assembled_solver_selection={'solver_type':solver.solver_type,
                'solver':'TGS' if curved else 'PGS','source_body_kinds':kinds,
                'policy':'TGS for spherical point contacts; PGS for box/cylinder convex meshes and linkage assemblies',
                'torsional_friction_overrides':False}
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=self.device, dt=self.dt,physx=solver))
        self.cylinder_collision_mode=None
        if self.is_assembled and any(obj.get('kind')=='primitive_cylinder' for obj in self.assembled_manifest['objects']):
            import omni.physx
            from omni.physx.bindings import _physx
            key=_physx.SETTING_COLLISION_APPROXIMATE_CYLINDERS
            settings=self.sim.carb_settings
            before=settings.get(key)
            # An explicit mesh collider below replaces only the original
            # Cylinder collision. Keep the global primitive setting unchanged.
            settings.set_bool(key,False)
            self.cylinder_collision_mode={'setting_key':key,'before':before,'after':settings.get(key),
                'default_value':settings.get(_physx.SETTING_COLLISION_APPROXIMATE_CYLINDERS_DEFAULT),
                'loaded_module_path':omni.physx.__file__,'loaded_binding_path':_physx.__file__,
                'representation':'explicit convex mesh collider inside the original rigid body',
                'scope':'runtime collision representation only; original USD, rendered shape, size and material preserved',
                'reason':'Use explicit GPU convex hulls after retained primitive-cylinder settling failures'}
        self.origin = torch.tensor([self.profile["frame_origin_w_m"]], device=self.device)
        if self.is_assembled:
            import hashlib
            from pathlib import Path
            from pxr import UsdPhysics, UsdGeom
            from geomind.scene_assembly.usd import measure_table
            stage = self.sim.stage
            _check(not stage.GetPrimAtPath("/World/AssembledObjects").IsValid(), "Simulation stage already contains assembly objects")
            root_layer = stage.GetRootLayer()
            root_layer.subLayerPaths.append(str(Path(self.resolved.scene["usd_path"]).resolve()))
            stage.SetEditTarget(root_layer)
            physics_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
            active_physics = self.sim.cfg.physics_prim_path
            disabled = []
            for path in physics_paths:
                if path != active_physics:
                    stage.OverridePrim(path).SetActive(False)
                    disabled.append(path)
            physics = UsdPhysics.Scene.Get(stage, active_physics)
            physics.CreateGravityDirectionAttr((0, 0, -1)); physics.CreateGravityMagnitudeAttr(9.81)
            from geomind.scene_assembly.visibility import apply_scene_visibility
            visualization = apply_scene_visibility(stage, self.assembled_manifest)
            hidden = visualization["hidden_prims"]
            if self.cylinder_collision_mode is not None:
                from geomind.scene_assembly.cylinder_collisions import install_cylinder_mesh_colliders
                from geomind.scene_assembly.body_geometry import points_for
                clouds={binding['prim_path']:points_for(self.assembled_body_points,index)
                        for index,binding in enumerate(self.assembled_bindings)}
                self.cylinder_collision_mode['explicit_mesh_adapter']=install_cylinder_mesh_colliders(
                    stage,self.assembled_manifest,clouds)
            self.assembled_table = measure_table(stage)
            from geomind.scene_assembly.usd import _background_obstacles
            self.assembled_table['retained_obstacles']=[item for item in _background_obstacles(stage,self.assembled_table['prim_path'])
                if not item['prim_path'].startswith('/World/AssembledObjects')]
            assembled_children = stage.GetPrimAtPath("/World/AssembledObjects").GetChildren()
            _check(len(assembled_children) == self.object_count, "Actual assembly object count changed during loading")
            self.scene_provenance = {
                "mode": "AssembledScene", "source_usd": self.resolved.scene["usd_path"],
                "source_sha256": hashlib.sha256(Path(self.resolved.scene["usd_path"]).read_bytes()).hexdigest(),
                "assembly_manifest": self.profile["assembly_manifest_path"],
                "assembly_manifest_sha256": self.profile["assembly_manifest_sha256"],
                "loader": "composition of the saved assembly layer; existing rigid prims bound with spawn=None",
                "source_usd_layers_sha256": self.assembled_manifest["provenance"].get("used_usd_layers_sha256", {}),
                "bound_objects": deepcopy(self.profile["assembly_object_bindings"]),
                "actual_object_count": len(assembled_children), "objects_respawned": False,
                "helper_supports_spawned": False, "layout_rerandomized": False,
                "source_physics_scene_paths": physics_paths, "active_physics_scene_path": active_physics,
                "disabled_source_physics_scenes": disabled, "visualization_hidden_prims": hidden,
                "visualization": visualization,
                "contact_dynamics": deepcopy(ASSEMBLED_CONTACT_DYNAMICS),
                "contact_dynamics_layer": "runtime root composition layer; source USD is read only",
                "contact_dynamics_applied_to": [item["prim_path"] for item in self.assembled_bindings],
                "solver_selection":deepcopy(self.assembled_solver_selection),
                "cylinder_collision_mode":deepcopy(self.cylinder_collision_mode),
            }
        elif self.profile["scene_mode"] == "RegisteredScene":
            from .registered_scene import spawn_registered_scene
            self.scene_provenance = spawn_registered_scene(self.resolved.scene)
            from geomind.scene_assembly.usd import measure_table
            surface = measure_table(self.sim.stage, '/World/RegisteredScene/furniture_table')
            self.scene_provenance['support_surface'] = {
                **surface, 'surface_triangles': surface['surface_triangles'].tolist()}
        else:
            self.scene_provenance = {"mode": "ProceduralTable"}
            ground = sim_utils.GroundPlaneCfg()
            ground.func("/World/Ground", ground)
            table_size = tuple(self.profile["table_size_m"])
            table = sim_utils.CuboidCfg(size=table_size, collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)))
            table_position = list(self.profile["frame_origin_w_m"])
            table_position[2] -= table_size[2] / 2
            table.func("/World/Table", table, translation=tuple(table_position))
            light = sim_utils.DomeLightCfg(intensity=2000)
            light.func("/World/Light", light)
        self.robot_name = self.resolved.scene["robot"]
        robot_cfg, arm_expr, hand_name, self.finger_expr = robot_profile(self.robot_name)
        robot_cfg.prim_path = "/World/Robot"
        robot_cfg.init_state.pos = tuple(self.profile["robot_base_w_m"])
        self.robot_asset = robot_cfg.spawn.usd_path
        self.robot = Articulation(robot_cfg)
        self.robots = {"left": self.robot}
        if self.robot_name == "piper_dual":
            secondary_cfg = robot_cfg.copy()
            secondary_cfg.prim_path = "/World/RobotRight"
            from .robot_profiles import PIPER_DUAL_RIGHT_HOME
            secondary_cfg.init_state.joint_pos.update(dict(zip([f"joint{i}" for i in range(1,7)], PIPER_DUAL_RIGHT_HOME)))
            secondary_cfg.init_state.pos = (robot_cfg.init_state.pos[0], -robot_cfg.init_state.pos[1], robot_cfg.init_state.pos[2])
            self.robots["right"] = Articulation(secondary_cfg)
        self.active_arm = "left"
        if self.robot_name == "piper":
            table_height = self.profile["tabletop_height_m"] if self.is_assembled else self.profile["frame_origin_w_m"][2]
            stand_height = self.profile["robot_base_w_m"][2] - table_height
            if stand_height > .001:
                stand = sim_utils.CuboidCfg(size=(.12, .12, stand_height),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.2,.2,.22)))
                pos = list(self.profile["robot_base_w_m"]); pos[2] -= stand_height/2
                stand.func("/World/PiperMount", stand, translation=tuple(pos))
        colours = {"red_block": (0.9, 0.1, 0.1), "blue_block": (0.1, 0.2, 0.9),
                   "green_block": (0.1, 0.8, 0.2), "yellow_block": (0.9, 0.8, 0.1)}
        self.bodies, self.supports = [], []
        size = tuple(self.profile["object_size_m"])
        for index, asset in enumerate(self.asset_order):
            if self.is_assembled:
                binding = self.assembled_bindings[index]
                import numpy as np
                prim = self.sim.stage.GetPrimAtPath(binding["prim_path"])
                scale = prim.GetAttribute("xformOp:scale").Get()
                source_object = next(obj for obj in self.assembled_manifest["objects"] if obj["prim_path"] == binding["prim_path"])
                _check(scale is not None and np.allclose(scale, source_object.get("scale", self.assembled_manifest["asset"]["scale"]), atol=1e-7),
                       "Binding an assembled body changed its authored catalog scale")
                from pxr import PhysxSchema
                dynamics = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                dynamics.CreateLinearDampingAttr(ASSEMBLED_CONTACT_DYNAMICS["linear_damping"])
                dynamics.CreateAngularDampingAttr(ASSEMBLED_CONTACT_DYNAMICS["angular_damping"])
                dynamics.CreateSolverPositionIterationCountAttr(ASSEMBLED_CONTACT_DYNAMICS["solver_position_iteration_count"])
                dynamics.CreateSolverVelocityIterationCountAttr(ASSEMBLED_CONTACT_DYNAMICS["solver_velocity_iteration_count"])
                body_cfg = RigidObjectCfg(prim_path=binding["prim_path"], spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(binding["root_position_w_m"]),
                        rot=tuple(binding["root_orientation_wxyz"])))
                self.bodies.append(RigidObject(body_cfg))
                continue
            spec = self.asset_specs[index]
            if spec:
                from .object_assets import spawn_catalog_object
                size = tuple(spec["dimensions_m"])
                spawn = sim_utils.UsdFileCfg(usd_path=spec["usd_path"], scale=tuple(spec["scale"]))
                spawn.func = spawn_catalog_object
                spawn._body_properties = sim_utils.RigidBodyPropertiesCfg(linear_damping=1.2, angular_damping=4.0)
                spawn._mass_properties = sim_utils.MassPropertiesCfg(mass=self.profile["object_mass_kg"])
            else:
                size = tuple(self.profile["object_size_m"])
                spawn = sim_utils.CuboidCfg(size=size, rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=32, solver_velocity_iteration_count=32) if self.resolved.episode["physical_execution"]["action"] == "PickLiftChangedObject" else sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=self.profile["object_mass_kg"]),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colours[asset]))
            body_cfg = RigidObjectCfg(prim_path=f"/World/Object_{index}", spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4 + index * 0.1, 0, 1.0)))
            self.bodies.append(RigidObject(body_cfg))
            half_height = -spec['bottom_offset_m'] if spec else size[2] / 2
            if abs(self.resolved.layout['objects'][index]['position_m'][2] - half_height) < 1e-6:
                # On-table objects use the real table mesh. Do not create a
                # hidden helper collider that could mask missing table support.
                self.supports.append(None)
                continue
            support_cfg = RigidObjectCfg(prim_path=f"/World/Support_{index}",
                spawn=sim_utils.CuboidCfg(size=(size[0] * 1.2, size[1] * 1.2, 0.02),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4 + index * 0.1, 0, 0.9)))
            self.supports.append(RigidObject(support_cfg))
        if not self.is_assembled:
            support_paths = [body.cfg.prim_path for body in self.supports if body is not None]
            self.scene_provenance['helper_supports'] = {'enabled': bool(support_paths), 'prim_paths': support_paths}
            if not any(self.asset_specs):
                self.scene_provenance['diagnostic_objects'] = {
                    'dimensions_m': list(self.profile['object_size_m']),
                    'support_policy': 'elevated_helpers' if support_paths else 'real_table_only'}
        self.camera = None
        if self.recording_config is not None:
            from isaaclab.sensors import Camera, CameraCfg
            video = self.recording_config
            self.camera = Camera(CameraCfg(prim_path="/World/TableCamera", update_period=0.0,
                height=video["height"], width=video["width"], data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=video["focal_length_mm"], horizontal_aperture=24.0,
                                               clipping_range=(0.05, 20.0))))
        self._build_wrist_camera_mounts()
        self.sim.reset()
        self.wrist_parent_body_ids = {}
        for spec in self.wrist_camera_mounts:
            robot = self.robots[spec['arm']]
            body_ids, body_names = robot.find_bodies(spec['parent_prim_path'].rsplit('/', 1)[-1])
            _check(len(body_ids) == 1, 'Wrist camera parent must identify one measured robot body')
            self.wrist_parent_body_ids[spec['name']] = body_ids[0]
        from pxr import UsdPhysics,PhysxSchema
        roots=[]
        for prim in self.sim.stage.Traverse():
            if str(prim.GetPath()).startswith('/World/Robot') and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api=PhysxSchema.PhysxArticulationAPI(prim)
                roots.append({'prim_path':str(prim.GetPath()),
                    'position_iterations':api.GetSolverPositionIterationCountAttr().Get(),
                    'velocity_iterations':api.GetSolverVelocityIterationCountAttr().Get()})
        api=PhysxSchema.PhysxSceneAPI(self.sim.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path))
        self.robot_solver_schema={'source':'composed_USD_after_sim_reset','articulation_roots':roots,
            'solver_type':api.GetSolverTypeAttr().Get(),
            'external_forces_every_iteration':api.GetEnableExternalForcesEveryIterationAttr().Get(),
            'scene_min_position_iterations':api.GetMinPositionIterationCountAttr().Get(),
            'scene_max_position_iterations':api.GetMaxPositionIterationCountAttr().Get(),
            'scene_min_velocity_iterations':api.GetMinVelocityIterationCountAttr().Get(),
            'scene_max_velocity_iterations':api.GetMaxVelocityIterationCountAttr().Get()}
        if self.camera is not None:
            self.camera.set_world_poses_from_view(
                torch.tensor([video["eye_w_m"]], device=self.device),
                torch.tensor([video["look_at_w_m"]], device=self.device))
            for _ in range(5):
                self.sim.render()
        self.arm_ids, _ = self.robot.find_joints(arm_expr, preserve_order=True)
        hand_ids, _ = self.robot.find_bodies(hand_name)
        _check(len(self.arm_ids) == (6 if self.robot_name in {"piper", "piper_dual"} else 7) and len(hand_ids) == 1 and self.robot.is_fixed_base,
               "Unexpected Franka articulation topology")
        self.hand_id = hand_ids[0]
        self.tcp_offset = torch.tensor([[0.0, 0.0, 0.107]], device=self.device)
        if self.robot_name in {"piper", "piper_dual"}:
            self.tcp_offset = torch.tensor([[0.0, 0.0, .138]], device=self.device)
        if self.robot_name == "franka_robotiq":
            # Diagnostic TCP at the centre between the open fingertip pads,
            # 140 mm along hand z (USD finger geometry spans z=91..148 mm).
            self.tcp_offset = torch.tensor([[0.0, 0.0, 0.14]], device=self.device)
        if self.robot_name == "fr3":
            from pxr import UsdGeom, Usd
            stage = self.sim.stage
            hand_prim = next(p for p in stage.Traverse() if p.GetName() == "fr3_hand")
            tcp_prim = next(p for p in stage.Traverse() if p.GetName() == "fr3_hand_tcp")
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            relative = cache.GetLocalToWorldTransform(tcp_prim) * cache.GetLocalToWorldTransform(hand_prim).GetInverse()
            self.tcp_offset = torch.tensor([list(relative.ExtractTranslation())], device=self.device)
            print("FR3 TCP offset:", self.tcp_offset, "joints:", self.robot.joint_names, flush=True)
        self.tool_quaternion = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.device)
        if self.robot_name == "piper":
            self.tool_quaternion = torch.tensor([[0., 2**-.5, 2**-.5, 0.]], device=self.device)
        if self.robot_name == "piper_dual":
            from .robot_profiles import PIPER_DUAL_TOOL_QUATERNIONS
            self.tool_quaternion = torch.tensor([PIPER_DUAL_TOOL_QUATERNIONS["left"]], device=self.device)
        self.controller = DifferentialIKController(DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=self.device)

    def _build_wrist_camera_mounts(self):
        """Author local camera children on every actual wrist, even without RGB recording."""
        import isaaclab.sim as sim_utils
        from isaaclab.utils.math import convert_camera_frame_orientation_convention
        from pxr import UsdGeom, UsdPhysics
        from .wrist_cameras import wrist_camera_specs
        roots = {'left': '/World/Robot'}
        if self.robot_name == 'piper_dual':
            roots['right'] = '/World/RobotRight'
        self.wrist_camera_mounts = wrist_camera_specs(self.robot_name, roots)
        self.wrist_cameras = {}
        if self.recording_config is not None:
            from isaaclab.sensors import Camera, CameraCfg
            simulation = self.sim

            class MountedCamera(Camera):
                def _update_poses(sensor, env_ids):
                    if not simulation.is_fabric_enabled():
                        return super()._update_poses(env_ids)
                    # Read the rendered child transform directly. XFormPrim's
                    # Fabric view lazily calls SetWorldXformFromUsd(), which
                    # overrides a moving child's mount with stale USD world
                    # coordinates. Reading the hierarchy does not create that
                    # world override, or derive camera pose from the parent.
                    import usdrt
                    hierarchy = sensor._view._get_fabric_hierarchy()
                    hierarchy.update_world_xforms()
                    positions, rotations = [], []
                    for index in env_ids.tolist():
                        matrix = hierarchy.get_world_xform(usdrt.Sdf.Path(sensor._view.prim_paths[index]))
                        quaternion = matrix.RemoveScaleShear().ExtractRotationQuat()
                        positions.append(list(matrix.ExtractTranslation()))
                        rotations.append([quaternion.GetReal(), *quaternion.GetImaginary()])
                    sensor._data.pos_w[env_ids] = sensor._data.pos_w.new_tensor(positions)
                    sensor._data.quat_w_world[env_ids] = convert_camera_frame_orientation_convention(
                        sensor._data.quat_w_world.new_tensor(rotations), origin='opengl', target='world')

        for spec in self.wrist_camera_mounts:
            parent = self.sim.stage.GetPrimAtPath(spec['parent_prim_path'])
            _check(parent.IsValid() and parent.HasAPI(UsdPhysics.RigidBodyAPI),
                   f"Missing actual rigid wrist camera parent: {spec['parent_prim_path']}")
            # Instance proxies cannot own an authored child. Only the selected
            # wrist hierarchy is expanded; mesh and collision data are unchanged.
            ancestor = parent
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.IsInstance():
                    ancestor.SetInstanceable(False)
                ancestor = ancestor.GetParent()
            parent = self.sim.stage.GetPrimAtPath(spec['parent_prim_path'])
            _check(not parent.IsInstanceProxy(), 'Wrist camera parent is still an instance proxy')
            _check(not self.sim.stage.GetPrimAtPath(spec['prim_path']).IsValid(),
                   'Wrist camera prim already exists; refusing to replace an unknown camera')
            width = self.recording_config['width'] if self.recording_config else 512
            height = self.recording_config['height'] if self.recording_config else 512
            cfg = sim_utils.PinholeCameraCfg(focal_length=spec['focal_length_mm'],
                horizontal_aperture=spec['horizontal_aperture_mm'],
                vertical_aperture=spec['horizontal_aperture_mm'] * height / width,
                clipping_range=tuple(spec['clipping_range_m']))
            rotation = self.torch.tensor([spec['quaternion_wxyz_ros']], dtype=self.torch.float32)
            opengl = convert_camera_frame_orientation_convention(rotation, origin='ros', target='opengl')[0].tolist()
            cfg.func(spec['prim_path'], cfg, translation=tuple(spec['position_m']), orientation=tuple(opengl))
            camera_prim = self.sim.stage.GetPrimAtPath(spec['prim_path'])
            _check(camera_prim.IsA(UsdGeom.Camera) and camera_prim.GetParent() == parent,
                   'Wrist camera was not mounted below the requested rigid body')
            if self.recording_config is not None:
                self.wrist_cameras[spec['name']] = MountedCamera(CameraCfg(prim_path=spec['prim_path'],
                    update_period=0.0, height=height, width=width, data_types=['rgb'], spawn=None,
                    update_latest_camera_pose=True))
        self.scene_provenance['wrist_camera_mounts'] = deepcopy(self.wrist_camera_mounts)

    def _wrist_camera_pose(self, spec, camera):
        data = camera.data
        parent = self.robots[spec['arm']].data.body_state_w[0, self.wrist_parent_body_ids[spec['name']], :7]
        return {'position_w_m': data.pos_w[0].cpu().tolist(),
                'quaternion_wxyz_ros': data.quat_w_ros[0].cpu().tolist(),
                'parent_position_w_m': parent[:3].cpu().tolist(),
                'parent_quaternion_wxyz': parent[3:7].cpu().tolist(),
                'mount_position_m': list(spec['position_m']),
                'mount_quaternion_wxyz_ros': list(spec['quaternion_wxyz_ros'])}

    def reset(self, *, resolved: ResolvedEpisode | None = None):
        _check(self.recorder is None, "Finish the previous video before resetting")
        if resolved is not None:
            _check(_fingerprint(resolved.scene) == self.scene_hash, "Cannot change physical scene without rebuilding backend")
            _check([o["asset"] for o in resolved.layout["objects"]] == self.asset_order,
                   "Cannot change physical object topology/assets without rebuilding backend")
            _check(supports_physics_task(resolved.task["task_id"]), "Unsupported physical task")
            validate_execution(resolved.episode, resolved.scene)
            _check(_fingerprint([o.get("physical_asset") for o in resolved.layout["objects"]]) == self.asset_signature,
                   "Cannot change physical asset descriptors without rebuilding")
            self.resolved = resolved
        torch = self.torch
        rng = random.Random(self.resolved.episode["seed"])
        jitter = self.resolved.layout["randomization"]["position_jitter_m"]
        self.ids = [o["id"] for o in self.resolved.layout["objects"]]
        self.object_index = {identity: index for index, identity in enumerate(self.ids)}
        self.targets = {}
        self.history = {}
        self.history_physics_ticks = {}
        self.result = None
        self.instance_id = "physx_" + uuid.uuid4().hex
        self.step_index = 0
        self.physics_step_index = 0
        self.trace = []
        self.control_trace = []
        self.dual_control_trace = []
        if self.robot_name == "piper_dual":
            self.active_arm = self.resolved.episode["physical_execution"]["active_arm"]
            self.robot = self.robots[self.active_arm]
            from .robot_profiles import PIPER_DUAL_TOOL_QUATERNIONS
            self.tool_quaternion = torch.tensor([PIPER_DUAL_TOOL_QUATERNIONS[self.active_arm]], device=self.device)
        self.robot_targets = {}
        for name, robot in self.robots.items():
            target = robot.data.default_joint_pos.clone()
            robot.write_joint_state_to_sim(target, torch.zeros_like(target))
            robot.reset()
            self.robot_targets[name] = target
        self.object_state_trace = []
        self.closed = False
        self.controller.reset()
        self._joint_target = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(self._joint_target, torch.zeros_like(self._joint_target))
        self.robot.reset()
        initial_positions = []
        self.reset_object_pose_writes = 0
        for index, obj in enumerate(self.resolved.layout["objects"]):
            if self.is_assembled:
                binding = self.assembled_bindings[index]
                pose = torch.tensor([[*binding["root_position_w_m"], *binding["root_orientation_wxyz"]]],
                                    dtype=torch.float32, device=self.device)
                self.bodies[index].write_root_pose_to_sim(pose)
                if self.assembled_dynamic[obj['id']]:
                    self.bodies[index].write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))
                self.bodies[index].reset()
                self.reset_object_pose_writes += 1
                initial_positions.append(pose[0, :3].cpu().tolist())
                continue
            local = [v + rng.uniform(-jitter, jitter) for v in obj["position_m"]]
            spec = self.asset_specs[index]
            half_height = -spec["bottom_offset_m"] if spec else self.profile["object_size_m"][2] / 2
            on_table = abs(local[2] - half_height) < 1e-6
            _check(on_table or local[2] >= half_height + 0.02,
                   "Object must rest on table or leave space for an elevated support")
            world = torch.tensor([local], device=self.device) + self.origin
            if self.supports[index] is not None:
                support_pose = self.supports[index].data.default_root_state[:, :7].clone()
                support_pose[:, :3] = world
                support_pose[:, 2] -= half_height + 0.01
                self.supports[index].write_root_pose_to_sim(support_pose)
            else:
                _check(on_table, 'Object without a helper must start on the real table')
            pose = self.bodies[index].data.default_root_state[:, :7].clone()
            pose[:, :3] = world
            pose[:, 2] += self.profile["settle_drop_height_m"]
            self.bodies[index].write_root_pose_to_sim(pose)
            self.bodies[index].write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))
            self.bodies[index].reset()
            initial_positions.append(pose[0, :3].cpu().tolist())
        self.initial_positions_w = initial_positions
        # Flush reset articulation/body transforms before the first contact
        # step, including when a second case reuses the existing simulation.
        # forward() synchronizes kinematics without advancing physics time.
        self.sim.forward()
        minimum=int(self.profile['settle_steps'])
        reset_stable_required=math.ceil(.5/self.dt) if self.is_assembled else 1
        maximum=max(minimum,math.ceil(5/self.dt)) if self.is_assembled else minimum
        stable=0;spins=[]
        for index in range(maximum):
            self._physics_tick(label=0)
            velocities=[float(torch.linalg.vector_norm(body.data.root_lin_vel_w).item()) for body in self.bodies]
            spins=[float(torch.linalg.vector_norm(body.data.root_ang_vel_w).item()) for body in self.bodies]
            good=max(velocities)<=self.profile['settle_velocity_tolerance_m_s'] and (not self.is_assembled or max(spins)<=.1)
            stable=stable+1 if good else 0
            if index+1>=minimum and (not self.is_assembled or stable>=reset_stable_required):break
        self.reset_settle_info={'minimum_steps':minimum,'maximum_steps':maximum,'actual_steps':index+1,
            'stable_steps':stable,'required_stable_steps':reset_stable_required,
            'linear_speed_m_s':dict(zip(self.ids,velocities)),'angular_speed_rad_s':dict(zip(self.ids,spins))}
        _check(good and (not self.is_assembled or stable>=reset_stable_required),f'Objects did not physically settle: {self.reset_settle_info}')
        self.settled_velocities_m_s = velocities
        self._sync_positions()
        self.initial_joint_positions = self.robot.data.joint_pos[0, self.arm_ids].cpu().tolist()
        if self.resolved.protocol is None:
            self.result = solve(self.resolved, self.snapshot())
        if self.resolved.episode["physical_execution"]["action"] == "PickPlaceMidpoint":
            self.result["instruction"] = "把绿色方块放到红蓝两个方块的正中间。"
        self._set_assembled_instruction()
        return self.observation(), {"backend": self.backend, "headless": True,
            "contract_hash": self.resolved.contract_hash, "settled_velocities_m_s": velocities,
            "settling":deepcopy(self.reset_settle_info),
            "reset_object_pose_writes": self.reset_object_pose_writes,
            "objects_respawned": False if self.is_assembled else None,
            "reset_pose_policy": "restore measured original rigid poses only during reset" if self.is_assembled else "authored layout and drop"}

    def _physics_tick(self, *, label: int):
        torch = self.torch
        for identity, index in self.object_index.items():
            body = self.bodies[index]
            if self.is_assembled and (not self.resolved.protocol or not self.assembled_dynamic[identity]):
                _check(not self.targets if not self.resolved.protocol else identity not in self.targets,
                       "Static assembled body cannot be driven by scene events")
                body.write_data_to_sim()
                continue
            force = torch.zeros((1, 1, 3), device=self.device)
            if identity in self.targets:
                target = self.targets[identity]
                error = target - body.data.root_pos_w
                force[:, 0, :] = (self.profile["event_stiffness_n_m"] * error
                                  - self.profile["event_damping_ns_m"] * body.data.root_lin_vel_w)
                mass = float(body.data.default_mass[0].sum().item())
                force[:, 0, 2] += mass * 9.81
                force = force.clamp(-5.0, 5.0)
            body.set_external_force_and_torque(force, torch.zeros_like(force), is_global=True)
            body.write_data_to_sim()
        self.robot.set_joint_position_target(self._joint_target)
        for name, robot in self.robots.items():
            if robot is not self.robot:
                robot.set_joint_position_target(self.robot_targets[name])
            robot.write_data_to_sim()
        self.sim.step(render=False)
        for robot in self.robots.values():
            robot.update(self.dt)
        for body in self.bodies:
            body.update(self.dt)
        self.physics_step_index += 1
        if self.physics_step_index==1 and self.cylinder_collision_mode is not None:
            import omni.physx
            measured=omni.physx.get_physxunittests_interface().get_physics_stats()
            self.scene_provenance['cylinder_collision_mode']['physics_shape_statistics_after_first_tick']=dict(measured)
        self.object_state_trace.append([self.physics_step_index, label, *torch.cat([body.data.root_state_w[0] for body in self.bodies]).cpu().tolist()])
        tcp, quaternion = self._tcp_pose()
        q = self.robot.data.joint_pos[0, self.arm_ids]
        positions = torch.cat([body.data.root_pos_w[0] for body in self.bodies])
        _check(bool(torch.isfinite(q).all() and torch.isfinite(tcp).all() and torch.isfinite(positions).all()),
               "Nonfinite physical state")
        self.trace.append([self.physics_step_index, self.step_index, label, *q.cpu().tolist(),
                           *tcp[0].cpu().tolist(), *positions.cpu().tolist()])
        self.control_trace.append([
            *self._joint_target[0].cpu().tolist(),
            *self.robot.data.joint_pos[0].cpu().tolist(),
            *self.robot.data.joint_vel[0].cpu().tolist(),
            *quaternion[0].cpu().tolist()])
        if self.robot_name == "piper_dual":
            row = [self.physics_step_index, label]
            for name, robot in self.robots.items():
                target = self._joint_target if robot is self.robot else self.robot_targets[name]
                row += [*target[0].cpu().tolist(), *robot.data.joint_pos[0].cpu().tolist(),
                        *robot.data.joint_vel[0].cpu().tolist()]
            self.dual_control_trace.append(row)
        if self.physics_step_index != self._video_deferred_tick:
            self._capture_video(stage=label)

    def start_recording(self, prefix):
        _check(self.camera is not None, "Camera rendering was not enabled at application startup")
        from .video import EpisodeVideo
        self.sim.render()
        self.camera.reset()
        self.camera.update(0.0, force_recompute=True)
        data = self.camera.data
        calibration = {"position_w_m": data.pos_w[0].cpu().tolist(),
                       "quaternion_wxyz_ros": data.quat_w_ros[0].cpu().tolist(),
                       "intrinsic_matrix": data.intrinsic_matrices[0].cpu().tolist()}
        self.recorder = EpisodeVideo(prefix, self.recording_config, calibration,
                                     start_tick=self.physics_step_index, dt=self.dt)
        from pathlib import Path
        from isaaclab.utils.math import quat_apply
        for spec in self.wrist_camera_mounts:
            camera = self.wrist_cameras[spec['name']]
            camera.reset()
            camera.update(0.0, force_recompute=True)
            data = camera.data
            position = data.pos_w[0].cpu().tolist()
            forward = quat_apply(data.quat_w_ros, self.torch.tensor([[0., 0., 1.]], device=self.device))[0].cpu().tolist()
            calibration = {'position_w_m': position,
                'quaternion_wxyz_ros': data.quat_w_ros[0].cpu().tolist(),
                'intrinsic_matrix': data.intrinsic_matrices[0].cpu().tolist(),
                'camera_name': spec['name'], 'parent_prim_path': spec['parent_prim_path'], 'mount': deepcopy(spec),
                'extrinsics_source': 'actual camera world transform read from Fabric hierarchy without a world override' if self.sim.is_fabric_enabled()
                                     else 'actual camera XFormPrim world transform from USD'}
            config = dict(self.recording_config, eye_w_m=position,
                          look_at_w_m=[p+v for p,v in zip(position,forward)], focal_length_mm=spec['focal_length_mm'])
            path = Path(prefix).with_name(Path(prefix).name + '_' + spec['name'])
            self.wrist_recorders[spec['name']] = EpisodeVideo(path, config, calibration,
                start_tick=self.physics_step_index, dt=self.dt)
        self._capture_video(stage=1 if self.resolved.protocol is not None else 2)

    def _capture_video(self, *, stage: int):
        if self.recorder is None:
            return
        recorder = self.recorder
        if (self.physics_step_index - recorder.start_tick) % recorder.stride:
            return
        self.sim.render()
        self.camera.update(1 / self.recording_config["fps"] if recorder.frame_count else 0.0, force_recompute=True)
        rgb = self.camera.data.output["rgb"][0, :, :, :3].contiguous().cpu().numpy()
        phase = self.phase()
        recorder.append(rgb, tick=self.physics_step_index, protocol_step=self.step_index,
                        phase=phase["id"], scene_visible=phase["scene_visible"], stage=stage,
                        sensor_frame=int(self.camera.frame[0].item()))
        for spec in self.wrist_camera_mounts:
            camera = self.wrist_cameras[spec['name']]
            wrist_recorder = self.wrist_recorders[spec['name']]
            camera.update(1 / self.recording_config['fps'] if wrist_recorder.frame_count else 0.0, force_recompute=True)
            wrist_rgb = camera.data.output['rgb'][0, :, :, :3].contiguous().cpu().numpy()
            wrist_recorder.append(wrist_rgb, tick=self.physics_step_index, protocol_step=self.step_index,
                phase=phase['id'], scene_visible=phase['scene_visible'], stage=stage,
                sensor_frame=int(camera.frame[0].item()), camera_pose=self._wrist_camera_pose(spec, camera))

    def finish_recording(self):
        if self.recorder is None:
            return None
        recorder, self.recorder = self.recorder, None
        wrists, self.wrist_recorders = self.wrist_recorders, {}
        errors, wrist_results = [], {}
        result = None
        try:
            result = recorder.close()
        except Exception as exc:
            errors.append(exc)
        for name, wrist in wrists.items():
            try:
                wrist_results[name] = wrist.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
        result['wrist_cameras'] = wrist_results
        result['all_views_share_physics_ticks'] = all(
            v['frame_count'] == result['frame_count'] and v['start_physics_tick'] == result['start_physics_tick']
            and v['last_physics_tick'] == result['last_physics_tick'] for v in result['wrist_cameras'].values())
        return result

    def _tcp_pose(self):
        from isaaclab.utils.math import combine_frame_transforms
        hand = self.robot.data.body_state_w[:, self.hand_id, :7]
        return combine_frame_transforms(hand[:, :3], hand[:, 3:7], self.tcp_offset)

    def _sync_positions(self):
        if self.is_assembled:
            from geomind.scene_assembly.audit import measured_bounds
            from geomind.scene_assembly.body_geometry import points_for
            self.positions = {}
            for identity, index in self.object_index.items():
                pose = self.bodies[index].data.root_state_w[0, :7].detach().cpu().numpy()
                low, high = measured_bounds(points_for(self.assembled_body_points, index), pose[:3], pose[3:7])
                self.positions[identity] = ((low + high) * .5).tolist()
            return
        self.positions = {identity: (self.bodies[index].data.root_pos_w[0] - self.origin[0]).cpu().tolist()
                          for identity, index in self.object_index.items()}

    def _set_assembled_instruction(self):
        if self.is_assembled and self.result is not None:
            from .resolver import is_registered_task
            if is_registered_task(self.resolved.task["task_id"]):
                from .registered_tasks import public_instruction
                self.result['instruction'] = public_instruction(self.resolved.episode['parameters']['legacy_task_spec'])
                return  # The original registered task supplies its own instruction.
            action = self.resolved.episode["physical_execution"]["action"]
            if action in {"PickLiftSelectedObject", "PickPlaceObject"}:
                from geomind.scene_assembly.task_binding import _instruction
                operation = {"type": "pick" if action == "PickLiftSelectedObject" else "place"}
                if action == "PickPlaceObject":
                    operation["moving_object"] = self.resolved.episode["physical_execution"]["moving_object"]
                labels = {obj["id"]: obj.get("label", obj["id"]) for obj in self.resolved.layout["objects"]}
                self.result["instruction"] = _instruction(self.resolved.task["task_id"],
                    self.resolved.episode["parameters"], operation, labels)

    def snapshot(self):
        self._sync_positions()
        return Snapshot.from_dict({"instance_id": self.instance_id, "contract_hash": self.resolved.contract_hash,
            "frame": self.resolved.layout["frame"], "step": self.step_index, "positions_m": deepcopy(self.positions)})

    def observation(self):
        self._sync_positions()
        self._set_assembled_instruction()
        if self.result is not None and self.resolved.episode["physical_execution"]["action"] == "PickLiftChangedObject":
            self.result["instruction"] = "记住桌上方块的位置，等画面恢复后，把刚才被移动的方块拿起来。"
        obs = super().observation()
        from .resolver import is_registered_task
        if is_registered_task(self.resolved.task['task_id']):
            from .registered_tasks import public_instruction,task_reference_view
            spec=self.resolved.episode['parameters']['legacy_task_spec']
            obs['task_reference']=task_reference_view(spec)
            obs['axes']={'x':'+x','y':'+y','z':'+z'}
            if self.result is None:
                obs['instruction'] = public_instruction(spec)
                if self.phase()['type'] == 'Delay' and self.phase()['scene_visible']:
                    obs['instruction'] += ' 请观察物体的连续移动，移动结束后再操作。'
        if self.resolved.episode["physical_execution"]["action"] == "PickLiftChangedObject":
            obs["instruction"] = "记住桌上方块的位置，等画面恢复后，把刚才被移动的方块拿起来。"
        tcp, _ = self._tcp_pose()
        obs["robot"] = {"joint_positions_rad": self.robot.data.joint_pos[0, self.arm_ids].cpu().tolist(),
                        "tcp_position_m": (tcp[0] - self.origin[0]).cpu().tolist()}
        if self.is_assembled:
            visible = {item["object_id"]: item for item in obs["objects"]}
            obs["objects"] = [visible[identity] for identity in self.assembled_public_order if identity in visible]
            obs.update(axes={"right": "+x", "forward": "+y", "up": "+z"}, units="m",
                       input_modality="structured_state_with_robot_proprioception_and_scene_image",
                       reference_point="GeometricCenter", reference_point_definition="WorldAxisAlignedBoundingBoxCenter",
                       input_contract={"structured_object_positions_available": True,
                                       "image_contains_object_id_labels": False, "visual_only": False})
        return obs

    def step(self, answer: dict | None = None):
        _check(not self.closed, "Call reset before stepping; episode may already be finished")
        if self.phase().get("allow_response", False):
            _check(answer is not None, "A response is required in response phase")
            semantic = evaluate_answer(self.resolved, self.result, answer)
            physical = self.execute_answer(answer)
            success = semantic["success"] and physical["success"]
            self.closed = True
            info = {"success": success, "semantic": semantic, "physical": physical, "backend": self.backend}
            return self.observation(), float(success), True, False, info
        _check(answer is None, "Responses are not allowed before retrieval")
        self._video_deferred_tick = self.physics_step_index + self.protocol_decimation
        for _ in range(self.protocol_decimation):
            self._physics_tick(label=1)
        self.step_index += 1
        observe, delay, _ = self.resolved.protocol["phases"]
        if self.step_index == observe["duration_steps"]:
            reference = f"{observe['id']}.End"
            self.history[reference] = self.snapshot()
            self.history_physics_ticks[reference] = self.physics_step_index
        for event in delay.get("events", []):
            if self.step_index == observe["duration_steps"] + event["after_steps"]:
                target = self.torch.tensor([event["position_m"]], device=self.device) + self.origin
                self.targets[event["object_id"]] = target
        if self.phase().get("allow_response", False):
            # Scene intervention forces end before freezing measured query GT or
            # beginning robot contact. They are never used to transport a grasp.
            if self.is_assembled:
                self.targets.clear()
                # Flush the protocol boundary that was deliberately deferred;
                # the final settling boundary is captured after GT is frozen.
                self._capture_video(stage=1)
                settle_ticks=math.ceil(1./self.dt)
                self._video_deferred_tick=self.physics_step_index+settle_ticks
                for _ in range(settle_ticks):
                    self._physics_tick(label=1)
                self._sync_positions()
                if self.resolved.task["task_id"] == "SelectByAdjacency_v0":
                    from geomind.scene_assembly.registered_export import _draw_edges
                    mapping = self.resolved.episode["parameters"]["object_id_map"]
                    objects = [{"object_id": mapping[alias], "position_w_m": pos}
                               for alias, pos in self.positions.items()]
                    from pxr import Usd
                    with Usd.EditContext(self.sim.stage, self.sim.stage.GetSessionLayer()):
                        _draw_edges(self.sim.stage, self.resolved.episode["parameters"]["legacy_task_spec"],
                                    objects, self.profile["tabletop_height_m"])
            self.result = solve(self.resolved, self.snapshot(), history=self.history)
        self._video_deferred_tick = None
        self._capture_video(stage=1)
        return self.observation(), 0.0, False, False, {}

    def _answer_point(self, answer: dict):
        if "object_id" in answer:
            _check(answer["object_id"] in self.object_index, "Physical response references unknown object")
            if self.is_assembled:
                self._sync_positions()
                return self.torch.tensor([self.positions[answer["object_id"]]], device=self.device)
            return self.bodies[self.object_index[answer["object_id"]]].data.root_pos_w.clone()
        _check(answer["frame"] == self.resolved.layout["frame"], "Physical answer frame mismatch")
        point = _position(answer["position_m"])
        bounds = self.resolved.scene["workspace"]["bounds_m"]
        _check(all(bounds[axis][0] <= point[i] <= bounds[axis][1] for i, axis in enumerate(("x", "y", "z"))),
               "Physical answer outside declared workspace")
        return self.torch.tensor([point], device=self.device) + self.origin

    def execute_answer(self, answer: dict) -> dict:
        if self.resolved.episode["physical_execution"]["action"] in {"PickLiftSelectedObject", "PickPlaceObject"}:
            from .assembled_actions import execute_assembled_action
            return execute_assembled_action(self, answer)
        if self.resolved.episode["physical_execution"]["action"] in {"PickPlaceMidpoint", "PickLiftChangedObject"}:
            from .placement import execute_midpoint_placement
            return execute_midpoint_placement(self, answer)
        from isaaclab.utils.math import quat_apply, subtract_frame_transforms
        torch = self.torch
        execution = self.resolved.episode["physical_execution"]
        motion_trace_start = len(self.trace) - 1
        offset = torch.tensor([execution["target_offset_m"]], device=self.device)
        command = self._answer_point(answer) + offset
        gt = self.result["ground_truth"]
        expected_answer = {"object_id": gt["object_id"]} if gt["type"] == "Object" else {
            "position_m": gt["position_m"], "frame": gt["frame"]}
        expected = self._answer_point(expected_answer) + offset
        tcp, _ = self._tcp_pose()
        clear_z = max(float(tcp[0, 2]), float(command[0, 2]) + execution["approach_clearance_m"])
        lift = tcp.clone(); lift[:, 2] = clear_z
        approach = command.clone(); approach[:, 2] = clear_z
        waypoints = [lift, approach, command]
        counts, total, final_dwell = [], 0, 0
        for stage, goal_tcp in enumerate(waypoints):
            hand_goal = goal_tcp - quat_apply(self.tool_quaternion, self.tcp_offset)
            root = self.robot.data.root_state_w[:, :7]
            position_b, quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], hand_goal, self.tool_quaternion)
            self.controller.set_command(torch.cat((position_b, quat_b), dim=-1))
            dwell, stage_steps = 0, 0
            required = execution["dwell_steps"] if stage == 2 else 3
            while total < execution["max_steps"]:
                hand = self.robot.data.body_state_w[:, self.hand_id, :7]
                position_b, quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], hand[:, :3], hand[:, 3:7])
                jacobian = self.robot.root_physx_view.get_jacobians()[:, self.hand_id - 1][:, :, self.arm_ids]
                q = self.robot.data.joint_pos[:, self.arm_ids]
                desired = self.controller.compute(position_b, quat_b, jacobian, q)
                desired = q + (desired - q).clamp(-execution["joint_step_limit_rad"], execution["joint_step_limit_rad"])
                limits = self.robot.data.soft_joint_pos_limits[:, self.arm_ids]
                desired = torch.maximum(torch.minimum(desired, limits[:, :, 1]), limits[:, :, 0])
                self._joint_target[:, self.arm_ids] = desired
                self._physics_tick(label=2 + stage)
                total += 1; stage_steps += 1
                actual, orientation = self._tcp_pose()
                distance = float(torch.linalg.vector_norm(actual - goal_tcp).item())
                dot = float((orientation * self.tool_quaternion).sum().abs().clamp(0, 1).item())
                angle = 2 * math.acos(dot)
                within = distance <= execution["position_tolerance_m"] and angle <= execution["orientation_tolerance_rad"]
                dwell = dwell + 1 if within else 0
                if dwell >= required:
                    break
            counts.append(stage_steps)
            if dwell < required:
                break
            if stage == 2:
                final_dwell = dwell
        actual, orientation = self._tcp_pose()
        # Freeze the query-time measured target; never move GT to fit execution drift.
        error = float(torch.linalg.vector_norm(actual - expected).item())
        angle = 2 * math.acos(float((orientation * self.tool_quaternion).sum().abs().clamp(0, 1).item()))
        q_final = self.robot.data.joint_pos[0, self.arm_ids].cpu().tolist()
        moved = max(abs(a - b) for a, b in zip(q_final, self.initial_joint_positions))
        rows = self.trace[motion_trace_start:]
        measured_joint_step = max((abs(b[j] - a[j]) for a, b in zip(rows, rows[1:]) for j in range(3, 3 + len(self.arm_ids))), default=0.0)
        success = (final_dwell >= execution["dwell_steps"] and error <= execution["position_tolerance_m"]
                   and angle <= execution["orientation_tolerance_rad"])
        return {"success": success, "action": execution["action"], "position_error_m": error,
                "position_tolerance_m": execution["position_tolerance_m"], "orientation_error_rad": angle,
                "orientation_tolerance_rad": execution["orientation_tolerance_rad"], "dwell_steps": final_dwell,
                "required_dwell_steps": execution["dwell_steps"], "motion_steps": total, "waypoint_steps": counts,
                "target_tcp_w_m": expected[0].cpu().tolist(),
                "target_policy": "query_time_frozen",
                "planned_waypoints_w_m": [p[0].cpu().tolist() for p in waypoints],
                "termination_reason": "dwell_satisfied" if success else "motion_budget_exhausted",
                "collision_check": "PhysX colliders enabled; contact-free motion not certified", "actual_tcp_w_m": actual[0].cpu().tolist(),
                "max_joint_displacement_rad": moved,
                "max_measured_joint_step_rad": measured_joint_step,
                "joint_state_writes_during_motion": 0, "object_pose_writes_after_reset": 0}

    def export_evidence(self):
        evidence = super().export_evidence()
        evidence["backend"] = self.backend
        evidence["physical_clock"] = {"dt_s": self.dt, "current_tick": self.physics_step_index,
                                      "history_ticks": dict(self.history_physics_ticks)}
        return evidence

    def close(self):
        self.finish_recording()
        self.closed = True
        self.sim.stop()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
