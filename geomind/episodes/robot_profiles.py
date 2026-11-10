"""Robot bindings for the native target-reaching backend (lazy simulator imports)."""
from geomind_lab.utils.portable_paths import isaac_assets_root, repo_assets_root

def robot_profile(name):
    if name == 'franka':
        from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
        cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
        from pathlib import Path
        local = isaac_assets_root()/'Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd'
        if local.is_file():
            cfg.spawn.usd_path = str(local)
        return cfg, 'panda_joint[1-7]', 'panda_hand', 'panda_finger_joint.*'
    if name == 'franka_robotiq':
        from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG, FRANKA_PANDA_HIGH_PD_CFG
        cfg = FRANKA_ROBOTIQ_GRIPPER_CFG.copy()
        cfg.spawn.usd_path = str(isaac_assets_root()/'Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd')
        cfg.spawn.func = spawn_robotiq_with_mount_filter
        # Resolve velocity residuals in the closed linkage/contact constraints.
        # Keep the asset, actuators, timestep and task thresholds unchanged.
        cfg.spawn.articulation_props.solver_position_iteration_count = 32
        cfg.spawn.articulation_props.solver_velocity_iteration_count = 32
        for group in ('panda_shoulder', 'panda_forearm'):
            cfg.actuators[group] = FRANKA_PANDA_HIGH_PD_CFG.actuators[group].copy()
        return cfg, 'panda_joint[1-7]', 'panda_hand', 'finger_joint'
    if name in {'piper', 'piper_dual'}:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from pathlib import Path
        usd = repo_assets_root() / 'robots/piper_prepared/usd/piper.usd'
        cfg = ArticulationCfg(spawn=sim_utils.UsdFileCfg(usd_path=str(usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=12, solver_velocity_iteration_count=2)),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={
                **dict(zip([f'joint{i}' for i in range(1,7)], [-3.0814451387220895e-08, 1.8639251171486537, -1.2862463950913159, -6.047024535421171e-08, 1.0803840779425815, 1.5707963744238749])),
                'gripper': .08, 'gripper_joint1': .04, 'gripper_joint2': -.04}),
            actuators={'arm': ImplicitActuatorCfg(joint_names_expr=['joint[1-6]'],
                stiffness=400., damping=40., effort_limit_sim=100., velocity_limit_sim=1.),
                'hand': ImplicitActuatorCfg(joint_names_expr=['gripper.*'],
                stiffness=1000., damping=40., effort_limit_sim=10., velocity_limit_sim=.1)})
        if name == 'piper_dual':
            cfg.init_state.joint_pos.update(dict(zip([f'joint{i}' for i in range(1,7)], [-0.3135353519959186, 0.9004803543146747, -0.6924313372435882, 1.189619157001312, 0.761433940766165, -1.3735119602089483])))
        return cfg, 'joint[1-6]', 'gripper_base', 'gripper'
    if name == 'fr3':
        from isaaclab_assets.robots.fr3 import FRANKA_FR3_HIGH_PD_CFG
        cfg = FRANKA_FR3_HIGH_PD_CFG.copy()
        cfg.spawn.usd_path = str(isaac_assets_root()/'Assets/Isaac/4.5/Isaac/Robots/Franka/FR3/fr3.usd')
        # Match the Panda comparison home pose; all seven angles are inside
        # the FR3 USD joint limits. Keep the FR3 asset and actuator settings.
        from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
        cfg.init_state.joint_pos = {
            key.replace('panda_', 'fr3_'): value
            for key, value in FRANKA_PANDA_HIGH_PD_CFG.init_state.joint_pos.items()
        }
        return cfg, 'fr3_joint[1-7]', 'fr3_hand', 'fr3_finger_joint.*'
    raise ValueError(f'Unsupported robot: {name}')


def filter_robotiq_mount(stage, root_path):
    """Filter fixed mount and two housing/linkage overlaps; preserve other contacts."""
    from pxr import UsdPhysics
    mount = stage.GetPrimAtPath(root_path + '/Robotiq_2F_85_edit/Robotiq_2F_85/base_link')
    wrist = stage.GetPrimAtPath(root_path + '/panda_link7')
    if not mount.IsValid() or not wrist.IsValid():
        raise ValueError('Robotiq mount topology changed; refusing an unfiltered spawn')
    UsdPhysics.FilteredPairsAPI.Apply(mount).CreateFilteredPairsRel().AddTarget(wrist.GetPath())
    # The closed linkage overlaps the housing in the authored open pose.
    gripper = root_path + '/Robotiq_2F_85_edit/Robotiq_2F_85/'
    for name in ('left_inner_knuckle', 'right_inner_knuckle'):
        link = stage.GetPrimAtPath(gripper + name)
        if not link.IsValid():
            raise ValueError('Robotiq linkage topology changed')
        UsdPhysics.FilteredPairsAPI.Apply(mount).CreateFilteredPairsRel().AddTarget(link.GetPath())


def spawn_robotiq_with_mount_filter(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    filter_robotiq_mount(prim.GetStage(), str(prim.GetPath()))
    return prim


PIPER_DUAL_RIGHT_HOME = [0.3135353519959186, 0.9004803543146747, -0.6924313372435882, -1.189619157001312, 0.761433940766165, 1.3735119602089483]

# Mirrored inward yaw, tool approach pitched 20 degrees below horizontal.
PIPER_DUAL_TOOL_QUATERNIONS = {'left': [0.559980384587607, -0.17729695222251712, 0.7997348700634734, 0.12414466244781326], 'right': [0.559980384587607, 0.17729695222251712, 0.7997348700634734, -0.12414466244781326]}
