"""Bounded local settling validation of an initial assembled layout, without a robot."""
from __future__ import annotations
from geomind_lab.utils.portable_paths import configure_material_search_path as _portable_material_paths

from pathlib import Path
import hashlib
import json
import os

import numpy as np

from .assets import ROOT
from .body_geometry import load_body_clouds, points_for


def mesh_bounds_from_rigid_pose(points, position, quaternion):
    """Normalize only the geometry quaternion; retain raw state in its archive."""
    from pxr import Gf
    q=np.asarray(quaternion,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-10:
        raise ValueError('Invalid rigid-pose geometry quaternion')
    q=q/np.linalg.norm(q)
    rotation=np.asarray(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(float(q[0]),Gf.Vec3d(*map(float,q[1:]))))))
    transformed=np.asarray(points)@rotation+np.asarray(position)
    return transformed.min(axis=0),transformed.max(axis=0)


def verify(output,token):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    body_points=load_body_clouds(manifest)
    _portable_material_paths()
    from geomind.core.envs.isaaclab_native_delegate import _IsaacLabAppRuntime
    _IsaacLabAppRuntime.ensure_started(headless=True,enable_cameras=True)
    app=_IsaacLabAppRuntime._simulation_app
    import carb
    import omni.usd
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Gf,UsdGeom,UsdLux,Usd
    from PIL import Image
    import imageio.v2 as imageio
    from .usd import measure_table,supported_xy

    settings=carb.settings.get_settings()
    settings.set('/renderer/mdl/searchPaths/system',os.environ['MDL_SYSTEM_PATH'])
    settings.set('/rtx/post/histogram/enabled',False)
    context=omni.usd.get_context()
    context.open_stage(manifest['scene_usda'])
    stage=context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
    support_surface=measure_table(stage)
    from .visibility import apply_scene_visibility
    visualization = apply_scene_visibility(stage, manifest)
    table=manifest['table'];bounds=table['bounds_xy_m'];z=table['height_m']
    cx=(bounds[0]+bounds[1])*.5;cy=(bounds[2]+bounds[3])*.5
    width=max(bounds[1]-bounds[0],bounds[3]-bounds[2],1.0)
    eye=[cx+width*.85,cy-width*.85,z+width*1.10];look=[cx,cy,z-.08]
    camera=UsdGeom.Camera.Define(stage,'/AssemblyPreviewCamera')
    camera.CreateFocalLengthAttr(24);camera.CreateHorizontalApertureAttr(30)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
    UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*look),Gf.Vec3d(0,0,1)).GetInverse())
    light=UsdLux.DomeLight.Define(stage,'/AssemblyPreviewFill');light.CreateIntensityAttr(650)
    product=rep.create.render_product(str(camera.GetPath()),(768,768))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
    world=World(physics_dt=1/120,rendering_dt=1/120,stage_units_in_meters=1.0,device='cpu')
    bodies=[]
    for i,obj in enumerate(manifest['objects']):
        body=SingleRigidPrim(prim_path=obj['prim_path'],name=f'assembly_object_{i:03d}',reset_xform_properties=False)
        world.scene.add(body);bodies.append(body)
        actual_scale=np.asarray(body.get_local_scale(),dtype=float)
        if not np.allclose(actual_scale,obj.get('scale',manifest['asset']['scale']),atol=1e-7):
            raise RuntimeError('Physics initialization changed the requested asset scale')
    world.reset()
    # Renderer compilation can be slow; take explicit measured steps throughout.
    for _ in range(20):world.step(render=True)
    video_path=output/'settling.mp4'
    writer=imageio.get_writer(str(video_path),fps=30,codec='libx264',quality=8,macro_block_size=1,ffmpeg_log_level='error')
    rows=[];stable=0;frames=0;last_image=None;max_xy=0.0;peak_xy=0.0;failure=None
    per_object=[]
    try:
        for tick in range(600):
            world.step(render=tick%4==0)
            states=[];passed=True;metrics=[]
            for index,(body,expected) in enumerate(zip(bodies,manifest['objects'])):
                pos,quat=body.get_world_pose()
                linear=body.get_linear_velocity();angular=body.get_angular_velocity()
                pos=np.asarray(pos,dtype=float);quat=np.asarray(quat,dtype=float)
                linear=np.asarray(linear,dtype=float);angular=np.asarray(angular,dtype=float)
                states.append(np.r_[pos,quat,linear,angular])
                xy_error=float(np.linalg.norm(pos[:2]-np.asarray(expected['position_w_m'][:2])))
                low,high=mesh_bounds_from_rigid_pose(points_for(body_points,index),pos,quat)
                centre=(low+high)*.5
                height_error=abs(float(low[2])-z)
                support_ok=supported_xy(support_surface,centre[:2],(high[:2]-low[:2])*.5)
                orientation_error=float(2*np.arccos(np.clip(abs(np.dot(quat,expected['orientation_wxyz'])),0,1)))
                speed=float(np.linalg.norm(linear));spin=float(np.linalg.norm(angular))
                good=xy_error<=.012 and height_error<=.006 and orientation_error<=.12 and speed<=.01 and spin<=.1 and support_ok
                passed=passed and good
                metrics.append({'prim_path':expected['prim_path'],'xy_error_m':xy_error,'support_height_error_m':height_error,
                                'orientation_error_rad':orientation_error,'linear_speed_m_s':speed,'angular_speed_rad_s':spin,'passed':good,
                                'position_w_m':pos.tolist(),'orientation_wxyz':quat.tolist(),
                                'measured_bbox_min_w_m':low.tolist(),'measured_bbox_max_w_m':high.tolist(),
                                'footprint_support_samples_passed':support_ok})
            rows.append(states)
            stable=stable+1 if passed else 0
            per_object=metrics
            max_xy=max(m['xy_error_m'] for m in metrics)
            peak_xy=max(peak_xy,max_xy)
            if tick%4==0:
                image=np.asarray(rgb.get_data())
                if image.ndim==3 and image.shape[:2]==(768,768):
                    last_image=image[:,:,:3].astype(np.uint8)
                    writer.append_data(last_image);frames+=1
        if last_image is None or last_image.std()<2:
            raise RuntimeError('No usable renderer image returned')
        Image.fromarray(last_image).save(output/'preview.png')
        world.pause()
        camera_transform=UsdGeom.Xformable(camera).GetOrderedXformOps()[0]
        camera_transform.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(cx+.32,cy-.32,z+.55),Gf.Vec3d(cx,cy,z+.025),Gf.Vec3d(0,0,1)).GetInverse())
        for _ in range(20):app.update()
        close_image=np.asarray(rgb.get_data())[:,:,:3].astype(np.uint8)
        if close_image.shape!=(768,768,3) or close_image.std()<2:
            raise RuntimeError('No usable detail renderer image returned')
        Image.fromarray(close_image).save(output/'detail_preview.png')
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
        import traceback;traceback.print_exc()
    finally:
        writer.close()
    trace=np.asarray(rows,dtype=np.float64)
    np.savez_compressed(output/'object_states.npz',data=trace,dt_s=1/120,
        columns=np.asarray(['x','y','z','qw','qx','qy','qz','vx','vy','vz','wx','wy','wz']),
        prim_paths=np.asarray([o['prim_path'] for o in manifest['objects']]))
    passed=bool(failure is None and stable>=60 and len(rows)==600 and frames==150)
    report={'executed':True,'passed':passed,'run_token':token,'backend':'isaacsim_physx','kind':'initial_layout_settling',
        'visualization':visualization,
        'physics_steps':600,'warmup_steps':20,'dt_s':1/120,'stable_steps':stable,'required_stable_steps':60,
        'position_error_max_m':max_xy,'object_count':len(bodies),'objects':per_object,'failure':failure,
        'peak_position_error_during_settling_m':peak_xy,
        'preview':str(output/'preview.png') if (output/'preview.png').exists() else None,
        'detail_preview':str(output/'detail_preview.png') if (output/'detail_preview.png').exists() else None,
        'video':str(video_path),'video_frames':frames,'video_fps':30,'state_trace':str(output/'object_states.npz'),
        'object_pose_writes_after_reset':0,'robot_actions_executed':False,
        'notes':'Initial scene settling only. Objects are real dynamic rigid bodies; background furniture is kinematic. Support height, XY layout, orientation and velocity are measured. No robot action or contact-free certification.'}
    report['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [output/'preview.png',output/'detail_preview.png',video_path,output/'object_states.npz',Path(manifest['asset_body_vertices'])] if p.exists()}
    (output/'physics_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('ASSEMBLY_PHYSICS_DONE',passed,stable,flush=True)
    rgb.detach([product.path]);product.destroy()
    app.close(wait_for_replicator=False)
    return report
