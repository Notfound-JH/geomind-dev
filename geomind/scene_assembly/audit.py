"""Offline recomputation of measured initial-layout settling, without a simulator."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
from .body_geometry import load_body_clouds, points_for


def measured_bounds(points,position,quaternion):
    """Quaternion-vector expression, independent of the runtime's Gf matrix."""
    q=np.asarray(quaternion,dtype=float)
    q=q/np.linalg.norm(q)
    w=q[0];v=q[1:]
    transformed=(w*w-np.dot(v,v))*points+2*(points@v)[:,None]*v+2*w*np.cross(v,points)+position
    return transformed.min(axis=0),transformed.max(axis=0)


def grid_dimensions_match(centres, rows, cols, *, band_span_m=1e-6):
    """Verify ordered row/column bands without requiring perfectly equal cells.

    Uses the request's row-major order, independent of production clustering.
    Swapped cells, overlapping row bands and excessive wobble are rejected.
    """
    centres=np.asarray(centres,dtype=float)
    if centres.shape!=(rows*cols,2) or not np.isfinite(centres).all():return False
    grid=centres.reshape(rows,cols,2)
    row_low=grid[:,:,0].min(axis=1);row_high=grid[:,:,0].max(axis=1)
    col_low=grid[:,:,1].min(axis=0);col_high=grid[:,:,1].max(axis=0)
    return bool(np.all(row_high-row_low<=band_span_m+1e-8) and
                np.all(col_high-col_low<=band_span_m+1e-8) and
                np.all(row_low[1:]>row_high[:-1]+1e-6) and
                np.all(col_low[1:]>col_high[:-1]+1e-6))


def score_window(states,manifest,points):
    expected=manifest['objects'];table=manifest['table'];checks=[]
    for row in states:
        good=True
        for index,(actual,goal) in enumerate(zip(row,expected)):
            low,high=measured_bounds(points_for(points,index),actual[:3],actual[3:7])
            xy=np.linalg.norm(actual[:2]-np.asarray(goal['position_w_m'][:2]))
            angle=2*np.arccos(np.clip(abs(np.dot(actual[3:7],goal['orientation_wxyz'])),0,1))
            linear=np.linalg.norm(actual[7:10]);angular=np.linalg.norm(actual[10:13])
            bounds=table['bounds_xy_m']
            within=low[0]>=bounds[0] and high[0]<=bounds[1] and low[1]>=bounds[2] and high[1]<=bounds[3]
            good=bool(good and xy<=.012 and abs(low[2]-table['height_m'])<=.006 and angle<=.12 and linear<=.01 and angular<=.1 and within)
        checks.append(good)
    dwell=0
    for good in reversed(checks):
        if not good:break
        dwell+=1
    return dwell


def audit(output):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    report=json.loads((output/'physics_report.json').read_text())
    states=np.load(output/'object_states.npz',allow_pickle=False)['data']
    cloud_manifest=dict(manifest,asset_body_vertices=str(output/'asset_body_vertices.npz'))
    points=load_body_clouds(cloud_manifest)
    count=manifest['request']['object']['count']
    if states.shape!=(600,count,13) or not np.isfinite(states).all():
        raise ValueError('Unexpected measured-state trace schema')
    dwell=score_window(states,manifest,points)
    if dwell!=report['stable_steps']:
        raise ValueError(f'Offline dwell disagrees: {dwell} vs {report["stable_steps"]}')
    # This is an offline injected wrong-layout control, not physical execution.
    wrong=states.copy();wrong[:,0,0]+=.05
    wrong_dwell=score_window(wrong,manifest,points)
    if wrong_dwell!=0:raise ValueError('Offline scorer accepted a misplaced object')
    expected_centres=np.asarray([o['bbox_center_xy_m'] for o in manifest['objects']])
    layout=manifest['request']['layout']
    grid_check=None
    measured_grid_check=None
    if layout['type']=='grid':
        randomized=manifest.get('layout_randomization',{}).get('enabled',False)
        band=.01 if randomized else 1e-6
        grid_check=grid_dimensions_match(expected_centres,layout['rows'],layout['cols'],band_span_m=band)
        if not grid_check:raise ValueError('Generated grid dimensions disagree with the explicit request')
        final_centres=[]
        for index,actual in enumerate(states[-1]):
            lo,hi=measured_bounds(points_for(points,index),actual[:3],actual[3:7]);final_centres.append(((lo+hi)*.5)[:2])
        measured_grid_check=grid_dimensions_match(final_centres,layout['rows'],layout['cols'],band_span_m=band+.024)
        if not measured_grid_check:raise ValueError('Measured settled layout lost the requested grid structure')
    files_verified=[]
    for name,expected_hash in report['sha256'].items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest()!=expected_hash:
            raise ValueError(f'Artifact changed: {name}')
        files_verified.append(name)
    if hashlib.sha256(Path(manifest['scene_usda']).read_bytes()).hexdigest()!=manifest['provenance']['scene_usda_sha256']:
        raise ValueError('Saved assembly changed after validation')
    for path,expected_hash in manifest['provenance'].get('used_usd_layers_sha256',{}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected_hash:
            raise ValueError(f'Assembly USD layer changed: {path}')
    result={'passed':bool(report['passed'] and dwell>=60),'mode':'offline recomputation, no physical replay',
        'measured_rows':len(states),'object_count':count,'stable_steps_recomputed':dwell,
        'grid_dimensions_independently_checked':grid_check,'offline_wrong_layout_control_rejected':wrong_dwell==0,
        'measured_grid_dimensions_independently_checked':measured_grid_check,
        'wrong_control_physics_executed':False,'artifact_hashes_verified':files_verified,
        'limitations':'Offline support check uses tabletop bounding bounds; exact mesh support samples were checked in the physics worker. Not a contact-free certificate.'}
    (output/'offline_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
