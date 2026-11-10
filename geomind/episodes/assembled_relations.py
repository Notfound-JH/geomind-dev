"""Measured adapters of the registered OnTop/Inside termination contracts.

The inequalities match mdp/terminations.py; all inputs are real rigid poses and
body geometry. Contact follows the original gap/overlap predicate, without
claiming an independent contact sensor measurement.
"""
from __future__ import annotations
import numpy as np
from geomind.scene_assembly.body_geometry import points_for


def rotation(q):
    w,x,y,z=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def measure_relation(contract, reference_index, target_index, points, states, centres, low, high, *, released):
    """Return the original relation predicate and its measured support height."""
    r,t=reference_index,target_index
    spec=contract;kind=spec['resolver']
    stable=(np.linalg.norm(states[t,7:10])<=spec.get('max_linear_speed_after_settle',.05)
            and np.linalg.norm(states[t,10:13])<=spec.get('max_angular_speed_after_settle',.4))
    result={'resolver':kind,'released':bool(released),'stable':bool(stable)}
    valid=stable and (released or not spec.get('must_be_released',False))
    if kind=='support_top_placement':
        overlap=np.maximum(0.,np.minimum(high[t,:2],high[r,:2])-np.maximum(low[t,:2],low[r,:2]))
        ratio=float(np.prod(overlap)/max(float(np.prod(high[t,:2]-low[t,:2])),1e-12))
        gap=abs(float(low[t,2]-high[r,2]))
        contact=gap<=spec.get('max_support_gap',.015) and ratio>=spec.get('min_support_projection_overlap_ratio',.72)
        above=centres[t,2]>high[r,2]+.25*(high[t,2]-low[t,2])
        if spec.get('contact_required',False):valid=valid and contact
        if spec.get('support_required',False):valid=valid and contact and above
        result.update(gap_m=gap,overlap_ratio=ratio,support_height_w_m=float(high[r,2]))
    elif kind=='containment_placement':
        reference_rotation=rotation(states[r,3:7]);target_rotation=rotation(states[t,3:7])
        ref_points=np.asarray(points_for(points,r));target_points=np.asarray(points_for(points,t))
        ref_local=(ref_points.min(axis=0)+ref_points.max(axis=0))*.5
        target_local=(target_points.min(axis=0)+target_points.max(axis=0))*.5
        ref_centre=states[r,:3]+reference_rotation@ref_local
        target_centre=states[t,:3]+target_rotation@target_local
        relative=reference_rotation.T@(target_centre-ref_centre)
        half=(target_points.max(axis=0)-target_points.min(axis=0))*.5
        projected=np.abs(reference_rotation.T@target_rotation)@half
        available=np.asarray(spec['interior_half_extents_xy'])-projected[:2]-spec.get('xy_inner_margin',0.)
        inside=bool(np.all(available>=0) and np.all(np.abs(relative[:2])<=available+1e-6))
        gap=float(relative[2]-projected[2]-spec.get('interior_floor_top_offset',0.))
        clearance=float(spec.get('opening_plane_z_offset',0.)-relative[2]-projected[2])
        valid=valid and inside and gap<=spec.get('max_floor_gap',.02) and gap>=-spec.get('max_floor_penetration',.004)
        valid=valid and clearance>=spec.get('min_top_clearance_below_opening',0.)
        support=ref_centre+reference_rotation@np.array([0.,0.,spec.get('interior_floor_top_offset',0.)])
        result.update(inside=inside,floor_gap_m=gap,top_clearance_m=clearance,support_height_w_m=float(support[2]))
    else:raise ValueError('Unsupported original relation resolver')
    result['passed']=bool(valid)
    return result
