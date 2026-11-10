"""
track_clutter-only motion planning: smooth XY paths + separation / crossing resolution.

Designed for batched Isaac Lab envs (fixed-size keyframe tensors).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from geomind_lab.mimic.envs.state_reindexing_logic import (
    _gather_rank_indices,
    _resolve_target_rank_position,
)

TRACK_CLUTTER_MAX_KEYFRAMES = 48


def resolve_track_clutter_protocol_grasp_object_index(
    *,
    initial_rank_indices: torch.Tensor,
    active_object_mask: torch.Tensor,
    target_rank: Any,
) -> torch.Tensor:
    """Map protocol ``target_rank`` + pre-merge rank layout to simulator object indices (-1 if unknown)."""
    if active_object_mask.ndim != 2 or initial_rank_indices.ndim != 2:
        raise ValueError("active_object_mask and initial_rank_indices must be 2D")
    device = active_object_mask.device
    num_envs = int(active_object_mask.shape[0])
    out = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    active_counts = active_object_mask.sum(dim=1).to(dtype=torch.long)
    rank_positions, supported = _resolve_target_rank_position(target_rank, active_counts)
    gathered = _gather_rank_indices(initial_rank_indices, rank_positions)
    ok = supported & (gathered >= 0)
    out = torch.where(ok, gathered, out)
    return out

# Subdivide piecewise-linear paths so no edge exceeds this length before keyframe packing.
_MAX_POLYLINE_EDGE_M = 0.024


def _interp_polyline(p_nodes: np.ndarray, xy: np.ndarray, u: float) -> np.ndarray:
    u = float(np.clip(u, 0.0, 1.0))
    if p_nodes.size <= 1:
        return xy[0].copy()
    idx = int(np.searchsorted(p_nodes, u, side="right") - 1)
    idx = int(np.clip(idx, 0, len(p_nodes) - 2))
    p0, p1 = float(p_nodes[idx]), float(p_nodes[idx + 1])
    denom = max(p1 - p0, 1e-9)
    t = (u - p0) / denom
    return (1.0 - t) * xy[idx] + t * xy[idx + 1]


def _arc_length_params(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xy.shape[0] <= 1:
        return np.array([0.0, 1.0]), xy
    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = max(float(cum[-1]), 1e-9)
    return cum / total, xy


def _chaikin_open(xy: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Corner cutting for smoother polylines (endpoints preserved)."""
    pts = np.asarray(xy, dtype=np.float64)
    for _ in range(iterations):
        n = pts.shape[0]
        if n < 2:
            break
        new_pts = [pts[0]]
        for i in range(n - 1):
            p, q = pts[i], pts[i + 1]
            new_pts.append(0.75 * p + 0.25 * q)
            new_pts.append(0.25 * p + 0.75 * q)
        new_pts.append(pts[-1])
        pts = np.stack(new_pts, axis=0)
    return pts.astype(np.float64)


def _insert_plateau(p_nodes: np.ndarray, xy: np.ndarray, u_center: float, hold_dp: float) -> tuple[np.ndarray, np.ndarray]:
    hold_dp = float(max(hold_dp, 1e-5))
    u_center = float(np.clip(u_center, hold_dp + 1e-6, 1.0 - hold_dp - 1e-6))
    pos = _interp_polyline(p_nodes, xy, u_center)
    u_end = u_center + hold_dp
    p2 = p_nodes.copy()
    xy2 = xy.copy()
    mask = p2 > u_center
    p2[mask] += hold_dp
    p_all = np.concatenate([[u_center, u_end], p2])
    xy_all = np.concatenate([[pos], [pos], xy2])
    order = np.argsort(p_all)
    return p_all[order], xy_all[order]


def _renormalize_path(p_nodes: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if p_nodes.size == 0:
        return p_nodes, xy
    lo, hi = float(p_nodes.min()), float(p_nodes.max())
    span = max(hi - lo, 1e-9)
    return (p_nodes - lo) / span, xy


def _resample_uniform(p_nodes: np.ndarray, xy: np.ndarray, k_out: int) -> tuple[np.ndarray, np.ndarray]:
    if k_out <= 1:
        return np.array([0.0]), xy[:1]
    us = np.linspace(0.0, 1.0, k_out)
    out_xy = np.stack([_interp_polyline(p_nodes, xy, float(u)) for u in us])
    return us, out_xy


def _densify_polyline_max_edge(xy: np.ndarray, max_edge: float) -> np.ndarray:
    """Insert vertices along long edges so piecewise-linear sampling never skips large chords."""
    if xy.shape[0] < 2:
        return xy
    pts: list[np.ndarray] = [xy[0].astype(np.float64)]
    max_edge = float(max(max_edge, 1e-6))
    for i in range(xy.shape[0] - 1):
        a = xy[i].astype(np.float64)
        b = xy[i + 1].astype(np.float64)
        seg = b - a
        ln = float(np.linalg.norm(seg))
        if ln < 1e-12:
            pts.append(b.copy())
            continue
        n_seg = max(1, int(np.ceil(ln / max_edge)))
        if n_seg <= 1:
            pts.append(b.copy())
            continue
        for j in range(1, n_seg):
            t = float(j) / float(n_seg)
            pts.append((1.0 - t) * a + t * b)
        pts.append(b.copy())
    return np.stack(pts, axis=0)


def _segment_intersection(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> tuple[float, float] | None:
    """Return (t,s) on segments a0+t*(a1-a0), b0+s*(b1-b0) if proper intersection in (0,1)^2 interior-ish."""
    d = (a1 - a0).astype(np.float64)
    e = (b1 - b0).astype(np.float64)
    denom = d[0] * e[1] - d[1] * e[0]
    if abs(denom) < 1e-12:
        return None
    r = (b0 - a0).astype(np.float64)
    t = (r[0] * e[1] - r[1] * e[0]) / denom
    s = (r[0] * d[1] - r[1] * d[0]) / denom
    eps = 1e-7
    if -eps <= t <= 1.0 + eps and -eps <= s <= 1.0 + eps:
        t = float(np.clip(t, 0.0, 1.0))
        s = float(np.clip(s, 0.0, 1.0))
        return t, s
    return None


def _point_to_polyline_u(xy: np.ndarray, pt: np.ndarray) -> float:
    """Closest-point projection → global arc-length fraction u in [0,1]."""
    if xy.shape[0] <= 1:
        return 0.0
    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total = float(np.sum(seg_len))
    if total < 1e-12:
        return 0.0
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    best_d2 = float("inf")
    best_u = 0.5
    for i in range(xy.shape[0] - 1):
        p0, p1 = xy[i], xy[i + 1]
        ab = p1 - p0
        denom = float(np.dot(ab, ab))
        if denom < 1e-18:
            continue
        t = float(np.dot(pt - p0, ab) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        proj = p0 + t * ab
        d2 = float(np.sum((pt - proj) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            arc = cum[i] + t * seg_len[i]
            best_u = arc / total
    return float(np.clip(best_u, 0.0, 1.0))


def _segment_distance_threshold(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    min_dist: float,
) -> bool:
    """True if segment endpoints / overlap imply possible collision."""
    ts = np.linspace(0.0, 1.0, 5)
    for t in ts:
        pa = a0 + t * (a1 - a0)
        for s in ts:
            pb = b0 + s * (b1 - b0)
            if float(np.linalg.norm(pa - pb)) < min_dist:
                return True
    return False


def _find_crossing_or_near_hit(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    min_dist: float,
) -> tuple[float, float] | None:
    """Returns (u_on_a, u_on_b) at first intersection or close-approach center."""
    for ia in range(xy_a.shape[0] - 1):
        for jb in range(xy_b.shape[0] - 1):
            a0, a1 = xy_a[ia], xy_a[ia + 1]
            b0, b1 = xy_b[jb], xy_b[jb + 1]
            intr = _segment_intersection(a0, a1, b0, b1)
            if intr is not None:
                ta, sb = intr
                mid_a = a0 + ta * (a1 - a0)
                ua = _point_to_polyline_u(xy_a, mid_a)
                ub = _point_to_polyline_u(xy_b, mid_a)
                return ua, ub
            if _segment_distance_threshold(a0, a1, b0, b1, min_dist):
                mid_a = 0.5 * (a0 + a1)
                ua = _point_to_polyline_u(xy_a, mid_a)
                ub = _point_to_polyline_u(xy_b, mid_a)
                return ua, ub
    return None


def _resolve_pairwise_clearance(
    paths_p: list[np.ndarray],
    paths_xy: list[np.ndarray],
    *,
    min_dist: float,
    samples: int,
    hold_dp: float,
    max_iter: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Simultaneous timeline sampling; defer higher object index with plateau."""
    n = len(paths_p)
    waits = 0
    us = np.linspace(0.0, 1.0, samples)
    for _ in range(max_iter):
        hit: tuple[int, float] | None = None
        for u in us:
            pts = [_interp_polyline(paths_p[i], paths_xy[i], float(u)) for i in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if float(np.linalg.norm(pts[i] - pts[j])) < min_dist:
                        hit = (max(i, j), float(u))
                        break
                if hit is not None:
                    break
            if hit is not None:
                break
        if hit is None:
            break
        wobj, u_bad = hit
        paths_p[wobj], paths_xy[wobj] = _insert_plateau(paths_p[wobj], paths_xy[wobj], u_bad, hold_dp)
        paths_p[wobj], paths_xy[wobj] = _renormalize_path(paths_p[wobj], paths_xy[wobj])
        waits += 1
    return paths_p, paths_xy, waits


def _resolve_pairwise_clearance_subset(
    paths_p: list[np.ndarray],
    paths_xy: list[np.ndarray],
    subset: list[int],
    *,
    min_dist: float,
    samples: int,
    hold_dp: float,
    max_iter: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Same as full clearance but only for object pairs inside ``subset`` (e.g. identity blocks)."""
    waits = 0
    us = np.linspace(0.0, 1.0, samples)
    for _ in range(max_iter):
        hit: tuple[int, float] | None = None
        for u in us:
            pts = [_interp_polyline(paths_p[i], paths_xy[i], float(u)) for i in range(len(paths_p))]
            for ii in range(len(subset)):
                for jj in range(ii + 1, len(subset)):
                    i = subset[ii]
                    j = subset[jj]
                    if float(np.linalg.norm(pts[i] - pts[j])) < min_dist:
                        hit = (max(i, j), float(u))
                        break
                if hit is not None:
                    break
            if hit is not None:
                break
        if hit is None:
            break
        wobj, u_bad = hit
        paths_p[wobj], paths_xy[wobj] = _insert_plateau(paths_p[wobj], paths_xy[wobj], u_bad, hold_dp)
        paths_p[wobj], paths_xy[wobj] = _renormalize_path(paths_p[wobj], paths_xy[wobj])
        waits += 1
    return paths_p, paths_xy, waits


def _resolve_crossings_and_near(
    paths_p: list[np.ndarray],
    paths_xy: list[np.ndarray],
    *,
    min_dist: float,
    hold_dp: float,
    max_iter: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Geometric segment checks for crossings / grazing approaches."""
    n = len(paths_xy)
    waits = 0
    for _ in range(max_iter):
        hit: tuple[int, float] | None = None
        for i in range(n):
            for j in range(i + 1, n):
                pair = _find_crossing_or_near_hit(paths_xy[i], paths_xy[j], min_dist)
                if pair is None:
                    continue
                ua, ub = pair
                wait_obj = max(i, j)
                u_bad = ub if wait_obj == j else ua
                hit = (wait_obj, float(u_bad))
                break
            if hit is not None:
                break
        if hit is None:
            break
        wobj, u_bad = hit
        paths_p[wobj], paths_xy[wobj] = _insert_plateau(paths_p[wobj], paths_xy[wobj], u_bad, hold_dp)
        paths_p[wobj], paths_xy[wobj] = _renormalize_path(paths_p[wobj], paths_xy[wobj])
        waits += 1
    return paths_p, paths_xy, waits


def _build_initial_random_paths(
    starts: np.ndarray,
    targets: np.ndarray,
    rng: np.random.RandomState,
    *,
    bounds_xy: tuple[tuple[float, float], tuple[float, float]],
    n_mid_min: int = 2,
    n_mid_max: int = 4,
    jitter: float = 0.035,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = starts.shape[0]
    paths_p: list[np.ndarray] = []
    paths_xy: list[np.ndarray] = []
    (xmin, xmax), (ymin, ymax) = bounds_xy
    for i in range(n):
        s = starts[i]
        t = targets[i]
        if float(np.linalg.norm(t - s)) < 1.0e-5:
            p_k, xy_k = _arc_length_params(np.stack([s, t], axis=0))
            paths_xy.append(xy_k)
            paths_p.append(p_k)
            continue
        n_mid = int(rng.randint(n_mid_min, n_mid_max + 1))
        mids: list[np.ndarray] = []
        for _ in range(n_mid):
            a = float(rng.uniform(0.12, 0.88))
            base = (1.0 - a) * s + a * t
            noise = rng.randn(2).astype(np.float64) * jitter
            pt = base + noise
            pt[0] = float(np.clip(pt[0], xmin, xmax))
            pt[1] = float(np.clip(pt[1], ymin, ymax))
            mids.append(pt)
        pts = np.stack([s] + mids + [t], axis=0)
        pts = _chaikin_open(pts, iterations=2)
        p_k, xy_k = _arc_length_params(pts)
        paths_xy.append(xy_k)
        paths_p.append(p_k)
    return paths_p, paths_xy


def build_track_clutter_keyframes(
    *,
    initial_xy: torch.Tensor,
    final_xy: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    base_seed: int | None,
    max_keyframes: int = TRACK_CLUTTER_MAX_KEYFRAMES,
    min_separation: float = 0.106,
    identity_pair_min_dist: float = 0.148,
    active_object_mask: torch.Tensor | None = None,
    grasp_object_index: torch.Tensor | None = None,
    distractor_pair_min_dist: float = 0.198,
    bounds_xy: tuple[tuple[float, float], tuple[float, float]] = ((0.18, 0.74), (-0.37, 0.37)),
    duration_scale_per_wait: float = 0.09,
    max_duration_scale: float = 2.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if initial_xy.ndim != 3 or final_xy.ndim != 3:
        raise ValueError("initial_xy and final_xy must be (E, N, 2)")
    num_envs, num_objects, _ = initial_xy.shape
    kf_p = torch.full((num_envs, num_objects, max_keyframes), -1.0, device=device, dtype=dtype)
    kf_xy = torch.zeros((num_envs, num_objects, max_keyframes, 2), device=device, dtype=dtype)
    kf_count = torch.zeros((num_envs, num_objects), device=device, dtype=torch.long)
    duration_scale = torch.ones((num_envs,), device=device, dtype=dtype)

    seed0 = 0 if base_seed is None else int(base_seed)

    for e in range(num_envs):
        rng = np.random.RandomState(seed0 + e * 100003)
        starts = initial_xy[e].detach().float().cpu().numpy()
        targets = final_xy[e].detach().float().cpu().numpy()
        paths_p, paths_xy = _build_initial_random_paths(starts, targets, rng, bounds_xy=bounds_xy)

        paths_p, paths_xy, w1 = _resolve_pairwise_clearance(
            paths_p,
            paths_xy,
            min_dist=min_separation,
            samples=112,
            hold_dp=0.040,
            max_iter=220,
        )
        paths_p, paths_xy, w2 = _resolve_crossings_and_near(
            paths_p,
            paths_xy,
            min_dist=min_separation * 0.98,
            hold_dp=0.038,
            max_iter=120,
        )
        paths_p, paths_xy, w3 = _resolve_pairwise_clearance(
            paths_p,
            paths_xy,
            min_dist=min_separation,
            samples=128,
            hold_dp=0.036,
            max_iter=120,
        )
        n_wait = w1 + w2 + w3
        if active_object_mask is not None:
            aid = torch.nonzero(active_object_mask[e], as_tuple=False).squeeze(-1)
            if aid.numel() >= 2:
                subset = [int(aid[k].item()) for k in range(aid.numel())]
                paths_p, paths_xy, wid = _resolve_pairwise_clearance_subset(
                    paths_p,
                    paths_xy,
                    subset,
                    min_dist=max(float(identity_pair_min_dist), float(min_separation)),
                    samples=136,
                    hold_dp=0.035,
                    max_iter=200,
                )
                n_wait += wid
            if grasp_object_index is not None and aid.numel() == 3:
                gid = int(grasp_object_index[e].item())
                if gid >= 0:
                    distractors = [int(aid[k].item()) for k in range(aid.numel()) if int(aid[k].item()) != gid]
                    if len(distractors) == 2:
                        pair_md = max(
                            float(distractor_pair_min_dist),
                            float(identity_pair_min_dist),
                            float(min_separation),
                        )
                        paths_p, paths_xy, wid2 = _resolve_pairwise_clearance_subset(
                            paths_p,
                            paths_xy,
                            distractors,
                            min_dist=pair_md,
                            samples=144,
                            hold_dp=0.038,
                            max_iter=240,
                        )
                        n_wait += wid2
        scale = min(1.0 + duration_scale_per_wait * float(n_wait), max_duration_scale)
        duration_scale[e] = float(scale)

        for o in range(num_objects):
            pp, xy = _renormalize_path(paths_p[o], paths_xy[o])
            xy = _densify_polyline_max_edge(xy, _MAX_POLYLINE_EDGE_M)
            pp, xy = _arc_length_params(xy)
            pp, xy = _resample_uniform(pp, xy, max_keyframes)
            k = min(max_keyframes, pp.shape[0])
            kf_count[e, o] = k
            kf_p[e, o, :k] = torch.as_tensor(pp[:k], device=device, dtype=dtype)
            kf_xy[e, o, :k] = torch.as_tensor(xy[:k], device=device, dtype=dtype)

    return kf_p, kf_xy, kf_count, duration_scale


def repel_active_identity_targets_xy(
    final_xy: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    identity_initial_xy: torch.Tensor | None = None,
    min_planar_sep: float = 0.278,
    iterations: int = 28,
    x_bounds: tuple[float, float] = (0.158, 0.724),
    y_bounds: tuple[float, float] = (-0.362, 0.362),
    hard_passes: int = 42,
) -> torch.Tensor:
    """Spread identity targets in the table XY plane only.

    Pairwise clearance is Euclidean distance in (x, y); height/Z is out of scope here and
    not used (objects are treated at one planar height).
    """
    if final_xy.ndim != 3 or active_mask.ndim != 2:
        raise ValueError("final_xy must be (E,N,2), active_mask (E,N)")
    out = final_xy.clone()
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    md = float(min_planar_sep)
    for e in range(out.shape[0]):
        idx = torch.nonzero(active_mask[e], as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue
        pts = out[e, idx.long(), :].clone()
        init_plane = (
            identity_initial_xy[e, idx.long(), :].clone()
            if identity_initial_xy is not None
            else pts.clone()
        )
        for _ in range(iterations):
            for ia in range(idx.numel()):
                for ib in range(ia + 1, idx.numel()):
                    va = pts[ia].clone()
                    vb = pts[ib].clone()
                    d = vb - va
                    dist = float(torch.linalg.norm(d).item())
                    if dist >= md - 1e-7:
                        continue
                    if dist < 1e-10:
                        u0 = init_plane[ib] - init_plane[ia]
                        un = float(torch.linalg.norm(u0).item())
                        if un < 1e-8:
                            bump = pts.new_tensor([md * 0.5, 0.0])
                        else:
                            bump = u0 / un * (md * 0.5)
                        pts[ia] = va - 0.5 * bump
                        pts[ib] = vb + 0.5 * bump
                    else:
                        u = d / dist
                        push = 0.5 * (md - dist)
                        pts[ia] = va - u * push
                        pts[ib] = vb + u * push
            pts[:, 0].clamp_(xmin, xmax)
            pts[:, 1].clamp_(ymin, ymax)

        for _ in range(hard_passes):
            for ia in range(idx.numel()):
                for ib in range(ia + 1, idx.numel()):
                    va = pts[ia].clone()
                    vb = pts[ib].clone()
                    d = vb - va
                    dist = float(torch.linalg.norm(d).item())
                    if dist >= md - 2e-5:
                        continue
                    if dist < 1e-10:
                        u0 = init_plane[ib] - init_plane[ia]
                        un = float(torch.linalg.norm(u0).item())
                        if un < 1e-8:
                            bump = pts.new_tensor([md * 0.55, 0.0])
                        else:
                            bump = u0 / un * (md * 0.55)
                        pts[ia] = va - 0.5 * bump
                        pts[ib] = vb + 0.5 * bump
                    else:
                        u = d / dist
                        push = 0.52 * (md - dist + 2e-5)
                        pts[ia] = va - u * push
                        pts[ib] = vb + u * push
                    pts[:, 0].clamp_(xmin, xmax)
                    pts[:, 1].clamp_(ymin, ymax)
            pair_min = float("inf")
            for ia in range(idx.numel()):
                for ib in range(ia + 1, idx.numel()):
                    pair_min = min(pair_min, float(torch.linalg.norm(pts[ib] - pts[ia]).item()))
            if pair_min >= md - 5e-5:
                break

        out[e, idx.long(), :] = pts
    return out


def repel_distractor_identity_pair_xy(
    final_xy: torch.Tensor,
    active_mask: torch.Tensor,
    grasp_object_index: torch.Tensor,
    *,
    distractor_min_planar_sep: float = 0.328,
    identity_initial_xy: torch.Tensor | None = None,
    iterations: int = 36,
    x_bounds: tuple[float, float] = (0.158, 0.724),
    y_bounds: tuple[float, float] = (-0.362, 0.362),
    hard_passes: int = 48,
) -> torch.Tensor:
    """Stronger XY separation only between the two non-protocol-grasp identity blocks (when resolvable).

    Does not move the grasp target object; widens clearance for the distractor pair to reduce contacts
    while ``track_clutter`` scripted motion runs.
    """
    if final_xy.ndim != 3 or active_mask.ndim != 2 or grasp_object_index.ndim != 1:
        raise ValueError("bad shapes for repel_distractor_identity_pair_xy")
    out = final_xy.clone()
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    md = float(distractor_min_planar_sep)
    for e in range(out.shape[0]):
        g = int(grasp_object_index[e].item())
        if g < 0:
            continue
        idx = torch.nonzero(active_mask[e], as_tuple=False).squeeze(-1)
        if idx.numel() != 3:
            continue
        did = [int(idx[i].item()) for i in range(idx.numel()) if int(idx[i].item()) != g]
        if len(did) != 2:
            continue
        i0, i1 = did[0], did[1]
        pts = torch.stack((out[e, i0].clone(), out[e, i1].clone()), dim=0)
        init_plane = (
            torch.stack(
                (identity_initial_xy[e, i0].clone(), identity_initial_xy[e, i1].clone()),
                dim=0,
            )
            if identity_initial_xy is not None
            else pts.clone()
        )
        for _ in range(iterations):
            va, vb = pts[0].clone(), pts[1].clone()
            d = vb - va
            dist = float(torch.linalg.norm(d).item())
            if dist >= md - 1e-7:
                pass
            elif dist < 1e-10:
                u0 = init_plane[1] - init_plane[0]
                un = float(torch.linalg.norm(u0).item())
                bump = pts.new_tensor([md * 0.5, 0.0]) if un < 1e-8 else u0 / un * (md * 0.5)
                pts[0] = va - 0.5 * bump
                pts[1] = vb + 0.5 * bump
            else:
                u = d / dist
                push = 0.5 * (md - dist)
                pts[0] = va - u * push
                pts[1] = vb + u * push
            pts[:, 0].clamp_(xmin, xmax)
            pts[:, 1].clamp_(ymin, ymax)
        for _ in range(hard_passes):
            va, vb = pts[0].clone(), pts[1].clone()
            d = vb - va
            dist = float(torch.linalg.norm(d).item())
            if dist >= md - 2e-5:
                break
            if dist < 1e-10:
                u0 = init_plane[1] - init_plane[0]
                un = float(torch.linalg.norm(u0).item())
                bump = pts.new_tensor([md * 0.55, 0.0]) if un < 1e-8 else u0 / un * (md * 0.55)
                pts[0] = va - 0.5 * bump
                pts[1] = vb + 0.5 * bump
            else:
                u = d / dist
                push = 0.52 * (md - dist + 2e-5)
                pts[0] = va - u * push
                pts[1] = vb + u * push
            pts[:, 0].clamp_(xmin, xmax)
            pts[:, 1].clamp_(ymin, ymax)
        out[e, i0] = pts[0]
        out[e, i1] = pts[1]
    return out


def interpolate_track_clutter_xy_row(
    kf_p: torch.Tensor,
    kf_xy: torch.Tensor,
    kf_count: torch.Tensor,
    env_id: int,
    object_index: int,
    progress: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    k = int(kf_count[env_id, object_index].item())
    if k <= 0:
        return torch.zeros((2,), device=device, dtype=dtype)
    if k == 1:
        return kf_xy[env_id, object_index, 0].to(device=device, dtype=dtype)
    pp = kf_p[env_id, object_index, :k]
    xy = kf_xy[env_id, object_index, :k]
    u = torch.clamp(
        torch.as_tensor(progress, device=pp.device, dtype=pp.dtype),
        0.0,
        1.0,
    )
    idx = int(torch.searchsorted(pp, u, right=True).item()) - 1
    idx = max(0, min(idx, k - 2))
    p0, p1 = pp[idx], pp[idx + 1]
    t = (u - p0) / torch.clamp(p1 - p0, min=1e-9)
    seg = (1.0 - t) * xy[idx] + t * xy[idx + 1]
    return seg.to(device=device, dtype=dtype)


def flatten_final_xy_from_slots(
    step_object_indices: torch.Tensor,
    step_target_xy: torch.Tensor,
    num_objects: int,
) -> torch.Tensor:
    num_envs = step_object_indices.shape[0]
    device = step_object_indices.device
    dtype = step_target_xy.dtype
    out = torch.zeros((num_envs, num_objects, 2), device=device, dtype=dtype)
    for e in range(num_envs):
        for slot in range(step_object_indices.shape[1]):
            oid = int(step_object_indices[e, slot].item())
            if 0 <= oid < num_objects:
                out[e, oid] = step_target_xy[e, slot]
    return out
