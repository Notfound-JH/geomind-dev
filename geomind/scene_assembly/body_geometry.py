"""Load hash-bound body-local geometry for homogeneous or mixed assemblies."""
from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path

import numpy as np


def _cloud(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError('Invalid assembled body-point cloud; expected finite Nx3 vertices')
    return points


def load_body_clouds(manifest):
    """Return one scaled body-local cloud per manifest object, in manifest order.

    Each object may name a ``body_points_key`` in the one NPZ artifact; older
    homogeneous assemblies use ``points_m`` for every object. The complete
    archive is hash-checked before any geometry is accepted, including keys
    that are not selected by the objects.
    """
    path = Path(manifest['asset_body_vertices'])
    expected = manifest.get('provenance', {}).get('asset_body_vertices_sha256')
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('Original body-point hash is missing')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Body-point hash changed')
    objects = manifest.get('objects')
    if not isinstance(objects, list) or not objects:
        raise ValueError('Body geometry requires a nonempty manifest object list')
    clouds = []
    cache = {}
    with np.load(path, allow_pickle=False) as archive:
        for obj in objects:
            key = obj.get('body_points_key', 'points_m')
            if not isinstance(key, str) or not key or key not in archive.files:
                raise ValueError(f'Body-point key is missing or invalid: {key!r}')
            if key not in cache:
                cache[key] = _cloud(archive[key])
            clouds.append(cache[key])
    return clouds


def points_for(points, index):
    """Select one cloud; a legacy Nx3 array is shared by all object indices."""
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError('Body-cloud index must be a nonnegative integer')
    if isinstance(points, np.ndarray):
        return _cloud(points if points.ndim == 2 else points[index])
    # Inspect just the first item, avoiding a full stack/copy of every object
    # cloud on each measured physics tick. A list of XYZ rows is shared.
    if len(points) and np.asarray(points[0]).ndim == 1:
        return _cloud(points)
    return _cloud(points[index])
