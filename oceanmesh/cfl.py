"""CFL / timestep utilities on unstructured meshes (port plan P2).

Ports of OceanMesh2D msh.CalcCFL, msh.GetBarLengths and
msh.bound_courant_number (mesh decimation/refinement to satisfy
Courant bounds).
"""

import logging

import numpy as np

from .fix_mesh import fix_mesh
from .mesh_improve import area_length_quality

logger = logging.getLogger(__name__)

__all__ = ["bound_courant_number", "calc_cfl", "get_bar_lengths"]

GRAV = 9.807
_R_EARTH = 6378206.4


def _bars(cells):
    b = np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]],
                        cells[:, [2, 0]]])
    return np.unique(np.sort(b, axis=1), axis=0)


def get_bar_lengths(points, cells, geographic=True):
    """Unique mesh edges and their lengths (Haversine when
    ``geographic``, else Euclidean). Port of msh.GetBarLengths."""
    bars = _bars(np.asarray(cells, dtype=int))
    p = np.asarray(points, dtype=float)
    if geographic:
        lon1, lat1 = np.deg2rad(p[bars[:, 0]]).T
        lon2, lat2 = np.deg2rad(p[bars[:, 1]]).T
        a = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2)
             * np.sin((lon2 - lon1) / 2) ** 2)
        L = 2 * _R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    else:
        L = np.linalg.norm(p[bars[:, 0]] - p[bars[:, 1]], axis=1)
    return bars, L


def calc_cfl(points, cells, depth, dt=None, geographic=True,
             wave_amplitude=1.0):
    """Per-node Courant number for ``dt`` seconds (or, when ``dt`` is
    None, the maximum stable timestep at Cr=1). ``depth`` is
    positive-down. Port of msh.CalcCFL: wavespeed
    ``sqrt(g*max(b,1)) + amp*sqrt(g/max(b,1))``, nodal dx = shortest
    connected bar."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(cells, dtype=int)
    b = np.maximum(np.asarray(depth, dtype=float), 1.0)
    bars, L = get_bar_lengths(p, t, geographic=geographic)
    dx = np.full(len(p), np.inf)
    np.minimum.at(dx, bars[:, 0], L)
    np.minimum.at(dx, bars[:, 1], L)
    u = np.sqrt(GRAV * b) + wave_amplitude * np.sqrt(GRAV / b)
    if dt is None:
        return dx / u  # max stable dt per node at Cr = 1
    return dt * u / dx


def bound_courant_number(points, cells, depth, dt, cr_max=0.5,
                         cr_min=0.0, max_iter=10, geographic=True,
                         depth_fn=None):
    """Port of msh.bound_courant_number: iteratively DECIMATE nodes
    whose Courant number exceeds ``cr_max`` (collapse into their
    nearest connected neighbour) and REFINE around nodes below
    ``cr_min`` (longest-edge midpoint insertion), until the bounds
    hold or ``max_iter`` is reached. ``depth_fn(points)->depth``
    re-evaluates bathymetry for new/kept nodes (nearest-carry when
    omitted). Boundary conditions must be rebuilt afterwards, as in
    OM2D."""
    p = np.asarray(points, dtype=float).copy()
    t = np.asarray(cells, dtype=int).copy()
    b = np.asarray(depth, dtype=float).copy()

    for it in range(max_iter):
        cr = calc_cfl(p, t, b, dt, geographic=geographic)
        over = np.where(cr > cr_max)[0]
        under = (np.where(cr < cr_min)[0] if cr_min > 0
                 else np.array([], dtype=int))
        if len(over) == 0 and len(under) == 0:
            logger.info(f"Courant bounds satisfied after {it} passes")
            break
        logger.info(
            f"pass {it + 1}: {len(over)} nodes Cr>{cr_max}, "
            f"{len(under)} nodes Cr<{cr_min}"
        )
        bars, L = get_bar_lengths(p, t, geographic=geographic)
        if len(over):
            # collapse each offending node into its nearest neighbour
            # (process a conflict-free subset per pass)
            target = {}
            L_of = {}
            for (a, c), ln in zip(bars, L):
                for v, w in ((int(a), int(c)), (int(c), int(a))):
                    if ln < L_of.get(v, np.inf):
                        L_of[v] = ln
                        target[v] = w
            done = set()
            remap = np.arange(len(p))
            for v in over:
                v = int(v)
                w = target.get(v)
                if w is None or v in done or w in done:
                    continue
                remap[v] = w
                done.update((v, w))
            t = remap[t]
            keep = ~((t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2])
                     | (t[:, 0] == t[:, 2]))
            t = t[keep]
        new_pts = []
        if len(under):
            seen = set()
            for v in under:
                inc = bars[(bars == int(v)).any(axis=1)]
                if not len(inc):
                    continue
                ln = L[(bars == int(v)).any(axis=1)]
                a, c = inc[int(np.argmax(ln))]
                key = (int(a), int(c))
                if key in seen:
                    continue
                seen.add(key)
                new_pts.append(0.5 * (p[int(a)] + p[int(c)]))
        p, t, _ = fix_mesh(p, t, delete_unused=True)
        if depth_fn is not None:
            b = np.asarray(depth_fn(p), dtype=float)
        else:
            from scipy.spatial import cKDTree

            tree = cKDTree(np.asarray(points, dtype=float))
            _, idx = tree.query(p)
            b = np.asarray(depth, dtype=float)[idx]
        if new_pts:
            # re-triangulate locally by full Delaunay of the point set
            # (OM2D uses tridiv octree + retriangulation)
            from scipy.spatial import Delaunay

            p = np.vstack([p, np.asarray(new_pts)])
            if depth_fn is not None:
                b = np.asarray(depth_fn(p), dtype=float)
            else:
                _, idx = tree.query(p)
                b = np.asarray(depth, dtype=float)[idx]
            t = Delaunay(p).simplices
            q = area_length_quality(p, t)
            t = t[q > 1e-6]
    return p, t, b
