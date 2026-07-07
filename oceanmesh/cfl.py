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


def _smoothmesh(p, t, constr, maxit=20, tol=0.01):
    """Port of utilities/smoothmesh.m: Laplacian smoothing with
    boundary nodes and ``constr`` (indices) pinned, converging on
    the relative edge-length change."""
    import scipy.sparse as sp

    from .fix_mesh import fix_mesh as _fm

    p, t, _ = _fm(p, t, delete_unused=True)
    n = len(p)
    S = sp.coo_matrix(
        (np.ones(6 * len(t)),
         (t[:, [0, 0, 1, 1, 2, 2]].ravel(),
          t[:, [1, 2, 0, 2, 0, 1]].ravel())),
        shape=(n, n),
    ).tocsr()
    S.data[:] = 1.0
    W = np.asarray(S.sum(axis=1)).ravel()
    if (W == 0).any():
        raise ValueError("Invalid mesh. Hanging nodes found.")
    e = np.vstack([t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]])
    e = np.sort(e, axis=1)
    eu, counts = np.unique(e, axis=0, return_counts=True)
    bnd = np.unique(eu[counts == 1])
    L = np.maximum(
        np.sqrt(((p[eu[:, 0]] - p[eu[:, 1]]) ** 2).sum(1)),
        np.finfo(float).eps,
    )
    constr = np.asarray(constr, dtype=int)
    for _ in range(maxit):
        pnew = (S @ p) / W[:, None]
        pnew[bnd] = p[bnd]
        if len(constr):
            pnew[constr] = p[constr]
        p = pnew
        Lnew = np.maximum(
            np.sqrt(((p[eu[:, 0]] - p[eu[:, 1]]) ** 2).sum(1)),
            np.finfo(float).eps,
        )
        move = np.max(np.abs((Lnew - L) / Lnew))
        if move < tol:
            break
        L = Lnew
    return p, t


def _mesh_boundary_polygon(p, t):
    """Ordered NaN-delimited boundary polygon(s) of a mesh
    (get_boundary_of_mesh analog), loops with < 3 points dropped."""
    e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    es = np.sort(e, axis=1)
    eu, counts = np.unique(es, axis=0, return_counts=True)
    bedges = eu[counts == 1]
    adj = {}
    for a, c in bedges:
        adj.setdefault(int(a), []).append(int(c))
        adj.setdefault(int(c), []).append(int(a))
    unused = {tuple(sorted(be)) for be in bedges.tolist()}
    parts = []
    while unused:
        a0, c0 = next(iter(unused))
        loop = [a0, c0]
        unused.discard((min(a0, c0), max(a0, c0)))
        while True:
            cur, prev = loop[-1], loop[-2]
            nxts = [v for v in adj.get(cur, [])
                    if (min(cur, v), max(cur, v)) in unused]
            if not nxts:
                break
            nxt = nxts[0]
            loop.append(nxt)
            unused.discard((min(cur, nxt), max(cur, nxt)))
            if nxt == loop[0]:
                break
        if len(loop) >= 4:
            parts.append(p[np.asarray(loop, dtype=int)])
            parts.append(np.array([[np.nan, np.nan]]))
    if not parts:
        return np.empty((0, 2))
    return np.vstack(parts)


def _decimate_tria(p, t, bad, bad_pts):
    """DecimateTria (msh.m:3013-3051): drop the offending points,
    retriangulate inside the mesh's own boundary polygon, prune
    quality<0.1 boundary elements, then locally smooth around the
    deleted sites (constr = everything outside their 12-NN)."""
    from _delaunay_class import DelaunayTriangulation as DT
    from scipy.spatial import cKDTree

    from .fix_mesh import fix_mesh as _fm
    from .fix_mesh import simp_qual
    from .geometry import inpoly2
    from . import edges as om_edges

    poly = _mesh_boundary_polygon(p, t)
    pedges = om_edges.get_poly_edges(poly)

    keep_pts = p[~bad]
    dtri = DT()
    dtri.insert(keep_pts.ravel().tolist())
    pm = np.asarray(dtri.get_finite_vertices())
    tm = np.asarray(dtri.get_finite_cells(), dtype=int)
    pmid = pm[tm].mean(axis=1)
    inside, _ = inpoly2(pmid, np.nan_to_num(poly), pedges)
    tm = tm[inside]
    pm, tm, _ = _fm(pm, tm, delete_unused=True)

    # delete poor boundary elements iteratively (msh.m:3031-3042)
    while True:
        q = simp_qual(pm, tm)
        e = np.vstack([tm[:, [0, 1]], tm[:, [1, 2]], tm[:, [2, 0]]])
        es = np.sort(e, axis=1)
        eu, counts = np.unique(es, axis=0, return_counts=True)
        bnodes = np.unique(eu[counts == 1])
        touching = np.isin(tm, bnodes).any(axis=1)
        badb = touching & (q < 0.1)
        if not badb.any():
            break
        tm = tm[~badb]
        pm, tm, _ = _fm(pm, tm, delete_unused=True)

    # local smoothing (msh.m:3043-3049)
    if len(bad_pts) < 0.1 * len(pm):
        _, idx = cKDTree(pm).query(bad_pts, k=min(12, len(pm)))
        near = np.unique(np.asarray(idx).ravel())
        constr = np.setdiff1d(np.arange(len(pm)), near)
        pm, tm = _smoothmesh(pm, tm, constr, maxit=50, tol=0.01)
    else:
        logger.warning(
            "Adaptation would lose too much connectivity; try a "
            "smaller timestep"
        )
    return pm, tm


def bound_courant_number(points, cells, depth, dt, cr_max=0.5,
                         cr_min=0.0, max_iter=10, geographic=True,
                         depth_fn=None, _depth=0):
    """Faithful port of msh.bound_courant_number
    (msh.m:2822-3075): DECIMATE where Cr > cr_max (DecimateTria +
    clean('passive') with escalating valence bound), REFINE where
    Cr < cr_min (tridiv2 red/green split of the touching
    triangles), bathymetry re-mapped each pass through a
    linear/nearest scattered interpolant built from the INPUT
    mesh; finishes with one recursive call so both bounds hold.
    Boundary conditions must be rebuilt afterwards, as in OM2D."""
    from pyproj import Transformer
    from scipy.interpolate import (LinearNDInterpolator,
                                   NearestNDInterpolator)

    from .clean import om2d_default_clean
    from .fix_mesh import fix_mesh as _fm
    from .tridiv2 import tridiv2

    p = np.asarray(points, dtype=float).copy()
    t = np.asarray(cells, dtype=int).copy()
    b = np.asarray(depth, dtype=float).copy()

    cr = calc_cfl(p, t, b, dt, geographic=geographic)
    if cr.max() <= cr_max and (cr_min <= 0 or cr.min() >= cr_min):
        logger.info("Courant number constraints already satisfied")
        return p, t, b

    # m_proj('Trans', mesh extent) sandwich (msh.m:2894-2906)
    lon0 = 0.5 * (p[:, 0].min() + p[:, 0].max())
    lat0 = 0.5 * (p[:, 1].min() + p[:, 1].max())
    if geographic:
        _tr = Transformer.from_crs(
            "EPSG:4326",
            f"+proj=tmerc +lon_0={lon0} +lat_0={lat0} "
            "+ellps=WGS84 +units=m",
            always_xy=True,
        )

        def _to(q):
            x, y = _tr.transform(q[:, 0], q[:, 1])
            return np.column_stack([x, y])

        def _from(q):
            x, y = _tr.transform(q[:, 0], q[:, 1],
                                 direction="INVERSE")
            return np.column_stack([x, y])
    else:
        def _to(q):
            return np.array(q, copy=True)

        def _from(q):
            return np.array(q, copy=True)

    pp = _to(p)
    # scatteredInterpolant(...,'linear','nearest') (msh.m:2909)
    _lin = LinearNDInterpolator(pp, b)
    _nea = NearestNDInterpolator(pp, b)

    def F(q):
        v = _lin(q)
        miss = ~np.isfinite(v)
        if miss.any():
            v[miss] = _nea(q[miss])
        return v

    # ---- bound the MAXIMUM Cr by decimation (msh.m:2912-2949)
    if cr_max > np.finfo(float).eps:
        con = 9
        badnump = np.inf
        it = 0
        while True:
            p_ll = _from(pp)
            cr = calc_cfl(p_ll, t, b, dt, geographic=geographic)
            bad = cr > cr_max
            badnum = int(bad.sum())
            logger.info(
                f"max-Cr pass {it}: {badnum} violations "
                f"(max Cr {cr.max():.2f})"
            )
            if it == max_iter or badnum == 0:
                break
            if badnump - badnum <= 0:
                con += 1
            it += 1
            bad_pts = pp[bad]
            pp2, t2 = _decimate_tria(pp, t, bad, bad_pts)
            pp2, t2 = om2d_default_clean(
                pp2, t2, min_qual_bound=0.1, dj_cutoff=0.0,
                smooth=False, con=con,
            )
            if len(t2) == 0 or len(pp2) < 3:
                logger.warning(
                    "decimation degenerated the mesh; stopping the "
                    "max-Cr loop"
                )
                break
            pp, t = pp2, t2
            b = F(pp)
            badnump = badnum

    # ---- bound the MINIMUM Cr by refinement (msh.m:2951-2989)
    if cr_min > np.finfo(float).eps:
        it = 0
        while True:
            p_ll = _from(pp)
            cr = calc_cfl(p_ll, t, b, dt, geographic=geographic)
            bad = cr < cr_min
            badnum = int(bad.sum())
            logger.info(
                f"min-Cr pass {it}: {badnum} violations "
                f"(min Cr {cr.min():.2f})"
            )
            if it == max_iter or badnum == 0:
                break
            it += 1
            tdiv = np.isin(t, np.where(bad)[0]).any(axis=1)
            pp, t = tridiv2(pp, t, tdiv)
            pp, t, _ = _fm(pp, t, delete_unused=True)
            b = F(pp)

    p = _from(pp)

    # recursive final check (msh.m:3003): the .m recurses
    # UNBOUNDED — the early return above (both bounds satisfied)
    # is the terminator; cap at 10 as a runaway guard only
    if _depth < 10:
        return bound_courant_number(
            p, t, b, dt, cr_max=cr_max, cr_min=cr_min,
            max_iter=max_iter, geographic=geographic,
            depth_fn=depth_fn, _depth=_depth + 1,
        )
    return p, t, b
