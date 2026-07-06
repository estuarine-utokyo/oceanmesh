"""Mesh merging and local remeshing (port plan P4).

Ports of OceanMesh2D msh.plus (type 'arb': inset-priority merge),
msh.cat, msh.minus and msh.remesh_patch/reconstructEdgefx.
All functions take/return (points, cells) numpy arrays in a common
planar CRS.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "cat_meshes",
    "merge_meshes",
    "minus_meshes",
    "remesh_patch",
]


def _boundary_polygons(points, cells):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from .boundary_conditions import boundary_loops

    polys = []
    for loop in boundary_loops(np.asarray(cells)):
        if len(loop) >= 3:
            g = Polygon(points[loop])
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty:
                polys.append(g)
    return unary_union(polys)


def _fix(points, cells):
    from .fix_mesh import fix_mesh

    p, t, _ = fix_mesh(np.asarray(points, float),
                       np.asarray(cells, int), delete_unused=True)
    return p, t


def cat_meshes(p1, t1, p2, t2, tol=1e-8):
    """Port of msh.cat: concatenate two meshes, welding vertices
    closer than ``tol`` (exact-match seams)."""
    from scipy.spatial import cKDTree

    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    j = cKDTree(p1).query_ball_point(p2, tol)
    remap = np.arange(len(p2)) + len(p1)
    keep = np.ones(len(p2), bool)
    for i, g in enumerate(j):
        if g:
            remap[i] = g[0]
            keep[i] = False
    new_index = np.full(len(p2), -1, int)
    new_index[keep] = np.arange(keep.sum()) + len(p1)
    remap[keep] = new_index[keep]
    p = np.vstack([p1, p2[keep]])
    t2r = remap[np.asarray(t2, int)]
    t = np.vstack([np.asarray(t1, int), t2r])
    # drop exact duplicate elements
    t = np.unique(np.sort(t, axis=1), axis=0)
    return _fix(p, t)


def minus_meshes(p1, t1, p2, t2, buffer=0.0):
    """Port of msh.minus: delete elements of mesh 1 whose centroid
    lies within mesh 2's footprint (optionally buffered)."""
    import shapely

    poly2 = _boundary_polygons(np.asarray(p2, float),
                               np.asarray(t2, int))
    if buffer:
        poly2 = poly2.buffer(buffer)
    p1 = np.asarray(p1, float)
    t1 = np.asarray(t1, int)
    cen = p1[t1].mean(axis=1)
    inside = shapely.contains_xy(poly2, cen[:, 0], cen[:, 1])
    return _fix(p1, t1[~inside])


def merge_meshes(p1, t1, p2, t2, lock_dist=None, smooth=True):
    """Port of msh.plus type 'arb': merge INSET mesh 1 into BASE
    mesh 2 with mesh-1 priority. Base elements touching the inset
    footprint are removed, the union point set is re-triangulated,
    triangles outside both footprints are discarded, and only the
    seam band (nodes within ``lock_dist`` of both inputs) is
    smoothed (LUR with everything else pinned)."""
    import shapely
    from scipy.spatial import Delaunay, cKDTree

    p1 = np.asarray(p1, float)
    t1 = np.asarray(t1, int)
    p2 = np.asarray(p2, float)
    t2 = np.asarray(t2, int)
    poly1 = _boundary_polygons(p1, t1)
    poly2 = _boundary_polygons(p2, t2)

    # delete base elements with ANY vertex inside the inset
    v_in = shapely.contains_xy(
        poly1, p2[:, 0], p2[:, 1]
    )
    t2_keep = t2[~v_in[t2].any(axis=1)]
    used2 = np.unique(t2_keep)
    p2k = p2[used2]

    # drop base points hugging the inset boundary (they survive the
    # footprint filter when they sit ON the boundary and produce
    # sliver triangles): closer to an inset node than half that
    # node's own local spacing -> inset priority, remove.
    tree1 = cKDTree(p1)
    d, j = tree1.query(p2k)
    d1n, _ = tree1.query(p1, k=2)
    local_h = d1n[:, 1]
    p2k = p2k[d > 0.5 * local_h[j]]
    h1 = np.median(
        np.linalg.norm(p1[t1[:, 0]] - p1[t1[:, 1]], axis=1)
    )
    pm = np.vstack([p1, p2k])

    tm = Delaunay(pm).simplices
    cen = pm[tm].mean(axis=1)
    keep = shapely.contains_xy(
        poly1.buffer(1e-9), cen[:, 0], cen[:, 1]
    ) | shapely.contains_xy(
        poly2.buffer(1e-9), cen[:, 0], cen[:, 1]
    )
    pm, tm = _fix(pm, tm[keep])

    # seam hygiene (OM2D deletes low-connectivity seam vertices):
    # drop degenerate slivers, then collapse remaining thin ones
    a, b, c = pm[tm[:, 0]], pm[tm[:, 1]], pm[tm[:, 2]]
    area2 = np.abs(np.cross(b - a, c - a))
    tm = tm[area2 > 1e-14 * np.median(area2)]
    from .mesh_improve import collapse_thin_triangles

    pm, tm = collapse_thin_triangles(pm, tm, min_qual=0.05)
    pm, tm = _fix(pm, tm)

    if smooth:
        if lock_dist is None:
            lock_dist = 2.0 * h1
        d1, _ = cKDTree(p1).query(pm)
        d2, _ = cKDTree(p2).query(pm)
        seam = (d1 > 1e-12) & (d2 > 1e-12)
        near = (d1 < lock_dist) & (d2 < lock_dist)
        free = seam & near
        from .mesh_improve import direct_smoother_lur

        pm, _ = direct_smoother_lur(pm, tm, pfix=pm[~free])
    logger.info(
        f"merge_meshes: inset {len(t1)} + base {len(t2)} -> "
        f"{len(tm)} elements"
    )
    return pm, tm


def _reconstruct_sizing(points, cells):
    """Port of msh.reconstructEdgefx: nodal size = incident
    circumradius (last-wins like the MATLAB loop; linear scattered
    interpolant)."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    p = np.asarray(points, float)
    t = np.asarray(cells, int)
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    la = np.linalg.norm(b - c, axis=1)
    lb = np.linalg.norm(a - c, axis=1)
    lc = np.linalg.norm(a - b, axis=1)
    area = 0.5 * np.abs(np.cross(b - a, c - a))
    cr = la * lb * lc / np.maximum(4.0 * area, 1e-30)
    z = np.zeros(len(p))
    z[t[:, 0]] = cr
    z[t[:, 1]] = cr
    z[t[:, 2]] = cr
    lin = LinearNDInterpolator(p, z)
    near = NearestNDInterpolator(p, z)

    def fh(q):
        q = np.atleast_2d(q)
        v = lin(q)
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = near(q[bad])
        return v

    return fh


def remesh_patch(points, cells, poly, target_h=None, grade=0.15,
                 max_iter=100, seed=0, constrain_ring=True):
    """Port of msh.remesh_patch: re-mesh the part of the mesh
    inside polygon ``poly`` (an (N, 2) array) and stitch it back.
    Sizing comes from the existing local circumradii unless
    ``target_h`` (uniform) is given."""
    import shapely
    from shapely.geometry import Polygon

    from .mesh_generator import generate_mesh
    from .signed_distance_function import Domain

    p = np.asarray(points, float)
    t = np.asarray(cells, int)
    ring = np.asarray(poly, float)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    pg = Polygon(ring)

    cen = p[t].mean(axis=1)
    inside = shapely.contains_xy(pg, cen[:, 0], cen[:, 1])
    if not inside.any():
        raise ValueError("polygon contains no elements")
    sub_p, sub_t = _fix(p, t[inside])
    hole_p, hole_t = _fix(p, t[~inside])

    # the actual cavity outline (mesh-conforming, not the input poly)
    cavity = _boundary_polygons(sub_p, sub_t)

    if target_h is not None:
        # grade from the existing boundary sizes down/up to
        # target_h so the seam has no size jump:
        # h = max(target, h_old - grade * dist_to_cavity_boundary)
        f_old = _reconstruct_sizing(sub_p, sub_t)

        def fh(q):
            import shapely as _sh

            q = np.atleast_2d(q)
            d = _sh.distance(
                _sh.points(q[:, 0], q[:, 1]), cavity.boundary
            )
            return np.maximum(float(target_h),
                              np.asarray(f_old(q)) - grade * d)
        h0 = float(target_h)
    else:
        fh = _reconstruct_sizing(sub_p, sub_t)
        h0 = float(np.nanpercentile(fh(sub_p), 10))

    def fd(q, box=None):
        q = np.atleast_2d(q)
        d = shapely.distance(
            shapely.points(q[:, 0], q[:, 1]), cavity.boundary
        )
        sgn = np.where(
            shapely.contains_xy(cavity, q[:, 0], q[:, 1]), -1.0, 1.0
        )
        return sgn * d

    x0, y0, x1, y1 = cavity.bounds
    domain = Domain((x0, x1, y0, y1), fd)
    gen_kw = {}
    if constrain_ring:
        # OM2D-'match' semantics: the cavity ring is kept VERBATIM
        # (ring edges as constrained Delaunay edges, NOT
        # lock_boundary), so the regenerated patch boundary
        # coincides exactly with the hole mesh and stitching is a
        # weld (cat) — the hole mesh is never re-triangulated.
        from .boundary_conditions import boundary_loops

        loops = boundary_loops(sub_t)
        ring_pts = []
        ring_edges = []
        off = 0
        for lp in loops:
            n = len(lp)
            ring_pts.append(sub_p[lp])
            ring_edges.extend(
                [[off + i, off + (i + 1) % n] for i in range(n)]
            )
            off += n
        gen_kw["pfix"] = np.vstack(ring_pts)
        gen_kw["egfix"] = np.asarray(ring_edges, dtype=int)
    new_p, new_t = generate_mesh(
        domain, fh, min_edge_length=h0, max_iter=max_iter,
        seed=seed, **gen_kw,
    )
    if constrain_ring:
        merged = cat_meshes(new_p, new_t, hole_p, hole_t,
                            tol=1e-3)
    else:
        merged = merge_meshes(new_p, new_t, hole_p, hole_t)
    logger.info(
        f"remesh_patch: {int(inside.sum())} elements -> "
        f"{len(new_t)} regenerated"
    )
    return merged
