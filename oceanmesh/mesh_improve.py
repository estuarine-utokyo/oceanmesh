"""OM2D msh-class mesh improvement utilities (port plan P2).

Ports of OceanMesh2D utilities operating on bare ``(points, cells)``
arrays: collapse_thin_triangles.m, direct_smoother_lur.m, the msh
``bound_con_int`` valence bound (behavioural port via edge flips +
local smoothing), ``flipEdge``, and RCM renumbering (msh.renum).
"""

import logging

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .edges import get_boundary_edges
from .fix_mesh import fix_mesh

logger = logging.getLogger(__name__)

__all__ = [
    "area_length_quality",
    "bound_connectivity",
    "collapse_thin_triangles",
    "direct_smoother_lur",
    "flip_edge",
    "renumber_rcm",
]


def area_length_quality(points, cells):
    """OM2D gettrimeshquan qm: 4*sqrt(3)*A/(l1^2+l2^2+l3^2)
    (equilateral = 1)."""
    tri = points[cells]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 1]
    e3 = tri[:, 0] - tri[:, 2]
    area = 0.5 * np.abs(e1[:, 0] * (-e3[:, 1]) - e1[:, 1] * (-e3[:, 0]))
    den = (e1**2).sum(1) + (e2**2).sum(1) + (e3**2).sum(1)
    den[den == 0] = np.finfo(float).eps
    return 4.0 * np.sqrt(3.0) * area / den


def collapse_thin_triangles(points, cells, min_qual=0.10, pfix=None):
    """Exact port of utilities/collapse_thin_triangles.m: while any
    element has MEAN-RATIO quality (gettrimeshquan qm, ~simp_qual)
    below ``min_qual``, collapse its shortest edge; REVERT collapses
    that would create index-degenerate elements
    (collapse_thin_triangles.m:33-38).

    ``pfix`` (fork feature, no .m counterpart — the .m has no CDT
    constraints to protect): coordinates of protected vertices.
    A protected vertex may be a merge TARGET but is never removed;
    a collapse between two protected vertices is skipped."""
    from .fix_mesh import fix_mesh as _fix_mesh
    from .fix_mesh import simp_qual

    p = np.array(points, dtype=float)
    t = np.array(cells, dtype=int)
    protected = frozenset()
    if pfix is not None and len(pfix) > 0:
        from scipy.spatial import cKDTree

        _, _idx = cKDTree(p).query(np.asarray(pfix, dtype=float))
        protected = frozenset(int(v) for v in np.atleast_1d(_idx))
    qm = simp_qual(p, t)
    kount = 0
    guard = 0
    while (qm < min_qual).any():
        guard += 1
        if guard > 10 * len(cells):
            logger.warning("collapse_thin_triangles: guard tripped")
            break
        tid = int(np.where(qm < min_qual)[0][0])
        tri = t[tid]
        ee = np.array([[tri[0], tri[1]], [tri[0], tri[2]],
                       [tri[1], tri[2]]])
        evec = p[ee[:, 0]] - p[ee[:, 1]]
        meid = int(np.argmin((evec**2).sum(1)))
        rm, keep = int(ee[meid, 0]), int(ee[meid, 1])
        if rm in protected:
            if keep in protected:
                # never collapse a constrained edge
                qm[tid] = 1.0
                continue
            rm, keep = keep, rm
        t_old = t.copy()
        t = np.delete(t, tid, axis=0)
        t[t == rm] = keep
        degen = ((t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2])
                 | (t[:, 0] == t[:, 2]))
        if degen.any():
            # OM2D reverts (collapse_thin_triangles.m:33-38)
            t = t_old
            qm[tid] = 1.0
            continue
        qm = np.delete(qm, tid)
        touched = ((t == keep).any(axis=1))
        if touched.any():
            qm[touched] = simp_qual(p, t[touched])
        kount += 1
    p, t, _ = _fix_mesh(p, t, delete_unused=True)
    logger.info(f"Removed {kount} thin triangles")
    return p, t


def direct_smoother_lur(points, cells, pfix=None):
    """Port of utilities/direct_smoother_lur.m: one implicit (direct
    solve) smoothing step. Boundary vertices and ``pfix`` positions
    are pinned with a large penalty; interior vertices solve a
    per-element stiffness system (D/T blocks, mu=1)."""
    p = np.array(points, dtype=float)
    t = np.asarray(cells, dtype=int)
    # normalize coordinates: kinf pinning is an absolute penalty,
    # and at UTM scale (coords ~1e6, K entries ~1e5) the stiffness
    # ratio collapses — 'pinned' vertices drifted tens of metres.
    shift = p.mean(axis=0)
    scale = float(np.abs(p - shift).max())
    scale = scale if scale > 0 else 1.0
    p = (p - shift) / scale
    if pfix is not None and len(pfix) > 0:
        pfix = (np.asarray(pfix, dtype=float) - shift) / scale
    n = len(p)
    mu = 1.0
    kinf = 1.0e12

    D = 2.0 * np.eye(2)
    T = np.array([[-1.0, -np.sqrt(3.0)], [np.sqrt(3.0), -1.0]])
    ke = [[D, T, T.T], [T.T, D, T], [T, T.T, D]]

    rows, cols, vals = [], [], []
    for r in range(3):
        for s in range(3):
            block = ke[r][s]
            gi = 2 * t[:, r]
            gj = 2 * t[:, s]
            for a in range(2):
                for b in range(2):
                    rows.append(gi + a)
                    cols.append(gj + b)
                    vals.append(np.full(len(t), block[a, b]))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n)).tolil()

    bnd = np.unique(get_boundary_edges(t))
    fixed = set(int(v) for v in bnd)
    if pfix is not None and len(pfix) > 0:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(p).query(np.asarray(pfix, dtype=float))
        fixed.update(int(v) for v in np.atleast_1d(idx))

    F = np.zeros(2 * n)
    for v in fixed:
        F[2 * v: 2 * v + 2] += kinf * p[v]
        K[2 * v, 2 * v] = kinf
        K[2 * v + 1, 2 * v + 1] = kinf

    c = sp.linalg.spsolve(K.tocsr(), F)
    out = c.reshape(-1, 2)
    # penalty pinning is approximate (~1/kinf residual); pinned
    # vertices are FIXED by contract — snap them back exactly
    fixed_idx = np.fromiter(fixed, dtype=int)
    out[fixed_idx] = p[fixed_idx]
    out = out * scale + shift
    return out, np.asarray(cells, dtype=int)


def _vertex_valence(cells, n):
    return np.bincount(np.asarray(cells).ravel(), minlength=n)


def flip_edge(points, cells, elem_a, elem_b):
    """Port of msh.flipEdge (OPNML line_swap): swap the shared edge
    of two adjacent elements. Returns new cells; raises ValueError
    when the elements do not share exactly one edge or the flip
    would produce an invalid (non-convex quad) configuration."""
    t = np.array(cells, dtype=int)
    ta, tb = t[elem_a], t[elem_b]
    shared = np.intersect1d(ta, tb)
    if len(shared) != 2:
        raise ValueError("elements do not share an edge")
    a_only = int(np.setdiff1d(ta, shared)[0])
    b_only = int(np.setdiff1d(tb, shared)[0])
    s1, s2 = int(shared[0]), int(shared[1])

    def _area(i, j, k):
        (x1, y1), (x2, y2), (x3, y3) = points[i], points[j], points[k]
        return 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    # the quad a_only-s1-b_only-s2 must be convex
    new1 = (a_only, s1, b_only)
    new2 = (a_only, b_only, s2)
    A1, A2 = _area(*new1), _area(*new2)
    if A1 == 0 or A2 == 0 or np.sign(A1) != np.sign(A2):
        raise ValueError("flip would tangle the mesh")
    if A1 < 0:
        new1 = new1[::-1]
        new2 = new2[::-1]
    t[elem_a] = new1
    t[elem_b] = new2
    return t


def bound_connectivity(points, cells, max_valence=7, pfix=None,
                       max_iter=20, region_nodes=None):
    """Behavioural port of msh.bound_con_int (OPNML
    reduce_con_int): reduce interior vertex valence to at most
    ``max_valence`` by flipping edges incident to over-connected
    vertices, followed by one implicit smoothing pass. OM2D applies
    case-wise local retriangulations; edge flips reproduce the same
    valence transfers on the interior."""
    p = np.array(points, dtype=float)
    t = np.array(cells, dtype=int)
    n = len(p)
    bnd = set(int(v) for v in np.unique(get_boundary_edges(t)))
    protected = set(bnd)
    if pfix is not None and len(pfix) > 0:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(p).query(np.asarray(pfix, dtype=float))
        protected.update(int(v) for v in np.atleast_1d(idx))

    n_flips = 0
    for _ in range(max_iter):
        val = _vertex_valence(t, n)
        over = [v for v in np.where(val > max_valence)[0]
                if v not in bnd]
        if region_nodes is not None:
            _reg = set(int(q) for q in region_nodes)
            over = [v for v in over if v in _reg]
        if not over:
            break
        progressed = False
        # element -> neighbors via shared edges
        from collections import defaultdict

        edge_map = defaultdict(list)
        for ei, tri in enumerate(t):
            for k in range(3):
                a, b = sorted((int(tri[k]), int(tri[(k + 1) % 3])))
                edge_map[(a, b)].append(ei)
        for v in over:
            # flip an edge (v, w) shared by two elements, where the
            # flip reduces v's valence: choose w with max valence
            nbrs = []
            for (a, b), els in edge_map.items():
                if len(els) == 2 and (a == v or b == v):
                    w = b if a == v else a
                    nbrs.append((w, els))
            nbrs.sort(key=lambda x: -val[x[0]])
            for w, els in nbrs:
                try:
                    t_new = flip_edge(p, t, els[0], els[1])
                except ValueError:
                    continue
                val_new = _vertex_valence(t_new, n)
                if val_new[v] < val[v] and (
                        val_new > max_valence
                ).sum() <= (val > max_valence).sum():
                    t = t_new
                    n_flips += 1
                    progressed = True
                    break
            if progressed:
                break
        if not progressed:
            break
    if n_flips:
        pin = (np.asarray(pfix, dtype=float)
               if pfix is not None and len(pfix) else None)
        if region_nodes is not None:
            # region-restricted call: the implicit smoothing pass
            # must not relocate vertices outside the region either
            # (a global pass moved interior nodes ~50 m across the
            # whole mesh — the phantom-C4 source in remesh_patch)
            _reg = set(int(q) for q in region_nodes)
            outside = np.array(
                [v for v in range(len(p)) if v not in _reg], int
            )
            pin_r = p[outside]
            if pin is not None:
                pin_r = np.vstack([pin_r, pin])
            p, t = direct_smoother_lur(p, t, pfix=pin_r)
        else:
            p, t = direct_smoother_lur(p, t, pfix=pin)
    logger.info(f"bound_connectivity: {n_flips} edge flips")
    return p, t


def renumber_rcm(points, cells, node_fields=None):
    """Port of msh.renum: Reverse Cuthill-McKee bandwidth-minimizing
    node renumbering. ``node_fields`` is an optional list of per-node
    arrays remapped alongside. Returns (points, cells, fields,
    permutation)."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(cells, dtype=int)
    n = len(p)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    A = sp.coo_matrix(
        (np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)
    ).tocsr()
    perm = np.asarray(reverse_cuthill_mckee(A, symmetric_mode=True))
    inv = np.empty(n, dtype=int)
    inv[perm] = np.arange(n)
    p2 = p[perm]
    t2 = inv[t]
    fields = None
    if node_fields is not None:
        fields = [np.asarray(f)[perm] for f in node_fields]
    return p2, t2, fields, perm
