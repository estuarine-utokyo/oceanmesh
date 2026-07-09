"""1:1 port of OceanMesh2D utilities/smooth2d.m (mesh2d, Darren
Engwirda): "hill-climbing" mesh smoothing with spring updates,
element-quality guarantees and vertex density control.

Structure, constants and variable names follow the .m line by line
(smooth2d.m:80-641). Engine substitutions only:
- deltri2 -> CGAL constrained Delaunay (verified binding) with the
  centroid-in-part test done by the referee-verified inpoly2
- triscr2 == area_length_quality (identical formula,
  GEOM_UTIL/mesh-cost/triscr2.m: 4/sqrt(3) * area / (lrms))
"""

import logging

import numpy as np

from .fix_mesh import fix_mesh

logger = logging.getLogger(__name__)

__all__ = ["smooth2d"]


def _tricon2(tria):
    """mesh-util/tricon2.m (reduced): unique edge table for a
    triangulation. Returns edge (E,2) int and, per edge, the count
    of adjacent triangles (edge 'degree'; 1 = boundary edge)."""
    e = np.vstack([tria[:, [0, 1]], tria[:, [1, 2]], tria[:, [2, 0]]])
    e = np.sort(e, axis=1)
    # int64-key encoding: identical lexicographic order/counts as
    # np.unique(axis=0) but ~10x faster (axis-unique's structured
    # sort took minutes per call at 28M edge rows, dominating
    # smooth2d on multi-million-node meshes)
    n = np.int64(e.max()) + 1
    key = e[:, 0].astype(np.int64) * n + e[:, 1].astype(np.int64)
    uk, counts = np.unique(key, return_counts=True)
    edge = np.column_stack([uk // n, uk % n]).astype(tria.dtype)
    return edge, counts


def _triscr2(pp, tt):
    """GEOM_UTIL/mesh-cost/triscr2.m: 4*sqrt(3)/3 * area / lrms."""
    if len(tt) == 0:
        return np.empty(0)
    a, b, c = pp[tt[:, 0]], pp[tt[:, 1]], pp[tt[:, 2]]
    area = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )
    lrms = (
        ((b - a) ** 2).sum(1)
        + ((c - b) ** 2).sum(1)
        + ((c - a) ** 2).sum(1)
    ) / 3.0
    scal = 4.0 * np.sqrt(3.0) / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        tscr = scal * area / lrms
    return np.where(np.isfinite(tscr), tscr, 0.0)


def _deltri2(vert, conn, node, pslg):
    """mesh-util/deltri2.m via the CGAL constrained Delaunay
    binding: triangulate vert with the constrained edges conn, keep
    triangles whose centroid lies inside the PSLG boundary."""
    from _delaunay_class import DelaunayTriangulation as DT

    try:
        from _delaunay_class import (
            ConstrainedDelaunayTriangulation as CDT,
        )
        have_cdt = True
    except Exception:  # pragma: no cover
        have_cdt = False

    from .geometry import inpoly2

    if have_cdt and len(conn):
        dt = CDT()
        dt.insert(vert.ravel().tolist())
        segs = np.hstack(
            [vert[conn[:, 0]], vert[conn[:, 1]]]
        ).ravel().tolist()
        dt.insert_constraints(segs)
    else:
        dt = DT()
        dt.insert(vert.ravel().tolist())
    p = np.asarray(dt.get_finite_vertices())
    t = np.asarray(dt.get_finite_cells(), dtype=int)
    pmid = p[t].mean(axis=1)
    inside, _ = inpoly2(pmid, node, pslg)
    t = t[inside]
    return p, t


def smooth2d(vert, tria, conn=None, iters=32, vtol=1.0e-2,
             disp_every=4):
    """smooth2d.m main loop. vert (N,2), tria (M,3); conn = the
    constrained (boundary) edges — default: boundary edges of the
    input triangulation. Returns (vert, tria)."""
    vert = np.array(vert, dtype=float)
    tria = np.array(tria, dtype=int)

    edge_all, counts = _tricon2(tria)
    if conn is None:
        conn = edge_all[counts < 2]
    conn = np.asarray(conn, dtype=int)

    # polygon bnds (smooth2d.m:133-151): the PSLG for containment
    node = vert.copy()
    pslg = conn.copy()

    # inflate bbox (smooth2d.m:154-166): 4 far-away box corners so
    # the (C)DT hull never clips the domain
    vmin = vert.min(axis=0)
    vmax = vert.max(axis=0)
    vdel = vmax - vmin
    vmin2 = vmin - 0.5 * vdel
    vmax2 = vmax + 0.5 * vdel
    vbox = np.array(
        [
            [vmin2[0], vmin2[1]],
            [vmax2[0], vmin2[1]],
            [vmax2[0], vmax2[1]],
            [vmin2[0], vmax2[1]],
        ]
    )
    vert = np.vstack([vert, vbox])

    for it in range(1, iters + 1):
        # inflate adj. (smooth2d.m:180)
        edge, counts = _tricon2(tria)
        nvrt = len(vert)
        nedg = len(edge)

        # compute scr. (smooth2d.m:187)
        oscr = _triscr2(vert, tria)

        # vertex degree via edge incidence
        vdeg = np.bincount(edge.ravel(), minlength=nvrt)
        free = vdeg == 0

        conn_mask = np.zeros(nvrt, dtype=bool)
        if len(conn):
            conn_mask[conn.ravel()] = True

        vold = vert.copy()
        for _isub in range(max(2, min(8, it))):
            # HFUN at vertices = mean adjacent edge length
            evec = vert[edge[:, 1]] - vert[edge[:, 0]]
            elen = np.sqrt((evec**2).sum(1))
            esum = np.zeros(nvrt)
            np.add.at(esum, edge[:, 0], elen)
            np.add.at(esum, edge[:, 1], elen)
            hvrt = esum / np.maximum(vdeg, 1)
            hvrt[free] = np.inf
            hmid = 0.5 * (hvrt[edge[:, 0]] + hvrt[edge[:, 1]])

            # relative edge extensions (smooth2d.m:216-222)
            with np.errstate(invalid="ignore"):
                scal = 1.0 - elen / hmid
            scal = np.clip(scal, -1.0, 1.0)
            scal = np.where(np.isfinite(scal), scal, 0.0)

            # projected points from each end (smooth2d.m:225-229)
            ipos = vert[edge[:, 0]] - 0.67 * scal[:, None] * evec
            jpos = vert[edge[:, 1]] + 0.67 * scal[:, None] * evec

            w = np.maximum(np.abs(scal), np.finfo(float).eps**0.75)

            vnew = np.zeros_like(vert)
            np.add.at(vnew, edge[:, 0], w[:, None] * ipos)
            np.add.at(vnew, edge[:, 1], w[:, None] * jpos)
            vsum = np.zeros(nvrt)
            np.add.at(vsum, edge[:, 0], w)
            np.add.at(vsum, edge[:, 1], w)
            vsum = np.maximum(vsum, np.finfo(float).eps**0.75)
            vnew = vnew / vsum[:, None]

            # fixed points (smooth2d.m:247-252)
            vnew[conn_mask] = vert[conn_mask]
            vnew[free] = vert[free]
            vert = vnew

        # hill-climber (smooth2d.m:262-306): unwind vertex updates
        # where the element score got worse and is below stol
        nscr = np.ones(len(tria))
        btri = np.ones(len(tria), dtype=bool)
        umax = 8
        for undo in range(1, umax + 1):
            nscr[btri] = _triscr2(vert, tria[btri])
            stol = min(0.90, 0.70 + it * 0.025)
            btri = (nscr <= stol) & (nscr < oscr)
            if not btri.any():
                break
            bvrt = np.zeros(len(vert), dtype=bool)
            bvrt[np.unique(tria[btri])] = True
            if undo != umax:
                bnew = 0.75**undo
            else:
                bnew = 0.0
            bold = 1.0 - bnew
            vert[bvrt] = bold * vold[bvrt] + bnew * vert[bvrt]
            btri = bvrt[tria].any(axis=1)
        oscr = nscr

        # convergence data (smooth2d.m:320-338)
        vdel_sq = ((vert - vold) ** 2).sum(1)
        evec = vert[edge[:, 1]] - vert[edge[:, 0]]
        elen = np.sqrt((evec**2).sum(1))
        esum = np.zeros(nvrt)
        np.add.at(esum, edge[:, 0], elen)
        np.add.at(esum, edge[:, 1], elen)
        hvrt = esum / np.maximum(vdeg, 1)
        hvrt[free] = np.inf
        hmid = 0.5 * (hvrt[edge[:, 0]] + hvrt[edge[:, 1]])
        scal = elen / hmid
        emid = 0.5 * (vert[edge[:, 0]] + vert[edge[:, 1]])

        # |deg|-based prune (smooth2d.m:341-346)
        keep = np.zeros(nvrt, dtype=bool)
        keep[vdeg > 4] = True
        keep[conn_mask] = True
        keep[free] = True

        # density control (smooth2d.m:349-386)
        lmax = 5.0 / 4.0
        lmin = 1.0 / lmax
        less = scal <= lmin
        more = scal >= lmax
        vbnd = conn_mask
        ebad = vbnd[edge[:, 0]] | vbnd[edge[:, 1]]
        less[ebad] = False
        more[ebad] = False

        # force as disjoint (smooth2d.m:364-379)
        lidx = np.where(less)[0]
        for epos in lidx:
            inod, jnod = edge[epos]
            if keep[inod] and keep[jnod]:
                keep[inod] = False
                keep[jnod] = False
            else:
                less[epos] = False
        ebad_m = keep[edge[less, 0]] & keep[edge[less, 1]]
        midx = np.where(less)[0][ebad_m]
        more[midx] = False

        # reindex vert/tria for the quality preserver
        # (smooth2d.m:389-415)
        redo = np.zeros(nvrt, dtype=int)
        kidx = np.where(keep)[0]
        litop = len(kidx)
        lsel = np.where(less)[0]
        redo[kidx] = np.arange(1, litop + 1)
        redo[edge[lsel, 0]] = litop + 1 + np.arange(len(lsel))
        redo[edge[lsel, 1]] = litop + 1 + np.arange(len(lsel))
        vnew2 = np.vstack([vert[keep], emid[lsel]])
        tnew = redo[tria] - 1
        ttmp = np.sort(tnew, axis=1)
        okay = (np.diff(ttmp, axis=1) != 0).all(axis=1) & (
            ttmp[:, 0] >= 0
        )
        tnew = tnew[okay]

        # quality preserver (smooth2d.m:418-441)
        nscr2 = _triscr2(vnew2, tnew)
        stol_q = 0.80
        tbad = (nscr2 < stol_q) & (nscr2 < oscr[okay])
        vbad = np.zeros(len(vnew2), dtype=bool)
        vbad[np.unique(tnew[tbad])] = True
        lsel2 = np.where(less)[0]
        e_r0 = redo[edge[lsel2, 0]] - 1
        e_r1 = redo[edge[lsel2, 1]] - 1
        ebad2 = vbad[e_r0] | vbad[e_r1]
        less[lsel2[ebad2]] = False
        keep[edge[lsel2[ebad2], :].ravel()] = True

        # final reindex (smooth2d.m:444-462)
        lsel = np.where(less)[0]
        vert = np.vstack(
            [vert[keep], emid[lsel], emid[np.where(more)[0]]]
        )
        old2new = np.full(nvrt, -1, dtype=int)
        kidx = np.where(keep)[0]
        old2new[kidx] = np.arange(len(kidx))
        old2new[edge[lsel, 0]] = len(kidx) + np.arange(len(lsel))
        old2new[edge[lsel, 1]] = len(kidx) + np.arange(len(lsel))
        cmap = old2new[conn]
        good = (cmap >= 0).all(axis=1)
        conn2 = cmap[good]

        # rebuild the (constrained) triangulation
        # (smooth2d.m:465-471)
        vert, tria = _deltri2(vert, conn2, node, pslg)
        # conn indices must be refreshed against the new vert order
        edge_all, counts = _tricon2(tria)
        conn = edge_all[counts < 2]
        conn_mask = np.zeros(len(vert), dtype=bool)
        if len(conn):
            conn_mask[conn.ravel()] = True

        # progress / convergence (smooth2d.m:474-491)
        with np.errstate(invalid="ignore", divide="ignore"):
            movei = vdel_sq / (
                np.where(np.isfinite(hvrt), hvrt, 1.0) ** 2
            )
        nmov = int((movei > vtol**2).sum())
        if disp_every and it % disp_every == 0:
            logger.info(
                "smooth2d iter %3d: MOVE(X)=%d DTRI(X)=%d",
                it, nmov, len(tria),
            )
        if nmov == 0:
            break

    # prune unused verts (smooth2d.m:499-511)
    vert, tria, _ = fix_mesh(vert, tria, delete_unused=True)
    return vert, tria
