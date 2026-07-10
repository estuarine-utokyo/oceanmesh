"""Local patch remeshing (OM2D Example_11 workflow):
extract_subdomain -> regenerate inside -> stitch back.

The regeneration engine is selectable:
- "jigsaw": jigsawpy (Engwirda) frontal-Delaunay — the successor
  of the mesh2d engine OM2D calls in mesh2dgen.m. Best quality;
  follows the sizing function tightly. Optional dependency.
- "distmesh": this package's OM2D-parity generator with the patch
  boundary held fixed (pfix/egfix). Statistically closest to the
  mesh2d output (same L0mult family).
"""
import logging

import numpy as np
import scipy.spatial

from . import edges as om_edges
from .clean import _external_topology
from .fix_mesh import fix_mesh
from .geometry import inpoly2

logger = logging.getLogger(__name__)

__all__ = ["extract_subdomain", "remesh_patch"]


def _ring_edges(ring):
    return om_edges.get_poly_edges(
        np.vstack([ring, [[np.nan, np.nan]]]))


def extract_subdomain(points, cells, polygon, keep_inverse=False):
    """Elements whose centroid lies inside `polygon` (msh
    extract_subdomain semantics)."""
    ring = np.vstack([polygon, polygon[:1]])
    e = _ring_edges(ring)
    cen = points[cells].mean(axis=1)
    ins, _ = inpoly2(cen,
                     np.nan_to_num(np.vstack([ring,
                                              [[np.nan, np.nan]]])),
                     e)
    return cells[~ins] if keep_inverse else cells[ins]


def _boundary_loops(points, cells):
    bedges, _ = _external_topology(points, cells)
    bedges = np.asarray(bedges, dtype=int)
    adj = {}
    for a, b in bedges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    unused = {tuple(sorted(ed)) for ed in bedges}
    loops = []
    while unused:
        e0 = unused.pop()
        loop = [e0[0], e0[1]]
        while True:
            cur = loop[-1]
            nxts = [n for n in adj[cur]
                    if tuple(sorted((cur, n))) in unused]
            if not nxts:
                break
            unused.discard(tuple(sorted((cur, nxts[0]))))
            loop.append(nxts[0])
            if nxts[0] == loop[0]:
                break
        if loop[0] == loop[-1]:
            loop = loop[:-1]
        loops.append(np.asarray(loop, dtype=int))
    loops.sort(key=len, reverse=True)
    return loops


def _remesh_jigsaw(bnd, sizing, crs_proj):
    import jigsawpy

    x, y = crs_proj.transform(bnd[:, 0], bnd[:, 1])
    bm = np.column_stack([x, y])
    opts = jigsawpy.jigsaw_jig_t()
    geom = jigsawpy.jigsaw_msh_t()
    geom.mshID = "euclidean-mesh"
    geom.ndims = 2
    geom.vert2 = np.array([((px, py), 0) for px, py in bm],
                          dtype=jigsawpy.jigsaw_msh_t.VERT2_t)
    n = len(bm)
    geom.edge2 = np.array([((i, (i + 1) % n), 0) for i in range(n)],
                          dtype=jigsawpy.jigsaw_msh_t.EDGE2_t)
    hmat = jigsawpy.jigsaw_msh_t()
    hmat.mshID = "euclidean-grid"
    hmat.ndims = 2
    xv = np.linspace(bm[:, 0].min() - 5e3, bm[:, 0].max() + 5e3, 512)
    yv = np.linspace(bm[:, 1].min() - 5e3, bm[:, 1].max() + 5e3, 512)
    X, Y = np.meshgrid(xv, yv, indexing="ij")
    lon, lat = crs_proj.transform(X.ravel(), Y.ravel(),
                                  direction="INVERSE")
    hv = np.asarray(sizing.eval(np.column_stack([lon, lat]))) * 111e3
    hmat.xgrid = xv
    hmat.ygrid = yv
    hmat.value = np.ascontiguousarray(
        hv.reshape(X.shape).T, dtype=np.float64)
    opts.hfun_scal = "absolute"
    opts.hfun_hmax = float(np.nanmax(hv))
    opts.hfun_hmin = float(np.nanmin(hv))
    opts.mesh_dims = 2
    mesh = jigsawpy.jigsaw_msh_t()
    jigsawpy.lib.jigsaw(opts, geom, mesh, None, hmat)
    lon2, lat2 = crs_proj.transform(mesh.vert2["coord"][:, 0],
                                    mesh.vert2["coord"][:, 1],
                                    direction="INVERSE")
    return np.column_stack([lon2, lat2]), mesh.tria3["index"]


def _remesh_distmesh(bnd, sizing, seed=0):
    from .mesh_generator import generate_mesh
    from .signed_distance_function import Domain

    tree = scipy.spatial.cKDTree(bnd)
    ring = np.vstack([bnd, bnd[:1]])
    e = _ring_edges(ring)
    poly_nn = np.nan_to_num(np.vstack([ring, [[np.nan, np.nan]]]))

    def _fd(q):
        q = np.asarray(q, dtype=float)
        d, _ = tree.query(q, k=1, workers=-1)
        ins, _ = inpoly2(q, poly_nn, e)
        return np.where(ins, -d, d)

    bb = (float(bnd[:, 0].min()), float(bnd[:, 0].max()),
          float(bnd[:, 1].min()), float(bnd[:, 1].max()))
    dom = Domain(bb, _fd, covering=_fd, crs="EPSG:4326")
    dom.boubox_ring = bnd
    n = len(bnd)
    egfix = np.column_stack([np.arange(n), (np.arange(n) + 1) % n])
    return generate_mesh(dom, sizing, pfix=bnd, egfix=egfix,
                         max_iter=50, seed=seed, cleanup="none")


def remesh_patch(points, cells, polygon, sizing, engine="jigsaw",
                 seed=0, weld_tol=1e-8):
    """Regenerate the mesh inside `polygon` and stitch it back.

    Returns (new_points, new_cells).
    """
    from pyproj import Transformer

    t_in = extract_subdomain(points, cells, polygon)
    t_out = extract_subdomain(points, cells, polygon,
                              keep_inverse=True)
    if len(t_in) == 0:
        logger.warning("remesh_patch: polygon contains no elements")
        return points, cells
    loops = _boundary_loops(points, t_in)
    bnd = points[loops[0]]

    if engine == "jigsaw":
        try:
            lo0 = 0.5 * (bnd[:, 0].min() + bnd[:, 0].max())
            la0 = 0.5 * (bnd[:, 1].min() + bnd[:, 1].max())
            proj = Transformer.from_crs(
                "EPSG:4326",
                f"+proj=tmerc +lon_0={lo0} +lat_0={la0} "
                "+ellps=WGS84 +units=m", always_xy=True)
            pp, tt = _remesh_jigsaw(bnd, sizing, proj)
        except ImportError:
            logger.warning("jigsawpy unavailable — falling back to "
                           "the distmesh engine")
            pp, tt = _remesh_distmesh(bnd, sizing, seed=seed)
    else:
        pp, tt = _remesh_distmesh(bnd, sizing, seed=seed)

    keep_idx = np.unique(t_out.reshape(-1))
    remap = -np.ones(len(points), dtype=int)
    remap[keep_idx] = np.arange(len(keep_idx))
    P = np.vstack([points[keep_idx], pp])
    T = np.vstack([remap[t_out], np.asarray(tt, dtype=int)
                   + len(keep_idx)])
    tr_ = scipy.spatial.cKDTree(P)
    parent = np.arange(len(P))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in tr_.query_pairs(weld_tol):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    root = np.array([find(i) for i in range(len(P))])
    uniq, inv = np.unique(root, return_inverse=True)
    P2 = P[uniq]
    T2 = inv[root[T.reshape(-1)]].reshape(-1, 3)
    P2, T2, _ = fix_mesh(P2, T2, dim=2, delete_unused=True)
    logger.info(
        f"remesh_patch[{engine}]: {len(t_in)} -> "
        f"{len(T2) - len(t_out)} patch elements, stitched "
        f"NP={len(P2):,}")
    return P2, T2
