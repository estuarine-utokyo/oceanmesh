"""Bathymetry-on-mesh utilities (port plan P3).

Ports of OceanMesh2D msh.interp (GridData cell-averaging core),
msh.lim_bathy_slope (utilities/limgrad.m graph limiter),
utilities/Unstruc_Bath_Slope.m and geodata.extractContour.
Depth convention: positive down (ADCIRC/FVCOM style).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "contour_to_shapefile",
    "extract_contour",
    "interp_bathymetry",
    "lim_bathy_slope",
    "unstructured_slopes",
]


def _local_sizes(points, cells):
    from .cfl import get_bar_lengths

    bars, L = get_bar_lengths(points, cells, geographic=False)
    dx = np.full(len(points), np.inf)
    np.minimum.at(dx, bars[:, 0], L)
    np.minimum.at(dx, bars[:, 1], L)
    return dx


def interp_bathymetry(
    points,
    cells,
    dem,
    method="cell-averaging",
    N=1.0,
    min_depth=None,
    max_depth=None,
    invert=True,
    nan_fill=True,
):
    """Interpolate DEM elevations onto mesh nodes (port of
    msh.interp / GridData).

    'cell-averaging' (OM2D 'CA', the default there too): each node
    averages all DEM cells within ``N * local_mesh_size / 2`` of it;
    nodes whose stencil is empty fall back to the DEM interpolant.
    ``invert=True`` returns positive-down depth from a negative-down
    DEM. ``min_depth``/``max_depth`` clamp the RESULT (positive
    down), matching OM2D's mindepth/maxdepth.
    """
    p = np.asarray(points, dtype=float)
    t = np.asarray(cells, dtype=int)

    if method == "cell-averaging":
        # Faithful GridData.m 'CA' (GridData.m:257-385):
        # per-node stencil = the bounding box of the CONNECTED
        # ELEMENT CENTROIDS mirrored about the node
        # (pcon_max/pcon_min, :270-285), enlarged by N; DEM cells
        # inside that index window are averaged, working on the
        # DEM's own axis VECTORS (valid for non-uniform lattices);
        # DEM_Z = -values (positive-down) and min/max depth clamp
        # the DEM CELLS BEFORE averaging (:311-321).
        xa, ya = dem.create_vectors()
        xa = np.asarray(xa, dtype=float)
        ya = np.asarray(ya, dtype=float)
        zz = np.asarray(dem.values, dtype=float)
        zd = -zz if invert else np.array(zz, copy=True)
        if min_depth is not None:
            zd[zd < min_depth] = min_depth
        if max_depth is not None:
            zd[zd > max_depth] = max_depth

        pmid = p[t].mean(axis=1)
        # per-node bbox of connected centroids
        cmin = np.full((len(p), 2), np.inf)
        cmax = np.full((len(p), 2), -np.inf)
        for k in range(3):
            np.minimum.at(cmin, t[:, k], pmid)
            np.maximum.at(cmax, t[:, k], pmid)
        lone = ~np.isfinite(cmin[:, 0])
        cmin[lone] = p[lone]
        cmax[lone] = p[lone]
        # mirror about the vertex (GridData.m:277-281)
        pcon_max = np.maximum(cmax, 2 * p - cmin)
        pcon_min = np.minimum(cmin, 2 * p - cmax)
        # enlarge by N (GridData.m:284-285)
        pcon_max = N * pcon_max + (1 - N) * p
        pcon_min = N * pcon_min + (1 - N) * p

        sx = 1 if xa[-1] >= xa[0] else -1
        sy = 1 if ya[-1] >= ya[0] else -1
        xs = xa[::sx]
        ys = ya[::sy]

        def _win(vec, lo, hi):
            i0 = np.searchsorted(vec, np.minimum(lo, hi)) - 1
            i1 = np.searchsorted(vec, np.maximum(lo, hi))
            i0 = np.clip(i0, 0, len(vec) - 1)
            i1 = np.clip(i1, i0 + 1, len(vec))
            return i0, i1

        ix0, ix1 = _win(xs, pcon_min[:, 0], pcon_max[:, 0])
        iy0, iy1 = _win(ys, pcon_min[:, 1], pcon_max[:, 1])

        zs = zd[::sx, ::sy]
        valid = np.isfinite(zs)
        zfill = np.where(valid, zs, 0.0)
        # O(1) window means via 2-D cumulative sums
        S = np.zeros((zs.shape[0] + 1, zs.shape[1] + 1))
        C = np.zeros_like(S)
        S[1:, 1:] = np.cumsum(np.cumsum(zfill, axis=0), axis=1)
        C[1:, 1:] = np.cumsum(np.cumsum(valid, axis=0), axis=1)

        def wsum(A, i0, i1, j0, j1):
            return (A[i1, j1] - A[i0, j1] - A[i1, j0] + A[i0, j0])

        ssum = wsum(S, ix0, ix1, iy0, iy1)
        cnt = wsum(C, ix0, ix1, iy0, iy1)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = ssum / cnt
        b[cnt == 0] = np.nan
    elif method in ("nearest", "linear"):
        z = np.asarray(dem.eval(p), dtype=float)
        b = -z if invert else z
        if min_depth is not None:
            b = np.maximum(b, min_depth)
        if max_depth is not None:
            b = np.minimum(b, max_depth)
    else:
        raise ValueError(f"unknown method {method}")

    missing = ~np.isfinite(b)
    if missing.any() and nan_fill:
        # GridData nanfill: nearest valid mesh node (:389-394)
        from scipy.spatial import cKDTree

        good = np.where(np.isfinite(b))[0]
        if len(good):
            _, j = cKDTree(p[good]).query(p[missing])
            b[missing] = b[good][j]
    logger.info(
        f"interp_bathymetry[{method}]: depth p10/p50/p90 = "
        f"{np.nanpercentile(b, [10, 50, 90]).round(2)}"
    )
    return b


def lim_bathy_slope(points, cells, depth, dfdx=0.10, imax=100,
                    geographic=False):
    """Port of msh.lim_bathy_slope / utilities/limgrad.m: limit
    |b(n2) - b(n1)| <= dfdx * L over every mesh edge by lowering the
    HIGHER value, iterating an active set until convergence."""
    from .cfl import get_bar_lengths

    b = np.asarray(depth, dtype=float).copy()
    bars, L = get_bar_lengths(points, np.asarray(cells),
                              geographic=geographic)
    ftol = np.nanmin(np.abs(b)) * np.sqrt(np.finfo(float).eps)
    for it in range(imax):
        hi = np.maximum(b[bars[:, 0]], b[bars[:, 1]])
        lo = np.minimum(b[bars[:, 0]], b[bars[:, 1]])
        excess = hi - lo - dfdx * L
        active = excess > ftol
        if not active.any():
            logger.info(f"lim_bathy_slope converged in {it} passes")
            break
        cap = lo[active] + dfdx * L[active]
        a_hi = np.where(b[bars[active, 0]] >= b[bars[active, 1]],
                        bars[active, 0], bars[active, 1])
        np.minimum.at(b, a_hi, cap)
    return b


def unstructured_slopes(points, cells, depth):
    """Port of utilities/Unstruc_Bath_Slope.m: nodal bathymetric
    slope components as area-weighted averages of the per-element
    linear gradients."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(cells, dtype=int)
    b = np.asarray(depth, dtype=float)
    x1, y1 = p[t[:, 0]].T
    x2, y2 = p[t[:, 1]].T
    x3, y3 = p[t[:, 2]].T
    twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    twoA[twoA == 0] = np.finfo(float).eps
    b1, b2, b3 = b[t[:, 0]], b[t[:, 1]], b[t[:, 2]]
    gx = ((y2 - y3) * b1 + (y3 - y1) * b2 + (y1 - y2) * b3) / twoA
    gy = ((x3 - x2) * b1 + (x1 - x3) * b2 + (x2 - x1) * b3) / twoA
    w = np.abs(twoA)
    bx = np.zeros(len(p))
    by = np.zeros(len(p))
    ws = np.zeros(len(p))
    for k in range(3):
        np.add.at(bx, t[:, k], gx * w)
        np.add.at(by, t[:, k], gy * w)
        np.add.at(ws, t[:, k], w)
    ws[ws == 0] = np.finfo(float).eps
    return bx / ws, by / ws


def extract_contour(dem, level=0.0):
    """Port of geodata.extractContour: iso-elevation contours of the
    DEM as a NaN-delimited (N, 2) lon/lat polyline array (usable as
    a PSLG / shoreline source)."""
    from skimage import measure

    zg = np.asarray(dem.values, dtype=float)
    contours = measure.find_contours(zg, level)
    xv, yv = dem.create_vectors()
    segs = []
    for c in contours:
        # c is (row=ix, col=iy) in grid index space
        xs = np.interp(c[:, 0], np.arange(len(xv)), xv)
        ys = np.interp(c[:, 1], np.arange(len(yv)), yv)
        segs.append(np.column_stack([xs, ys]))
        segs.append(np.array([[np.nan, np.nan]]))
    if not segs:
        return np.empty((0, 2))
    return np.vstack(segs)


def contour_to_shapefile(dem, level, out_path, min_length=10):
    """Floodplain workflow helper (OM2D fp): write the DEM contour
    at ``level`` (e.g. +10 m upland limit) as polyline shapefile
    usable as a Shoreline source, so the domain extends over the
    floodplain up to that contour. MergeFP == merge_meshes,
    interpFP == interp_bathymetry."""
    import geopandas as gpd
    from shapely.geometry import LineString

    arr = extract_contour(dem, level=level)
    lines = []
    run = []
    for q in arr:
        if np.isnan(q[0]):
            if len(run) >= max(2, min_length):
                lines.append(LineString(run))
            run = []
        else:
            run.append(tuple(q))
    if len(run) >= max(2, min_length):
        lines.append(LineString(run))
    gdf = gpd.GeoDataFrame(geometry=lines, crs=dem.crs)
    gdf.to_file(out_path)
    logger.info(
        f"contour_to_shapefile: {len(lines)} polylines at "
        f"z={level} -> {out_path}"
    )
    return out_path
