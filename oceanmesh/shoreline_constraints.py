"""High-fidelity shoreline constraints (port of OceanMesh2D V6.0 #264).

OceanMesh2D's ``geodata(..., 'high_fidelity', 1)`` resamples the
shoreline polylines at the local target edge length and injects the
result as fixed points/edges into the DistMesh iteration, so the
final mesh boundary lies EXACTLY on the (resampled) shoreline.

This module ports the point half of that feature: it turns a
:class:`oceanmesh.Shoreline`'s ``mainland`` + ``inner`` polylines into
a ``pfix`` array for :func:`oceanmesh.generate_mesh`, which already
re-snaps fixed points on every iteration, zeroes their forces, and —
via ``clean(..., pfix=...)`` / ``laplacian2(..., pfix=...)`` —
protects them during cleaning. (Fixed EDGES need a constrained
Delaunay path in the C++ layer and are not ported yet; in practice a
chain of fixed points at local-h spacing recovers the boundary edges.)

Mirrors the MATLAB behaviour where it matters:

* resampling at the LOCAL size from the edge-length function
  (``mesh1d.m``'s greedy arc-length marching is used instead of the
  1-D force iteration — the spacing target is identical);
* polylines shorter than ``min_points`` local elements are skipped
  (``mesh1d.m:37-45``);
* closing duplicates are dropped and near-coincident fixed points are
  deduplicated (``meshgen.m:592`` errors on duplicates).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["shoreline_to_fixed_points"]


def _split_nan_delimited(arr):
    """Split an (N, 2) NaN-delimited vertex array into polylines."""
    if arr is None or len(arr) == 0:
        return []
    arr = np.asarray(arr, dtype=float)
    isnan = np.isnan(arr[:, 0]) | np.isnan(arr[:, 1])
    polylines = []
    start = 0
    for i in range(len(arr)):
        if isnan[i]:
            if i - start >= 2:
                polylines.append(arr[start:i])
            start = i + 1
    if len(arr) - start >= 2:
        polylines.append(arr[start:])
    return polylines


def _resample_at_local_h(pts, fh, min_points):
    """Greedy arc-length marching: place points along ``pts`` spaced by
    the local target size ``fh``. Returns None when the polyline is
    shorter than ``min_points`` local elements."""
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    keep = seg_len > 0
    if not keep.all():
        pts = np.vstack([pts[:1], pts[1:][keep]])
        seg = np.diff(pts, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
    if len(pts) < 2:
        return None
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])

    h_mid = float(np.asarray(fh(pts[len(pts) // 2 : len(pts) // 2 + 1])).ravel()[0])
    if not np.isfinite(h_mid) or h_mid <= 0:
        return None
    if total < min_points * h_mid:
        return None

    s_list = [0.0]
    s = 0.0
    while True:
        xy = np.array([
            np.interp(s, cum, pts[:, 0]),
            np.interp(s, cum, pts[:, 1]),
        ])
        h = float(np.asarray(fh(xy[None, :])).ravel()[0])
        if not np.isfinite(h) or h <= 0:
            h = h_mid
        s_next = s + h
        if s_next >= total - 0.5 * h:
            break
        s_list.append(s_next)
        s = s_next
    if len(s_list) + 1 < min_points:
        return None
    s_arr = np.asarray(s_list + [total])
    out = np.column_stack([
        np.interp(s_arr, cum, pts[:, 0]),
        np.interp(s_arr, cum, pts[:, 1]),
    ])
    closed = np.allclose(pts[0], pts[-1])
    if closed:
        out = out[:-1]
    return out


def shoreline_to_fixed_points(
    shoreline,
    edge_length,
    *,
    min_points=5,
    dedupe_tol_frac=0.25,
):
    """Resample a :class:`Shoreline`'s mainland + inner polylines at the
    local target edge length and return the points as a ``pfix`` array
    for :func:`generate_mesh`.

    Parameters
    ----------
    shoreline:
        ``oceanmesh.Shoreline`` (uses ``mainland`` and ``inner``).
    edge_length:
        Sizing :class:`Grid` (its ``eval`` is used) or a callable
        mapping ``(n, 2)`` points to local sizes in coordinate units.
    min_points:
        Polylines shorter than this many local elements are skipped,
        mirroring OceanMesh2D's ``mesh1d``.
    dedupe_tol_frac:
        Fixed points closer than this fraction of the local size are
        merged (the MATLAB implementation hard-errors on duplicates).

    Returns
    -------
    ``(n, 2)`` float array (possibly empty) for ``generate_mesh(pfix=...)``.
    """
    fh = getattr(edge_length, "eval", edge_length)
    polylines = []
    for attr in ("mainland", "inner"):
        polylines.extend(_split_nan_delimited(getattr(shoreline, attr, None)))

    chunks = []
    n_skipped = 0
    for poly in polylines:
        out = _resample_at_local_h(poly, fh, min_points)
        if out is None:
            n_skipped += 1
        else:
            chunks.append(out)
    if not chunks:
        logger.info(
            "shoreline_to_fixed_points: no polylines survived "
            f"({n_skipped} below {min_points} local elements)"
        )
        return np.empty((0, 2))

    pfix = np.vstack(chunks)
    # Deduplicate near-coincident points (chain junctions / touching
    # polylines) — generate_mesh snaps each pfix to its closest vertex,
    # so coincident pairs would fight over one vertex.
    h_at = np.asarray(fh(pfix)).ravel()
    h_at = np.where(np.isfinite(h_at) & (h_at > 0), h_at, np.nanmedian(h_at))
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pfix)
        pairs = tree.query_pairs(r=float(np.nanmax(h_at)) * dedupe_tol_frac)
        drop = set()
        for i, j in pairs:
            tol = dedupe_tol_frac * min(h_at[i], h_at[j])
            if np.hypot(*(pfix[i] - pfix[j])) < tol and j not in drop:
                drop.add(j)
        if drop:
            keep = np.setdiff1d(np.arange(len(pfix)), np.fromiter(drop, int))
            pfix = pfix[keep]
    except ImportError:  # pragma: no cover
        pass

    logger.info(
        f"shoreline_to_fixed_points: {len(pfix)} fixed points from "
        f"{len(chunks)} polylines ({n_skipped} skipped as sub-resolution)"
    )
    return pfix
