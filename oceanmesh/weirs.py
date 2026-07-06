"""Weir/barrier geometry (port plan P4).

Port of OceanMesh2D GenerateWeirGeometry: a weir crestline becomes
a thin "racetrack" outline (pfix + egfix) meshed as an internal
hole, plus the across-weir node pairing (ibconn) used for ADCIRC
ibtype-24 internal barrier boundaries.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["build_weir_geometry", "match_weir_nodes"]

_DEG = 1.0 / 111e3


def _resample(line, spacing):
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.ceil(s[-1] / spacing)) + 1, 2)
    si = np.linspace(0.0, s[-1], n)
    return np.column_stack(
        [np.interp(si, s, line[:, 0]), np.interp(si, s, line[:, 1])]
    )


def build_weir_geometry(crestline, width_m, spacing_m=None,
                        geographic=True):
    """Port of GenerateWeirGeometry. Returns (pfix, egfix, pairs):
    the closed racetrack outline points/edges to pass to the mesh
    generator, and the (K, 4) array of paired [x_up, y_up, x_dn,
    y_dn] points across the crest (ADCIRC ibconn)."""
    f = _DEG if geographic else 1.0
    width = 0.5 * width_m * f
    spacing = (spacing_m if spacing_m else 2.0 * width_m) * f
    line = _resample(np.asarray(crestline, float), spacing)
    dx = np.gradient(line[:, 0])
    dy = np.gradient(line[:, 1])
    nrm = np.column_stack([-dy, dx])
    nrm /= np.maximum(
        np.linalg.norm(nrm, axis=1)[:, None], 1e-30
    )
    above = line[1:-1] + width * nrm[1:-1]
    below = line[1:-1] - width * nrm[1:-1]
    pairs = np.hstack([above, below])
    pfix = np.vstack([line[0], below, line[-1], above[::-1]])
    n = len(pfix)
    egfix = np.column_stack(
        [np.arange(n), np.roll(np.arange(n), -1)]
    )
    logger.info(
        f"weir: {len(line)} crest pts -> {n} outline pts, "
        f"{len(pairs)} pairs (width {width_m} m)"
    )
    return pfix, egfix, pairs


def match_weir_nodes(points, pairs, tol=None):
    """Map the across-weir point pairs onto MESH node ids after
    generation (the racetrack points are pfix, so they survive
    exactly). Returns (front_ids, back_ids)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(points, float))
    pairs = np.asarray(pairs, float)
    d1, front = tree.query(pairs[:, :2])
    d2, back = tree.query(pairs[:, 2:])
    if tol is not None and (d1.max() > tol or d2.max() > tol):
        raise ValueError(
            f"weir nodes drifted: {max(d1.max(), d2.max()):.3g} > "
            f"{tol}"
        )
    return front.astype(int), back.astype(int)
