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


def _my_interpm(line, spacing):
    """1:1 port of utilities/my_interpm.m: insert-only
    densification with the PER-COMPONENT (Chebyshev) segment
    measure — nin = ceil(max(|dlat|,|dlon|)/maxdiff) - 1 interior
    points per segment (my_interpm.m:14-16), NOT Euclidean."""
    out = []
    for a, b in zip(line[:-1], line[1:]):
        ni = int(np.ceil(
            max(abs(b[0] - a[0]), abs(b[1] - a[1])) / spacing
        )) - 1
        if ni <= 0:
            out.append(a)
        else:
            xs = np.linspace(a[0], b[0], ni + 2)
            ys = np.linspace(a[1], b[1], ni + 2)
            out.extend(np.column_stack([xs, ys])[:ni + 1])
    out.append(line[-1])
    return np.asarray(out)


def build_weir_geometry(crestline, width_m, spacing_m=None,
                        geographic=True):
    """1:1 port of GenerateWeirGeometry.m. Returns (pfix, egfix,
    pairs): the racetrack outline points/ring edges for the mesh
    generator, and the (K, 4) array of paired [x_up, y_up, x_dn,
    y_dn] points across the crest (ADCIRC ibconn).

    NB the .m normalizes the offset directions by the MATRIX
    2-norm (``u = v./norm(v)``, GenerateWeirGeometry.m:10), NOT
    row-wise, so the face separation varies along the crest and
    is generally much smaller than ``width_m``; replicated
    verbatim for parity.
    """
    f = _DEG if geographic else 1.0
    width = 0.5 * width_m * f
    spacing = (spacing_m if spacing_m else 2.0 * width_m) * f
    line = _my_interpm(np.asarray(crestline, float), spacing)
    dx = np.gradient(line[:, 0])
    dy = np.gradient(line[:, 1])
    v = np.column_stack([-dy, dx])
    u = v / np.linalg.norm(v, 2)
    above = line[1:-1] + width * u[1:-1]
    below = line[1:-1] - width * u[1:-1]
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
