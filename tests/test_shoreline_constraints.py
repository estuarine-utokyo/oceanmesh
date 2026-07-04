"""Tests for shoreline_to_fixed_points (high-fidelity port, #264)."""

import numpy as np

from oceanmesh.shoreline_constraints import (
    _resample_at_local_h,
    _split_nan_delimited,
    shoreline_to_fixed_points,
)


class _FakeShoreline:
    def __init__(self, mainland, inner):
        self.mainland = mainland
        self.inner = inner


def _const_h(h):
    return lambda pts: np.full(len(np.atleast_2d(pts)), h)


def test_split_nan_delimited():
    nan = np.nan
    arr = np.array([
        [0, 0], [1, 0], [2, 0], [nan, nan],
        [5, 5], [6, 5], [nan, nan],
        [9, 9],  # single point: dropped
    ])
    polys = _split_nan_delimited(arr)
    assert len(polys) == 2
    assert len(polys[0]) == 3 and len(polys[1]) == 2


def test_resample_spacing_matches_local_h():
    pts = np.column_stack([np.linspace(0.0, 10.0, 101), np.zeros(101)])
    out = _resample_at_local_h(pts, _const_h(1.0), min_points=5)
    d = np.hypot(*np.diff(out, axis=0).T)
    assert abs(len(out) - 11) <= 1
    assert d.min() > 0.4 and d.max() < 1.6
    # Endpoints preserved exactly (open polyline).
    assert np.allclose(out[0], [0.0, 0.0])
    assert np.allclose(out[-1], [10.0, 0.0])


def test_short_polyline_skipped():
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert _resample_at_local_h(pts, _const_h(1.0), min_points=5) is None


def test_closed_ring_has_no_duplicate_endpoint():
    th = np.linspace(0, 2 * np.pi, 200)
    ring = np.column_stack([np.cos(th), np.sin(th)])
    out = _resample_at_local_h(ring, _const_h(0.2), min_points=5)
    assert not np.allclose(out[0], out[-1])


def test_driver_collects_mainland_and_inner_and_dedupes():
    nan = np.nan
    mainland = np.array([[0, 0], [10, 0], [nan, nan]])
    th = np.linspace(0, 2 * np.pi, 100)
    island = np.column_stack([3 + np.cos(th), 3 + np.sin(th)])
    inner = np.vstack([island, [[nan, nan]]])
    shore = _FakeShoreline(mainland, inner)
    pfix = shoreline_to_fixed_points(shore, _const_h(0.5), min_points=5)
    assert len(pfix) > 20
    # No near-duplicates below the dedupe tolerance.
    from scipy.spatial import cKDTree

    assert len(cKDTree(pfix).query_pairs(r=0.5 * 0.2)) == 0
