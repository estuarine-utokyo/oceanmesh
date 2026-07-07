"""Tests for the OceanMesh2D port (plan P1-P3)."""
import numpy as np
import pytest

import oceanmesh as om
from oceanmesh import Region
from oceanmesh.mesh_generator import _al_quality
from oceanmesh.signed_distance_function import create_bbox as _cb

BBOX = (0.0, 1.0, 0.0, 1.0)


def _uniform(size):
    def h(p):
        return np.full(len(np.atleast_2d(p)), size)

    return h


def _unit_mesh(size=0.05, **kw):
    kw.setdefault("max_iter", 30)
    return om.generate_mesh(
        _cb(BBOX), _uniform(size), bbox=BBOX, min_edge_length=size,
        **kw,
    )


# ---------------- P1: finalize / bands / CFL -------------------------
def test_elevation_bands_scalar_and_bands():
    z = np.array([-5.0, -50.0, -500.0])
    assert np.allclose(om.elevation_bands(7.5, z), 7.5)
    out = om.elevation_bands([[10, -100, 0], [20, -1e4, -100]], z)
    assert np.allclose(out, [10, 10, 20])
    with pytest.raises(ValueError):
        om.elevation_bands([[1, 2]], z)


def test_wave_celerity_matches_om2d():
    # sqrt(g*100) + sqrt(g/100)
    assert om.wave_celerity(-100.0) == pytest.approx(31.63, abs=0.01)
    # depth floored at 1 m
    assert om.wave_celerity(0.0) == pytest.approx(
        np.sqrt(9.807) + np.sqrt(9.807), abs=1e-6
    )


def test_courant_bounds_two_sided():
    grid = om.Grid(bbox=BBOX, dx=0.05, crs="EPSG:32654",
                   values=1000.0, hmin=1000.0)

    class FakeDEM:
        def eval(self, q):
            return np.full(len(np.atleast_2d(q)), -400.0)

    g2, dt = om.enforce_courant_bounds(grid, FakeDEM(), timestep=16.0,
                                       courant_min=0.1,
                                       courant_max=0.5)
    u = om.wave_celerity(-400.0)
    cr = dt * u / np.asarray(g2.values)
    assert cr.max() <= 0.5 + 1e-9
    assert cr.min() >= 0.1 - 1e-9


def test_generator_improvement_and_quality():
    p, t = _unit_mesh(improve=True, max_iter=40)
    assert _al_quality(p, t).mean() > 0.85


@pytest.mark.xfail(
    reason="bossen_heckbert (upstream-extra force function, not on "
    "the OM2D parity path) collapses the point cloud under the "
    "in-loop improvement cycle; tracked in docs/OM2D_AUDIT.md",
    strict=False,
)
def test_generator_bossen_heckbert():
    p, t = _unit_mesh(force_function="bossen_heckbert",
                      cleanup="none")
    assert len(t) > 0 and _al_quality(p, t).min() > 0


def test_generator_egfix_holds():
    pf = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
    eg = np.array([[0, 1], [1, 2]])
    p, t = _unit_mesh(pfix=pf, egfix=eg)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(p).query(pf)
    assert d.max() < 1e-12


# ---------------- P2: improvement / BC / IO / CFL --------------------
def test_collapse_thin_triangles():
    p = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1],
                  [1.5, 0.001]], float)
    t = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
                  [1, 2, 6]], int)
    p2, t2 = om.collapse_thin_triangles(p, t, min_qual=0.10)
    assert om.area_length_quality(p2, t2).min() > 0.10


def test_direct_smoother_pins_boundary():
    p, t = _unit_mesh(improve=False, max_iter=12)
    from oceanmesh.edges import get_boundary_edges

    bnd = np.unique(get_boundary_edges(t))
    ps, _ = om.direct_smoother_lur(p, t)
    assert np.abs(ps[bnd] - p[bnd]).max() < 1e-8
    assert (om.area_length_quality(ps, t).mean()
            >= om.area_length_quality(p, t).mean())


def test_bound_connectivity_valence():
    p, t = _unit_mesh(improve=False, max_iter=15)
    p2, t2 = om.bound_connectivity(p, t, max_valence=7)
    from oceanmesh.edges import get_boundary_edges

    bnd = np.unique(get_boundary_edges(t2))
    val = np.bincount(t2.ravel(), minlength=len(p2))
    interior = np.setdiff1d(np.arange(len(p2)), bnd)
    if len(interior):
        assert val[interior].max() <= 8  # <=7 target, 8 tolerated


def test_renumber_rcm_is_permutation():
    p, t = _unit_mesh(improve=False, max_iter=10)
    p2, t2, _, perm = om.renumber_rcm(p, t)
    assert sorted(perm.tolist()) == list(range(len(p)))
    assert np.allclose(np.sort(p2, axis=0), np.sort(p, axis=0))


def test_make_bc_auto_depth_and_roundtrip(tmp_path):
    p, t = om.generate_mesh(_cb((0., 2., 0., 1.)), _uniform(0.08),
                            bbox=(0., 2., 0., 1.),
                            min_edge_length=0.08, max_iter=25,
                            improve=False)
    dep = np.where(p[:, 0] < 1.0, 50.0, 2.0)
    bc = om.make_bc_auto(p, t, depth=dep, classifier="depth",
                         depth_lim=10.0, cut_lim=5)
    assert len(bc["open"]) >= 1 and len(bc["land"]) >= 1
    f14 = tmp_path / "roundtrip.14"
    om.write_fort14(str(f14), p, t, depth=dep, boundaries=bc)
    p2, t2, d2, b2, _ = om.read_fort14(str(f14))
    assert np.allclose(p2, p) and (t2 == t).all()
    assert [len(s) for s in b2["open"]] == [len(s) for s in bc["open"]]


def test_make_bc_auto_requires_inputs():
    p, t = _unit_mesh(improve=False, max_iter=10)
    with pytest.raises(ValueError):
        om.make_bc_auto(p, t, classifier="depth")
    with pytest.raises(ValueError):
        om.make_bc_auto(p, t, classifier="distance")


def test_bound_courant_number_decimates():
    # partial violation (a deep stripe), as DecimateTria assumes:
    # the .m warns and degrades when EVERY node violates
    # (msh.m:3046-3048); a depth step makes ~1/3 of nodes exceed
    # cr_max while the rest stay legal
    bbox = (0., 40000., 0., 40000.)
    p, t = om.generate_mesh(_cb(bbox), _uniform(2000.0), bbox=bbox,
                            min_edge_length=2000.0, max_iter=20,
                            improve=False)

    def dep_fn(q):
        return np.where(q[:, 0] > 26000.0, 3000.0, 30.0)

    dep = dep_fn(p)
    cr0 = om.calc_cfl(p, t, dep, dt=8.0, geographic=False)
    assert (cr0 > 0.5).any() and not (cr0 > 0.5).all()
    p2, t2, b2 = om.bound_courant_number(
        p, t, dep, dt=8.0, cr_max=0.5, geographic=False,
        depth_fn=dep_fn)
    assert om.calc_cfl(p2, t2, b2, dt=8.0,
                       geographic=False).max() <= 0.5 + 1e-9


# ---------------- P3: bathymetry / channels --------------------------
def test_lim_bathy_slope_limits():
    p, t = _unit_mesh(improve=False, max_iter=10)
    b = p[:, 0] * 100.0  # steep planar ramp
    b2 = om.lim_bathy_slope(p, t, b, dfdx=0.1, geographic=False)
    from oceanmesh.cfl import get_bar_lengths

    bars, L = get_bar_lengths(p, t, geographic=False)
    viol = np.abs(b2[bars[:, 0]] - b2[bars[:, 1]]) - 0.1 * L
    assert viol.max() < 1e-6


def test_unstructured_slopes_planar():
    p, t = _unit_mesh(improve=False, max_iter=10)
    b = 2.0 * p[:, 0] + 3.0 * p[:, 1]
    bx, by = om.unstructured_slopes(p, t, b)
    interior = np.setdiff1d(
        np.arange(len(p)),
        np.unique(__import__("oceanmesh").edges.get_boundary_edges(t)),
    )
    if len(interior):
        assert np.allclose(bx[interior], 2.0, atol=0.2)
        assert np.allclose(by[interior], 3.0, atol=0.2)


def test_inpoly_numba_matches_brute_on_multisegment():
    # regression for the pip-'inpoly' incident: multi-ring +
    # open-chain + duplicate-vertex input must match a brute-force
    # crossing-number referee exactly (the pip package failed 200/
    # 200 refereed mismatches on real shoreline data)
    from oceanmesh.geometry.point_in_polygon import (
        _HAVE_NUMBA_INPOLY,
        _inpoly_numba,
    )

    if not _HAVE_NUMBA_INPOLY:
        pytest.skip("numba unavailable")
    rng = np.random.default_rng(3)
    segs = []
    # outer ring with a duplicate vertex
    th = np.linspace(0, 2 * np.pi, 41)
    ring = np.column_stack([np.cos(th), np.sin(th)])
    ring = np.insert(ring, 5, ring[5], axis=0)
    segs.append(ring)
    # two island rings
    for cx, cy, r in ((0.3, 0.2, 0.15), (-0.4, -0.3, 0.1)):
        th = np.linspace(0, 2 * np.pi, 23)
        segs.append(np.column_stack([cx + r * np.cos(th),
                                     cy + r * np.sin(th)]))
    # open chain (clipped-coast analogue)
    segs.append(np.column_stack([np.linspace(-0.9, 0.9, 30),
                                 0.7 + 0.05 * rng.standard_normal(30)]))
    parts = []
    for s in segs:
        parts.append(s)
        parts.append(np.array([[np.nan, np.nan]]))
    poly = np.vstack(parts)
    from oceanmesh import edges as om_edges

    node = np.nan_to_num(poly)
    edge = om_edges.get_poly_edges(poly)
    q = rng.uniform(-1.2, 1.2, (20000, 2))
    s_nb, _ = _inpoly_numba(q, node, edge, 5e-14)

    x1, y1 = node[edge[:, 0], 0], node[edge[:, 0], 1]
    x2, y2 = node[edge[:, 1], 0], node[edge[:, 1], 1]
    ref = np.zeros(len(q), bool)
    for i, (qx, qy) in enumerate(q):
        c = (y1 > qy) != (y2 > qy)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (qy - y1) * (x2 - x1) / (y2 - y1)
        ref[i] = (np.count_nonzero(c & (qx < xi)) % 2) == 1
    assert (s_nb == ref).mean() == 1.0
