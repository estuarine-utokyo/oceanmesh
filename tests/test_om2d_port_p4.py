"""Tests for port plan P4: merging, remesh_patch, weirs, floodplain."""
import numpy as np
import pytest

import oceanmesh as om
from oceanmesh.mesh_improve import area_length_quality
from oceanmesh.signed_distance_function import create_bbox as _cb


def _uni(s):
    def h(p):
        return np.full(len(np.atleast_2d(p)), s)

    return h


def _mesh(bbox, size):
    return om.generate_mesh(_cb(bbox), _uni(size), bbox=bbox,
                            min_edge_length=size, max_iter=30,
                            improve=False)


@pytest.fixture(scope="module")
def base():
    return _mesh((0.0, 2.0, 0.0, 1.0), 0.10)


@pytest.fixture(scope="module")
def inset():
    return _mesh((0.7, 1.3, 0.3, 0.7), 0.04)


def test_merge_inset_priority(base, inset):
    bp, bt = base
    ip, it = inset
    mp, mt = om.merge_meshes(ip, it, bp, bt)
    q = area_length_quality(mp, mt)
    assert q.min() > 0.2 and q.mean() > 0.8
    # inset nodes survive verbatim
    from scipy.spatial import cKDTree

    d, _ = cKDTree(mp).query(ip)
    assert d.max() < 1e-9


def test_cat_welds_shared_seam():
    lp, lt = _mesh((0.0, 1.0, 0.0, 1.0), 0.25)
    rp = lp * np.array([-1.0, 1.0]) + np.array([2.0, 0.0])
    cp, ct = om.cat_meshes(lp, lt, rp, lt[:, ::-1], tol=1e-9)
    assert len(cp) < 2 * len(lp)  # seam nodes welded


def test_minus_carves_footprint(base, inset):
    bp, bt = base
    ip, it = inset
    p, t = om.minus_meshes(bp, bt, ip, it)
    assert len(t) < len(bt)
    import shapely
    from shapely.geometry import box

    cen = p[t].mean(axis=1)
    inner = box(0.75, 0.35, 1.25, 0.65)
    assert not shapely.contains_xy(inner, cen[:, 0],
                                   cen[:, 1]).any()


def test_remesh_patch_quality(base):
    bp, bt = base
    poly = np.array([[0.5, 0.25], [1.5, 0.25], [1.5, 0.75],
                     [0.5, 0.75]])
    p, t = om.remesh_patch(bp, bt, poly, target_h=0.05)
    q = area_length_quality(p, t)
    assert q.min() > 0.3 and len(t) > len(bt)


def test_remesh_patch_errors(base):
    bp, bt = base
    with pytest.raises(ValueError):
        om.remesh_patch(bp, bt,
                        np.array([[5.0, 5.0], [6.0, 5.0],
                                  [6.0, 6.0]]))


def test_weir_geometry_and_pairs(tmp_path):
    crest = np.array([[0.3, 0.5], [0.7, 0.5]])
    pf, eg, pairs = om.build_weir_geometry(crest, width_m=2000.0,
                                           spacing_m=8000.0)
    assert len(eg) == len(pf)  # closed ring
    p, t = om.generate_mesh(
        _cb((0.0, 1.0, 0.0, 1.0)), _uni(0.05),
        bbox=(0.0, 1.0, 0.0, 1.0), min_edge_length=0.05,
        max_iter=30, pfix=pf, egfix=eg, improve=False,
    )
    fr, bk = om.match_weir_nodes(p, pairs, tol=1e-9)
    assert np.allclose(p[fr], pairs[:, :2])
    assert np.allclose(p[bk], pairs[:, 2:])
    f14 = tmp_path / "weir.14"
    om.write_fort14(
        str(f14), p, t, depth=np.ones(len(p)),
        boundaries={"open": [], "land": [], "island": []},
        weirs=[{"front": fr, "back": bk,
                "crest": np.full(len(fr), 2.5)}],
    )
    assert " 24 = " in f14.read_text()


def test_weir_zero_width_error():
    with pytest.raises(Exception):
        om.build_weir_geometry(np.array([[0, 0], [1, 0]]),
                               width_m=0.0)


def test_contour_to_shapefile(tmp_path):
    from oceanmesh import Region

    reg = Region((-95.16, -95.02, 29.51, 29.61), 4326)
    dem = om.DEM("tests/galv_sub.nc", bbox=reg)
    out = tmp_path / "fp.shp"
    om.contour_to_shapefile(dem, level=0.0, out_path=str(out),
                            min_length=3)
    import geopandas as gpd

    assert len(gpd.read_file(out)) > 0
