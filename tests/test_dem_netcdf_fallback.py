"""Contracts of the coordinate-based NetCDF DEM fallback.

The Tokyo Bay regression (2026-07-11): GDAL mis-georeferenced a
plain lon/lat NetCDF (self-sliced SRTM15_kanto_15s) and, once the
spacing gate routed it through the xarray fallback, the fallback
returned the raster in file order — upside down after DEM's
np.fliplr(). These tests pin both contracts with tiny synthetic
files.
"""
import numpy as np
import pytest
import xarray as xr

from oceanmesh.geodata import (
    _netcdf_coord_steps,
    _try_subset_netcdf_with_xarray,
)


def _write_nc(path, lat_ascending=True):
    lon = np.linspace(130.05, 131.95, 20)
    lat = np.linspace(30.05, 31.95, 20)
    if not lat_ascending:
        lat = lat[::-1]
    # z encodes latitude so orientation errors are visible:
    # z = -100 * lat_index_from_south (south shallow, north deep)
    zsouth = -100.0 * np.argsort(np.argsort(lat))
    z = np.tile(zsouth[:, None], (1, len(lon)))
    ds = xr.Dataset(
        {"z": (("lat", "lon"), z)},
        coords={"lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)
    return lon, lat


def test_coord_steps(tmp_path):
    f = tmp_path / "toy.nc"
    _write_nc(f)
    steps = _netcdf_coord_steps(f)
    assert steps is not None
    assert steps[0] == pytest.approx(1.9 / 19, rel=1e-6)
    assert steps[1] == pytest.approx(1.9 / 19, rel=1e-6)


@pytest.mark.parametrize("lat_ascending", [True, False])
def test_fallback_row_order_contract(tmp_path, lat_ascending):
    """topobathy_xy must be (nx, ny) with y NORTH-first, matching
    the rasterio path before DEM.__init__'s np.fliplr()."""
    f = tmp_path / "toy.nc"
    _write_nc(f, lat_ascending=lat_ascending)
    out = _try_subset_netcdf_with_xarray(
        f, (130.0, 132.0, 30.0, 32.0), "EPSG:4326"
    )
    assert out is not None
    bbox_out, dx, dy, txy = out
    assert txy.shape == (20, 20)
    # y north-first: row 0 in y is the NORTHERNMOST cell => most
    # negative z under the encoding above
    assert txy[0, 0] == pytest.approx(-100.0 * 19)   # north
    assert txy[0, -1] == pytest.approx(0.0)          # south
    assert bbox_out[2] == pytest.approx(30.05)
    assert bbox_out[3] == pytest.approx(31.95)
