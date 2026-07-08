import errno
import logging
import os
from pathlib import Path

import fiona
import geopandas as gpd
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import rasterio
from affine import Affine
import rasterio.crs
import rasterio.warp
import shapely.geometry
import shapely.validation
from pyproj import CRS
from rasterio.windows import from_bounds

from .grid import Grid
from .region import Region, to_lat_lon

nan = np.nan
fiona_version = fiona.__version__

logger = logging.getLogger(__name__)

__all__ = ["Shoreline", "DEM", "get_polygon_coordinates", "create_circle_coords"]


def _bbox_overlaps_bounds(bbox_vals, bounds):
    xmin, xmax, ymin, ymax = bbox_vals
    ixmin = max(xmin, bounds.left)
    ixmax = min(xmax, bounds.right)
    iymin = max(ymin, bounds.bottom)
    iymax = min(ymax, bounds.top)
    return (ixmin < ixmax) and (iymin < iymax)


def _transform_bbox_to_src(bbox_vals, src_crs, dst_crs):
    if src_crs is None or dst_crs is None or src_crs == dst_crs:
        return bbox_vals

    from pyproj import Transformer

    xmin, xmax, ymin, ymax = bbox_vals
    transformer = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
    xs, ys = transformer.transform([xmin, xmax], [ymin, ymax])
    return (min(xs), max(xs), min(ys), max(ys))


def _prepare_region_bbox_and_crs(bbox, crs):
    region = None
    region_crs = None
    region_bbox = None

    if bbox is not None:
        assert isinstance(bbox, Region), "bbox must be a Region class object"
        region = bbox
        region_crs = region.crs
        region_bbox = region.total_bounds
        if crs != "EPSG:4326":
            logger.warning(
                "DEM: Both a Region object and an explicit 'crs' were provided; the Region's CRS will take precedence (crs=%s).",
                region_crs,
            )
        crs = CRS.from_user_input(region_crs).to_string()
        bbox = region_bbox

    return bbox, crs, region, region_crs, region_bbox


def _pick_netcdf_open_target(dem_path, bbox_vals, crs_str):
    """Choose a stable rasterio open target for NetCDF inputs.

    GDAL NetCDF subdataset ordering differs across builds/wheels. Prefer a
    subdataset whose bounds overlap the requested bbox, otherwise choose the
    largest-area subdataset.
    """

    if dem_path.suffix.lower() not in {".nc", ".nc4"}:
        return dem_path

    try:
        with rasterio.open(dem_path) as container:
            subdatasets = list(getattr(container, "subdatasets", []) or [])
    except Exception:
        return dem_path

    if not subdatasets:
        return dem_path

    desired_bbox = None
    desired_crs = None
    if bbox_vals is not None:
        desired_bbox = tuple(map(float, bbox_vals))
        desired_crs = CRS.from_user_input(crs_str)

    best = None
    best_area = -1.0

    for sd in subdatasets:
        try:
            with rasterio.open(sd) as src:
                if src.width <= 1 or src.height <= 1:
                    continue
                b = src.bounds
                if not np.isfinite([b.left, b.right, b.bottom, b.top]).all():
                    continue

                area = float((b.right - b.left) * (b.top - b.bottom))

                if desired_bbox is not None and desired_crs is not None:
                    bbox_in_sd = desired_bbox
                    if src.crs is not None and src.crs != desired_crs:
                        try:
                            bbox_in_sd = _transform_bbox_to_src(
                                desired_bbox, src.crs, desired_crs
                            )
                        except Exception:
                            bbox_in_sd = None

                    if bbox_in_sd is not None and _bbox_overlaps_bounds(bbox_in_sd, b):
                        if area > best_area:
                            best = sd
                            best_area = area
                        continue

                if area > best_area:
                    best = sd
                    best_area = area
        except Exception:
            continue

    return best or dem_path


def _xr_open_dataset(path):
    try:
        import xarray as xr
    except Exception:
        return None

    try:
        return xr.open_dataset(path)
    except Exception:
        return None


def _xr_pick_first_raster_data_array(ds):
    for name, var in ds.data_vars.items():
        if getattr(var, "ndim", 0) >= 2:
            return ds[name]
    return None


def _xr_reduce_to_2d(da):
    while getattr(da, "ndim", 0) > 2:
        da = da.isel({da.dims[0]: 0})
    return da


def _xr_pick_coord_name(ds, da, candidates):
    for candidate in candidates:
        if candidate in da.coords:
            return candidate
        if candidate in ds.coords:
            return candidate
    return None


def _xr_find_overlap_index_range(axis, lo, hi):
    axis = np.asarray(axis)
    if axis.size == 0:
        return None

    if axis[0] <= axis[-1]:
        idx = np.where((axis >= lo) & (axis <= hi))[0]
    else:
        idx = np.where((axis <= hi) & (axis >= lo))[0]

    if idx.size == 0:
        return None
    return int(idx.min()), int(idx.max())


def _xr_median_spacing(axis):
    axis = np.asarray(axis, dtype=float)
    if axis.size <= 1:
        return np.nan
    return float(np.nanmedian(np.abs(np.diff(axis))))


def _xr_subset_to_outputs(sub, xname, yname):
    arr = np.asarray(sub.values, dtype=np.float64)
    if tuple(sub.dims) == (xname, yname):
        arr = np.transpose(arr, (1, 0))

    topobathy_xy = np.transpose(arr, (1, 0))

    x_sub = np.asarray(sub.coords[xname].values, dtype=float)
    y_sub = np.asarray(sub.coords[yname].values, dtype=float)
    dx = _xr_median_spacing(x_sub)
    dy = _xr_median_spacing(y_sub)
    if not (np.isfinite(dx) and np.isfinite(dy) and dx > 0 and dy > 0):
        return None

    bbox_out = (
        float(np.nanmin(x_sub)),
        float(np.nanmax(x_sub)),
        float(np.nanmin(y_sub)),
        float(np.nanmax(y_sub)),
    )
    return bbox_out, dx, dy, topobathy_xy


def _try_subset_netcdf_with_xarray(dem_path, bbox_vals, crs_str):
    """Fallback subsetting for NetCDF when rasterio bounds are unreliable.

    Returns (bbox, dx, dy, topobathy_xy) where topobathy_xy is shaped (nx, ny)
    like the rasterio code path before the final np.fliplr() in DEM.
    """

    if dem_path.suffix.lower() not in {".nc", ".nc4"}:
        return None

    ds = _xr_open_dataset(dem_path)
    if ds is None:
        return None

    da = _xr_pick_first_raster_data_array(ds)
    if da is None:
        return None
    da = _xr_reduce_to_2d(da)

    coord_candidates_x = ["x", "lon", "longitude", "Long", "LONGITUDE"]
    coord_candidates_y = ["y", "lat", "latitude", "Lat", "LATITUDE"]

    xname = _xr_pick_coord_name(ds, da, coord_candidates_x)
    yname = _xr_pick_coord_name(ds, da, coord_candidates_y)
    if xname is None or yname is None:
        return None

    x = np.asarray(da.coords[xname].values)
    y = np.asarray(da.coords[yname].values)
    if x.ndim != 1 or y.ndim != 1:
        return None

    xmin, xmax, ymin, ymax = map(float, bbox_vals)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    if not (
        max(xmin, x_min) < min(xmax, x_max) and max(ymin, y_min) < min(ymax, y_max)
    ):
        return None

    xrng = _xr_find_overlap_index_range(x, xmin, xmax)
    yrng = _xr_find_overlap_index_range(y, ymin, ymax)
    if xrng is None or yrng is None:
        return None

    x0, x1 = xrng
    y0, y1 = yrng
    sub = da.isel({xname: slice(x0, x1 + 1), yname: slice(y0, y1 + 1)})
    return _xr_subset_to_outputs(sub, xname, yname)


def _read_dem_array_and_meta(dem_path, bbox, crs, region_bbox, region_crs):
    open_target = _pick_netcdf_open_target(dem_path, bbox, crs)

    with rasterio.open(open_target) as src:
        nodata_value = src.nodata
        meta = src.meta

        src_crs = src.crs
        desired_crs = CRS.from_user_input(crs)

        # Entire DEM is read in
        if bbox is None:
            bbox_ds = src.bounds  # left, bottom, right, top
            bbox_out = (bbox_ds.left, bbox_ds.right, bbox_ds.bottom, bbox_ds.top)
            topobathy = src.read(1)
            topobathy_xy = np.transpose(topobathy, (1, 0))
        else:
            if region_bbox is None:
                region_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])

            region_crs_for_transform = (
                region_crs if region_crs is not None else desired_crs
            )
            region_bbox_src = _transform_bbox_to_src(
                region_bbox, src_crs, region_crs_for_transform
            )

            ds_bounds = src.bounds  # (left, bottom, right, top)
            ds_bbox_conv = (
                ds_bounds.left,
                ds_bounds.right,
                ds_bounds.bottom,
                ds_bounds.top,
            )

            xmin = max(region_bbox_src[0], ds_bbox_conv[0])
            xmax = min(region_bbox_src[1], ds_bbox_conv[1])
            ymin = max(region_bbox_src[2], ds_bbox_conv[2])
            ymax = min(region_bbox_src[3], ds_bbox_conv[3])

            if not (xmin < xmax and ymin < ymax):
                if (src_crs is None) or src.transform == Affine.identity:
                    logger.warning(
                        "DEM appears un-georeferenced; applying synthetic georeference using provided bbox extents %s.",
                        region_bbox,
                    )
                    topobathy = src.read(1)
                    ny, nx = topobathy.shape  # rasterio returns (rows, cols)
                    dx_syn = (region_bbox[1] - region_bbox[0]) / (nx - 1)
                    dy_syn = (region_bbox[3] - region_bbox[2]) / (ny - 1)
                    meta["transform"] = Affine(
                        dx_syn, 0, region_bbox[0], 0, -dy_syn, region_bbox[3]
                    )
                    bbox_out = region_bbox
                    topobathy_xy = np.transpose(topobathy, (1, 0))
                else:
                    if dem_path.suffix.lower() in {".nc", ".nc4"}:
                        fb = _try_subset_netcdf_with_xarray(
                            dem_path, region_bbox_src, crs
                        )
                        if fb is not None:
                            bbox_out, dx_fb, dy_fb, topobathy_xy = fb
                            meta["transform"] = Affine(
                                dx_fb, 0, bbox_out[0], 0, -dy_fb, bbox_out[3]
                            )
                        else:
                            raise ValueError(
                                "Transformed DEM clipping bbox does not overlap raster bounds. "
                                f"Region bbox (in raster CRS)={region_bbox_src}, raster bounds={ds_bbox_conv}."
                            )
                    else:
                        raise ValueError(
                            "Transformed DEM clipping bbox does not overlap raster bounds. "
                            f"Region bbox (in raster CRS)={region_bbox_src}, raster bounds={ds_bbox_conv}."
                        )
            else:
                bounds_for_window = (xmin, ymin, xmax, ymax)
                try:
                    window = from_bounds(*bounds_for_window, transform=src.transform)
                except Exception as e:
                    raise RuntimeError(
                        "Failed to create window for DEM subset. "
                        f"Bounds={bounds_for_window}, transform={src.transform}."
                    ) from e
                topobathy = src.read(1, window=window, masked=True)
                topobathy_xy = np.transpose(topobathy, (1, 0))
                bbox_out = (xmin, xmax, ymin, ymax)

        # Warn if user requested output CRS different from raster CRS (no reprojection performed here)
        if (
            (src_crs is not None)
            and (desired_crs is not None)
            and (src_crs != desired_crs)
            and not ((src_crs is None) or src.transform == Affine.identity)
        ):
            logger.warning(
                "DEM opened in its native CRS %s but requested CRS %s differs. "
                "Values are NOT reprojected; proceeding in native CRS.",
                src_crs,
                desired_crs,
            )
            crs = src_crs.to_string()

    return bbox_out, crs, meta, nodata_value, topobathy_xy


def _infer_crs_from_coordinates(bbox):
    """
    Heuristically infer whether a bbox looks like geographic (WGS84) or projected.

    Parameters
    ----------
    bbox : tuple
        (xmin, xmax, ymin, ymax)

    Returns
    -------
    is_geographic : bool
        True if coordinates appear geographic (lon/lat degrees), False if likely projected.

    Notes
    -----
    This is a coarse heuristic intended to catch obvious cases:
      - If x in [-180, 180] and y in [-90, 90] -> geographic
      - If any |coord| > 360 -> projected
      - If horizontal magnitudes are in [180, 360] (typical of meters-based UTM ranges) -> projected
    Ambiguous cases default to geographic to preserve historical behavior.
    """
    try:
        xmin, xmax, ymin, ymax = bbox
    except Exception:
        return True

    xs = (xmin, xmax)
    ys = (ymin, ymax)

    # Any very large absolute values strongly indicates projected units (meters)
    if any(abs(v) > 360 for v in xs + ys):
        return False

    # Clear geographic ranges
    if all(-180.0 <= v <= 180.0 for v in xs) and all(-90.0 <= v <= 90.0 for v in ys):
        return True

    # Values between 180 and 360 degrees on x suggest projected values mislabeled
    if any(180.0 < abs(v) <= 360.0 for v in xs + ys):
        return False

    # Default: assume geographic to remain backward compatible
    return True


def create_circle_coords(radius, center, arc_res):
    """
    Given a radius and a center point, creates a numpy array of coordinates
    defining a circle in a CCW direction with a given arc resolution.

    Parameters:
    radius (float): the radius of the circle
    center (tuple): the (x,y) coordinates of the center point
    arc_res (float): the arc resolution of the circle in degrees

    Returns:
    numpy.ndarray: an array of (x,y) coordinates defining the circle
    """
    # Define the angle array with the given arc resolution
    angles = np.arange(0, 360 + arc_res, arc_res) * np.pi / 180

    # Calculate the (x,y) coordinates of the circle points
    x_coords = center[0] + radius * np.cos(angles)
    y_coords = center[1] + radius * np.sin(angles)

    # Combine the (x,y) coordinates into a single array
    coords = np.column_stack((x_coords, y_coords))

    return coords


def get_polygon_coordinates(vector_file):
    """Get the coordinates of a polygon from a vector file or plain csv file"""
    # detect if file is a shapefile or a geojson or geopackage
    if (
        vector_file.endswith(".shp")
        or vector_file.endswith(".geojson")
        or vector_file.endswith(".gpkg")
    ):
        gdf = gpd.read_file(vector_file)
        polygon = np.array(gdf.iloc[0].geometry.exterior.coords.xy).T
    elif vector_file.endswith(".csv"):
        polygon = np.loadtxt(vector_file, delimiter=",")
    return polygon


def _convert_to_array(lst):
    """Converts a list of numpy arrays to a np array"""
    return np.concatenate(lst, axis=0)


def _convert_to_list(arr):
    """Converts a nan-delimited numpy array to a list of numpy arrays"""
    a = np.insert(arr, 0, [[nan, nan]], axis=0)
    tmp = [a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a[:, 0]))]
    return [np.append(a, [[nan, nan]], axis=0) for a in tmp]


def _create_boubox(bbox):
    """Create a bounding box from domain extents `bbox`. Path orientation will be CCW."""
    if isinstance(bbox, tuple):
        xmin, xmax, ymin, ymax = bbox
        return [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ]
    return bbox


def _create_ranges(start, stop, N, endpoint=True):
    """Vectorized alternative to numpy.linspace
    https://stackoverflow.com/questions/40624409/vectorized-np-linspace-for-multiple-start-and-stop-values
    """
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return steps[:, None] * np.arange(N) + start[:, None]


def _nan_segments(polys):
    """Yield NaN-delimited segments (each with its trailing NaN row
    when present) from a stacked polys array."""
    arr = np.asarray(polys, dtype=float)
    isn = np.isnan(arr[:, 0])
    start = 0
    for i in range(len(arr)):
        if isn[i]:
            if i > start:
                yield arr[start:i + 1]
            start = i + 1
    if start < len(arr):
        yield arr[start:]


def _cull_small_features(polys, boubox, h0, island_mult=4.0,
                         mainland_mult=100.0):
    """OM2D Read_shapefile area culling on raw polygons.

    Islands wholly inside the domain polygon with shoelace area
    < island_mult*h0^2 and partially-inside (mainland) fragments
    < mainland_mult*h0^2 are dropped (Read_shapefile.m:219,231).
    Open segments are kept untouched. h0 is in the same units as
    the coordinates (degrees).
    """
    from shapely.geometry import Polygon as _Pg

    b = np.asarray(boubox, dtype=float)
    b = b[~np.isnan(b[:, 0])]
    try:
        bpoly = _Pg(b)
        if not bpoly.is_valid:
            bpoly = bpoly.buffer(0)
    except Exception:
        return polys
    a_island = island_mult * h0 * h0
    a_main = mainland_mult * h0 * h0
    out = []
    for seg in _nan_segments(polys):
        pts = seg[~np.isnan(seg[:, 0])]
        closed = len(pts) > 3 and np.allclose(pts[0], pts[-1])
        if not closed:
            out.append(seg)
            continue
        try:
            pg = _Pg(pts)
            if not pg.is_valid:
                pg = pg.buffer(0)
            area = pg.area
            inside = bpoly.contains(pg)
        except Exception:
            out.append(seg)
            continue
        if inside and area < a_island:
            continue
        if not inside and area < a_main:
            continue
        out.append(seg)
    if not out:
        return polys
    return np.vstack(out)


def _resample_segments(polys, spacing):
    """OM2D my_interpm parity: resample every NaN-delimited segment
    to ~uniform ``spacing`` (degrees) — DECIMATING excess detail as
    well as filling gaps. Without decimation a fine global
    shoreline (GSHHS_f over the W-Pacific) enters feature/SDF
    machinery with millions of vertices and stalls sizing for a
    10-50 km nest."""
    out = []
    isnan = np.isnan(polys[:, 0])
    start = 0
    for idx in list(np.where(isnan)[0]) + [len(polys)]:
        seg = polys[start:idx]
        start = idx + 1
        n = len(seg)
        if n == 0:
            continue
        if n < 3:
            out.append(seg)
            out.append(np.array([[np.nan, np.nan]]))
            continue
        closed = bool(np.allclose(seg[0], seg[-1]))
        d = np.hypot(*np.diff(seg, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(d)])
        L = s[-1]
        if L <= spacing:
            out.append(seg)
            out.append(np.array([[np.nan, np.nan]]))
            continue
        m = max(int(np.ceil(L / spacing)) + 1, 4 if closed else 2)
        si = np.linspace(0.0, L, m)
        res = np.column_stack([np.interp(si, s, seg[:, 0]),
                               np.interp(si, s, seg[:, 1])])
        if closed:
            res[-1] = res[0]
        out.append(res)
        out.append(np.array([[np.nan, np.nan]]))
    if not out:
        return polys
    return np.vstack(out)


def _densify(poly, maxdiff, bbox, radius=0):
    """Fills in any gaps in latitude or longitude arrays
    that are greater than a `maxdiff` (degrees) apart
    """
    logger.debug("Entering:_densify")

    boubox = _create_boubox(bbox)
    path = mpltPath.Path(boubox, closed=True)
    inside = path.contains_points(poly, radius=0.1)  # add a small radius
    lon, lat = poly[:, 0], poly[:, 1]
    nx = len(lon)
    dlat = np.abs(lat[1:] - lat[:-1])
    dlon = np.abs(lon[1:] - lon[:-1])
    nin = np.ceil(np.maximum(dlat, dlon) / maxdiff) - 1
    nin[~inside[1:]] = 0  # no need to densify outside of bbox please
    # handle negative values
    nin[nin < 0] = 0
    sumnin = np.nansum(nin)
    if sumnin == 0:
        return np.hstack((lon[:, None], lat[:, None]))
    nout = sumnin + nx

    lonout = np.full((int(nout)), nan, dtype=float)
    latout = np.full((int(nout)), nan, dtype=float)

    n = 0
    for i in range(nx - 1):
        ni = nin[i]
        if ni == 0 or np.isnan(ni):
            latout[n] = lat[i]
            lonout[n] = lon[i]
            nstep = 1
        else:
            ni = int(ni)
            icoords = _create_ranges(
                np.array([lat[i], lon[i]]),
                np.array([lat[i + 1], lon[i + 1]]),
                ni + 2,
            )
            latout[n : n + ni + 1] = icoords[0, : ni + 1]
            lonout[n : n + ni + 1] = icoords[1, : ni + 1]
            nstep = ni + 1
        n += nstep

    latout[-1] = lat[-1]
    lonout[-1] = lon[-1]

    logger.debug("Exiting:_densify")

    return np.hstack((lonout[:, None], latout[:, None]))


def _poly_area(x, y):
    """Calculates area of a polygon"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _poly_length(coords):
    """Calculates circumference of a polygon"""
    if all(np.isclose(coords[0, :], coords[-1, :])):
        c = coords
    else:
        c = np.vstack((coords, coords[0, :]))

    return np.sum(np.sqrt(np.sum(np.diff(c, axis=0) ** 2, axis=1)))


def _classify_shoreline(bbox, boubox, polys, h0, minimum_area_mult, stereo=False):
    """Classify segments in numpy.array `polys` as either `inner` or
    `mainland` per OM2D Read_shapefile.m:110-236: a ring wholly
    inside the domain polygon is an island (`inner`), ANY other ring
    is `mainland`, kept whole — no boolean clipping. The returned
    domain ring is the densified `boubox` itself; the meshing sign
    is the even-odd parity over [boubox; mainland; inner]
    (Read_shapefile.m:283 stacks outer=[boubox;mainland] and
    meshgen.m:329-331 appends inner), which handles arbitrary
    nesting (e.g. Hackensack wetland channels: water-in-land-in-
    domain) that the previous shapely difference() approach dropped.
    NB: removes `inner` geometry with area < minimum_area_mult*h0**2.
    """
    logger.debug("Entering:_classify_shoreline")

    if stereo:
        return _classify_shoreline_stereo(
            bbox, boubox, polys, h0, minimum_area_mult
        )

    _AREAMIN = minimum_area_mult * h0**2

    if len(boubox) == 0:
        boubox = _create_boubox(bbox)
        boubox = np.asarray(boubox)
    elif not _is_path_ccw(boubox):
        boubox = np.flipud(boubox)

    boubox = _densify(boubox, h0 / 2, bbox, radius=0.1)

    isNaN = np.sum(np.isnan(boubox), axis=1) > 0
    if any(isNaN):
        boubox = np.delete(boubox, isNaN, axis=0)
    del isNaN

    from . import edges as _om_edges
    from .geometry import inpoly2 as _inpoly2

    ring = np.vstack((boubox, boubox[0], [[nan, nan]]))
    e_box = _om_edges.get_poly_edges(ring)
    ring_nn = np.nan_to_num(ring)

    inner = np.empty(shape=(0, 2))
    mainland = np.empty(shape=(0, 2))

    islands = []
    main_parts = []
    for poly in _convert_to_list(polys):
        pts = poly[:-2, :]
        if len(pts) < 3:
            continue
        inside, _ = _inpoly2(pts, ring_nn, e_box)
        if inside.all():
            # wholly inside -> island (Read_shapefile.m:217-228)
            if abs(_poly_area(pts[:, 0], pts[:, 1])) >= _AREAMIN:
                islands.append(poly)
        else:
            # partially inside or outside -> mainland, kept WHOLE
            # (Read_shapefile.m:229-241; no geometric clipping)
            main_parts.append(poly)

    # Merge islands that OVERLAP the mainland (share >= 3 vertices
    # at 1e-5 tolerance) into the mainland via polygon union,
    # keeping holes (Read_shapefile.m:253-281: ismembertol +
    # polyshape union). Overlapping rings double-count in the
    # even-odd parity, flipping the overlap region to land.
    if main_parts and islands:
        main_vset = set()
        for mp in main_parts:
            fp = mp[~np.isnan(mp[:, 0])]
            for r in np.round(fp / 1e-5).astype(np.int64):
                main_vset.add((int(r[0]), int(r[1])))
        merged_polys = []
        keep_islands = []
        for isl in islands:
            fp = isl[~np.isnan(isl[:, 0])]
            r = np.round(fp / 1e-5).astype(np.int64)
            nshared = sum(
                (int(a), int(b)) in main_vset for a, b in r
            )
            if nshared > 2:
                merged_polys.append(fp)
            else:
                keep_islands.append(isl)
        if merged_polys:
            logger.info(
                f"merging {len(merged_polys)} islands overlapping "
                "the mainland (Read_shapefile.m:253-281)"
            )
            from shapely.ops import unary_union

            geoms = []
            for mp in main_parts:
                fp = mp[~np.isnan(mp[:, 0])]
                g = shapely.geometry.Polygon(fp)
                if not g.is_valid:
                    g = g.buffer(0)
                geoms.append(g)
            for fp in merged_polys:
                g = shapely.geometry.Polygon(fp)
                if not g.is_valid:
                    g = g.buffer(0)
                geoms.append(g)
            u = unary_union(geoms)
            if u.geom_type == "Polygon":
                u = shapely.geometry.MultiPolygon([u])
            main_parts = []
            for g in u.geoms:
                rings = [np.asarray(g.exterior.coords)]
                rings += [np.asarray(i.coords) for i in g.interiors]
                for rg in rings:
                    main_parts.append(
                        np.vstack((rg, [[nan, nan]]))
                    )
            islands = keep_islands

    inner = (np.vstack(islands) if islands
             else np.empty(shape=(0, 2)))
    mainland = (np.vstack(main_parts) if main_parts
                else np.empty(shape=(0, 2)))

    logger.debug("Exiting:classify_shoreline")
    return inner, mainland, ring


def _classify_shoreline_stereo(bbox, boubox, polys, h0, minimum_area_mult):
    """Pre-Read_shapefile-port behavior, kept for the global stereo
    path only (Antarctica boubox replacement relies on the shapely
    difference mechanics)."""
    logger.debug("Entering:_classify_shoreline_stereo")

    stereo = True
    _AREAMIN = minimum_area_mult * h0**2

    if len(boubox) == 0:
        boubox = _create_boubox(bbox)
        boubox = np.asarray(boubox)
    elif not _is_path_ccw(boubox):
        boubox = np.flipud(boubox)

    boubox = _densify(boubox, h0 / 2, bbox, radius=0.1)

    # Remove nan's (append again at end)
    isNaN = np.sum(np.isnan(boubox), axis=1) > 0
    if any(isNaN):
        boubox = np.delete(boubox, isNaN, axis=0)
    del isNaN

    inner = np.empty(shape=(0, 2))
    inner[:] = nan
    mainland = np.empty(shape=(0, 2))
    mainland[:] = nan

    polyL = _convert_to_list(polys)
    bSGP = shapely.geometry.Polygon(boubox)

    for poly in polyL:
        pSGP = shapely.geometry.Polygon(poly[:-2, :])
        if bSGP.contains(pSGP):
            if stereo:
                # convert back to Lat/Lon coordinates for the area testing
                area = _poly_area(*to_lat_lon(*np.asarray(pSGP.exterior.xy)))
            else:
                area = pSGP.area
            if area >= _AREAMIN:
                inner = np.append(inner, poly, axis=0)
        elif pSGP.overlaps(bSGP):
            if stereo:
                bSGP = pSGP
            else:
                # GEOS difference() raises TopologyException ("side
                # location conflict") when either operand carries the
                # near-degenerate rings that the tiny-buffer repairs
                # above can produce; make_valid() both sides first.
                from shapely.errors import GEOSException
                from shapely.validation import make_valid

                if not bSGP.is_valid:
                    bSGP = make_valid(bSGP)
                if not pSGP.is_valid:
                    pSGP = make_valid(pSGP)
                try:
                    bSGP = bSGP.difference(pSGP)
                except GEOSException:
                    bSGP = make_valid(bSGP).difference(
                        make_valid(pSGP.buffer(0))
                    )
                # Append polygon segment to mainland
                mainland = np.vstack((mainland, poly))
                # Clip polygon segment from boubox and regenerate path

    out = np.empty(shape=(0, 2))

    if bSGP.geom_type == "GeometryCollection":
        # make_valid()/difference() can emit collections holding
        # degenerate lines/points; keep only the polygonal parts.
        _polys = []
        for _g in bSGP.geoms:
            if _g.geom_type == "Polygon":
                _polys.append(_g)
            elif _g.geom_type == "MultiPolygon":
                _polys.extend(_g.geoms)
        bSGP = shapely.geometry.MultiPolygon(_polys)

    if bSGP.geom_type == "Polygon":
        # Convert to `MultiPolygon`
        bSGP = shapely.geometry.MultiPolygon([bSGP])

    # MultiPolygon members can be accessed via iterator protocol using `in`.
    for b in bSGP.geoms:
        xy = np.asarray(b.exterior.coords)
        xy = np.vstack((xy, xy[0]))
        out = np.vstack((out, xy, [nan, nan]))

    logger.debug("Exiting:classify_shoreline")

    return inner, mainland, out


def _coarsen_outside(polys, boubox, inflation=1.10, jump=200):
    """Port of OceanMesh2D @geodata/private/coarsen_polygon.m:
    decimate polyline vertices lying outside the ``inflation``-scaled
    bounding polygon by jumping ``jump`` vertices at a time when a
    long run (>= ``jump``) of consecutive outside points is found.
    Keeps full detail inside; detail outside the domain of interest
    cannot influence the mesh but slows every geometric query."""
    from . import edges as _edges
    from .geometry import inpoly2

    bou = np.asarray(boubox, dtype=float)
    finite = bou[~np.isnan(bou[:, 0])]
    c = finite.mean(axis=0)
    inflated = np.vstack([
        (finite - c) * inflation + c,
        [np.nan, np.nan],
    ])
    e_box = _edges.get_poly_edges(inflated)
    bou_nn = np.nan_to_num(inflated)

    out_segments = []
    idx = np.where(np.isnan(polys[:, 0]))[0]
    start = 0
    for stop in idx:
        seg = polys[start:stop]
        start = stop + 1
        if len(seg) == 0:
            continue
        inside, _ = inpoly2(seg, bou_nn, e_box)
        keep = []
        j = 0
        n = len(seg)
        while j < n - 1:
            if inside[j]:
                keep.append(j)
                j += 1
                continue
            bd = min(j + jump, n - 1)
            ext = min(jump, bd - j)
            if ext > 0 and not inside[j:bd].any():
                keep.append(j)
                keep.append(j + ext)
                j += ext
            else:
                keep.append(j)
                j += 1
        keep.append(n - 1)
        dec = seg[sorted(set(keep))]
        if len(dec) < 4:
            # never decimate a ring below a valid linearring
            dec = seg
        out_segments.append(dec)
        out_segments.append(np.array([[np.nan, np.nan]]))
    if not out_segments:
        return polys
    return np.vstack(out_segments)


def _moving_average_smooth(polys, window=5):
    """OM2D-style moving-average shoreline smoothing (geodata
    ``window`` option; an alternative to Chaikin corner cutting).
    Rings are smoothed cyclically; open lines keep their ends."""
    if window <= 1:
        return polys
    out = []
    idx = np.where(np.isnan(polys[:, 0]))[0]
    start = 0
    kernel = np.ones(window) / window
    for stop in idx:
        seg = polys[start:stop]
        start = stop + 1
        if len(seg) < window + 2:
            out.append(seg)
            out.append(np.array([[np.nan, np.nan]]))
            continue
        closed = np.allclose(seg[0], seg[-1])
        if closed:
            body = seg[:-1]
            pad = window // 2
            ext = np.vstack([body[-pad:], body, body[:pad]])
            sm = np.column_stack([
                np.convolve(ext[:, 0], kernel, mode="valid"),
                np.convolve(ext[:, 1], kernel, mode="valid"),
            ])[: len(body)]
            sm = np.vstack([sm, sm[:1]])
        else:
            sm = seg.copy()
            inner_len = len(seg) - 2 * (window // 2)
            if inner_len > 0:
                smoothed = np.column_stack([
                    np.convolve(seg[:, 0], kernel, mode="valid"),
                    np.convolve(seg[:, 1], kernel, mode="valid"),
                ])
                sm[window // 2: window // 2 + len(smoothed)] = smoothed
        out.append(sm)
        out.append(np.array([[np.nan, np.nan]]))
    return np.vstack(out)


def _chaikins_corner_cutting(coords, refinements=5):
    """http://www.cs.unc.edu/~dm/UNC/COMP258/LECTURES/Chaikins-Algorithm.pdf"""
    logger.debug("Entering:_chaikins_corner_cutting")
    coords = np.array(coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    logger.debug("Exiting:_chaikins_corner_cutting")
    return coords


def _smooth_shoreline(polys, N):
    """Smoothes the shoreline segment-by-segment using
    a `N` refinement Chaikins Corner cutting algorithm.
    """
    logger.debug("Entering:_smooth_shoreline")

    polys = _convert_to_list(polys)
    out = []
    for poly in polys:
        tmp = _chaikins_corner_cutting(poly[:-1], refinements=N)
        tmp = np.append(tmp, [[nan, nan]], axis=0)
        out.append(tmp)

    logger.debug("Exiting:_smooth_shoreline")

    return _convert_to_array(out)


def _clip_polys_2(polys, bbox, delta=0.10):
    """Clip segments in `polys` that intersect with `bbox`.
    Clipped segments need to extend outside `bbox` to avoid
    false positive `all(inside)` cases. Solution here is to
    add a small offset `delta` to the `bbox`.
    """
    logger.debug("Entering:_clip_polys_2")

    # Inflate bounding box to allow clipped segment to overshoot original box.
    bbox = (bbox[0] - delta, bbox[1] + delta, bbox[2] - delta, bbox[3] + delta)
    boubox = np.asarray(_create_boubox(bbox))
    path = mpltPath.Path(boubox)
    polys = _convert_to_list(polys)

    out = []

    for poly in polys:
        p = poly[:-1, :]

        inside = path.contains_points(p)

        iRemove = []

        _keepLL = True
        _keepUL = True
        _keepLR = True
        _keepUR = True

        if all(inside):
            out.append(poly)
        elif any(inside):
            for j in range(0, len(p)):
                if not inside[j]:  # snap point to inflated domain bounding box
                    px = p[j, 0]
                    py = p[j, 1]
                    if not (bbox[0] < px and px < bbox[1]) or not (
                        bbox[2] < py and py < bbox[3]
                    ):
                        if (
                            _keepLL and px < bbox[0] and py < bbox[2]
                        ):  # is over lower-left
                            p[j, :] = [bbox[0], bbox[2]]
                            _keepLL = False
                        elif (
                            _keepUL and px < bbox[0] and bbox[3] < py
                        ):  # is over upper-left
                            p[j, :] = [bbox[0], bbox[3]]
                            _keepUL = False
                        elif (
                            _keepLR and bbox[1] < px and py < bbox[2]
                        ):  # is over lower-right
                            p[j, :] = [bbox[1], bbox[2]]
                            _keepLR = False
                        elif (
                            _keepUR and bbox[1] < px and bbox[3] < py
                        ):  # is over upper-right
                            p[j, :] = [bbox[1], bbox[3]]
                            _keepUR = False
                        else:
                            iRemove.append(j)

            logger.info(f"Simplify polygon: length {len(p)} --> {len(p)}")
            # Remove colinear||duplicate vertices
            if len(iRemove) > 0:
                p = np.delete(p, iRemove, axis=0)
                del iRemove

            line = p

            # Close polygon
            if not all(np.isclose(line[0, :], line[-1, :])):
                line = np.append(line, [line[0, :], [nan, nan]], axis=0)
            else:
                line = np.append(line, [[nan, nan]], axis=0)

            out.append(line)

    logger.debug("Exiting:_clip_polys_2")

    return _convert_to_array(out)


def _clip_polys(polys, bbox, delta=0.10):
    """Clip segments in `polys` that intersect with `bbox`.
    Clipped segments need to extend outside `bbox` to avoid
    false positive `all(inside)` cases. Solution here is to
    add a small offset `delta` to the `bbox`.
    Dependencies: shapely.geometry and numpy
    """

    logger.debug("Entering:_clip_polys")

    # Inflate bounding box to allow clipped segment to overshoot original box.
    bbox = (bbox[0] - delta, bbox[1] + delta, bbox[2] - delta, bbox[3] + delta)
    boubox = np.asarray(_create_boubox(bbox))
    polyL = _convert_to_list(polys)

    out = np.empty(shape=(0, 2))

    b = shapely.geometry.Polygon(boubox)

    for poly in polyL:
        mp = shapely.geometry.Polygon(poly[:-2, :])
        if not mp.is_valid:
            logger.warning(
                "Shapely.geometry.Polygon "
                + f"{shapely.validation.explain_validity(mp)}."
                + " Applying tiny buffer to make valid."
            )
            mp = mp.buffer(1.0e-6)  # ~0.1m
            if mp.geom_type == "Polygon":
                mp = shapely.geometry.MultiPolygon([mp])
        else:
            mp = shapely.geometry.MultiPolygon([mp])

        for p in mp.geoms:
            pi = p.intersection(b)
            if b.contains(p):
                out = np.vstack((out, poly))
            elif not pi.is_empty:
                # assert(pi.geom_type,'MultiPolygon')
                if pi.geom_type == "Polygon":
                    pi = shapely.geometry.MultiPolygon([pi])

                for ppi in pi.geoms:
                    xy = np.asarray(ppi.exterior.coords)
                    xy = np.vstack((xy, xy[0]))
                    out = np.vstack((out, xy, [nan, nan]))

                del (ppi, xy)
            del pi
        del (p, mp)

    logger.debug("Exiting:_clip_polys")

    return out


def _nth_simplify(polys, bbox):
    """Collapse segments in `polys` outside of `bbox`"""
    logger.debug("Entering:_nth_simplify")

    boubox = np.asarray(_create_boubox(bbox))
    path = mpltPath.Path(boubox)
    polys = _convert_to_list(polys)
    out = []
    for poly in polys:
        j = 0
        inside = path.contains_points(poly[:-2, :])
        line = np.empty(shape=(0, 2))
        while j < len(poly[:-2]):
            if inside[j]:  # keep point (in domain)
                line = np.append(line, [poly[j, :]], axis=0)
            else:  # pt is outside of domain
                bd = min(
                    j + 50, len(inside) - 1
                )  # collapses 50 pts to 1 vertex (arbitary)
                exte = min(50, bd - j)
                if sum(inside[j:bd]) == 0:  # next points are all outside
                    line = np.append(line, [poly[j, :]], axis=0)
                    line = np.append(line, [poly[j + exte, :]], axis=0)
                    j += exte
                else:  # otherwise keep
                    line = np.append(line, [poly[j, :]], axis=0)
            j += 1
        line = np.append(line, [[nan, nan]], axis=0)
        out.append(line)

    logger.debug("Exiting:_nth_simplify")
    return _convert_to_array(out)


def _is_path_ccw(_p):
    """Compute curve orientation from first two line segment of a polygon.
    Source: https://en.wikipedia.org/wiki/Curve_orientation
    """
    detO = 0.0
    O3 = np.ones((3, 3))

    i = 0
    while (i + 3 < _p.shape[0]) and np.isclose(detO, 0.0):
        # Colinear vectors detected. Try again with next 3 indices.
        O3[:, 1:] = _p[i : (i + 3), :]
        detO = np.linalg.det(O3)
        i += 1

    if np.isclose(detO, 0.0):
        raise RuntimeError("Cannot determine orientation from colinear path.")

    return detO > 0.0


def _is_overlapping(bbox1, bbox2):
    """Determines if two axis-aligned boxes intersect"""
    x1min, x1max, y1min, y1max = bbox1
    x2min, x2max, y2min, y2max = bbox2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max


def remove_dup(arr: np.ndarray):
    """Remove duplicate element from np.ndarray"""
    result = np.concatenate((arr[np.nonzero(np.diff(arr))[0]], [arr[-1]]))

    return result


class Shoreline(Region):
    """
    The shoreline class extends :class:`Region` to store data
    that is later used to create signed distance functions to
    represent irregular shoreline geometries. This data
    is also involved in developing mesh sizing functions.

    Parameters
    ----------
    shp : str or pathlib.Path
        Path to shapefile containing shoreline data.
    bbox : tuple | numpy.ndarray | oceanmesh.Region
        Bounding box of the region of interest OR a Region object.
        If a Region object is passed, its bbox and crs are automatically used.
        If a tuple/array is passed, the expected bbox format is (xmin, xmax, ymin, ymax).
    h0 : float
        Minimum grid spacing.
    crs : str, optional
        Coordinate reference system to interpret bbox when bbox is a tuple/array.
        Ignored when bbox is a Region object (the Region's CRS is used).
    refinements : int, optional
        Number of refinements to apply to the shoreline. Default is 1.
    minimum_area_mult : float, optional
        Minimum area multiplier. Default is 4.0.
        Note that features with area less than h0*minimum_area_mult
        are removed.
    smooth_shoreline : bool, optional
        Smooth the shoreline. Default is True.
    stereo : bool, optional
        Use stereographic projection handling intended for global meshes. Set True for
        global world meshes (with EPSG:4326 inputs), and False for regional meshes. This
        flag is retained on the Shoreline instance and propagated to downstream Domain
        objects for validation in generate_multiscale_mesh.
    """

    def __init__(
        self,
        shp,
        bbox,
        h0,
        crs="EPSG:4326",
        refinements=1,
        minimum_area_mult=4.0,
        smooth_shoreline=True,
        smoothing_method="moving_average",
        smoothing_window=5,
        coarsen_outside=True,
        coarsen_inflation=1.10,
        stereo=False,
    ):
        if isinstance(shp, str):
            shp = Path(shp)

        # Determine bbox coordinates and CRS to use
        crs_to_use = crs

        if isinstance(bbox, Region):
            bbox_coords = bbox.bbox
            region_crs = bbox.crs
            if crs != "EPSG:4326":
                logger.warning(
                    "Shoreline: Both a Region object and an explicit 'crs' were provided; "
                    "the Region's CRS will take precedence (crs=%s).",
                    region_crs,
                )
            crs_to_use = region_crs
        else:
            bbox_coords = bbox
            # Backward compatibility for tuple/array input; optional auto-detection
            if isinstance(bbox_coords, tuple) and crs == "EPSG:4326":
                looks_geo = _infer_crs_from_coordinates(bbox_coords)
                if (
                    not looks_geo
                ):  # Only inspect shapefile native CRS when clearly projected
                    try:
                        native = gpd.read_file(shp).crs
                    except Exception:
                        native = None
                    if (native is not None) and (
                        CRS.from_user_input(native) != CRS.from_user_input("EPSG:4326")
                    ):
                        logger.info(
                            "Shoreline: bbox looks projected but 'crs' not specified; using shapefile's native CRS %s",
                            native,
                        )
                        crs_to_use = native

        # Build polygon representation and normalized bbox tuple
        if isinstance(bbox_coords, tuple):
            _boubox = np.asarray(_create_boubox(bbox_coords))
            bbox_tuple = bbox_coords
        else:
            _boubox = np.asarray(bbox_coords)
            if not _is_path_ccw(_boubox):
                _boubox = np.flipud(_boubox)
            bbox_tuple = (
                np.nanmin(_boubox[:, 0]),
                np.nanmax(_boubox[:, 0]),
                np.nanmin(_boubox[:, 1]),
                np.nanmax(_boubox[:, 1]),
            )

        super().__init__(bbox_tuple, crs_to_use)

        self.shp = shp
        self.h0 = h0
        # Retain stereo flag for downstream validation (e.g., multiscale mixing)
        self.stereo = bool(stereo)
        self.inner = []
        self.outer = []
        self.mainland = []
        self.boubox = _boubox
        self.refinements = refinements
        self.minimum_area_mult = minimum_area_mult

        polys = self._read()

        if stereo:
            self.bbox = (
                np.nanmin(polys[:, 0] * 0.99),
                np.nanmax(polys[:, 0] * 0.99),
                np.nanmin(polys[:, 1] * 0.99),
                np.nanmax(polys[:, 1] * 0.99),
            )  # so that bbox overlaps with antarctica > and becomes the outer boundary
            self.boubox = np.asarray(_create_boubox(self.bbox))

        # OM2D Read_shapefile area culls on RAW polygons: islands
        # wholly inside < 4*h0^2, partially-inside mainland
        # < 100*h0^2 (Read_shapefile.m:219,231)
        polys = _cull_small_features(
            polys, self.boubox, self.h0,
            island_mult=self.minimum_area_mult, mainland_mult=100.0
        )

        if coarsen_outside:
            # OM2D coarsen_polygon: decimate detail outside the
            # (inflated) domain polygon
            polys = _coarsen_outside(
                polys, self.boubox, inflation=coarsen_inflation
            )

        # OM2D my_interpm: densify (insert-only, NEVER decimate) so
        # gaps < h0/2 (geodata.m:382-397)
        polys = _densify(polys, 0.5 * self.h0, self.bbox)

        # OM2D smooth_coastline: 5-point boxcar AFTER densification
        # (geodata.m:400-415); chaikin retained as an opt-in only
        if smooth_shoreline:
            if smoothing_method == "moving_average":
                polys = _moving_average_smooth(
                    polys, window=smoothing_window
                )
            elif smoothing_method == "chaikin":
                polys = _smooth_shoreline(polys, self.refinements)
            elif smoothing_method is not None:
                raise ValueError(
                    "smoothing_method must be 'chaikin', "
                    "'moving_average', or None"
                )

        polys = _clip_polys(polys, self.bbox)

        # keep the PRE-densification region ring: _classify_shoreline
        # densifies/clips self.boubox, after which it no longer
        # reconstructs the region polygon (edge-wise NaN segments,
        # duplicate vertices) — the multiscale covering test needs
        # the true region (Obitsu I11/J11 incident).
        self.region_polygon = np.asarray(self.boubox, dtype=float)
        self.inner, self.mainland, self.boubox = _classify_shoreline(
            self.bbox, self.boubox, polys, self.h0 / 2, self.minimum_area_mult, stereo
        )

    @property
    def shp(self):
        return self.__shp

    @shp.setter
    def shp(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
        self.__shp = filename

    @property
    def refinements(self):
        return self.__refinements

    @refinements.setter
    def refinements(self, value):
        if value < 0:
            raise ValueError("Refinements must be > 0")
        self.__refinements = value

    @property
    def minimum_area_mult(self):
        return self.__minimum_area_mult

    @minimum_area_mult.setter
    def minimum_area_mult(self, value):
        if value <= 0.0:
            raise ValueError(
                "Minimum area multiplier * h0**2 to "
                " prune inner geometry must be > 0.0"
            )
        self.__minimum_area_mult = value

    @property
    def h0(self):
        return self.__h0

    @h0.setter
    def h0(self, value):
        if value <= 0:
            raise ValueError("h0 must be > 0")
        self.__h0 = value

    @staticmethod
    def transform_to(gdf, dst_crs):
        """Transform geodataframe ``gdf`` representing
        a shoreline to dst_crs
        """
        dst_crs = CRS.from_user_input(dst_crs)
        if gdf.crs is None:
            # no .prj sidecar (e.g. OM2D's PostSandyNCEI shapefile):
            # assume the data is already in dst_crs, as OM2D's
            # CRS-agnostic reader effectively does
            logger.warning(
                f"Shapefile has no CRS; assuming {dst_crs.srs}"
            )
            gdf = gdf.set_crs(dst_crs)
        elif not gdf.crs.equals(dst_crs):
            logger.info(f"Reprojecting shoreline from {gdf.crs} to {dst_crs}")
            gdf = gdf.to_crs(dst_crs)
        return gdf

    def _read(self):
        """Reads a ESRI Shapefile from `filename` ∩ `bbox`"""
        if not isinstance(self.bbox, tuple):
            _bbox = (
                np.amin(self.bbox[:, 0]),
                np.amax(self.bbox[:, 0]),
                np.amin(self.bbox[:, 1]),
                np.amax(self.bbox[:, 1]),
            )
        else:
            _bbox = self.bbox

        logger.debug("Entering: _read")

        msg = f"Reading in vector file: {self.shp}"
        logger.info(msg)

        # Load once to get native CRS, then transform if necessary
        gdf = gpd.read_file(self.shp)
        native_crs = gdf.crs
        # transform if necessary
        s = self.transform_to(gdf, self.crs)

        # Explode to remove multipolygons or multi-linestrings (if present)
        s = s.explode(index_parts=True)

        polys = []  # store polygons

        delimiter = np.empty((1, 2))
        delimiter[:] = np.nan
        re = numpy.array([0, 2, 1, 3], dtype=int)

        for g in s.geometry:
            # extent of geometry
            bbox2 = [g.bounds[r] for r in re]
            if _is_overlapping(_bbox, bbox2):
                if g.geom_type == "LineString":
                    poly = np.asarray(g.coords)
                elif g.geom_type == "Polygon":  # a polygon
                    poly = np.asarray(g.exterior.coords.xy).T
                else:
                    raise ValueError(f"Unsupported geometry type: {g.geom_type}")

                poly = remove_dup(poly)
                polys.append(np.vstack((poly, delimiter)))

        if len(polys) == 0:
            cur_crs = None
            try:
                cur_crs = CRS.from_user_input(self.crs).to_string()
            except Exception:
                cur_crs = str(self.crs)
            native_str = str(native_crs) if native_crs is not None else "None"
            raise ValueError(
                "Shoreline data does not intersect with bbox. "
                f"Shoreline CRS in use: {cur_crs}; shapefile native CRS: {native_str}; "
                f"bbox={_bbox}.\n"
                "If your bbox is in a different CRS, prefer passing a Region object: "
                "Shoreline(fname, region, h0).\n"
                "Alternatively, specify the CRS explicitly when using a tuple bbox: "
                "Shoreline(fname, bbox, h0, crs=YOUR_CRS)."
            )

        logger.debug("Exiting: _read")

        return _convert_to_array(polys)

    def plot(
        self,
        ax=None,
        xlabel=None,
        ylabel=None,
        title=None,
        file_name=None,
        show=True,
        xlim=None,
        ylim=None,
    ):
        """Visualize the content in the shp field of Shoreline"""
        flg1, flg2 = False, False

        if ax is None:
            fig, ax = plt.subplots()
            ax.axis("equal")

        if len(self.mainland) != 0:
            (line1,) = ax.plot(self.mainland[:, 0], self.mainland[:, 1], "k-")
            flg1 = True
        if len(self.inner) != 0:
            (line2,) = ax.plot(self.inner[:, 0], self.inner[:, 1], "r-")
            flg2 = True
        (line3,) = ax.plot(self.boubox[:, 0], self.boubox[:, 1], "g--")

        xmin, xmax, ymin, ymax = self.bbox
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=None,
            hatch="////",
            alpha=0.2,
        )

        border = 0.10 * (xmax - xmin)
        if ax is None:
            plt.xlim(xmin - border, xmax + border)
            plt.ylim(ymin - border, ymax + border)

        ax.add_patch(rect)

        if flg1 and flg2:
            ax.legend((line1, line2, line3), ("mainland", "inner", "outer"))
        elif flg1 and not flg2:
            ax.legend((line1, line3), ("mainland", "outer"))
        elif flg2 and not flg1:
            ax.legend((line2, line3), ("inner", "outer"))

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

        ax.set_aspect("equal", adjustable="box")

        if show:
            plt.show()
        if file_name is not None:
            plt.savefig(file_name)
        return ax


class DEM(Grid):
    """
    Digitial elevation model read in from a tif or NetCDF file
    """

    def __init__(self, dem, crs="EPSG:4326", bbox=None, extrapolate=False, backup=None):
        """Read in a DEM from a tif or NetCDF file for later use
        in developing mesh sizing functions.

        Parameters
        ----------
        dem : str or pathlib.Path
            Path to the DEM file
        crs : str, optional
            Coordinate reference system to interpret bbox when bbox is a tuple. If a Region
            object is provided for bbox, the Region's CRS is used and this parameter is ignored.
        bbox : oceanmesh.Region, optional
            Bounding box of the DEM. Prefer passing a Region object; when provided, both the
            bbox extents and Region CRS are used. If None, the entire DEM is read.
        extrapolate : bool, optional
            Extrapolate the DEM outside the bounding box, by default False
        """

        if isinstance(dem, str):
            dem = Path(dem)

        bbox, crs, _region, region_crs, region_bbox = _prepare_region_bbox_and_crs(
            bbox, crs
        )

        if dem.exists():
            logger.info(f"Reading in {dem}")

            bbox, crs, meta, nodata_value, topobathy = _read_dem_array_and_meta(
                dem,
                bbox=bbox,
                crs=crs,
                region_bbox=region_bbox,
                region_crs=region_crs,
            )
            self.meta = meta

            topobathy = topobathy.astype(np.float64)
            topobathy[topobathy == nodata_value] = np.nan
        elif not dem.exists():
            raise FileNotFoundError(f"File {dem} could not be located.")

        super().__init__(
            bbox=bbox,
            crs=crs,
            dx=self.meta["transform"][0],
            dy=abs(
                self.meta["transform"][4]
            ),  # Note: grid spacing in y-direction is negative.
            values=np.fliplr(topobathy),  # we need to flip the array
            extrapolate=extrapolate,  # user-specified potentially "dangerous" option
        )
        super().build_interpolant()

        if backup is not None:
            # OM2D 'backupdem' (Fb2): fill primary-DEM holes from a
            # secondary DEM so downstream sizing/interp never see NaN
            # where the backup has coverage.
            holes = ~np.isfinite(self.values)
            if holes.any():
                b2 = backup if isinstance(backup, DEM) else DEM(
                    backup, crs=crs, bbox=bbox, extrapolate=True
                )
                xg, yg = self.create_grid()
                fill = np.asarray(b2.eval(
                    np.column_stack([xg[holes], yg[holes]])
                ), dtype=float)
                vals = np.asarray(self.values, dtype=float)
                vals[holes] = fill
                self.values = vals
                super().build_interpolant()
                logger.info(
                    f"backup DEM filled {int(holes.sum())} cells"
                )

    def flip(self):
        """Flip the DEM upside down"""
        self.values = -self.values
        super().build_interpolant()
        return self

    def plot(self, coarsen=1, holding=False, **kwargs):
        """Visualize the DEM"""
        fig, ax, pc = super().plot(
            coarsen=coarsen,
            holding=True,
            cmap="terrain",
            **kwargs,
        )
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.set_aspect("equal")
        cbar = fig.colorbar(pc)
        cbar.set_label("Topobathymetric depth (m)")
        if not holding:
            plt.show()
        return fig, ax
