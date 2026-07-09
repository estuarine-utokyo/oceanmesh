"""OM2D-parity sizing finalization.

Ports the OceanMesh2D ``edgefx.finalize`` pipeline (fixed order:
min-combine -> nearshore cap -> max_el bounds -> floor -> gradation
-> CFL limiting -> interpolant) and its per-elevation-band
parameterization (Nx3 ``[value, zmin, zmax]`` rows) to Python.

References: OceanMesh2D @edgefx/edgefx.m (finalize, dt/Courant
handling), utilities/limgradStruct.m (spatially variable grade).
"""

import logging

import numpy as np
import skfmm
from _HamiltonJacobi import gradient_limit

from . import edges
from .geometry import inpoly2
from .grid import Grid, compute_minimum

logger = logging.getLogger(__name__)

__all__ = [
    "elevation_bands",
    "enforce_courant_bounds",
    "enforce_nearshore_max_edge",
    "finalize_sizing",
    "wave_celerity",
]

GRAV = 9.807
_M2_PERIOD = 12.42 * 3600.0


def elevation_bands(param, elevation, default=np.nan):
    """Expand an OM2D-style parameter (scalar or Nx3 rows of
    ``[value, zmin, zmax]``) onto an ``elevation`` array.

    Cells outside every band get ``default``.
    """
    arr = np.asarray(param, dtype=float)
    if arr.ndim == 0:
        return np.full_like(np.asarray(elevation, dtype=float),
                            float(arr))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            "banded parameter must be scalar or Nx3 [value, zmin, zmax]"
        )
    out = np.full_like(np.asarray(elevation, dtype=float), default)
    for value, zmin, zmax in arr:
        sel = (elevation >= zmin) & (elevation < zmax)
        out[sel] = value
    return out


def wave_celerity(depth, wave_amplitude=1.0):
    """OM2D wave speed: ``sqrt(g*|z|) + amp*sqrt(g/|z|)`` after the
    SIGNED clamp ``tmpz(tmpz > -1) = -1`` (edgefx.m:940-941): land
    and shallows become 1 m water. ``abs()`` alone turned mountain
    cells into kilometre-deep water (u ~70 m/s on Puerto Rico ->
    auto-dt 0.22 s vs OM2D 1.26 s)."""
    z = np.where(np.asarray(depth) > -1.0, -1.0, depth)
    d = np.abs(z)
    return np.sqrt(GRAV * d) + wave_amplitude * np.sqrt(GRAV / d)


def _dem_on_grid(grid, dem):
    xg, yg = grid.create_grid()
    z = np.asarray(
        dem.eval(np.column_stack([xg.ravel(), yg.ravel()]))
    ).reshape(xg.shape)
    return z


def _deg_factor(grid):
    """Metres per degree on this grid (OM2D's 111e3 convention when
    the CRS is geographic; 1.0 for projected grids)."""
    crs = str(grid.crs).upper() if grid.crs is not None else ""
    if "4326" in crs or "WGS84" in crs.replace(" ", ""):
        return 111e3
    return 1.0


def enforce_courant_bounds(
    grid,
    dem,
    timestep=0.0,
    courant_min=None,
    courant_max=0.5,
    wave_amplitude=1.0,
):
    """OM2D ``dt`` limiter (two-sided Courant bounds) on a combined
    sizing grid.

    ``timestep=0`` selects the automatic timestep
    ``dt = min(courant_max * h / u)`` over the grid (OM2D dt=0).
    Cells with ``Cr > courant_max`` are raised to ``u*dt/courant_max``
    (deep water coarsens); with ``courant_min`` set, cells with
    ``Cr < courant_min`` are lowered to ``u*dt/courant_min``.
    Sizes are handled in metres internally (OM2D planar-metres
    enforcement).
    """
    factor = _deg_factor(grid)
    hh_m = np.asarray(grid.values, dtype=float) * factor
    z = _dem_on_grid(grid, dem)
    # sanitize DEM: fill values (|z| ~ 1e6+) masqueraded as depth
    # and produced km/s celerities, collapsing the automatic dt by
    # 1000x; clamp to physical ocean depths
    z = np.where(np.isfinite(z), z, 0.0)
    z = np.clip(z, -11000.0, 11000.0)
    u = wave_celerity(z, wave_amplitude=wave_amplitude)

    dt = float(timestep)
    if dt == 0.0:
        dt = float(np.nanmin(courant_max * hh_m / u))
        logger.info(
            f"Automatic timestep selection: dt = {dt:.3f} s "
            f"(Cr_max = {courant_max})"
        )
    cr = dt * u / hh_m
    n_up = int((cr > courant_max).sum())
    hh_m = np.where(cr > courant_max, u * dt / courant_max, hh_m)
    n_dn = 0
    if courant_min is not None:
        if courant_min > courant_max:
            raise ValueError("courant_min > courant_max")
        n_dn = int((cr < courant_min).sum())
        hh_m = np.where(cr < courant_min, u * dt / courant_min, hh_m)
    logger.info(
        f"Courant bounds dt={dt:.3f}s: raised {n_up}, lowered {n_dn}"
    )
    grid.values = hh_m / factor
    return grid, dt


def enforce_nearshore_max_edge(grid, shoreline, max_edge_length_ns,
                               band_factor=2.0 / 3.0):
    """OM2D ``max_el_ns``: cap sizes within the nearshore band
    (|distance to coast| < ``band_factor * h0``, distances on the
    sizing lattice via fast marching)."""
    factor = _deg_factor(grid)
    xg, yg = grid.create_grid()
    phi = np.ones(grid.values.shape)
    points = np.vstack((shoreline.inner, shoreline.mainland))
    if len(points) == 0:
        return grid
    boubox = np.nan_to_num(shoreline.boubox)
    e_box = edges.get_poly_edges(shoreline.boubox)
    try:
        inside, _ = inpoly2(points, boubox, e_box)
        points = points[inside]
    except Exception:  # pragma: no cover - degenerate boubox
        pass
    indices = grid.find_indices(points, xg, yg)
    phi[indices] = -1.0
    dis = np.abs(skfmm.distance(phi, [grid.dx, grid.dy]))
    band = dis < band_factor * shoreline.h0
    cap = max_edge_length_ns / factor
    n = int((band & (np.asarray(grid.values) > cap)).sum())
    grid.values = np.where(band, np.minimum(grid.values, cap),
                           grid.values)
    logger.info(f"max_el_ns: capped {n} nearshore cells")
    return grid


try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _limgrad_struct_kernel(ny, xeglen, yeglen, ffun, fdfdx, imax):
        """Faithful port of utilities/limgradStruct.m (Engwirda/
        Roberts): active-set gradient limiting on the structured
        graph, 8-node stencil, per-row x edge lengths (anisotropy
        dx = h0*cos(lat)), bound by the LOWER node's dfdx."""
        n = ffun.shape[0]
        aset = np.zeros(n, np.int64)
        ftol = ffun.min() * np.sqrt(2.220446049250313e-16)
        dgeglen = np.sqrt(xeglen**2 + yeglen**2)
        eglen = 0.5 * (xeglen + yeglen)
        npos = np.empty(9, np.int64)
        elens = np.empty(9, np.float64)
        for it in range(1, imax + 1):
            aidx = np.where(aset == it - 1)[0]
            if aidx.size == 0:
                break
            order = np.argsort(ffun[aidx])
            for k in range(aidx.size):
                inod = aidx[order[k]]
                ipos = 1 + inod // ny
                jpos = inod - (ipos - 1) * ny + 1
                npos[0] = inod
                npos[1] = ipos * ny + (jpos - 1)
                npos[2] = (ipos - 2) * ny + (jpos - 1)
                npos[3] = (ipos - 1) * ny + min(jpos + 1, ny) - 1
                npos[4] = (ipos - 1) * ny + max(jpos - 1, 1) - 1
                npos[5] = ipos * ny + max(jpos - 1, 1) - 1
                npos[6] = ipos * ny + min(jpos + 1, ny) - 1
                npos[7] = (ipos - 2) * ny + min(jpos + 1, ny) - 1
                npos[8] = (ipos - 2) * ny + max(jpos - 1, 1) - 1
                j = jpos - 1
                elens[0] = eglen[j]
                elens[1] = xeglen[j]
                elens[2] = xeglen[j]
                elens[3] = yeglen
                elens[4] = yeglen
                elens[5] = dgeglen[j]
                elens[6] = dgeglen[j]
                elens[7] = dgeglen[j]
                elens[8] = dgeglen[j]
                for ne in range(1, 9):
                    nod2 = npos[ne]
                    if nod2 < 0 or nod2 >= n:
                        continue
                    nod1 = npos[0]
                    elen = elens[ne]
                    if ffun[nod1] > ffun[nod2]:
                        fun1 = ffun[nod2] + elen * fdfdx[nod2]
                        if ffun[nod1] > fun1 + ftol:
                            ffun[nod1] = fun1
                            aset[nod1] = it
                    else:
                        fun2 = ffun[nod1] + elen * fdfdx[nod2]
                        if ffun[nod2] > fun2 + ftol:
                            ffun[nod2] = fun2
                            aset[nod2] = it
        return ffun

    _HAVE_LIMGRAD_STRUCT = True
except Exception:  # pragma: no cover
    _HAVE_LIMGRAD_STRUCT = False


def _limgrad_struct(grid, gradation_field):
    """Run limgradStruct on a Grid: xeglen per row = dx*cos(lat)
    (edgefx.m:306-308 uses h0*cosd(lat) with dy=h0), column-major
    flattening (rows = y vary fastest), degrees throughout."""
    lats = grid.create_grid()[1][0, :]
    xeglen = grid.dx * np.cos(np.deg2rad(np.minimum(np.abs(lats), 85.0)))
    yeglen = float(grid.dy)
    ny = grid.values.shape[1]
    ffun = np.asarray(grid.values, dtype=float).flatten("F").copy()
    ffun = np.asarray(grid.values, dtype=float).ravel(order="C").copy()
    # rows must vary fastest: our values are (nx, ny) with y along
    # axis 1 -> C-order ravel gives y fastest, matching jpos=y
    fdfdx = np.asarray(gradation_field, dtype=float)
    if fdfdx.ndim == 0:
        fdfdx = np.full_like(ffun, float(fdfdx))
    else:
        fdfdx = fdfdx.ravel(order="C").astype(float)
    out = _limgrad_struct_kernel(ny, xeglen.astype(float), yeglen,
                                 ffun, fdfdx, 10000)
    return out.reshape(grid.values.shape, order="C")


def finalize_sizing(
    edge_lengths,
    dem=None,
    shoreline=None,
    *,
    hmin=None,
    max_edge_length=None,
    max_edge_length_nearshore=None,
    gradation=0.15,
    courant=None,
    weirs=None,
    verbose=True,
):
    """OM2D ``edgefx.finalize`` pipeline on a list of sizing grids.

    Order (fixed, as in OceanMesh2D):
      1. pointwise minimum across ``edge_lengths``
      2. nearshore cap (``max_edge_length_nearshore``, needs
         ``shoreline``)
      3. ``max_edge_length`` bound — scalar or Nx3 elevation bands
         (needs ``dem`` for the banded form)
      4. ``hmin`` floor
      5. gradation — scalar or Nx3 elevation bands (spatially
         variable grade via the extended Hamilton-Jacobi solver)
      6. Courant bounds (``courant`` = dict with keys ``timestep``
         (0 = auto), ``max`` (default 0.5), optional ``min``,
         ``wave_amplitude``; needs ``dem``)
      7. interpolant build

    Returns the finalized :class:`Grid` (and the effective timestep
    if Courant limiting ran, else None) as ``(grid, dt)``.
    """
    if not isinstance(edge_lengths, (list, tuple)):
        edge_lengths = [edge_lengths]
    grid = (compute_minimum(list(edge_lengths))
            if len(edge_lengths) > 1 else edge_lengths[0])
    factor = _deg_factor(grid)

    if max_edge_length_nearshore is not None:
        if shoreline is None:
            raise ValueError("max_edge_length_nearshore needs shoreline")
        grid = enforce_nearshore_max_edge(
            grid, shoreline, max_edge_length_nearshore
        )

    if max_edge_length is not None:
        # edgefx.m:855-870: limidx = hh > mx | isnan(hh) — the
        # max_el pass also SCRUBS NaN cells to max_el. np.minimum
        # propagates NaN instead (audit P1-10): NaN stripes from
        # DEM gaps (e.g. SRTM15+ over the Bahamas banks) punched
        # streaky holes into the ECGC mesh.
        vals = np.asarray(grid.values, dtype=float)
        arr = np.asarray(max_edge_length, dtype=float)
        if arr.ndim == 0:
            mx = float(arr) / factor
            limidx = (vals > mx) | ~np.isfinite(vals)
            vals[limidx] = mx
        else:
            if dem is None:
                raise ValueError("banded max_edge_length needs dem")
            z = _dem_on_grid(grid, dem)
            cap = elevation_bands(arr, z, default=np.inf) / factor
            limidx = (vals > cap) | (
                ~np.isfinite(vals) & np.isfinite(cap)
            )
            vals[limidx] = np.broadcast_to(cap, vals.shape)[limidx]
        grid.values = vals

    if weirs is not None:
        # edgefx.m:803-833: densify each crestline at h0/2 and SET
        # (hard override, not cap) hh = min_ele in a +-10-cell
        # window around every crest point
        from .weirs import _my_interpm

        xvec, yvec = grid.create_vectors()
        vals = np.asarray(grid.values, dtype=float)
        for w in weirs:
            crest = np.asarray(w["crestline"], dtype=float)
            spacing = float(
                w.get("min_ele_m") or w.get("spacing_m")
                or 2.0 * w["width_m"]
            ) / factor
            xy = _my_interpm(crest, 0.5 * float(grid.hmin))
            for q in xy:
                i = int(np.clip(np.searchsorted(xvec, q[0]), 0,
                                len(xvec) - 1))
                j = int(np.clip(np.searchsorted(yvec, q[1]), 0,
                                len(yvec) - 1))
                vals[max(0, i - 10):i + 11,
                     max(0, j - 10):j + 11] = spacing
        grid.values = vals

    if hmin is not None:
        grid.values = np.maximum(grid.values, hmin / factor)
        grid.hmin = hmin / factor

    grad = np.asarray(gradation, dtype=float)
    sz = (*grid.values.shape, 1)
    if grad.ndim == 0:
        if _HAVE_LIMGRAD_STRUCT:
            limited = _limgrad_struct(grid, grad).flatten("F")
        else:
            limited = gradient_limit(
                [*sz], grid.dx, float(grad), 10000,
                np.asarray(grid.values, dtype=float).flatten("F"),
            )
    else:
        if dem is None:
            raise ValueError("banded gradation needs dem")
        z = _dem_on_grid(grid, dem)
        dfdx = elevation_bands(grad, z, default=float(np.nanmax(grad[:, 0])))
        limited = gradient_limit(
            [*sz], grid.dx, dfdx.flatten("F"), 10000,
            np.asarray(grid.values, dtype=float).flatten("F"),
        )
    grid.values = np.reshape(limited, grid.values.shape, "F")

    dt_eff = None
    if courant is not None:
        if dem is None:
            raise ValueError("courant limiting needs dem")
        _ts = float(courant.get("timestep", 0.0))
        _cr_max = float(courant.get("max", 0.5))
        if _ts == 0.0:
            # edgefx.m:983-985 (audit P1-11): the automatic
            # timestep uses the DISTANCE/feature field hh_d
            # (the FIRST sizing function), floored at h0 — NOT
            # the wl/CFL-combined grid
            _feat = edge_lengths[0]
            _hd = np.asarray(
                getattr(_feat, "values_raw", _feat.values),
                dtype=float) * factor
            _hd = np.maximum(_hd, float(hmin) if hmin is not None
                             else _hd.min())
            _z = _dem_on_grid(_feat, dem)
            _z = np.where(np.isfinite(_z), _z, 0.0)
            _z = np.clip(_z, -11000.0, 11000.0)
            _u = wave_celerity(
                _z, wave_amplitude=float(
                    courant.get("wave_amplitude", 1.0)))
            _ts = float(np.nanmin(_cr_max * _hd / _u))
            logger.info(
                f"Automatic timestep from the distance field: "
                f"dt = {_ts:.4f} s (Cr_max = {_cr_max})"
            )
        grid, dt_eff = enforce_courant_bounds(
            grid, dem,
            timestep=_ts,
            courant_min=courant.get("min"),
            courant_max=_cr_max,
            wave_amplitude=float(courant.get("wave_amplitude", 1.0)),
        )

    grid.build_interpolant()
    if verbose:
        vals = np.asarray(grid.values) * factor
        logger.info(
            f"finalize_sizing: h [m] p1/p50/p99 = "
            f"{np.nanpercentile(vals, [1, 50, 99]).round(1)}"
            + (f", dt = {dt_eff:.2f} s" if dt_eff else "")
        )
    return grid, dt_eff
