# Policy: no silent fallbacks (2026-07-11, user directive)

Automatic fallbacks to an alternative implementation, solver, or
data path are FORBIDDEN. When the primary path fails or is detected
as unreliable, the code STOPS with an error that states (1) what was
detected, with numbers, and (2) the available options, including the
explicit opt-in parameter for the alternative. Alternatives exist
only as explicit user choices.

Rationale: a silent fallback lets an implementation that differs
from the intent complete as if the intent were fulfilled. The Tokyo
Bay incident that motivated the policy: GDAL mis-georeferenced two
self-sliced DEMs; an automatic switch to the xarray reader — itself
carrying a latent upside-down bug — produced a "successful" mesh
from garbage bathymetry.

Converted call sites:

| site | old behaviour | now |
|---|---|---|
| DEM/NetCDF subset (geodata) | auto-switch to coordinate-based reader on GDAL/coords mismatch | ValueError with both spacings; options `nc_reader='coords'` / `'gdal-unchecked'` |
| DEM bbox-no-overlap branch | auto xarray subset | ValueError + `nc_reader='coords'` hint |
| finalize_sizing gradation | numba missing -> silent legacy gradient_limit | ImportError; explicit `gradation_solver='legacy'` |
| _cull_small_features | degenerate boubox -> silently skip culling | ValueError naming the boubox |
| sizing-grid hmin (multiscale + generate_mesh) | silently recompute hmin from grid minimum | ValueError: set grid.hmin / pass min_edge_length |
| remesh_patch engine (precedent) | — | already ImportError + `engine='distmesh'` option |

Reviewed and kept (not divergent-implementation fallbacks):
- DEM subset npz cache: cache miss recomputes the identical result.
- `_pick_netcdf_open_target`: chooses among GDAL subdatasets of the
  same file; wrong picks are caught by the coordinate-spacing gate.
- Per-segment `except` in _cull_small_features: retains an
  unparseable segment uncculled (data-preserving, logged).

Known fixture note: `tests/galv_sub.nc` is mis-georeferenced under
GDAL (coords 1/3 arcsec, GDAL stretches to ~19x); its tests validate
sizing math on the stretched raster and now opt in explicitly with
`nc_reader='gdal-unchecked'`.
