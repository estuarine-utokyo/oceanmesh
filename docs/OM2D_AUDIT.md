# OceanMesh2D -> oceanmesh port audit

Date: 2026-07-07. Method: line-level reading of both code bases by three
parallel auditors (shoreline chain / build+clean chain / edgefx+builtins),
each verdict backed by file:line evidence. Trigger: coastal mesh quality
far below OM2D (boundary spike triangles, land/sea misassignments) in the
1:1 translation of `mesh_wide_varres_3r.m`.

## Executive summary

The fork is NOT missing large features. It is missing (a) one critical
**wiring** (the automatic post-build clean), (b) several **defaults and
orderings** that differ from OM2D, and (c) a handful of localized
**bugs**. Two long-held assumptions were refuted by code:

1. OM2D does **not** decimate/simplify the shoreline to resolution.
   `my_interpm` only inserts vertices (never removes;
   `utilities/my_interpm.m:25-26`). The "resolution-appropriate
   smoothness" of OM2D coastlines comes from: densify to h0/2 ->
   5-point boxcar moving average (`smooth_coastline` -> `fastsmooth`,
   effective length ~2.5*h0) -> area culling of small features ->
   mesh-scale boundary conformity + the automatic post-build clean.
2. OM2D's `meshgen.build` **always** ends with a full topological clean
   (`clean('default', djc=0.25)`, `@meshgen/meshgen.m:1063-1068`) —
   boundary-quality deletion loop (db=0.25), thin-triangle collapse
   (0.25), boundary traversability, valence bound (9), FEM smoothing,
   and a recursive re-clean while min quality < 0.25. The fork's
   `generate_mesh` returns **raw distmesh output** (only exterior
   removal + fix_mesh); the ported cleaning primitives exist but are
   never invoked by the generator, and `mesh_clean`'s boundary-quality
   default is 0.01 single-pass (vs 0.25 looped).

## P0 — direct causes of the observed coastline failures

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | No automatic post-build clean; boundary spikes (e.g. Urayasu K5/K6 triangles) survive | meshgen.m:1063-1068 vs mesh_generator.py:1040-1043 | Wire OM2D 'default' clean into `generate_mesh` tail: db=0.25 while-loop over boundary-touching elements, collapse_thin_triangles(0.25), make_mesh_boundaries_traversable (with the forced-entry pass of Make_Mesh_Boundaries_Traversable.m:77-79), bound_con(9), smoothing, recursive re-clean (mqa=0.25). Expose `cleanup=` kwarg. |
| 2 | Shoreline smoothing default: 1-pass Chaikin on raw vertices; OM2D = 5-pt boxcar AFTER densify | geodata.py:1088-1090,1172-1174 vs geodata.m:382-415 | Default `smoothing_method="moving_average"`, window 5, applied after densify-to-h0/2; align `_moving_average_smooth` end handling with fastsmooth(ends=1). |
| 3 | The fork's `_resample_segments` decimates to h0/2 — an operation OM2D does not have; deletes detail OM2D keeps | geodata.py:501-539 vs my_interpm.m:25-26 | Remove (revert to densify-only). Performance is recovered by P0-4. |
| 4 | Area culling too permissive / missing: islands culled at h0^2 (OM2D 4*h0^2; a /2 argument bug), mainland 100*h0^2 cull entirely absent | geodata.py:618 vs Read_shapefile.m:219,231 | Cull islands < 4*h0^2 and mainland fragments < 100*h0^2 on raw polygons before resampling. Also the true reason OM2D handles GSHHS_f fast at coarse h0. |
| 5 | Slope sizing clamp inverted: all water shallower than 50 m flattened to -50 => nearshore slope sizing disabled | edgefx.py:485-486 vs edgefx.m:456 (clamps only z>50 topography) | Clamp topography only; keep shelf gradients. |
| 6 | Rossby-radius low-pass is a no-op: filtered block overwritten just before gradient | edgefx.py:734 | Use the filtered array; delete pad rows/cols as OM2D does (edgefx.m:561-576). |

## P1 — systematic accuracy gaps (deviations, not crashes)

| # | Deviation | Evidence |
|---|---|---|
| 7 | Distance/feature/KD computations in raw lon/lat degrees; OM2D projects to Mercator metres first (WrapperForKsearch.m:13). ~18% anisotropy at 35N | edgefx.py:417,843 |
| 8 | Gradation limiter isotropic scalar degrees; OM2D limgradStruct uses dx=h0*cos(lat), dy=h0 metres, 8-node stencil | grid.py:117-119, finalize.py:249-265 vs edgefx.m:895-921 |
| 9 | Sizing interpolant extrapolates linearly outside bbox; OM2D griddedInterpolant(...,'nearest') holds edge values | grid.py:494-526 vs edgefx.m:1014 |
| 10 | NaN handling in min-combine / max_el: MATLAB min skips NaN and max_el pass scrubs NaN->max_el; np.minimum propagates NaN | finalize.py:198,212,218 vs edgefx.m:796,858-867 |
| 11 | Auto-dt computed on the combined grid over the full rect; OM2D uses the dis/fs grid; also OM2D edgefx grids stay finite outside the domain (no polygon-NaN) | finalize.py:82-131 vs edgefx.m:874-884 |
| 12 | Finalize order: OM2D = min -> weirs -> nearshore cap -> floor h0 -> max_el(NaN scrub) -> gradation -> CFL; fork floors last and applies max_el before weirs | finalize.py:164-287 vs edgefx.m:764-1017 |
| 13 | CA bathymetry: stencil = centroid bounding box (OM2D) vs circular half-edge radius (fork); min/maxdepth clamp per DEM cell BEFORE averaging (OM2D) vs after (fork); ignoreOL/slope_calc/lut unported | GridData.m:270-321 vs bathymetry.py:24-93 |
| 14 | make_mesh_boundaries_traversable lacks OM2D forced-entry pass; mesh_clean smooths with laplacian2 instead of direct_smoother_lur | clean.py:160-220 vs Make_Mesh_Boundaries_Traversable.m:77-79 |

## P2 — minor numeric differences

- wavelength `meters_per_degree`: `cos(2*lat)` missing `np.radians` (edgefx.py:923-927)
- g = 9.81 vs 9.807 in calc_cfl; wavelength g 9.81 vs 9.807
- default gradation 0.15 (fork) vs 0.20 (OM2D)
- Haversine sphere R=6378206.4 vs m_lldist ellipsoid
- densify threshold h0 (fork) vs h0/2 (OM2D); boubox densified at h0/4 effective
- legacy slope `-999` histcounts path unported
- `_clip_polys` and boubox boolean difference are fork-only extra steps
- filt2 boundary mode 'nearest' vs MATLAB conv2 reflect

## MATLAB builtins that matter (inventory)

| Builtin | OM2D use | Fork equivalent | Identical? |
|---|---|---|---|
| griddedInterpolant('linear','nearest') | edgefx interpolant | RegularGridInterpolator(fill_value=None) | NO (extrapolation mode) |
| scatteredInterpolant('linear','nearest') | bathy re-map in bound_courant | cKDTree nearest-carry | NO |
| inpoly (Engwirda) | all PIP | inpoly2 (same algorithm; numba kernel referee-verified) | YES |
| knnsearch via Mercator projection | all NN distances | cKDTree on raw degrees | NO (cos-lat) |
| my_interpm | densify-only | _densify (+ the P0-3 resampler to remove) | after P0-3: YES |
| min(...,[],3) NaN-skip | sizing combine | np.minimum.reduce (NaN-propagating) | NO |
| filt2 (Gaussian) | slope filters | scipy gaussian_filter | close |
| unique/accumarray/intersect rows | edges/dedup | np.unique/np.add.at equivalents | YES |
| m_lldist (ellipsoid) | bar lengths | Haversine sphere | ~ (sub-%) |
| EarthGradient | slope | _earth_gradient | YES |

## Implementation order

P0 items 1-6 first (each with a regression test; re-run the translation
acceptance after 1-4 and inspect the fixed checklist sites: Urayasu,
Futtsu, Kannonzaki, Tokyo/Yokohama ports, bay mouth). Then P1 7-12 in
that order; P2 opportunistically. Every geometry-touching change must
pass a real-multi-ring equivalence/referee test before merging (lesson
of the pip-inpoly incident).


## Known issues (parked)

- `bossen_heckbert` force function (upstream extra, unused by any
  OM2D example): its point cloud collapses to a handful of vertices
  under the in-loop improvement cycle regardless of seeding lattice;
  test marked xfail. Investigate when/if that force function is ever
  needed.
- `test_global_regional_multiscale_australia`: nondeterministic
  pass/fail on identical code (pre-existing flake).
