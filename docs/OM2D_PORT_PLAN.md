# OceanMesh2D -> oceanmesh: Full Port Plan

Status: DRAFT for approval (2026-07-06). Owner: fvcom-mesh-tools
project. Target branch: `main` (this fork). C++ builds follow the
Intel oneAPI icx policy already used for the CDT extension.

Goal (user directive): port the complete OceanMesh2D (MATLAB)
feature set to Python oceanmesh. This is the "stage 1" generator.
Stage 2 (SMS-equivalent automated finishing) is a separate,
original development on top and is NOT part of this plan.

Sources inventoried: OceanMesh2D @edgefx/@geodata/@meshgen/@msh +
utilities (MATLAB, ~/Github/OceanMesh2D) vs this fork
(~/Github/oceanmesh). Inventory date 2026-07-06.

## 1. Gap matrix

Legend: OK = parity exists; PART = partial; MISS = absent.
"Use" = importance for the Tokyo Bay FVCOM pipeline (H/M/L).

### 1.1 Sizing (edgefx)

| OM2D feature | fork status | Use | Action |
|---|---|---|---|
| `dis` distance sizing | OK (`distance_sizing_function`) | M | - |
| `fs` feature size (R elements/width) | OK (`feature_sizing_function`) | H | negative-`fs` auto mode: port (P1) |
| `wl` wavelength sizing | OK scalar (`wavelength_sizing_function`) | M | per-elevation-band (Nx3) form: port (P1) |
| `slp` slope sizing | OK scalar (`bathymetric_gradient_sizing_function`) | M | Nx3 band form: port (P1) |
| `fl` bathy filters (lowpass/band; Rossby) | PART (`filt2` lowpass/band; Rossby stubbed BOTH sides) | L | Rossby: port (P3) |
| `ch` channel/thalweg sizing (+`Channels`, `min_el_ch`, `AngOfRe`) | MISS | H (rivers) | port (P3) |
| `g` gradation | OK scalar (`enforce_mesh_gradation`) | H | spatially-variable / Nx3 band grade: port (P1) |
| `max_el` (global / per-band) | PART (per-function caps) | M | band form in finalize: port (P1) |
| `max_el_ns` nearshore cap | MISS | M | port (P1) |
| `dt` CFL limiter (auto dt=0; minCr/maxCr two-sided) | PART (fork `courant_sizing_function` = floor semantics only) | H | full OM2D semantics: port (P1) |
| weir spacing override in finalize | MISS (no weirs) | L | port with weirs (P4) |
| **`finalize` pipeline** (min-combine -> weirs -> max_el_ns -> max_el -> grade -> CFL -> interpolant, fixed order) | MISS (ad hoc in callers) | H | port as `SizingCollection.finalize()` API (P1, architectural) |

### 1.2 Geodata

| OM2D feature | fork status | Use | Action |
|---|---|---|---|
| shp read + outer/mainland/inner classification | OK (`Shoreline`) | H | - |
| polygon boubox (N x 2 non-rectangular domain) | PART (`Region`/`Shoreline` accept polygon; masking depth unverified) | H | verify + complete (P1) |
| DEM NetCDF/GeoTIFF reader | OK (`DEM`) | H | - |
| `backupdem` (Fb2 hole filling) | MISS | L | port (P3) |
| **`coarsen_polygon`** (coarsen coastline OUTSIDE 1.10x bbox) | MISS | H | port (P1). NOTE: this is OM2D's native answer to outside-region coastline simplification -- it replaces the failed `simplify_outside_region` operator in fvcom-mesh-tools (Yokosuka seam post-mortem, DESIGN_HISTORY section 4) |
| `smooth_coastline` (moving-average window) | PART (Chaikin instead) | M | add window-average option (P1, trivial) |
| `high_fidelity` (mesh1d -> pfix/egfix) | OK+ (fork `shoreline_constraints` + CDT egfix; arguably stronger than OM2D) | H | - |
| `close()` watertight outer, `extractContour` (DEM isoline -> geodata) | MISS | M | port (P3) |
| weirs (`GenerateWeirGeometry`, ibconn) | MISS | L | port (P4) |
| pslg direct input | PART | M | round out (P1) |

### 1.3 Mesh generation (meshgen)

| OM2D feature | fork status | Use | Action |
|---|---|---|---|
| multiscale ef/bou nests | OK (`generate_multiscale_mesh`) | H | validate against OM2D `smooth_outer`/`enforce_min_ef` semantics (P1) |
| per-nest coastline sources (own h0) | OK by construction (per-nest `Shoreline`) | H | wire through fvcom-mesh-tools engine (P1) |
| pfix/egfix constraints | OK+ (CGAL CDT) | H | - |
| **`heal_fixed_edges`** (delete thin tris at constrained edges, every 2 iters) | MISS | H | port (P1) -- directly targets our OBC/coast junction quality |
| Bossen-Heckbert force (OM2D default) | MISS (Persson-Strang active, B-H commented out) | M | port as option, A/B (P1) |
| improvement cycle every 10 iters: delete conn<=4 nodes, delete close pairs (LN<0.5), split long edges (LN>2), 10%-loss rewind | MISS | H | port (P1) -- OM2D's in-loop add/delete is a big quality lever our generator lacks |
| quality-based termination (min qual > 0.30) | MISS | M | port (P1) |
| initial seeding along outer polygon (box 1) | MISS/unverified | M | port (P1) |
| m_map projections ('trans', 'utm', ...) | PART (lonlat + stereo) | L | document; UTM handled downstream (no port) |
| `cleanup` presets on exit | PART (`mesh_clean`) | H | see 1.4 |
| `delaunay_elim_on_exit`, `big_mesh`, `improve_boundary` | stubs in OM2D itself | - | skip (documented no-ops) |

### 1.4 Mesh utilities (msh)

| OM2D feature | fork status | Use | Action |
|---|---|---|---|
| fort.14/2dm READERS | MISS (exist in fvcom-mesh-tools) | H | port into oceanmesh (P2) |
| fort.14 writer WITH boundary nodestrings; gr3; 2dm; ww3 | PART (`write_to_fort14` writes zero boundaries; t3s only) | H | port boundary-aware writers (P2) |
| `make_bc`/`makens` auto BC classification (distance/depth/both, cut_lim) | PART (`identify_ocean_boundary_sections`) | H | port full auto classifier (P2) |
| `clean` presets (passive/default/aggressive; db, djc, mqa) | PART (`mesh_clean` pipeline) | H | port preset semantics (P2) |
| `direct_smoother_lur` (implicit LU smoother, pfix-aware) | MISS (only `laplacian2`) | H | port (P2) -- maps onto our C1/C4 tails |
| `smooth2d` hill-climbing smoother | MISS | M | port (P2) |
| `bound_con_int` (valence bound <=7 + local spring/swap) | MISS | H | port (P2) -- our C5 gate |
| `collapse_thin_triangles` | MISS (fmesh has site-op collapse) | H | port (P2) -- our C1/C2 gates |
| `Fix_single_connec_edge_elements` | OK (`delete_faces_connected_to_one_face`) | - | - |
| `Make_Mesh_Boundaries_Traversable` | OK (`make_mesh_boundaries_traversable`) | - | - |
| `flipEdge` (manual edge swap) | MISS | M | port (P2) |
| `renum` (RCM bandwidth renumbering) | MISS | M | port (P2) -- FVCOM solver benefit |
| `CalcCFL` / `GetBarLengths` | MISS (equivalents in fmesh QA) | H | port (P2) |
| **`bound_courant_number`** (decimate/refine mesh to Cr bounds) | MISS | H | port (P2) -- mesh-level CFL enforcement, complements the sizing-level dt limiter |
| `interp` (GridData CA cell-averaging; mindepth/maxdepth; slope rms/abs; lut; N stencil) | MISS | H (depth phase) | port (P3) |
| `lim_bathy_slope`, `Unstruc_Bath_Slope` | MISS | M | port (P3) |
| `plus`/`minus`/`cat` mesh merging (Bowyer-Watson insets) | MISS | M | port (P4) |
| `remesh_patch`/`reconstructEdgefx`/`mesh_patch_smoother` | MISS | M | port (P4) -- SMS-like local rebuild; feeds stage 2 |
| `trim`/`pruneOverlandMesh`/`extract_subdomain`/`get_boundary_of_mesh` | PART | M | round out (P2/P3) |
| weir BCs (ibtype 24), `extractWeirs` | MISS | L | port (P4) |
| ADCIRC file makers (f11/f13/f15/f19/f20/f24/f5354, tide_fac, sponge calc, stations/KML) | MISS | - | OUT OF SCOPE: ADCIRC-model-specific I/O, not meshing. FVCOM-side equivalents live in fvcom-mesh-tools. Documented exclusion. |
| floodplain suite (`fp`, `interpFP`, `MergeFP`) | MISS | L | port (P4, last) |

## 2. Phasing

- **P1 - generation-core parity (first)**: sizing finalize API
  (bands, max_el_ns, two-sided CFL, min-combine order),
  `coarsen_polygon`, polygon-boubox completion, multiscale
  validation with per-nest shorelines, meshgen improvement cycle +
  `heal_fixed_edges` + B-H force option + quality termination +
  boundary seeding. Acceptance: Tokyo Bay 3-nest recipe
  (varres-parity parameters) builds end-to-end; A/B against the
  goto2023 sample (size-by-band, size-vs-depth) and against the
  current v5u single-nest pipeline.
- **P2 - mesh utilities**: clean parity (direct_smoother_lur,
  smooth2d, bound_con_int, collapse_thin_triangles, presets),
  renum, flipEdge, CalcCFL + bound_courant_number, make_bc auto,
  boundary-aware fort.14 writer + readers, 2dm/gr3.
  Acceptance: fmesh QA gates driven by oceanmesh-native utilities;
  round-trip read/write tests.
- **P3 - bathymetry & channels**: GridData CA interp suite,
  lim_bathy_slope, channel (`ch`) sizing + thalweg input, Rossby
  filters, extractContour, backupdem.
- **P4 - advanced**: weirs end-to-end, mesh merge (plus/minus/cat),
  remesh_patch + patch smoother, floodplain.

Each phase: pytest coverage (happy + error paths), English docs in
README section 6.x, and an A/B validation note. No GPL imports;
CGAL/pybind11 additions follow the existing icx build policy.

## 3. Known fork advantages to keep

CDT egfix (exact boundary conformity, beyond OM2D's healing
approach), Shapely-2 vectorized geometry, xarray/rasterio DEM I/O,
stereographic global meshing, `shoreline_constraints` local-h
resampling.
