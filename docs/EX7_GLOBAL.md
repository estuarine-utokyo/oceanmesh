# Example_7 Global 4-20 km (PASSED)

Date: 2026-07-11. Ladder step 15 (Example_7_Global, stereographic
generation, GSHHS_f_L1+L6 + SRTM15+). Status: PASSED (user visual
sign-off 2026-07-11 JST).

Result vs MATLAB golden (R2024a, GENKAI):

| metric | ours | golden |
|---|---|---|
| initial seeds | 1,464,832 | 1,462,514 (+0.16%) |
| final NP | 1,418,611 | 1,372,623 (+3.4%) |
| res-ratio p50 (at golden nodes) | 0.995 | — |
| band p50 (<5k / 5-10k / 10-19k / >=19k) | 1.069 / 0.988 / 0.987 / 0.991 | — |
| slope-only field vs ML dump | p10/50/90 = 0.999/1.000/1.006 | — |

Residual +3.4%: seeds and the sizing field match; the in-loop point
reduction is milder here (-3.2%) than MATLAB's (-6.2%) — identical
deletion rules (LN < 0.5, small connectivity), different stochastic
realisation. Distribution and spatial structure match (histogram,
ridge banding, island holes incl. Stewart Island / Oahu).

## Global-path fixes this step produced (fork code first exercised here)

1. rossby_radius_filter: longitude-wrap and pole-mirror index paths
   (float mult TypeError + three 0-based off-by-ones vs edgefx.m:549,557).
2. Stereo improvement step evaluated fh on raw stereo coordinates
   (LN uniform-in-plane -> runaway splitting 2.7M -> 6.0M); now
   converts via to_lat_lon like the force step (meshgen.m uses
   m_lldist + ll midpoints in every step).
3. Stereo seeding lattice: meshgen.m:686-731 physical equilateral
   lattice, INCLUDING its over-pole row lengths (m_lldist between
   points 180 deg of longitude apart crosses the pole: rows span
   2*(180-2|lat|) deg, not 360 cos(lat)); rejection is plain
   (h0/fh)^2 with no stereographic distortion factor. The previous
   square-degree lattice over-seeded 1.9x.
4. Stereo shoreline: _cull_small_features culled by stereo-frame
   area (deleted every island below continent scale — Stewart
   Island); now culls by inverse-projected lonlat area as
   geodata.m. Outer boundary ring selected deterministically (odd
   winding parity at the stereo origin, max |shoelace|) instead of
   the order-dependent overlaps-replacement chain.
5. Slope sizing: per-row dx = h0*cosd(min(|lat|,85)) (edgefx.m:458)
   — scalar cos(mean lat) sat 33% coarse on shelf slopes.
6. Slope filter default: Example_7 passes no 'fl' -> edgefx defval
   fl=0 -> filter OFF (we had assumed barotropic/50; Ex4 passes
   fl=-50 explicitly and keeps it).
7. grid_dx two-stage DEM sampling (ParseDEM stride decimation ->
   h0-lattice linear resample). Single-stage variants both fail:
   native bilinear +12% coarse, pure decimation -7% fine. With the
   two-stage port the slope-only field matches the ML dump at
   p50 = 1.000 in every band.
8. limgradStruct: cyclical dateline padding for global lattices
   (edgefx.m:890-937); the fork's extra "stereographic re-gradation"
   pass is NOT in OM2D and was removed from the runner (+15% nodes).

Diagnostic method: golden-vs-ours NP trajectory from logs -> seed
count isolation; field isolation via MATLAB edgefx dumps (full fh,
then slope-only); expected-node integrals (2/sqrt(3) * sum dA/fh^2)
to attribute gaps to field vs generator.
