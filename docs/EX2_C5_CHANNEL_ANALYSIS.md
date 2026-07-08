# Example_2_NY C5/D5 wetland divergence — root-cause analysis

Date: 2026-07-08. Context: acceptance-ladder step 3
(Examples/Example_2_NY.m, PostSandyNCEI, h0 = 30 m). After the
faithful-port fixes below, the Python and MATLAB meshes still
differed visibly in the Hackensack Meadowlands (atlas cells
C4/C5-D5): each implementation meshes a different subset of the
wetland lobes. This note records why, with the measurements that
close the question.

## Fixes that came out of the investigation (all committed)

1. `get_poly_edges` dropped the FINAL segment of every
   NaN-delimited part (upstream bug since 2022): every ring
   reached `inpoly2` as an open chain, producing odd
   crossing-parity stripes (first seen as full-width mesh voids
   over the Bahamas in TestECGC). Fixed + regression test.
2. Shoreline classification and the SDF sign were rebuilt to the
   OM2D semantics (Read_shapefile.m:110-236, :283;
   meshgen.m:329-331; dpoly.m:40-44): wholly-inside ring ->
   island, ANY other ring -> mainland kept WHOLE (no boolean
   clipping), sign = even-odd parity over
   `[boubox; mainland; inner]`. The previous shapely
   `difference()` flow could not represent depth-3 nesting
   (channel-in-land-in-domain) and silently dropped rings
   disjoint from the shrinking domain polygon — that deleted the
   entire Meadowlands channel network from the domain (0 inner
   points in C5 vs MATLAB's 5,386).
3. Pipeline order now matches OM2D: classify raw polygons first,
   then densify/smooth per category; the default path no longer
   clips mainland rings.
4. The Read_shapefile "merge overlapping mainland and inner"
   block (:253-281) is gated OFF for 2-D shapefiles, as upstream:
   its guard `~isempty(new_mainb)` only passes for 3-D (height)
   shapefiles. Verified: MATLAB gdat keeps all C5 wetland islands
   in `inner`.

After these fixes the classified ring sets are identical to
MATLAB's gdat (C5 window: inner 5,392 vs 5,386 points, mainland
10,802 vs 10,802), and MATLAB's own `inpoly` agrees with our
parity 4/4 on lobe probe points.

## Why one implementation keeps a wetland lobe and the other does not

Verdict: **generation-phase stochasticity in an under-resolved
channel maze; the cleaning rules are equivalent and contain no
discretionary judgment.**

Evidence chain (Example_2_NY, seed 0 vs the MATLAB run):

- Both RAW meshes cover the disputed NE lobe. Ours: NP 47,397 /
  NT 75,789; theirs (Precleaned_grid.mat, auto-saved by
  meshgen.build): NP 51,777 / NT 83,444.
- Kill-stage tracing (per-element deletion tags) shows the NE
  lobe in OUR raw dies as ONE 2,695-element DISCONNECTED
  component at the first Make_Mesh_Boundaries_Traversable pass —
  not through a quality-deletion cascade (db deletions in the
  lobe: ~300, scattered). The lobe was already severed from the
  main water body AT GENERATION: the single connecting channel
  failed to triangulate continuously.
- Symmetrically, THEIR raw mesh has a different severed patch
  (SE complex, 1,891 elements) that their clean removes — and
  that region survives in OUR mesh. The two final meshes differ
  complementarily, exactly as observed in the C5/D5 panels.
- Cross-clean test (same input, both cleaners): MATLAB
  clean('default') on OUR raw gives NP 41,012; our
  om2d_default_clean gives NP 40,992 (0.05% apart; disputed-lobe
  element counts 1,205 vs 1,204). The cleaning chains are
  equivalent.
- Sizing fields are equivalent (MATLAB edgefx dump vs ours,
  1852x1852 lattice): ratio p50 = 1.000 globally (322 vs 323 m);
  at the connecting channel p50 = 60 m vs 60 m (ratio 1.004).

Mechanism: the channels are 30-60 m wide while the local sizing
is ~60 m — exactly the one-element-across regime. Whether a
continuous ribbon of water-centroid triangles forms along the
channel is decided by vertex placement noise (the same
generator-equilibrium stochasticity that makes MATLAB itself
produce NP 6,017 vs 5,968 on repeated Example_1 runs). Once the
ribbon breaks anywhere, Make_Mesh_Boundaries_Traversable
(dj_cutoff) removes everything beyond the break as a
disconnected component — a whole-lobe amplification of a
one-element failure.

## Practical guidance

To keep such channels deterministically, give them >= 2 elements
across: local sizing <= half the channel width (here fh <= 20-30
m, e.g. h0 = 15 m or stronger feature refinement). This is a
model-setup requirement, not an implementation property; OM2D
behaves identically at the current settings.

## Methodology notes

- The even-odd referee is MATLAB `inpoly` itself (run it on the
  same NaN-stacked rings). `matplotlib.path.Path.contains_point`
  uses the NONZERO WINDING rule and must not be used to referee
  even-odd parity.
- OM2D auto-saves the pre-clean mesh to `Precleaned_grid.mat` in
  the working directory of every `meshgen.build` — use it for
  generation-vs-cleaning attribution.
- `msh()` needs an explicit `.14` suffix and fort.14 files must
  end with the four NOPE/NBOU section lines (zeros suffice).
