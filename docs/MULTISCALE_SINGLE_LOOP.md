# Multiscale meshing — OM2D single-loop port (Example_10)

Date: 2026-07-09. Ladder step 5 (Example_10_Multiscale_Smoother,
idealized 1 km box nested in a 10 km box). Status: PASSED (user
visual sign-off; NP 11,687 vs golden 11,519, +1.5 %; min quality
0.587; transition structure identical).

Two deviations were found by comparing against the MATLAB golden
mesh and were replaced by faithful ports:

## 1. Nest-to-nest sizing smoothing = smooth_outer.m

OM2D (meshgen.m:394 -> @meshgen/private/smooth_outer.m) does NOT
blend sizing fields. For every outer edgefx it
1. PASTES the finer nests' values onto the outer lattice
   (masked to the finer boubox), then
2. relaxes the whole outer grid with limgradStruct at the outer
   grid's own gradation.

Because the relaxation runs on the COARSE lattice, the transition
quantizes on coarse cells (1 -> 3 -> 5 -> 7 -> 9 km per 10 km cell
at g = 0.2 on Example_10). The previous IDW blend
(`blend_width`/`nnear`/`p`) spilled near-fine sizes 3-5 coarse
cells outward. After the port the outer sizing fields agree with
the MATLAB `ef{1}` dump CELL-EXACTLY (both produce the same
1/3/5/7/9 km staircase, symmetric on all sides).
`multiscale_sizing_function` keeps its signature; the blend
parameters are accepted but unused; `gradation` (scalar or
per-outer-grid list) threads through `generate_multiscale_mesh`.

## 2. One distmesh loop over all nests (no build-and-merge)

OM2D meshes ALL nests in a single distmesh loop; there is no
per-nest build or seam merge (utilities/split_bars.m is dead code
— nothing calls it). `generate_multiscale_mesh` now builds:

- composite fd (dpoly.m box loop): each point takes the signed
  distance of the DEEPEST box containing it; membership is
  inpoly against each domain's densified `boubox_ring` (polygon
  nests), with a rectangle fallback;
- composite fh: bar midpoints evaluate the deepest box's
  (smooth_outer-relaxed) sizing;
- per-box seeding (meshgen.m:664-748): each box lays its own
  equilateral lattice at its h0, rejects at (h0/fh)^2 anchored at
  ITS OWN h0, excludes deeper-box regions (they seed themselves
  finer), and box 1 additionally seeds its outline ring;

and calls `generate_mesh` ONCE with the composites, the union
bbox, the per-nest `min_edge_length` vector and the combined
initial points.

Effect on Example_10: the fine mesh now ends exactly AT the nest
boundary with the same 2-3 transition rows as OM2D; the previous
architecture (per-nest meshes merged afterwards) produced a
10-15 km wide spill of intermediate sizes outside the nest and
an asymmetric fringe.

## Incidental fixes

- `generate_mesh`: the `points=` kwarg was projected twice (the
  tmerc block already projects `opts["points"]`); the redundant
  projection corrupted injected points whenever the projection
  was active.
- `min_edge_length` may now be a per-nest vector (assert
  vectorized; `geps`/`ttol` gates use its minimum).

## Related observations (recorded for methodology)

- The OM2D golden for Example_10 lost its outer NW/NE frame
  corners (nearest vertex 0.151 deg from the corner) while ours
  kept them — the same unpinned-corner drift lottery documented
  in EX5B_JBAY_ANALYSIS.md, that time against OM2D. Frame corners
  are not fixed points in either implementation.

## Appendix: remesh_patch engines (Example_11)

`om.remesh_patch(points, cells, polygon, sizing, engine=...)`
regenerates the mesh inside a polygon and stitches it back with a
band-CDT seam (area-exact, verified non-manifold-free).

- `engine="jigsaw"` (default): jigsawpy — the successor of the
  mesh2d engine OM2D calls. OPTIONAL dependency, not bundled
  (non-OSI license): `mamba install -c conda-forge jigsawpy
  jigsaw`. Falls back to "distmesh" automatically when absent.
- `engine="distmesh"`: this package's OM2D-parity generator with
  the patch boundary held fixed; statistically closest to the
  mesh2d output.
