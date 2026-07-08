# Example_5b JBAY (weirs) — acceptance analysis and findings

Date: 2026-07-09. Ladder step 4: Examples/Example_5b_JBAY_w_weirs.m
(PostSandyNCEI, h0 = 15 m, two weirs, TestJBAY criteria).
Status: **PASSED** (user visual sign-off; 3 of 4 seed realizations
pass all six TestJBAY criteria, V distribution overlapping the
OM2D rng-shuffled distribution).

## Faithful-port fixes that came out of this step (all committed)

1. **Weir geometry bit-exact** (`oceanmesh/weirs.py`):
   GenerateWeirGeometry.m ported verbatim, including the
   my_interpm per-component (Chebyshev) insert-only densification
   and the `u = v./norm(v)` MATRIX-norm quirk (face separation
   varies along the crest and is much smaller than `width`).
   Verified pointwise against a MATLAB dump: max diff 0.0 for
   both weirs (pfix and ibconn).
2. **Seeding anchored at h0** (meshgen.m:668 `max_r0 = 1/h0^2`):
   the previous lattice-sampled-minimum anchor over-seeded 2.8x
   on JBAY (71,633 vs ~25,300 initial vertices) — unrecoverable
   in-loop because LN is median-normalized and uniform
   overcrowding looks normal to the improvement cycle.
3. **Stable pfix rows**: p(1:nfix) == pfix restored after every
   retriangulation (CGAL does not preserve insertion order; the
   old closest-node snapping let the fixed set drift).
4. **heal_fixed_edges exact** (meshgen.m:1153-1168 + the
   delIT < 5 escape valve at :867-878): without the valve the
   every-2nd-iteration `continue` starved the improvement cycle
   permanently on crowded constrained meshes.
5. **Clean pfix semantics** (msh.m:1103-1246): pfix only selects
   direct_smoother_lur and rides the recursion; the db loop,
   collapse_thin_triangles and bound_con_int run unconditionally.
6. **Densified outline-ring seeding** (meshgen.m:740-747): a
   name-collision bug had attached the 5-point region polygon
   instead of the densified boubox ring, so outline seeding was
   silently empty.
7. **Weir min_ele sizing override** (edgefx.m:803-833): OM2D SETS
   hh = min_ele in a +-10-cell window along the crest (a hard
   override, not a cap). The cap-only variant left a 15-40 m
   feature-sizing halo around the racetracks (caught by figure
   inspection of the west weir).

## Volume-variance attribution (the "-0.7 % bias" investigation)

- Cleaning, interpolation and the sizing field were proven
  equivalent on identical inputs (cross-clean on the same raw:
  NP delta 0.05-0.1 %, lobe-element delta 1; interp on their mesh:
  -0.01 %; fh ratio p50 = 1.000).
- MATLAB `-batch` resets the RNG to a fixed default each start:
  OM2D "reruns" share one rand stream and their +-0.15 % spread is
  numerical jitter, not realization variance. With rng('shuffle')
  OM2D x4 gives V in [2.0576, 2.0767] (+-0.46 %).
- 74 % of the missing coverage in low realizations is the TWO
  offshore frame corners. Nobody places exact corner vertices
  (nearest-vertex-to-corner: ours-best 1-10 m, MATLAB 24-591 m,
  ours-worst 270-1615 m): corner coverage is a CONTINUOUS lottery
  of projection-back accumulation along the frame, shared by both
  implementations.
- After fixes: ours x4 V = {2.051, 2.055, 2.062, 2.077} vs OM2D
  shuffled {2.058..2.077}; 3/4 pass all six criteria; min quality
  0.26-0.34.

## East-weir local sizing flips (user-observed D6 / D5-E5 spots)

The only visible mesh differences map 1:1 onto LOCAL sizing-field
blobs (figure `outputs/om2d_examples/jbay/eastweir_fh.png` in
fvcom-mesh-tools): ours 2-5x finer in a small cove NW of the
crest head; OM2D 1.3-1.5x finer NE of the knife-edge tip.
Mechanism: feature sizing = distance at MEDIAL-AXIS points / R,
with medial cells detected on the h0 lattice by threshold tests
(|grad d| < 0.90 AND d < -0.5 h0, the .m rule). At sub-lattice
degenerate features (knife-edge tip gap ~100 m, cove mouth of a
few h0, marsh ponds) the gradient sits exactly at the threshold,
and FP-level differences in the gridded d flip individual medial
cells in/out. One flipped cell changes the local feature width
~2x, and the gradation limiter (0.15) spreads it into a ~500 m
blob. The flips are symmetric (each implementation has spots the
other lacks) and confined to features at/below the lattice scale;
globally the fields agree at ratio p50 = 1.000. The .m itself
would flip these cells under equally tiny perturbations. Not a
porting defect.

## Reusable methodology

- MATLAB realization spread MUST be measured with rng('shuffle');
  the -batch default RNG makes reruns near-identical.
- meshgen.build auto-saves its pre-clean mesh to
  Precleaned_grid.mat — use it for generation-vs-cleaning
  attribution.
- Cross-clean (their clean on our raw via fort.14 with the 4-line
  empty NOPE/NBOU tail) isolates the cleaning chain.
- Per-element kill-stage tagging and OM_TRACE_SCALE /
  OM_TRACE_CORNER env traces (mesh_generator) localize where
  points/coverage are lost.
