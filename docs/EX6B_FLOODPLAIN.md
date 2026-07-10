# Example_6b GBAY floodplain — two-stage build (PASSED)

Date: 2026-07-10. Ladder step 14 (Example_6b_GBAY_w_floodplain).
Status: PASSED (user visual sign-off 2026-07-11 JST).

Result vs MATLAB golden (R2024a, GENKAI job 6180190):

| stage | ours NP | golden NP | diff | res-ratio p50 by band |
|---|---|---|---|---|
| 1 (underwater) | 49,526 | 49,564 | -0.08% | 0.99-1.03 |
| constraints | 12,927 pfix | 12,840 | +0.7% | overlay coincides |
| 2 (floodplain) | 226,551 | 230,555 | -1.7% | 0.98-1.06 |

## Ports/fixes this step produced

1. **extract_fixed_constraints** (boundary_conditions.py) — port of
   msh.extractFixedConstraints: exterior boundary edges minus edges
   touching open-boundary nodes, renumberEdges semantics.
2. **Banded gradation through limgradStruct** (finalize.py) — the
   Nx3 elevation-banded `g` previously routed through the fork's
   gradient_limit (different solver, no per-row cos(lat) edge
   lengths). limgradStruct.m natively supports spatially variable
   fdfdx; first exercised here (grade [[0.25,-inf,0],[0.05,0,inf]]).
3. **inpoly_flip caveat documented** (geodata.py) — the .m votes
   with the BFS-closed obj.outer, which this port does not build;
   the 10m-LMSL contour is borderline (43/100 raw-segment vote vs
   >50 in the .m). Such contour datasets set `inpoly_flip`
   explicitly from the MATLAB decision (golden log evidence).

## The trap that cost a day: geodata bbox default

Example_6b passes NO 'bbox' to geodata, so the meshing bbox is the
**DEM footprint** (-95.25..-94.30 x 28.85..29.80), not Example_6's
bbox. With the wrong bbox our mesh covered 47% of the golden's area
while every LOCAL statistic looked plausible (fh-field ratio p50 =
1.000 on the overlap; NP -15%). Checks that catch this in seconds:
compare total mesh AREA and node lon/lat RANGES against the golden
before any field comparison.

## Residual differences (accepted)

pfix disagreement is symmetric (golden-only 4.9%, ours-only 5.3%)
and confined to 1-2-element-wide creeks/ponds at the h0=60 m
resolution limit. All 624 golden-only points have NO boundary in
our stage-1 mesh (the extraction dropped nothing); sub-resolution
features live or die stochastically in both implementations (same
family as the EX5B corner-drift lottery). No directional bias.
