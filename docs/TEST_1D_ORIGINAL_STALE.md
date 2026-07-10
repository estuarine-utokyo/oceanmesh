# Tests/test_1d_original.m is stale upstream (cannot run)

Date: 2026-07-10. Ladder item "test_1d_original" (1-D distmesh,
mesh1d).

Verdict: the upstream test CANNOT run against current OceanMesh2D
and is excluded from the acceptance ladder. Evidence (MATLAB R2024a
on GENKAI, job 6180306):

```
関数 'mesh1d' (タイプ 'function_handle' の入力引数) が未定義です。
(Undefined function 'mesh1d' for input arguments of type
'function_handle'.)
```

Root causes, in order encountered:

1. `mesh1d.m` lives in `@meshgen/private/` — private class methods
   are not callable from scripts. The test predates the move.
2. Even if pathed in, the current `mesh1d(poly, fh0, h, fix, boubox,
   box_num0, ...)` signature is incompatible with the test's
   4-argument call: `nestedHFx` indexes `fh0{box_num}` (cell) and
   divides by 111e3 (degree conversion), and `geps = 1e-3 *
   h(box_num0)` reads the missing 6th argument.

Acceptance for our `oceanmesh/mesh1d.py` therefore rests on the
LIVE call path — meshgen.m:569's high-fidelity pfix/egfix chains —
validated by Example_13 (user visual sign-off, 2026-07-10).

Port-fidelity note fixed while auditing: `L0mult = 1 + 0.4/2^(dim-1)`
with dim=1 gives 1.4; the port carried the 2-D value 1.2. Corrected.
Behavioural deviation kept (documented): our port pins both chain
endpoints to the exact polyline ends; the .m locks the first/last
SORTED seeds, which may sit up to h0 short of the far end (chains
are later welded by fixgeo2 anyway).
