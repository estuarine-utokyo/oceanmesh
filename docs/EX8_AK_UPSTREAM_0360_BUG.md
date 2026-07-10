# Example_8_AK: upstream 0-360 regression (OM2D master)

Date: 2026-07-10. Ladder step 9 (Example_8_AK, dateline-straddling
0-360 boubox, pure distance sizing).

## Upstream defect (confirmed by direct probe)

Example_8 exists to demonstrate the 0-360 longitude format
(commit 53a2b190, "adding Alaska example"). On current OM2D
master (MATLAB R2024a) the geodata 0-360 path is broken:

    geodata('shp','GSHHS_f_L1','bbox',ak_outerpoly,'h0',5e3)
    mainland lon range: -180.00..180.00, points east of 180: 0
    inner    lon range:  163.40..180.00, points east of 180: 0

No shoreline east of the dateline survives classification (the
+360 shift never fires), so the golden mesh (NP 8,575) meshes
straight over the Alaska landmass and shows an artificial
vertical seam at lon 180 (63-70N).

## Our behaviour

The Python port collects negative-longitude geometries shifted
+360 when the bbox extends past lon 180 (geodata.m:469-500
intent) and wraps unprojected longitudes back into the 0-360
frame after the tmerc round-trip. The resulting mesh (NP 17,474)
resolves the full Alaska/Aleutian/Chukotka shoreline.

## Acceptance basis

West of the dateline both implementations carry the full
shoreline; node counts agree there: ours 3,875 vs OM2D 3,986
(-2.8 %). East of the dateline the entire difference is the
missing upstream shoreline. Treated like the get_poly_edges
closure bug: faithful intent ported, upstream defect documented.

## Fixes that came out of this example

- Shoreline: 0-360 collection (+360-shifted copies).
- Generator: wrap unprojected lon back into a 0-360 domain frame
  (pyproj normalises to [-180,180)).
- distance sizing: metric KD distance (dpoly parity) replacing
  the degree-isotropic fast-march.
- metric KD: drop non-finite projected land points (whole-kept
  continental polygons reach beyond tmerc's +/-90 deg validity).
