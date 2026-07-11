"""ADCIRC-style boundary classification and boundary-aware mesh I/O.

Ports of OceanMesh2D msh.make_bc('auto') and the fort.14 nodestring
writer/reader (readfort14/writefort14), plus 2dm and gr3 writers.
Boundaries are represented as a dict:
``{"open": [node-index arrays], "land": [...], "island": [...]}``.
"""

import logging

import numpy as np

from .edges import get_boundary_edges

logger = logging.getLogger(__name__)

__all__ = [
    "boundary_loops",
    "extract_fixed_constraints",
    "make_bc_auto",
    "read_fort14",
    "write_fort14",
    "write_2dm",
    "write_gr3",
]


def boundary_loops(cells):
    """Ordered closed boundary loops (lists of vertex indices) from
    the triangulation, walked edge-to-edge."""
    bedges = get_boundary_edges(np.asarray(cells, dtype=int))
    nxt = {}
    for a, b in bedges:
        nxt.setdefault(int(a), []).append(int(b))
        nxt.setdefault(int(b), []).append(int(a))
    unused = {tuple(sorted((int(a), int(b)))) for a, b in bedges}
    loops = []
    while unused:
        a, b = next(iter(unused))
        loop = [a, b]
        unused.discard(tuple(sorted((a, b))))
        while True:
            cur = loop[-1]
            prev = loop[-2]
            cands = [w for w in nxt[cur]
                     if w != prev
                     and tuple(sorted((cur, w))) in unused]
            if not cands:
                break
            w = cands[0]
            unused.discard(tuple(sorted((cur, w))))
            if w == loop[0]:
                break
            loop.append(w)
        loops.append(np.asarray(loop, dtype=int))
    return loops


def _loop_area(points, loop):
    x, y = points[loop, 0], points[loop, 1]
    return 0.5 * float(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


def make_bc_auto(
    points,
    cells,
    depth=None,
    shoreline=None,
    classifier="both",
    dist_lim=None,
    depth_lim=10.0,
    cut_lim=10,
):
    """Port of msh.make_bc('auto'): classify each outer-boundary
    vertex as OPEN (ocean) or LAND, split the outer loop at class
    changes, drop runs shorter than ``cut_lim``, and mark closed
    inner loops as islands.

    ``depth`` is positive-down nodal depth (metres); a vertex is
    ocean-deep when ``depth > depth_lim``. ``shoreline`` provides
    mainland points for the distance criterion (``dist_lim``
    defaults to ``10 * h0``). ``classifier``: 'distance', 'depth' or
    'both' (open requires all active criteria).
    """
    p = np.asarray(points, dtype=float)
    loops = boundary_loops(cells)
    outer = max(loops, key=lambda lp: abs(_loop_area(p, lp)))

    use_dist = classifier in ("distance", "both")
    use_depth = classifier in ("depth", "both")
    if use_depth and depth is None:
        raise ValueError("depth classifier requires nodal depths")
    if use_dist:
        if shoreline is None:
            raise ValueError("distance classifier requires shoreline")
        from scipy.spatial import cKDTree

        # makens.m:1469-1476: land = [gdat.mainland; gdat.inner] --
        # the distance criterion measures against mainland AND inner
        # (island/breakwater) shorelines
        parts = [np.asarray(shoreline.mainland, dtype=float)]
        inner = np.asarray(getattr(shoreline, "inner", []), dtype=float)
        if inner.size:
            parts.append(inner.reshape(-1, 2))
        land_pts = np.vstack(parts)
        land_pts = land_pts[~np.isnan(land_pts[:, 0])]
        tree = cKDTree(land_pts)
        if dist_lim is None:
            dist_lim = 10.0 * shoreline.h0

    is_open = np.ones(len(outer), dtype=bool)
    if use_dist:
        d, _ = tree.query(p[outer])
        is_open &= d > dist_lim
    if use_depth:
        is_open &= np.asarray(depth, dtype=float)[outer] > depth_lim

    out = {"open": [], "land": [], "island": []}
    if is_open.all() or (~is_open).all():
        key = "open" if is_open.all() else "land"
        out[key].append(outer)
    else:
        # rotate so the loop starts at a class change
        change = np.where(is_open != np.roll(is_open, 1))[0]
        start = int(change[0])
        ring = np.roll(outer, -start)
        cls = np.roll(is_open, -start)
        # contiguous runs
        runs = []
        s = 0
        for i in range(1, len(ring) + 1):
            if i == len(ring) or cls[i] != cls[s]:
                runs.append((s, i, bool(cls[s])))
                s = i
        # merge short runs into their neighbours (cut_lim)
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for k, (s, e, c) in enumerate(runs):
                if e - s < cut_lim:
                    runs.pop(k)
                    if not runs:
                        break
                    j = k % len(runs)
                    s2, e2, c2 = runs[j]
                    runs[j] = (min(s, s2), max(e, e2), c2)
                    changed = True
                    break
        for s, e, c in runs:
            seg = ring[s:e]
            # nodestrings share their junction vertices with the
            # neighbouring string (ADCIRC convention)
            e_ext = ring[e % len(ring)]
            seg = np.append(seg, e_ext)
            out["open" if c else "land"].append(seg)
    for lp in loops:
        if lp is not outer:
            out["island"].append(lp)
    logger.info(
        "make_bc auto: %d open, %d land, %d island strings",
        len(out["open"]), len(out["land"]), len(out["island"]),
    )
    return out


def extract_fixed_constraints(points, cells, open_strings=None):
    """Port of msh.extractFixedConstraints (msh.m:3928): every
    exterior boundary edge of the mesh, minus any edge touching an
    ocean (open) boundary node, becomes a fixed constraint for a
    subsequent meshgen call (Example_6b floodplain workflow).

    Parameters
    ----------
    points, cells:
        the source mesh
    open_strings: iterable of node-index arrays, optional
        the ``"open"`` entry of :func:`make_bc_auto`; when given,
        edges touching these nodes are removed so only shoreline
        (land/island) constraints remain

    Returns
    -------
    (pfix, egfix):
        pfix (M, 2) coordinates and egfix (K, 2) 0-based indices
        into pfix (renumberEdges semantics)
    """
    p = np.asarray(points, dtype=float)
    egfix = np.asarray(get_boundary_edges(cells))
    if open_strings is not None and len(open_strings) > 0:
        ocean = np.unique(
            np.concatenate([np.asarray(s).ravel() for s in open_strings])
        )
        keep = ~(np.isin(egfix[:, 0], ocean) | np.isin(egfix[:, 1], ocean))
        logger.info(
            "extract_fixed_constraints: removed %d ocean-boundary edges",
            int((~keep).sum()),
        )
        egfix = egfix[keep]
    uniq, inv = np.unique(egfix, return_inverse=True)
    pfix = p[uniq]
    egfix = inv.reshape(egfix.shape)
    return pfix, egfix


def write_fort14(filepath, points, cells, depth=None, boundaries=None,
                 weirs=None, title="Created with oceanmesh"):
    """ADCIRC/FVCOM fort.14 writer WITH boundary nodestrings (the
    packaged write_to_fort14 writes zero boundaries). ``boundaries``
    follows the make_bc_auto dict; islands get ibtype 21, land 20,
    open 0 (indices 0-based in memory, 1-based on file)."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(cells, dtype=int)
    b = (np.zeros(len(p)) if depth is None
         else np.asarray(depth, dtype=float))
    bnd = boundaries or {"open": [], "land": [], "island": []}
    lines = [title, f"{len(t)} {len(p)}"]
    for i, ((x, y), z) in enumerate(zip(p, b), start=1):
        lines.append(f"{i} {x:.10f} {y:.10f} {z:.6f}")
    for i, tri in enumerate(t, start=1):
        lines.append(
            f"{i} 3 {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}"
        )
    op = bnd.get("open", [])
    lines.append(f"{len(op)} = Number of open boundaries")
    lines.append(
        f"{sum(len(s) for s in op)} = Total number of open boundary nodes"
    )
    for k, seg in enumerate(op, start=1):
        lines.append(f"{len(seg)} = Number of nodes for open boundary {k}")
        lines.extend(str(int(v) + 1) for v in seg)
    land = list(bnd.get("land", []))
    isl = list(bnd.get("island", []))
    weirs = weirs or []
    lines.append(
        f"{len(land) + len(isl) + len(weirs)} = "
        "Number of land boundaries"
    )
    tot = (sum(len(s) for s in land) + sum(len(s) + 1 for s in isl)
           + sum(len(w['front']) for w in weirs))
    lines.append(f"{tot} = Total number of land boundary nodes")
    k = 0
    for seg in land:
        k += 1
        lines.append(f"{len(seg)} 20 = Number of nodes for land boundary {k}")
        lines.extend(str(int(v) + 1) for v in seg)
    for seg in isl:
        k += 1
        closed = np.append(seg, seg[0])
        lines.append(
            f"{len(closed)} 21 = Number of nodes for island boundary {k}"
        )
        lines.extend(str(int(v) + 1) for v in closed)
    for w in weirs:
        # ADCIRC ibtype 24 internal barrier: front node, back node
        # (ibconn), crest height, sub/super-critical coefficients
        k += 1
        fr = np.asarray(w["front"], dtype=int)
        bk = np.asarray(w["back"], dtype=int)
        cr = np.asarray(w.get("crest", np.zeros(len(fr))), float)
        csub = float(w.get("subcritical", 1.0))
        csup = float(w.get("supercritical", 1.0))
        lines.append(
            f"{len(fr)} 24 = Number of nodes for weir boundary {k}"
        )
        for a, b, c in zip(fr, bk, cr):
            lines.append(
                f"{int(a) + 1} {int(b) + 1} {c:.3f} {csub} {csup}"
            )
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Wrote {filepath}")


def read_fort14(filepath):
    """fort.14 reader (port of @msh/private/readfort14). Returns
    (points, cells, depth, boundaries dict, title); indices
    0-based."""
    with open(filepath) as f:
        tokens = f.read().split("\n")
    title = tokens[0].strip()
    ne, np_ = (int(v) for v in tokens[1].split()[:2])
    pts = np.empty((np_, 2))
    dep = np.empty(np_)
    for i in range(np_):
        parts = tokens[2 + i].split()
        pts[i] = (float(parts[1]), float(parts[2]))
        dep[i] = float(parts[3])
    cells = np.empty((ne, 3), dtype=int)
    for i in range(ne):
        parts = tokens[2 + np_ + i].split()
        cells[i] = [int(parts[2]) - 1, int(parts[3]) - 1,
                    int(parts[4]) - 1]
    pos = 2 + np_ + ne
    bnd = {"open": [], "land": [], "island": []}

    def _int_of(line):
        return int(line.split()[0])

    try:
        nope = _int_of(tokens[pos]); pos += 2
        for _ in range(nope):
            n = _int_of(tokens[pos]); pos += 1
            seg = [int(tokens[pos + j].split()[0]) - 1 for j in range(n)]
            pos += n
            bnd["open"].append(np.asarray(seg, dtype=int))
        nbou = _int_of(tokens[pos]); pos += 2
        for _ in range(nbou):
            parts = tokens[pos].split(); pos += 1
            n = int(parts[0])
            ibtype = int(parts[1]) if len(parts) > 1 and \
                parts[1].isdigit() else 20
            seg = [int(tokens[pos + j].split()[0]) - 1 for j in range(n)]
            pos += n
            key = "island" if ibtype in (1, 21) else "land"
            arr = np.asarray(seg, dtype=int)
            if key == "island" and len(arr) > 1 and arr[0] == arr[-1]:
                arr = arr[:-1]
            bnd[key].append(arr)
    except (IndexError, ValueError):
        logger.info("fort.14 has no/partial boundary tables")
    return pts, cells, dep, bnd, title


def write_2dm(filepath, points, cells, depth=None):
    """SMS 2dm writer (port of @msh/private/write2dm)."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(cells, dtype=int)
    b = (np.zeros(len(p)) if depth is None
         else np.asarray(depth, dtype=float))
    with open(filepath, "w") as f:
        f.write("MESH2D\n")
        for i, tri in enumerate(t, start=1):
            f.write(
                f"E3T {i} {tri[0] + 1} {tri[1] + 1} {tri[2] + 1} 1\n"
            )
        for i, ((x, y), z) in enumerate(zip(p, b), start=1):
            f.write(f"ND {i} {x:.10f} {y:.10f} {-z:.6f}\n")
    logger.info(f"Wrote {filepath}")


def write_gr3(filepath, points, cells, depth=None, boundaries=None,
              title="Created with oceanmesh"):
    """SCHISM gr3/hgrid writer (msh.write 'gr3': land ibtype
    20/21 -> 0/1)."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(cells, dtype=int)
    b = (np.zeros(len(p)) if depth is None
         else np.asarray(depth, dtype=float))
    bnd = boundaries or {"open": [], "land": [], "island": []}
    with open(filepath, "w") as f:
        f.write(f"{title}\n{len(t)} {len(p)}\n")
        for i, ((x, y), z) in enumerate(zip(p, b), start=1):
            f.write(f"{i} {x:.10f} {y:.10f} {z:.6f}\n")
        for i, tri in enumerate(t, start=1):
            f.write(f"{i} 3 {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")
        op = bnd.get("open", [])
        f.write(f"{len(op)} = Number of open boundaries\n")
        f.write(f"{sum(len(s) for s in op)} = Total open boundary nodes\n")
        for seg in op:
            f.write(f"{len(seg)}\n")
            for v in seg:
                f.write(f"{int(v) + 1}\n")
        land = list(bnd.get("land", []))
        isl = list(bnd.get("island", []))
        f.write(f"{len(land) + len(isl)} = Number of land boundaries\n")
        tot = sum(len(s) for s in land) + sum(len(s) + 1 for s in isl)
        f.write(f"{tot} = Total land boundary nodes\n")
        for seg in land:
            f.write(f"{len(seg)} 0\n")
            for v in seg:
                f.write(f"{int(v) + 1}\n")
        for seg in isl:
            closed = np.append(seg, seg[0])
            f.write(f"{len(closed)} 1\n")
            for v in closed:
                f.write(f"{int(v) + 1}\n")
    logger.info(f"Wrote {filepath}")
