"""1:1 port of OM2D utilities/GEOM_UTIL/mesh-util/tridiv2.m
(+ the tricon2.m edge numbering it relies on): conforming
red/green refinement of flagged triangles.

Red (>= 2 marked edges, closed transitively): 1 -> 4 children;
green (exactly 1 marked edge): 1 -> 2 children. New vertices are
the midpoints of marked edges.
"""

import numpy as np

__all__ = ["tridiv2"]


def _tricon2(tria):
    """tricon2.m: unique edge table; per-triangle edge ids for the
    corner pairs (1,2), (2,3), (3,1) in columns 3..5 (0-based)."""
    nt = len(tria)
    ee = np.vstack(
        [tria[:, [0, 1]], tria[:, [1, 2]], tria[:, [2, 0]]]
    )
    ee = np.sort(ee, axis=1)
    edge, jv = np.unique(ee, axis=0, return_inverse=True)
    tcon = np.column_stack(
        [jv[0 * nt:1 * nt], jv[1 * nt:2 * nt], jv[2 * nt:3 * nt]]
    )
    return edge, tcon


def tridiv2(vert, tria, tdiv):
    """Split the triangles flagged in ``tdiv`` (bool, len == NT).

    Returns (vert_new, tria_new)."""
    vert = np.asarray(vert, dtype=float)
    tria = np.asarray(tria, dtype=int)
    tdiv = np.asarray(tdiv, dtype=bool)

    edge, tcon = _tricon2(tria)
    ediv = np.zeros(len(edge), dtype=bool)
    ediv[tcon[tdiv].ravel()] = True

    # transitive closure (tridiv2.m:55-63): any triangle with >= 2
    # marked edges gets all 3 marked
    snum = int(ediv.sum())
    while True:
        div3 = ediv[tcon].sum(axis=1) >= 2
        ediv[tcon[div3].ravel()] = True
        snew = int(ediv.sum())
        if snew == snum:
            break
        snum = snew

    div1 = ediv[tcon].sum(axis=1) == 1

    # midpoint vertices for every marked edge (tridiv2.m:66-71)
    ivec = np.full(len(edge), -1, dtype=int)
    ivec[ediv] = len(vert) + np.arange(snum)
    emid = 0.5 * (vert[edge[ediv, 0]] + vert[edge[ediv, 1]])
    vert = np.vstack([vert, emid])

    e4, e5, e6 = tcon[:, 0], tcon[:, 1], tcon[:, 2]
    t1, t2, t3 = tria[:, 0], tria[:, 1], tria[:, 2]

    out = [tria[~div1 & ~div3]]

    # red 1->4 (tridiv2.m:79-112): children over corners and the
    # midpoints m12=ivec[e4], m23=ivec[e5], m31=ivec[e6]
    d = div3
    m12, m23, m31 = ivec[e4[d]], ivec[e5[d]], ivec[e6[d]]
    out.append(np.column_stack([t1[d], m12, m31]))
    out.append(np.column_stack([m12, t2[d], m23]))
    out.append(np.column_stack([m31, m23, t3[d]]))
    out.append(np.column_stack([m31, m12, m23]))

    # green 1->2 per which single edge is marked
    # (tridiv2.m:113-142)
    g = ediv[e4] & div1
    out.append(np.column_stack([ivec[e4[g]], t3[g], t1[g]]))
    out.append(np.column_stack([ivec[e4[g]], t2[g], t3[g]]))
    g = ediv[e5] & div1
    out.append(np.column_stack([ivec[e5[g]], t1[g], t2[g]]))
    out.append(np.column_stack([ivec[e5[g]], t3[g], t1[g]]))
    g = ediv[e6] & div1
    out.append(np.column_stack([ivec[e6[g]], t2[g], t3[g]]))
    out.append(np.column_stack([ivec[e6[g]], t1[g], t2[g]]))

    tria_new = np.vstack([o for o in out if len(o)])
    return vert, tria_new
