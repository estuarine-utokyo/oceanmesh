"""Port of @meshgen/private/mesh1d.m: 1-D distmesh along a
polyline's arclength, sizes from the (composite) edge-length
function. Produces the high-fidelity pfix/egfix chains."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["mesh1d"]


def mesh1d(poly, fh_eval, h0, max_iter=200):
    """1-D force equilibrium on the arclength axis of `poly`.

    poly : (N,2) polyline (finite vertices, lon/lat degrees)
    fh_eval : callable((M,2)) -> sizes in DEGREE units (composite,
        deepest-box-wins, as meshgen's nestedHFx/111e3)
    h0 : seeding interval in degree units (min h0 of all boxes)

    Returns (pout, egfix) or (None, None) when the part is too
    short (mesh1d.m: np < 5).
    """
    X = np.asarray(poly[:, 0], dtype=float)
    Y = np.asarray(poly[:, 1], dtype=float)
    dist = np.hypot(np.diff(X), np.diff(Y))
    u = np.concatenate([[0.0], np.cumsum(dist)])
    total = u[-1]
    if total / h0 < 5:
        return None, None

    # sizes along the line (mesh1d.m nestedHFx + griddedInterpolant)
    tpar = np.linspace(0.0, total, int(np.ceil(total / h0)))
    xn = np.interp(tpar, u, X)
    yn = np.interp(tpar, u, Y)
    hvals = np.asarray(fh_eval(np.column_stack([xn, yn])),
                       dtype=float)

    def fh(s):
        return np.interp(s, tpar, hvals)

    def fdist(s):
        return np.maximum(-s, s - total)  # my_1d_sdf

    # mesh1d.m: L0mult = 1 + 0.4/2^(dim-1) with dim=1 -> 1.4
    # (1.2 is the 2-D distmesh value)
    ptol, L0mult, deltat = 0.01, 1.4, 0.10
    geps = 1e-3 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0

    p = np.arange(0.0, total + h0, h0)
    p = p[fdist(p) < geps]
    r0 = fh(p)
    keep = np.random.rand(len(p)) < (r0.min() / r0)
    p = np.unique(p[keep])
    if len(p) < 2:
        return None, None
    # pin the endpoints
    p[0], p[-1] = 0.0, total

    for _ in range(max_iter):
        p = np.sort(p)
        pair = np.column_stack([np.arange(len(p) - 1),
                                np.arange(1, len(p))])
        bars = p[pair[:, 0]] - p[pair[:, 1]]
        L = np.abs(bars)
        L0 = fh(0.5 * (p[pair[:, 0]] + p[pair[:, 1]]))
        L0 = L0 * L0mult * (L.sum() / L0.sum())
        F = np.maximum(L0 - L, 0.0)
        Fbar = F * np.sign(bars)
        dp = np.zeros(len(p))
        np.add.at(dp, pair[:, 0], Fbar)
        np.add.at(dp, pair[:, 1], -Fbar)
        dp[0] = 0.0
        dp[-1] = 0.0
        p = p + deltat * dp
        d = fdist(p)
        out = d > 0
        out[0] = out[-1] = False
        if out.any():
            grad = (fdist(p[out] + deps) - d[out]) / deps
            p[out] = p[out] - d[out] * grad
        maxdp = deltat * np.abs(dp[d < -geps]).max() if (
            d < -geps).any() else 0.0
        if maxdp < ptol * h0:
            break

    p = np.unique(np.sort(p))
    pout = np.column_stack([np.interp(p, u, X), np.interp(p, u, Y)])
    n = len(pout)
    if n < 3:
        return None, None
    egfix = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    return pout, egfix
