import numpy as np

from oceanmesh import edges

nan = np.nan


def test_edges():
    poly = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [nan, nan],
            [0.2, 0.2],
            [0.4, 0.2],
            [0.3, 0.3],
            [0.2, 0.2],
            [nan, nan],
        ]
    )

    e = edges.get_poly_edges(poly)
    edges.draw_edges(poly, e)


def test_get_poly_edges_keeps_final_segment():
    # regression: the final segment of each part was dropped,
    # leaving every ring an open chain (ECGC stripe incident)
    import numpy as np
    from oceanmesh import edges

    sq_open = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
         [np.nan, np.nan]]
    )
    e = edges.get_poly_edges(sq_open)
    undirected = {tuple(sorted(r)) for r in e.tolist()}
    assert undirected == {(0, 1), (1, 2), (2, 3), (0, 3)}

    sq_closed = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0],
         [np.nan, np.nan]]
    )
    e = edges.get_poly_edges(sq_closed)
    lens = np.hypot(
        *(sq_closed[e[:, 0]] - sq_closed[e[:, 1]]).T
    )
    assert (lens > 0).all()
    undirected = {tuple(sorted(r)) for r in e.tolist()}
    assert (0, 1) in undirected and (3, 4) in undirected
    assert len(undirected) == 4

    # two parts, second without trailing NaN
    two = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [np.nan, np.nan],
         [2.0, 0.0], [3.0, 0.0], [2.5, 1.0]]
    )
    e = edges.get_poly_edges(two)
    undirected = {tuple(sorted(r)) for r in e.tolist()}
    assert {(0, 1), (1, 2), (0, 2)} <= undirected
    assert {(4, 5), (5, 6), (4, 6)} <= undirected
