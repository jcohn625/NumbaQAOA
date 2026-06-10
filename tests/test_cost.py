import numpy as np

from numba_qaoa import QAOASimulator


def direct_cost(state, w, h):
    n = h.shape[0]
    z = np.empty(n)
    for i in range(n):
        z[i] = -1.0 if ((state >> i) & 1) else 1.0
    e = np.dot(h, z)
    for i in range(n):
        for j in range(i + 1, n):
            e += w[i, j] * z[i] * z[j]
    return e


def test_cost_uses_upper_triangle_only():
    w = np.array(
        [
            [0.0, 1.0, -0.5],
            [99.0, 0.0, 0.25],
            [88.0, 77.0, 0.0],
        ]
    )
    h = np.array([0.2, -0.1, 0.4])
    sim = QAOASimulator(w, h, p=1)
    expected = np.array([direct_cost(s, w, h) for s in range(8)])
    np.testing.assert_allclose(sim.cost, expected)
