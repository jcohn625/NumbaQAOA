import numpy as np

from numba_qaoa import QAOASimulator
from numba_qaoa.banding import band_weight, banded_by_permutation, make_banded_surrogate


def finite_difference(sim, params, eps=1e-6):
    grad = np.empty_like(params)
    for i in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (sim.energy(plus) - sim.energy(minus)) / (2.0 * eps)
    return grad


def test_banded_by_permutation_keeps_requested_band():
    w = np.arange(25, dtype=float).reshape(5, 5)
    order = np.array([4, 2, 0, 1, 3])
    banded = banded_by_permutation(w, bandwidth=1, order=order)
    pos = np.empty(5, dtype=int)
    pos[order] = np.arange(5)

    for i in range(5):
        for j in range(i + 1, 5):
            expected = w[i, j] if abs(pos[i] - pos[j]) <= 1 else 0.0
            assert banded[i, j] == expected
            assert banded[j, i] == 0.0


def test_make_banded_surrogate_improves_or_matches_identity_order():
    w = np.zeros((5, 5))
    w[0, 4] = 10.0
    w[1, 3] = 8.0
    identity = np.arange(5)
    banded, order = make_banded_surrogate(w, bandwidth=1, local_search_passes=4)

    assert band_weight(w, order, 1) >= band_weight(w, identity, 1)
    assert np.count_nonzero(np.triu(banded, k=1)) <= 4


def test_banded_phase_full_objective_gradient_matches_finite_difference():
    rng = np.random.default_rng(17)
    n = 5
    p = 2
    w = rng.normal(size=(n, n))
    h = rng.normal(size=n)
    params = rng.normal(scale=0.2, size=2 * p)
    sim = QAOASimulator(w, h, p=p, max_bandwidth=1, permutation="spectral")

    assert sim.phase_cost is not sim.objective_cost
    assert np.count_nonzero(np.triu(sim.phase_w, k=1)) < np.count_nonzero(np.triu(w, k=1))

    fd = finite_difference(sim, params)
    for mode in ("adjoint", "layer", "full"):
        energy, grad = sim.energy_and_gradient(params, cache_mode=mode)
        assert np.isfinite(energy)
        np.testing.assert_allclose(grad, fd, atol=1e-6, rtol=1e-6)


def test_phase_energy_can_differ_from_objective_energy():
    rng = np.random.default_rng(19)
    n = 4
    p = 1
    w = rng.normal(size=(n, n))
    h = rng.normal(size=n)
    params = rng.normal(scale=0.2, size=2 * p)
    sim = QAOASimulator(w, h, p=p, max_bandwidth=1, permutation="identity")

    assert sim.energy(params) != sim.phase_energy(params)
