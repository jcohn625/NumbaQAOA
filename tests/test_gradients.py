import numpy as np

from numba_qaoa import QAOASimulator


def finite_difference(sim, params, eps=1e-6):
    grad = np.empty_like(params)
    for i in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (sim.energy(plus) - sim.energy(minus)) / (2.0 * eps)
    return grad


def test_gradient_matches_finite_difference_all_cache_modes():
    rng = np.random.default_rng(7)
    n = 4
    p = 3
    w = rng.normal(size=(n, n))
    h = rng.normal(size=n)
    params = rng.normal(size=2 * p)
    sim = QAOASimulator(w, h, p=p, cache_mode="adjoint")

    fd = finite_difference(sim, params)
    for mode in ("adjoint", "layer", "full"):
        energy, grad = sim.energy_and_gradient(params, cache_mode=mode)
        assert np.isfinite(energy)
        np.testing.assert_allclose(grad, fd, atol=1e-6, rtol=1e-6)


def test_gradient_modes_agree():
    rng = np.random.default_rng(11)
    n = 5
    p = 2
    w = rng.normal(size=(n, n))
    h = rng.normal(size=n)
    params = rng.normal(size=2 * p)
    sim = QAOASimulator(w, h, p=p)

    e0, g0 = sim.energy_and_gradient(params, cache_mode="adjoint")
    for mode in ("layer", "full"):
        e, g = sim.energy_and_gradient(params, cache_mode=mode)
        np.testing.assert_allclose(e, e0, atol=1e-12)
        np.testing.assert_allclose(g, g0, atol=1e-12)
