import numpy as np
import pytest

from numba_qaoa import QAOASimulator, goemans_williamson_reference, ising_energy


cp = pytest.importorskip("cvxpy")


def brute_force_energy(w, h):
    n = h.size
    best = np.inf
    for state in range(1 << n):
        spins = np.array([-1 if ((state >> i) & 1) else 1 for i in range(n)])
        best = min(best, ising_energy(spins, w, h))
    return best


def test_ising_energy_uses_upper_triangle():
    w = np.array(
        [
            [0.0, 2.0, -1.0],
            [100.0, 0.0, 0.5],
            [200.0, 300.0, 0.0],
        ]
    )
    h = np.array([0.25, -0.5, 0.75])
    spins = np.array([1, -1, 1])
    expected = 0.25 + 0.5 + 0.75 - 2.0 - 1.0 - 0.5
    np.testing.assert_allclose(ising_energy(spins, w, h), expected)


def test_goemans_williamson_reference_finds_small_exact_solution():
    w = np.array(
        [
            [0.0, 1.0, 0.4, 0.0],
            [0.0, 0.0, 1.2, 0.7],
            [0.0, 0.0, 0.0, 1.1],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    h = np.array([0.15, -0.2, 0.05, 0.1])
    result = goemans_williamson_reference(w, h, n_rounds=128, seed=5)
    np.testing.assert_allclose(result.energy, brute_force_energy(w, h), atol=1e-7)
    np.testing.assert_allclose(result.energy, ising_energy(result.spins, w, h))
    assert result.bitstring.shape == h.shape


def test_simulator_reference_solution_alias():
    w = np.array([[0.0, 1.0], [0.0, 0.0]])
    h = np.zeros(2)
    sim = QAOASimulator(w, h, p=1)
    result = sim.reference_solution(method="gnomes-williamson", n_rounds=16, seed=1)
    np.testing.assert_allclose(result.energy, -1.0, atol=1e-7)
