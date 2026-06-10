import numpy as np
from scipy.linalg import expm

from numba_qaoa import QAOASimulator


def dense_reference(w, h, params):
    n = h.shape[0]
    dim = 1 << n
    cost = np.empty(dim)
    for state in range(dim):
        z = np.array([-1.0 if ((state >> i) & 1) else 1.0 for i in range(n)])
        cost[state] = np.dot(h, z)
        for i in range(n):
            for j in range(i + 1, n):
                cost[state] += w[i, j] * z[i] * z[j]

    hc = np.diag(cost)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    ident = np.eye(2)
    hb = np.zeros((dim, dim))
    for q in range(n):
        mats = [ident] * n
        mats[n - 1 - q] = x
        term = mats[0]
        for mat in mats[1:]:
            term = np.kron(term, mat)
        hb += term

    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for layer in range(params.size // 2):
        psi = expm(-1j * params[2 * layer] * hc) @ psi
        psi = expm(-1j * params[2 * layer + 1] * hb) @ psi
    return psi, float(np.real(np.vdot(psi, hc @ psi)))


def test_state_and_energy_match_dense_reference():
    rng = np.random.default_rng(5)
    n = 3
    p = 2
    w = rng.normal(size=(n, n))
    h = rng.normal(size=n)
    params = rng.normal(size=2 * p)
    sim = QAOASimulator(w, h, p=p)

    psi_ref, energy_ref = dense_reference(w, h, params)
    psi = sim.build_state(params).copy()

    np.testing.assert_allclose(psi, psi_ref, atol=1e-12)
    np.testing.assert_allclose(sim.energy(params), energy_ref, atol=1e-12)
