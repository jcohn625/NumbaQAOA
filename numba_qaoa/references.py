from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceResult:
    method: str
    energy: float
    spins: np.ndarray
    bitstring: np.ndarray
    sdp_bound: float | None = None
    solver_status: str | None = None


def ising_energy(spins, w, h):
    spins = np.asarray(spins, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if w.shape != (spins.size, spins.size):
        raise ValueError("w must have shape (n, n)")
    if h.shape != (spins.size,):
        raise ValueError("h must have shape (n,)")

    energy = float(np.dot(h, spins))
    for i in range(spins.size):
        for j in range(i + 1, spins.size):
            energy += float(w[i, j] * spins[i] * spins[j])
    return energy


def goemans_williamson_reference(
    w,
    h=None,
    *,
    n_rounds=256,
    seed=None,
    solver=None,
    solver_kwargs=None,
):
    """SDP plus random-hyperplane rounding reference for Ising minimization.

    For positive pair couplings and no fields, minimizing this Ising Hamiltonian
    is equivalent to MaxCut up to a constant, giving the usual GW-style
    reference. With fields or signed couplings this is the same SDP rounding
    idea, but without the classic MaxCut approximation guarantee.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "goemans_williamson_reference requires cvxpy. Install it with "
            '`python -m pip install "numba-qaoa[gw]"` or `python -m pip install cvxpy`.'
        ) from exc

    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("w must be a square n x n matrix")
    n = w.shape[0]
    if h is None:
        h = np.zeros(n, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (n,):
        raise ValueError("h must be a length-n vector matching w")
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")

    # Add an anchor spin fixed to +1 after rounding so fields h_i z_i become
    # pair terms h_i z_i z_anchor in the relaxation.
    m = n + 1
    anchor = n
    jmat = np.zeros((m, m), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            val = 0.5 * w[i, j]
            jmat[i, j] = val
            jmat[j, i] = val
        field = 0.5 * h[i]
        jmat[i, anchor] = field
        jmat[anchor, i] = field

    x = cp.Variable((m, m), symmetric=True)
    objective = cp.Minimize(cp.sum(cp.multiply(jmat, x)))
    problem = cp.Problem(objective, [x >> 0, cp.diag(x) == 1])
    kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
    if solver is None:
        problem.solve(**kwargs)
    else:
        problem.solve(solver=solver, **kwargs)

    if x.value is None:
        raise RuntimeError(f"SDP solve failed with status {problem.status!r}")

    gram = np.asarray(x.value, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 0.0)
    vectors = eigvecs * np.sqrt(eigvals)

    rng = np.random.default_rng(seed)
    best_spins = None
    best_energy = np.inf

    anchor_vec = vectors[anchor]
    if np.linalg.norm(anchor_vec) > 0.0:
        candidates = [anchor_vec]
    else:
        candidates = []
    candidates.extend(rng.normal(size=(n_rounds, m)))

    for direction in candidates:
        rounded = np.sign(vectors @ direction)
        rounded[rounded == 0.0] = 1.0
        spins = rounded[:n] * rounded[anchor]
        energy = ising_energy(spins, w, h)
        if energy < best_energy:
            best_energy = energy
            best_spins = spins.copy()

    bitstring = (best_spins < 0.0).astype(np.int8)
    return ReferenceResult(
        method="goemans-williamson",
        energy=float(best_energy),
        spins=best_spins.astype(np.int8),
        bitstring=bitstring,
        sdp_bound=float(problem.value) if problem.value is not None else None,
        solver_status=problem.status,
    )
