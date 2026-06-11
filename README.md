# NumbaQAOA

Fast exact statevector QAOA simulator for Ising Hamiltonians.

The Quantum Approximate Optimization Algorithm (QAOA) prepares a variational
state by alternating phase evolution under a problem Hamiltonian with mixing
evolution under a driver Hamiltonian. This package simulates the full
statevector for Ising objectives of the form

```math
H_C = \sum_{i<j} w_{ij} Z_i Z_j + \sum_i h_i Z_i,
```

using the standard transverse-field mixer

```math
H_B = \sum_i X_i.
```

For depth $p$, the prepared state is

```math
\left|\psi(\gamma,\beta)\right\rangle =
\left(\prod_{\ell=1}^{p}
e^{-i \beta_\ell H_B}
e^{-i \gamma_\ell H_C}\right)
\left|+\right\rangle^{\otimes n}.
```

The input `w` may be a full `n x n` matrix, but only the upper triangle is used.

## Installation

Install the package in editable mode from the project root:

```bash
python -m pip install -e .
```

At minimum, make sure Numba is installed:

```bash
python -m pip install numba
```

The package also uses NumPy and SciPy. Optional extras include `cvxpy` for the
Goemans-Williamson reference and `matplotlib` for plotting.

## Quick Start

```python
import numpy as np
from numba_qaoa import QAOASimulator

w = np.array([
    [0.0, 1.0, -0.5],
    [0.0, 0.0, 0.8],
    [0.0, 0.0, 0.0],
])
h = np.array([0.1, -0.2, 0.3])

sim = QAOASimulator(w, h, p=2)
params = np.array([0.2, 0.4, -0.1, 0.3])

energy = sim.energy(params)
grad = sim.gradient(params)
result = sim.optimize(params, method="l-bfgs-b")
```

## Sampling

Sample bitstrings from the final state and evaluate their full-objective
energies:

```python
samples = sim.sample_bitstrings(params, n_samples=1000, seed=1)

print(samples["bitstrings"])
print(samples["energies"])
```

In a notebook, plot the sampled energy distribution:

```python
fig, ax, samples = sim.plot_sampled_energy_distribution(
    params,
    n_samples=1000,
    bins=40,
    seed=1,
)
```

## Banded Phase Surrogates

To use a sparse/banded Hamiltonian for the QAOA phase layers while still
evaluating the full dense objective:

```python
sim = QAOASimulator(
    w,
    h,
    p=6,
    max_bandwidth=2,
    permutation="spectral",
)

print(sim.energy(params))       # full objective
print(sim.phase_energy(params)) # banded phase Hamiltonian objective
print(sim.phase_permutation)    # logical order used to define the band
```

`max_bandwidth=k` keeps couplings whose permuted indices are at most `k` apart.
The default permutation heuristic uses a weighted spectral ordering followed by
adjacent-swap local search to keep as much upper-triangle coupling weight as
possible inside the band.

## Cache Modes

- `cache_mode="full"` saves every intermediate gate state. Fastest, most memory.
- `cache_mode="layer"` saves layer-boundary states only. Balanced default candidate.
- `cache_mode="adjoint"` stores no forward states during gradients. Lowest memory.
- `cache_mode="auto"` chooses based on a memory budget.

## CPU Parallelism

The CPU kernels use Numba `parallel=True` / `prange` for statevector-wide work:

- cost-vector construction from the upper-triangle edge list
- cost phase application
- X-mixer pair updates
- energy and gradient reductions
- state initialization and copies

You can control the number of Numba threads before running Python:

```bash
NUMBA_NUM_THREADS=8 python examples/basic_qaoa.py
```

or inside Python before kernels compile:

```python
import numba

numba.set_num_threads(8)
```

## Optimizers

Use either:

- `method="adam"` for the local lightweight Adam optimizer
- `method="l-bfgs-b"` for `scipy.optimize.minimize(..., method="L-BFGS-B")`

For layer-by-layer greedy initialization:

```python
result = sim.optimize_greedy(
    method="l-bfgs-b",
    expansion="append_random",
    seed=1,
    init_scale=0.1,
    scipy_options={"maxiter": 300},
)

print(result["fun"])
print(result["x"])
```

This optimizes depth 1, appends a small random new layer, optimizes depth 2, and
repeats until it reaches `sim.p`. Use `expansion="append_zero"` if you want the
new depth to start as an exact identity extension of the previous circuit.

## Reference Solvers

For a Goemans-Williamson-style SDP reference:

```python
ref = sim.reference_solution(method="goemans-williamson", n_rounds=512, seed=1)

print(ref.energy)
print(ref.bitstring)
print(ref.sdp_bound)
```

This uses the standard MaxCut SDP idea. For fields or signed couplings, it uses
the analogous Ising SDP relaxation with an anchor spin, so it is a useful
reference but not the classic MaxCut approximation guarantee.

## Local Scratch Work

Messy notebooks and exploratory scripts should live under `scratch/` or `notebooks/`.
Both folders are ignored by git.
