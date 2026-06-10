# NumbaQAOA

Fast exact statevector QAOA simulator for Ising Hamiltonians

```text
H_C = sum_{i<j} w_ij Z_i Z_j + sum_i h_i Z_i
```

The input `w` may be a full `n x n` matrix, but only the upper triangle is used.

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
