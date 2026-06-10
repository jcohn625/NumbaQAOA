import numpy as np
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from numba_qaoa import QAOASimulator


def main():
    w = np.array(
        [
            [0.0, 1.0, -0.5, 0.0],
            [0.0, 0.0, 0.8, 1.2],
            [0.0, 0.0, 0.0, -0.7],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    h = np.array([0.1, -0.2, 0.3, -0.1])
    sim = QAOASimulator(w, h, p=2)
    params0 = np.array([0.2, 0.4, -0.1, 0.3])

    print("initial energy:", sim.energy(params0))
    print("initial gradient:", sim.gradient(params0))

    result = sim.optimize(params0, method="l-bfgs-b", scipy_options={"maxiter": 50})
    print("final energy:", result.fun)
    print("final params:", result.x)


if __name__ == "__main__":
    main()
