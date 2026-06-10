import numpy as np

from numba_qaoa import QAOASimulator


def test_greedy_append_zero_preserves_previous_layers():
    w = np.array(
        [
            [0.0, 1.0, -0.5],
            [0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0],
        ]
    )
    h = np.array([0.2, -0.1, 0.3])
    sim = QAOASimulator(w, h, p=3)

    result = sim.optimize_greedy(
        np.array([0.1, -0.2]),
        method="adam",
        expansion="append_zero",
        steps=0,
    )

    np.testing.assert_allclose(result["x"], np.array([0.1, -0.2, 0.0, 0.0, 0.0, 0.0]))
    assert [stage["depth"] for stage in result["stages"]] == [1, 2, 3]


def test_greedy_lbfgsb_returns_full_depth_params():
    rng = np.random.default_rng(12)
    n = 4
    p = 3
    w = rng.normal(size=(n, n))
    h = rng.normal(size=n)
    sim = QAOASimulator(w, h, p=p)

    result = sim.optimize_greedy(
        method="l-bfgs-b",
        expansion="append_zero",
        cache_mode="layer",
        scipy_options={"maxiter": 10},
    )

    assert result["x"].shape == (2 * p,)
    assert result["jac"].shape == (2 * p,)
    assert len(result["stages"]) == p
    np.testing.assert_allclose(sim.params, result["x"])


def test_greedy_random_initialization_is_seeded():
    w = np.array([[0.0, 1.0], [0.0, 0.0]])
    h = np.zeros(2)
    sim = QAOASimulator(w, h, p=2)

    result_a = sim.optimize_greedy(method="adam", steps=0, seed=3, init_scale=0.2)
    result_b = sim.optimize_greedy(method="adam", steps=0, seed=3, init_scale=0.2)

    np.testing.assert_allclose(result_a["x"], result_b["x"])
    assert not np.allclose(result_a["x"][:2], 0.0)
