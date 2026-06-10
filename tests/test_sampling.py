import numpy as np
import pytest

from numba_qaoa import QAOASimulator


def test_sample_bitstrings_shapes_and_energies():
    rng = np.random.default_rng(21)
    n = 4
    p = 2
    w = rng.normal(size=(n, n))
    h = rng.normal(size=n)
    params = rng.normal(scale=0.2, size=2 * p)
    sim = QAOASimulator(w, h, p=p)

    samples = sim.sample_bitstrings(params, n_samples=25, seed=3)

    assert samples["states"].shape == (25,)
    assert samples["bitstrings"].shape == (25, n)
    assert samples["energies"].shape == (25,)
    assert samples["probabilities"].shape == (25,)
    np.testing.assert_allclose(samples["energies"], sim.objective_cost[samples["states"]])


def test_sample_bitstrings_is_seeded():
    rng = np.random.default_rng(22)
    n = 3
    p = 1
    w = rng.normal(size=(n, n))
    h = rng.normal(size=n)
    params = rng.normal(scale=0.2, size=2 * p)
    sim = QAOASimulator(w, h, p=p)

    a = sim.sample_bitstrings(params, n_samples=20, seed=9)
    b = sim.sample_bitstrings(params, n_samples=20, seed=9)

    np.testing.assert_array_equal(a["states"], b["states"])
    np.testing.assert_array_equal(a["bitstrings"], b["bitstrings"])


def test_sample_bitstrings_rejects_nonpositive_sample_count():
    sim = QAOASimulator(np.zeros((2, 2)), np.zeros(2), p=1)
    with pytest.raises(ValueError, match="n_samples"):
        sim.sample_bitstrings(np.zeros(2), n_samples=0)


def test_plot_sampled_energy_distribution_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)

    sim = QAOASimulator(np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros(2), p=1)
    fig, ax, samples = sim.plot_sampled_energy_distribution(
        np.array([0.2, 0.3]),
        n_samples=10,
        seed=2,
        bins=4,
    )

    assert fig is ax.figure
    assert samples["energies"].shape == (10,)
