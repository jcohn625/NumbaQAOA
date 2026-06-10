import numpy as np


def band_weight(w, order, bandwidth, power=2.0):
    w = np.asarray(w, dtype=np.float64)
    order = np.asarray(order, dtype=np.int64)
    pos = np.empty(order.size, dtype=np.int64)
    pos[order] = np.arange(order.size)

    total = 0.0
    for i in range(order.size):
        for j in range(i + 1, order.size):
            if abs(pos[i] - pos[j]) <= bandwidth:
                total += abs(w[i, j]) ** power
    return float(total)


def spectral_order(w, power=2.0):
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("w must be a square n x n matrix")
    n = w.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int64)

    upper = np.triu(w, k=1)
    weights = np.abs(upper + upper.T) ** power
    degrees = weights.sum(axis=1)
    if np.allclose(degrees, 0.0):
        return np.arange(n, dtype=np.int64)

    laplacian = np.diag(degrees) - weights
    _, eigvecs = np.linalg.eigh(laplacian)
    fiedler = eigvecs[:, 1]
    return np.argsort(fiedler).astype(np.int64)


def improve_order_adjacent_swaps(w, order, bandwidth, passes=4, power=2.0):
    order = np.asarray(order, dtype=np.int64).copy()
    best = band_weight(w, order, bandwidth, power=power)
    for _ in range(passes):
        improved = False
        for idx in range(order.size - 1):
            candidate = order.copy()
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            score = band_weight(w, candidate, bandwidth, power=power)
            if score > best:
                order = candidate
                best = score
                improved = True
        if not improved:
            break
    return order


def optimize_bandwidth_order(
    w,
    bandwidth,
    *,
    method="spectral",
    local_search_passes=4,
    power=2.0,
):
    w = np.asarray(w, dtype=np.float64)
    if bandwidth < 0:
        raise ValueError("bandwidth must be non-negative")
    if method == "identity":
        order = np.arange(w.shape[0], dtype=np.int64)
    elif method == "spectral":
        order = spectral_order(w, power=power)
    else:
        order = np.asarray(method, dtype=np.int64)
        if sorted(order.tolist()) != list(range(w.shape[0])):
            raise ValueError("custom permutation must contain each index exactly once")

    if local_search_passes > 0:
        order = improve_order_adjacent_swaps(
            w,
            order,
            bandwidth,
            passes=local_search_passes,
            power=power,
        )
    return order


def banded_by_permutation(w, bandwidth, order):
    w = np.asarray(w, dtype=np.float64)
    order = np.asarray(order, dtype=np.int64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("w must be a square n x n matrix")
    if order.shape != (w.shape[0],):
        raise ValueError("order must have shape (n,)")
    if bandwidth < 0:
        raise ValueError("bandwidth must be non-negative")

    pos = np.empty(order.size, dtype=np.int64)
    pos[order] = np.arange(order.size)

    out = np.zeros_like(w, dtype=np.float64)
    for i in range(w.shape[0]):
        for j in range(i + 1, w.shape[0]):
            if abs(pos[i] - pos[j]) <= bandwidth:
                out[i, j] = w[i, j]
    return out


def make_banded_surrogate(
    w,
    bandwidth,
    *,
    permutation="spectral",
    local_search_passes=4,
    power=2.0,
):
    order = optimize_bandwidth_order(
        w,
        bandwidth,
        method=permutation,
        local_search_passes=local_search_passes,
        power=power,
    )
    return banded_by_permutation(w, bandwidth, order), order
