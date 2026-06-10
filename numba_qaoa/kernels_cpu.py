import numpy as np
import numba as nb


@nb.njit(parallel=True)
def build_cost_upper_edges(cost, edge_i, edge_j, edge_w, h):
    dim = cost.shape[0]
    n = h.shape[0]
    for state in nb.prange(dim):
        e = 0.0
        for i in range(n):
            zi = 1.0
            if (state >> i) & 1:
                zi = -1.0
            e += h[i] * zi
        for edge in range(edge_w.shape[0]):
            i = edge_i[edge]
            j = edge_j[edge]
            zi = 1.0
            zj = 1.0
            if (state >> i) & 1:
                zi = -1.0
            if (state >> j) & 1:
                zj = -1.0
            e += edge_w[edge] * zi * zj
        cost[state] = e


@nb.njit(parallel=True)
def fill_plus_state(state):
    amp = 1.0 / np.sqrt(state.shape[0])
    for i in nb.prange(state.shape[0]):
        state[i] = amp + 0.0j


@nb.njit(parallel=True)
def copy_state(dst, src):
    for i in nb.prange(src.shape[0]):
        dst[i] = src[i]


@nb.njit(parallel=True)
def apply_cost_phase_inplace(state, cost, gamma):
    for i in nb.prange(state.shape[0]):
        angle = -gamma * cost[i]
        phase = np.cos(angle) + 1j * np.sin(angle)
        state[i] *= phase


@nb.njit(parallel=True)
def apply_x_mixer_inplace(state, beta, n):
    c = np.cos(beta)
    s = np.sin(beta)
    minus_i_s = -1j * s
    dim = state.shape[0]
    half_dim = dim >> 1
    for q in range(n):
        step = 1 << q
        low_mask = step - 1
        for pair in nb.prange(half_dim):
            low = pair & low_mask
            high = pair >> q
            i = (high << (q + 1)) | low
            j = i | step
            ai = state[i]
            aj = state[j]
            state[i] = c * ai + minus_i_s * aj
            state[j] = minus_i_s * ai + c * aj


@nb.njit(parallel=True)
def apply_cost_operator(dst, state, cost):
    for i in nb.prange(state.shape[0]):
        dst[i] = cost[i] * state[i]


@nb.njit(parallel=True)
def energy_from_state(state, cost):
    e = 0.0
    for i in nb.prange(state.shape[0]):
        prob = state[i].real * state[i].real + state[i].imag * state[i].imag
        e += prob * cost[i]
    return e


@nb.njit(parallel=True)
def energy_and_apply_cost_operator(dst, state, cost):
    e = 0.0
    for i in nb.prange(state.shape[0]):
        amp = state[i]
        dst[i] = cost[i] * amp
        prob = amp.real * amp.real + amp.imag * amp.imag
        e += prob * cost[i]
    return e


@nb.njit(parallel=True)
def gamma_gradient_inner(left, right, cost):
    imag_acc = 0.0
    for i in nb.prange(right.shape[0]):
        l = left[i]
        r = right[i]
        imag_acc += cost[i] * (l.real * r.imag - l.imag * r.real)
    return 2.0 * imag_acc


@nb.njit(parallel=True)
def beta_gradient_inner(left, right, n):
    imag_acc = 0.0
    dim = right.shape[0]
    half_dim = dim >> 1
    total_terms = n * half_dim
    for term_index in nb.prange(total_terms):
        q = term_index // half_dim
        pair = term_index - q * half_dim
        step = 1 << q
        low_mask = step - 1
        low = pair & low_mask
        high = pair >> q
        i = (high << (q + 1)) | low
        j = i | step
        li = left[i]
        lj = left[j]
        ri = right[i]
        rj = right[j]
        pair_imag = li.real * rj.imag - li.imag * rj.real
        pair_imag += lj.real * ri.imag - lj.imag * ri.real
        imag_acc += pair_imag
    return 2.0 * imag_acc


@nb.njit(parallel=True)
def norm2(state):
    acc = 0.0
    for i in nb.prange(state.shape[0]):
        acc += state[i].real * state[i].real + state[i].imag * state[i].imag
    return acc
