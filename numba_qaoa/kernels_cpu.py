import numpy as np
import numba as nb


@nb.njit
def build_cost_upper(cost, w, h):
    dim = cost.shape[0]
    n = h.shape[0]
    for state in range(dim):
        e = 0.0
        for i in range(n):
            zi = 1.0
            if (state >> i) & 1:
                zi = -1.0
            e += h[i] * zi
            for j in range(i + 1, n):
                zj = 1.0
                if (state >> j) & 1:
                    zj = -1.0
                e += w[i, j] * zi * zj
        cost[state] = e


@nb.njit
def fill_plus_state(state):
    amp = 1.0 / np.sqrt(state.shape[0])
    for i in range(state.shape[0]):
        state[i] = amp + 0.0j


@nb.njit
def copy_state(dst, src):
    for i in range(src.shape[0]):
        dst[i] = src[i]


@nb.njit
def apply_cost_phase_inplace(state, cost, gamma):
    for i in range(state.shape[0]):
        angle = -gamma * cost[i]
        phase = np.cos(angle) + 1j * np.sin(angle)
        state[i] *= phase


@nb.njit
def apply_x_mixer_inplace(state, beta, n):
    c = np.cos(beta)
    s = np.sin(beta)
    minus_i_s = -1j * s
    dim = state.shape[0]
    for q in range(n):
        step = 1 << q
        block = step << 1
        for base in range(0, dim, block):
            for offset in range(step):
                i = base + offset
                j = i + step
                ai = state[i]
                aj = state[j]
                state[i] = c * ai + minus_i_s * aj
                state[j] = minus_i_s * ai + c * aj


@nb.njit
def apply_cost_operator(dst, state, cost):
    for i in range(state.shape[0]):
        dst[i] = cost[i] * state[i]


@nb.njit
def energy_from_state(state, cost):
    e = 0.0
    for i in range(state.shape[0]):
        prob = state[i].real * state[i].real + state[i].imag * state[i].imag
        e += prob * cost[i]
    return e


@nb.njit
def gamma_gradient_inner(left, right, cost):
    acc = 0.0 + 0.0j
    for i in range(right.shape[0]):
        acc += np.conj(left[i]) * cost[i] * right[i]
    return 2.0 * acc.imag


@nb.njit
def beta_gradient_inner(left, right, n):
    acc = 0.0 + 0.0j
    dim = right.shape[0]
    for q in range(n):
        step = 1 << q
        block = step << 1
        for base in range(0, dim, block):
            for offset in range(step):
                i = base + offset
                j = i + step
                acc += np.conj(left[i]) * right[j]
                acc += np.conj(left[j]) * right[i]
    return 2.0 * acc.imag


@nb.njit
def norm2(state):
    acc = 0.0
    for i in range(state.shape[0]):
        acc += state[i].real * state[i].real + state[i].imag * state[i].imag
    return acc
