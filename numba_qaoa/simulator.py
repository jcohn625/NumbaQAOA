import numpy as np
from scipy.optimize import minimize

from .kernels_cpu import (
    apply_cost_phase_inplace,
    apply_x_mixer_inplace,
    beta_gradient_inner,
    build_cost_upper_edges,
    copy_state,
    energy_and_apply_cost_operator,
    energy_from_state,
    fill_plus_state,
    gamma_gradient_inner,
)
from .memory import available_memory_bytes, real_dtype_for_complex
from .optimizers import Adam
from .references import goemans_williamson_reference


class QAOASimulator:
    def __init__(
        self,
        w,
        h,
        p,
        *,
        dtype=np.complex128,
        cache_mode="auto",
        cost_mode="cached",
        max_cache_bytes=None,
    ):
        self.w = np.asarray(w, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.p = int(p)
        self.dtype = np.dtype(dtype)
        self.real_dtype = real_dtype_for_complex(self.dtype)
        self.cache_mode = cache_mode
        self.cost_mode = cost_mode
        self.max_cache_bytes = max_cache_bytes

        if self.w.ndim != 2 or self.w.shape[0] != self.w.shape[1]:
            raise ValueError("w must be a square n x n matrix")
        if self.h.ndim != 1 or self.h.shape[0] != self.w.shape[0]:
            raise ValueError("h must be a length-n vector matching w")
        if self.p < 0:
            raise ValueError("p must be non-negative")
        if self.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError("dtype must be np.complex64 or np.complex128")
        if cost_mode != "cached":
            raise NotImplementedError('only cost_mode="cached" is implemented in this first version')
        if cache_mode not in ("auto", "full", "layer", "adjoint"):
            raise ValueError('cache_mode must be "auto", "full", "layer", or "adjoint"')

        self.n = self.h.shape[0]
        if self.n >= 63:
            raise ValueError("n is too large for int64 basis indexing")
        self.dim = 1 << self.n
        self.num_params = 2 * self.p
        self.params = np.zeros(self.num_params, dtype=np.float64)

        self.edge_i, self.edge_j, self.edge_w = self._build_upper_edges()
        self.cost = np.empty(self.dim, dtype=self.real_dtype)
        build_cost_upper_edges(self.cost, self.edge_i, self.edge_j, self.edge_w, self.h)

        self.state = np.empty(self.dim, dtype=self.dtype)
        self._work = np.empty(self.dim, dtype=self.dtype)

    def memory_estimate(self, cache_mode=None):
        mode = self._resolve_cache_mode(cache_mode or self.cache_mode)
        cbytes = self.dtype.itemsize * self.dim
        rbytes = self.real_dtype.itemsize * self.dim
        base = rbytes + 3 * cbytes
        if mode == "full":
            cache = (2 * self.p + 1) * cbytes
        elif mode == "layer":
            cache = (self.p + 1) * cbytes
        else:
            cache = 0
        return {
            "mode": mode,
            "cost_bytes": rbytes,
            "work_bytes": base,
            "cache_bytes": cache,
            "total_bytes": base + cache,
        }

    def initial_state(self):
        state = np.empty(self.dim, dtype=self.dtype)
        fill_plus_state(state)
        return state

    def build_state(self, params=None, out=None):
        params = self._params(params)
        state = self.state if out is None else out
        fill_plus_state(state)
        for layer in range(self.p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            apply_cost_phase_inplace(state, self.cost, gamma)
            apply_x_mixer_inplace(state, beta, self.n)
        return state

    def energy(self, params=None):
        state = self.build_state(params)
        return float(energy_from_state(state, self.cost))

    def gradient(self, params=None, cache_mode=None):
        params = self._params(params)
        mode = self._resolve_cache_mode(cache_mode or self.cache_mode)
        if mode == "full":
            return self._gradient_full(params)[1]
        if mode == "layer":
            return self._gradient_layer(params)[1]
        return self._gradient_adjoint(params)[1]

    def energy_and_gradient(self, params=None, cache_mode=None):
        params = self._params(params)
        mode = self._resolve_cache_mode(cache_mode or self.cache_mode)
        if mode == "full":
            return self._gradient_full(params)
        if mode == "layer":
            return self._gradient_layer(params)
        return self._gradient_adjoint(params)

    def optimize(
        self,
        params0=None,
        *,
        method="adam",
        steps=1000,
        lr=1e-2,
        cache_mode=None,
        callback=None,
        scipy_options=None,
    ):
        params = self._params(params0).copy()
        method_key = method.lower()
        if method_key == "adam":
            opt = Adam(params.shape, lr=lr)
            history = []
            for step in range(steps):
                energy, grad = self.energy_and_gradient(params, cache_mode=cache_mode)
                history.append(energy)
                if callback is not None:
                    callback(step, params, energy, grad)
                params = opt.step(params, grad)
            final_energy, final_grad = self.energy_and_gradient(params, cache_mode=cache_mode)
            self.params = params.copy()
            return {
                "method": "adam",
                "x": params,
                "fun": final_energy,
                "jac": final_grad,
                "history": np.asarray(history),
            }
        if method_key in ("l-bfgs-b", "lbfgsb"):
            def fun_and_jac(x):
                return self.energy_and_gradient(x, cache_mode=cache_mode)

            result = minimize(
                fun_and_jac,
                params,
                method="L-BFGS-B",
                jac=True,
                options={} if scipy_options is None else scipy_options,
            )
            self.params = result.x.copy()
            return result
        raise ValueError('method must be "adam" or "l-bfgs-b"')

    def optimize_greedy(
        self,
        params0=None,
        *,
        method="l-bfgs-b",
        expansion="append_random",
        seed=None,
        init_scale=0.1,
        steps=1000,
        lr=1e-2,
        cache_mode=None,
        scipy_options=None,
        callback=None,
    ):
        """Optimize layer-by-layer from p=1 up to this simulator's p.

        Each stage optimizes a depth-k simulator. The depth-(k+1) initial point
        is built from the optimized depth-k parameters.
        """
        if self.p == 0:
            energy, grad = self.energy_and_gradient(self.params, cache_mode=cache_mode)
            return {
                "method": f"greedy-{method}",
                "x": self.params.copy(),
                "fun": energy,
                "jac": grad,
                "stages": [],
            }

        rng = np.random.default_rng(seed)
        params = self._first_greedy_params(params0, rng, init_scale)
        stages = []
        for depth in range(1, self.p + 1):
            if depth > 1:
                params = self._expand_greedy_params(params, expansion, rng, init_scale)
            depth_sim = self._clone_with_depth(depth)
            result = depth_sim.optimize(
                params,
                method=method,
                steps=steps,
                lr=lr,
                cache_mode=cache_mode,
                scipy_options=scipy_options,
            )
            params = np.asarray(result["x"] if isinstance(result, dict) else result.x).copy()
            energy = float(result["fun"] if isinstance(result, dict) else result.fun)
            stage = {
                "depth": depth,
                "x": params.copy(),
                "fun": energy,
                "result": result,
            }
            stages.append(stage)
            if callback is not None:
                callback(depth, stage)

        final_energy, final_grad = self.energy_and_gradient(params, cache_mode=cache_mode)
        self.params = params.copy()
        return {
            "method": f"greedy-{method}",
            "x": params,
            "fun": final_energy,
            "jac": final_grad,
            "stages": stages,
        }

    def reference_solution(self, method="goemans-williamson", **kwargs):
        method_key = method.lower()
        if method_key in ("goemans-williamson", "gw", "gnomes-williamson"):
            return goemans_williamson_reference(self.w, self.h, **kwargs)
        raise ValueError('method must be "goemans-williamson"')

    def _gradient_full(self, params):
        states = np.empty((2 * self.p + 1, self.dim), dtype=self.dtype)
        fill_plus_state(states[0])
        idx = 0
        for layer in range(self.p):
            idx += 1
            copy_state(states[idx], states[idx - 1])
            apply_cost_phase_inplace(states[idx], self.cost, params[2 * layer])
            idx += 1
            copy_state(states[idx], states[idx - 1])
            apply_x_mixer_inplace(states[idx], params[2 * layer + 1], self.n)

        left = np.empty(self.dim, dtype=self.dtype)
        energy = float(energy_and_apply_cost_operator(left, states[-1], self.cost))
        grad = np.empty(self.num_params, dtype=np.float64)

        idx = 2 * self.p
        for layer in range(self.p - 1, -1, -1):
            grad[2 * layer + 1] = beta_gradient_inner(left, states[idx], self.n)
            apply_x_mixer_inplace(left, -params[2 * layer + 1], self.n)
            idx -= 1
            grad[2 * layer] = gamma_gradient_inner(left, states[idx], self.cost)
            apply_cost_phase_inplace(left, self.cost, -params[2 * layer])
            idx -= 1
        return energy, grad

    def _gradient_layer(self, params):
        states = np.empty((self.p + 1, self.dim), dtype=self.dtype)
        fill_plus_state(states[0])
        for layer in range(self.p):
            copy_state(states[layer + 1], states[layer])
            apply_cost_phase_inplace(states[layer + 1], self.cost, params[2 * layer])
            apply_x_mixer_inplace(states[layer + 1], params[2 * layer + 1], self.n)

        left = np.empty(self.dim, dtype=self.dtype)
        right_mid = np.empty(self.dim, dtype=self.dtype)
        energy = float(energy_and_apply_cost_operator(left, states[-1], self.cost))
        grad = np.empty(self.num_params, dtype=np.float64)

        for layer in range(self.p - 1, -1, -1):
            grad[2 * layer + 1] = beta_gradient_inner(left, states[layer + 1], self.n)
            apply_x_mixer_inplace(left, -params[2 * layer + 1], self.n)

            copy_state(right_mid, states[layer])
            apply_cost_phase_inplace(right_mid, self.cost, params[2 * layer])
            grad[2 * layer] = gamma_gradient_inner(left, right_mid, self.cost)
            apply_cost_phase_inplace(left, self.cost, -params[2 * layer])
        return energy, grad

    def _gradient_adjoint(self, params):
        right = self.build_state(params, out=np.empty(self.dim, dtype=self.dtype))
        left = np.empty(self.dim, dtype=self.dtype)
        energy = float(energy_and_apply_cost_operator(left, right, self.cost))
        grad = np.empty(self.num_params, dtype=np.float64)

        for layer in range(self.p - 1, -1, -1):
            grad[2 * layer + 1] = beta_gradient_inner(left, right, self.n)
            apply_x_mixer_inplace(right, -params[2 * layer + 1], self.n)
            apply_x_mixer_inplace(left, -params[2 * layer + 1], self.n)

            grad[2 * layer] = gamma_gradient_inner(left, right, self.cost)
            apply_cost_phase_inplace(right, self.cost, -params[2 * layer])
            apply_cost_phase_inplace(left, self.cost, -params[2 * layer])
        return energy, grad

    def _resolve_cache_mode(self, mode):
        if mode != "auto":
            return mode
        budget = self.max_cache_bytes
        if budget is None:
            avail = available_memory_bytes()
            budget = None if avail is None else int(0.25 * avail)
        if budget is None:
            return "layer"
        cbytes = self.dtype.itemsize * self.dim
        if (2 * self.p + 1) * cbytes <= budget:
            return "full"
        if (self.p + 1) * cbytes <= budget:
            return "layer"
        return "adjoint"

    def _params(self, params):
        if params is None:
            params = self.params
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.num_params,):
            raise ValueError(f"params must have shape ({self.num_params},)")
        return params

    def _first_greedy_params(self, params0, rng, init_scale):
        if params0 is None:
            return rng.uniform(-init_scale, init_scale, size=2)
        params0 = np.asarray(params0, dtype=np.float64)
        if params0.shape == (2,):
            return params0.copy()
        if params0.shape == (self.num_params,):
            return params0[:2].copy()
        raise ValueError(f"params0 must have shape (2,) or ({self.num_params},)")

    def _expand_greedy_params(self, params, expansion, rng, init_scale):
        if expansion == "append_zero":
            return np.concatenate([params, np.zeros(2, dtype=np.float64)])
        if expansion == "append_random":
            new_layer = rng.uniform(-init_scale, init_scale, size=2)
            return np.concatenate([params, new_layer])
        if expansion == "repeat_last":
            return np.concatenate([params, params[-2:].copy()])
        raise ValueError('expansion must be "append_random", "append_zero", or "repeat_last"')

    def _clone_with_depth(self, p):
        obj = object.__new__(QAOASimulator)
        obj.w = self.w
        obj.h = self.h
        obj.p = int(p)
        obj.dtype = self.dtype
        obj.real_dtype = self.real_dtype
        obj.cache_mode = self.cache_mode
        obj.cost_mode = self.cost_mode
        obj.max_cache_bytes = self.max_cache_bytes
        obj.n = self.n
        obj.dim = self.dim
        obj.num_params = 2 * obj.p
        obj.params = np.zeros(obj.num_params, dtype=np.float64)
        obj.edge_i = self.edge_i
        obj.edge_j = self.edge_j
        obj.edge_w = self.edge_w
        obj.cost = self.cost
        obj.state = np.empty(obj.dim, dtype=obj.dtype)
        obj._work = np.empty(obj.dim, dtype=obj.dtype)
        return obj

    def _build_upper_edges(self):
        rows, cols = np.triu_indices(self.n, k=1)
        weights = self.w[rows, cols]
        keep = weights != 0.0
        return (
            np.ascontiguousarray(rows[keep], dtype=np.int64),
            np.ascontiguousarray(cols[keep], dtype=np.int64),
            np.ascontiguousarray(weights[keep], dtype=self.real_dtype),
        )
