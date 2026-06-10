import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark core NumbaQAOA simulator operations.")
    parser.add_argument("--package-path", default=None, help="Project path to import numba_qaoa from.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 14, 18])
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--cache-modes", nargs="+", default=["adjoint", "layer"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--density", type=float, default=1.0, help="Upper-triangle coupling density.")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex128")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json", action="store_true", help="Print JSON lines instead of a table.")
    return parser.parse_args()


def import_simulator(package_path):
    if package_path is not None:
        sys.path.insert(0, str(pathlib.Path(package_path).resolve()))
    from numba_qaoa import QAOASimulator

    return QAOASimulator


def make_problem(n, p, density, dtype, seed):
    rng = np.random.default_rng(seed + 1009 * n + 9176 * p)
    w = rng.normal(size=(n, n))
    mask = rng.random(size=(n, n)) < density
    w = np.triu(w * mask, k=1)
    h = rng.normal(size=n)
    params = rng.normal(scale=0.2, size=2 * p)
    return w, h, params, getattr(np, dtype)


def median_time(fn, repeats):
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def benchmark_case(QAOASimulator, n, p, cache_mode, repeats, density, dtype, seed):
    w, h, params, np_dtype = make_problem(n, p, density, dtype, seed)

    start = time.perf_counter()
    sim = QAOASimulator(w, h, p=p, dtype=np_dtype, cache_mode=cache_mode)
    init_seconds = time.perf_counter() - start

    sim.energy(params)
    sim.energy_and_gradient(params, cache_mode=cache_mode)

    energy_seconds, energy = median_time(lambda: sim.energy(params), repeats)
    grad_seconds, energy_grad = median_time(
        lambda: sim.energy_and_gradient(params, cache_mode=cache_mode), repeats
    )

    try:
        mem = sim.memory_estimate(cache_mode=cache_mode)
        memory_total_bytes = int(mem["total_bytes"])
        memory_cache_bytes = int(mem["cache_bytes"])
    except Exception:
        memory_total_bytes = None
        memory_cache_bytes = None

    return {
        "n": n,
        "dim": 1 << n,
        "p": p,
        "density": density,
        "dtype": dtype,
        "cache_mode": cache_mode,
        "init_seconds": init_seconds,
        "energy_seconds": energy_seconds,
        "gradient_seconds": grad_seconds,
        "energy": float(energy),
        "gradient_energy": float(energy_grad[0]),
        "memory_total_bytes": memory_total_bytes,
        "memory_cache_bytes": memory_cache_bytes,
    }


def print_table(rows):
    header = (
        "n  dim       p  mode      init(s)   energy(s)  grad(s)   memory(MB)"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['n']:<2d} {row['dim']:<9d} {row['p']:<2d} "
            f"{row['cache_mode']:<9s} "
            f"{row['init_seconds']:<9.4f} "
            f"{row['energy_seconds']:<10.4f} "
            f"{row['gradient_seconds']:<9.4f} "
            f"{format_memory(row['memory_total_bytes']):<}"
        )


def format_memory(memory_bytes):
    if memory_bytes is None:
        return "n/a"
    return f"{memory_bytes / 1e6:.1f}"


def main():
    args = parse_args()
    QAOASimulator = import_simulator(args.package_path)
    rows = []
    for n in args.sizes:
        for cache_mode in args.cache_modes:
            row = benchmark_case(
                QAOASimulator,
                n=n,
                p=args.p,
                cache_mode=cache_mode,
                repeats=args.repeats,
                density=args.density,
                dtype=args.dtype,
                seed=args.seed,
            )
            rows.append(row)
            if args.json:
                print(json.dumps(row, sort_keys=True))
    if not args.json:
        print_table(rows)


if __name__ == "__main__":
    main()
