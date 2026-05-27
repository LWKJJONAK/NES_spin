import argparse
import math

import numpy as np


DEFAULT_N = 20
DEFAULT_H_VALUES = [0.5, 1.0, 2.0]
DEFAULT_K = 4
DEFAULT_MAX_STATES = 1 << 22


def spin_signs(states, site):
    bits = ((states >> np.uint64(site)) & np.uint64(1)).astype(np.int8)
    return 1 - 2 * bits


def periodic_ising_diagonal(n):
    """Diagonal for rbm.cc PeriodicIsing: -sum_i sigma_z_i sigma_z_{i+1}."""
    dim = 1 << n
    states = np.arange(dim, dtype=np.uint64)
    diag = np.zeros(dim, dtype=float)
    for i in range(n):
        zi = spin_signs(states, i)
        zj = spin_signs(states, (i + 1) % n)
        diag -= zi * zj
    return diag


def apply_hamiltonian(n, h, v):
    diag = periodic_ising_diagonal(n)
    states = np.arange(diag.size, dtype=np.uint64)
    result = diag * v
    if h != 0:
        for i in range(n):
            flipped = states ^ np.uint64(1 << i)
            result = result - h * v[flipped]
    return result


def dense_hamiltonian(n, h):
    dim = 1 << n
    eye = np.eye(dim)
    return np.column_stack([apply_hamiltonian(n, h, eye[:, i]) for i in range(dim)])


def quasiparticle_sums(energies):
    sums = np.zeros(1, dtype=float)
    parity = np.zeros(1, dtype=np.int8)
    for energy in energies:
        sums = np.concatenate((sums, sums + energy))
        parity = np.concatenate((parity, 1 - parity))
    return sums, parity


def sector_energies(n, h, boundary, quasiparticle_parity):
    if boundary == "anti-periodic":
        momenta = [(2 * m + 1) * math.pi / n for m in range(n)]
    elif boundary == "periodic":
        momenta = [2 * m * math.pi / n for m in range(n)]
    else:
        raise ValueError(f"unknown boundary: {boundary}")

    eps = np.asarray(
        [2 * math.sqrt(1 + h * h - 2 * h * math.cos(k)) for k in momenta],
        dtype=float,
    )
    sums, parity = quasiparticle_sums(eps)
    base = -0.5 * float(np.sum(eps))
    return base + sums[parity == quasiparticle_parity]


def exact_periodic_tfim_energies(n, h):
    anti_periodic_even = sector_energies(n, h, "anti-periodic", 0)
    periodic_parity = 0 if h < 1 else 1
    periodic_sector = sector_energies(n, h, "periodic", periodic_parity)
    return np.sort(np.concatenate((anti_periodic_even, periodic_sector)))


def lowest_energies(n, h, k):
    dim = 1 << n
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > dim:
        raise ValueError(f"k={k} is larger than Hilbert dimension {dim}")

    return exact_periodic_tfim_energies(n, h)[:k]


def parse_h_values(values):
    if values is None:
        return DEFAULT_H_VALUES
    h_values = []
    for value in values:
        h_values.extend(float(item) for item in value.split(",") if item.strip())
    return h_values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Exact finite-size solver for the periodic transverse-field Ising model "
            "matching rbm.cc PeriodicIsing."
        )
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of spins.")
    parser.add_argument(
        "--h",
        nargs="*",
        help="Transverse fields. Accepts space-separated values or comma-separated groups.",
    )
    parser.add_argument("-k", "--num-energies", type=int, default=DEFAULT_K)
    parser.add_argument("--max-states", type=int, default=DEFAULT_MAX_STATES)
    return parser.parse_args()


def main():
    args = parse_args()
    dim = 1 << args.n
    if dim > args.max_states:
        raise SystemExit(
            f"N={args.n} gives Hilbert dimension {dim}, above --max-states={args.max_states}."
        )

    h_values = parse_h_values(args.h)
    print("Model: rbm.cc PeriodicIsing")
    print("Hamiltonian: H = -sum_i sigma_z_i sigma_z_{i+1} - h sum_i sigma_x_i")
    print("Solver: exact Jordan-Wigner finite-size spectrum")
    print(f"N = {args.n}, dim = {dim}, k = {args.num_energies}")
    for h in h_values:
        energies = lowest_energies(args.n, h, args.num_energies)
        print(f"\nh = {h:.16g}")
        for i, energy in enumerate(energies):
            print(f"E{i}/N = {energy / args.n:.16g}")


if __name__ == "__main__":
    main()
