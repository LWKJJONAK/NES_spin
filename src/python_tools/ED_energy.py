## Output the lowest k energies for a transverse-field Ising Hamiltonian with a given J matrix.
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import LinearOperator, eigsh


DEFAULT_N = 20
DEFAULT_MODE = 7
DEFAULT_J_FILE = "J_N20_mode7.mat"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_H = 2
DEFAULT_K = 4
DEFAULT_MAX_STATES = 1 << 22
DEFAULT_SCAN_H_MIN = 0.0
DEFAULT_SCAN_H_MAX = 4.0
DEFAULT_SCAN_H_POINTS = 41
DEFAULT_SCAN_OUTPUT = "ED_scan_N20_mode7"
DENSE_LIMIT = 4096


def load_j_matrix(path):
    data = sio.loadmat(path)
    if "J" not in data:
        raise KeyError(f"{path} does not contain a variable named 'J'")

    J = np.asarray(data["J"], dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"J must be a square matrix, got shape {J.shape}")
    return J


def spin_signs(states, site):
    bits = ((states >> np.uint64(site)) & np.uint64(1)).astype(np.int8)
    return 1 - 2 * bits


def ising_diagonal(J, add_nearest_neighbor=False):
    """Diagonal part of sum_ij J_ij sz_i sz_j in the computational basis."""
    n = J.shape[0]
    dim = 1 << n
    states = np.arange(dim, dtype=np.uint64)
    spins = [spin_signs(states, i) for i in range(n)]
    diag = np.full(dim, np.trace(J), dtype=float)

    for i in range(1, n):
        zi = spins[i]
        for j in range(i):
            coeff = J[i, j] + J[j, i]
            if coeff != 0:
                diag += coeff * zi * spins[j]

    if add_nearest_neighbor:
        for i in range(n):
            diag += spins[i] * spins[(i + 1) % n]

    return diag


def hamiltonian_operator(diag, h, n):
    """Matrix-free operator for H = diag - h * sum_i sx_i."""
    dim = diag.size
    states = np.arange(dim, dtype=np.uint64)

    def matvec(v):
        result = diag * v
        if h != 0:
            for i in range(n):
                flipped = states ^ np.uint64(1 << i)
                result = result - h * v[flipped]
        return result

    return LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)


def dense_lowest_energies(op, k):
    dim = op.shape[0]
    eye = np.eye(dim)
    ham = np.column_stack([op.matvec(eye[:, i]) for i in range(dim)])
    return np.linalg.eigvalsh(ham)[:k]


def lowest_energies(J, h, k, add_nearest_neighbor=False):
    n = J.shape[0]
    dim = 1 << n
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > dim:
        raise ValueError(f"k={k} is larger than Hilbert dimension {dim}")

    diag = ising_diagonal(J, add_nearest_neighbor=add_nearest_neighbor)
    if h == 0:
        return np.sort(diag)[:k]

    op = hamiltonian_operator(diag, h, n)
    if k >= dim:
        return dense_lowest_energies(op, k)
    if dim <= DENSE_LIMIT and k >= dim - 1:
        return dense_lowest_energies(op, k)

    energies = eigsh(op, k=k, which="SA", return_eigenvectors=False)
    return np.sort(energies)


def lowest_energies_from_diag(diag, h, n, k, v0=None):
    dim = diag.size
    if h == 0:
        return np.sort(diag)[:k], None

    op = hamiltonian_operator(diag, h, n)
    if k >= dim or (dim <= DENSE_LIMIT and k >= dim - 1):
        return dense_lowest_energies(op, k), None

    energies, vectors = eigsh(op, k=k, which="SA", return_eigenvectors=True, v0=v0)
    order = np.argsort(energies)
    return energies[order], vectors[:, order]


def output_base_path(output):
    path = Path(output)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path


def mode_pattern_from_j(J):
    sym_j = 0.5 * (J + J.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym_j)
    mode_vector = eigenvectors[:, np.argmin(eigenvalues)]
    pattern = np.sign(mode_vector)
    pattern[pattern == 0] = 1
    return pattern.astype(float), mode_vector


def z_observable_values(n, pattern):
    dim = 1 << n
    states = np.arange(dim, dtype=np.uint64)
    mz = np.zeros(dim, dtype=float)
    mode = np.zeros(dim, dtype=float)
    for i in range(n):
        z = spin_signs(states, i).astype(float)
        mz += z
        mode += pattern[i] * z
    return mz / n, mode / n


def flip_indices(n):
    dim = 1 << n
    states = np.arange(dim, dtype=np.uint64)
    return [states ^ np.uint64(1 << i) for i in range(n)]


def binder(m2, m4):
    if m2 <= 0:
        return np.nan
    return 1 - m4 / (3 * m2 * m2)


def z_moments(prob, values):
    abs_mean = np.sum(prob * np.abs(values))
    second = np.sum(prob * values**2)
    fourth = np.sum(prob * values**4)
    return abs_mean, second, fourth, binder(second, fourth)


def ground_observables_from_vector(psi, mz_values, mode_values, flips):
    prob = np.abs(psi) ** 2
    mz_abs, mz2, mz4, mz_binder = z_moments(prob, mz_values)
    mode_abs, mode2, mode4, mode_binder = z_moments(prob, mode_values)
    mx = 0.0
    for flipped in flips:
        mx += np.vdot(psi, psi[flipped]).real
    mx /= len(flips)
    return {
        "mx": mx,
        "mz_abs": mz_abs,
        "mz2": mz2,
        "mz_binder": mz_binder,
        "mode_abs": mode_abs,
        "mode2": mode2,
        "mode_binder": mode_binder,
    }


def ground_observables_from_diagonal(diag, mz_values, mode_values):
    tol = max(1e-12, abs(np.min(diag)) * 1e-12)
    ground = np.flatnonzero(diag <= np.min(diag) + tol)
    prob = np.zeros_like(diag)
    prob[ground] = 1 / len(ground)
    mz_abs, mz2, mz4, mz_binder = z_moments(prob, mz_values)
    mode_abs, mode2, mode4, mode_binder = z_moments(prob, mode_values)
    return {
        "mx": 0.0,
        "mz_abs": mz_abs,
        "mz2": mz2,
        "mz_binder": mz_binder,
        "mode_abs": mode_abs,
        "mode2": mode2,
        "mode_binder": mode_binder,
    }


def save_scan_results(base_path, h_values, energies_per_spin, gaps_per_spin, observables, j_file, n):
    headers = ["h"]
    headers.extend(f"E{i}_over_N" for i in range(energies_per_spin.shape[1]))
    headers.extend(gaps_per_spin.keys())
    headers.extend(observables.keys())

    columns = [h_values, energies_per_spin]
    columns.extend(gaps_per_spin[name] for name in gaps_per_spin)
    columns.extend(observables[name] for name in observables)
    table = np.column_stack(columns)
    csv_path = base_path.with_suffix(".csv")
    np.savetxt(csv_path, table, delimiter=",", header=",".join(headers), comments="")

    mat_path = base_path.with_suffix(".mat")
    sio.savemat(
        mat_path,
        {
            "h": h_values,
            "energies_over_N": energies_per_spin,
            **gaps_per_spin,
            **observables,
            "J_file": str(j_file),
            "N": n,
        },
    )
    return csv_path, mat_path


def plot_scan_results(base_path, h_values, energies_per_spin, gaps_per_spin, observables):
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for i in range(energies_per_spin.shape[1]):
        axes[0].plot(h_values, energies_per_spin[:, i], marker="o", markersize=3, label=f"E{i}/N")
    axes[0].set_ylabel("Energy / N")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, values in gaps_per_spin.items():
        axes[1].plot(h_values, values, marker="o", markersize=3, label=name)
    axes[1].set_ylabel("Gap / N")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(h_values, observables["mode2"], marker="o", markersize=3, label="mode2")
    axes[2].plot(h_values, observables["mode_abs"], marker="o", markersize=3, label="mode_abs")
    axes[2].plot(h_values, observables["mz2"], marker="o", markersize=3, label="mz2")
    axes[2].plot(h_values, observables["mx"], marker="o", markersize=3, label="mx")
    axes[2].set_xlabel("h")
    axes[2].set_ylabel("Order parameters")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = base_path.with_suffix(".png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return png_path


def scan_h_values(J, h_values, k, add_nearest_neighbor=False):
    if k < 2:
        raise ValueError("scan mode needs k >= 2 to compute a gap")

    n = J.shape[0]
    diag = ising_diagonal(J, add_nearest_neighbor=add_nearest_neighbor)
    energies = np.empty((len(h_values), k), dtype=float)
    pattern, _ = mode_pattern_from_j(J)
    mz_values, mode_values = z_observable_values(n, pattern)
    flips = flip_indices(n)
    observable_names = ["mx", "mz_abs", "mz2", "mz_binder", "mode_abs", "mode2", "mode_binder"]
    observables = {name: np.empty(len(h_values), dtype=float) for name in observable_names}
    v0 = None

    header = "h " + " ".join(f"E{i}/N" for i in range(k)) + " gap01/N gap20/N mode2 mx"
    print(header, flush=True)
    for idx, h in enumerate(h_values):
        vals, vecs = lowest_energies_from_diag(diag, h, n, k, v0=v0)
        energies[idx] = vals
        if vecs is not None:
            v0 = vecs[:, 0]
            obs = ground_observables_from_vector(vecs[:, 0], mz_values, mode_values, flips)
        else:
            obs = ground_observables_from_diagonal(diag, mz_values, mode_values)
        for name in observable_names:
            observables[name][idx] = obs[name]

        vals_per_spin = vals / n
        gap_per_spin = (vals[1] - vals[0]) / n
        gap20_per_spin = (vals[2] - vals[0]) / n if k >= 3 else np.nan
        row = [f"{h:.8g}"]
        row.extend(f"{value:.12g}" for value in vals_per_spin)
        row.append(f"{gap_per_spin:.12g}")
        row.append(f"{gap20_per_spin:.12g}")
        row.append(f"{obs['mode2']:.12g}")
        row.append(f"{obs['mx']:.12g}")
        print(" ".join(row), flush=True)

    energies_per_spin = energies / n
    gaps_per_spin = {"gap01_over_N": (energies[:, 1] - energies[:, 0]) / n}
    if k >= 3:
        gaps_per_spin["gap20_over_N"] = (energies[:, 2] - energies[:, 0]) / n
    return energies_per_spin, gaps_per_spin, observables


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the lowest k energies for a J-matrix transverse-field Ising Hamiltonian."
    )
    parser.add_argument(
        "j_file",
        nargs="?",
        help=(
            f"MAT file containing variable J. Default: {DEFAULT_J_FILE}; "
            "if --n or --mode is changed, default becomes J_N{N}_mode{mode}.mat."
        ),
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="N used only for the default file name.")
    parser.add_argument("--mode", type=int, default=DEFAULT_MODE, help="Mode used only for the default file name.")
    parser.add_argument(
        "-k",
        "--k",
        "--num-energies",
        dest="num_energies",
        type=int,
        default=DEFAULT_K,
        help="Number of lowest energies.",
    )
    parser.add_argument("--h", type=float, default=DEFAULT_H, help="Transverse field strength.")
    parser.add_argument("--scan", action="store_true", help="Scan gap versus h and save CSV/MAT/PNG.")
    parser.add_argument("--h-min", type=float, default=DEFAULT_SCAN_H_MIN, help="Minimum h for --scan.")
    parser.add_argument("--h-max", type=float, default=DEFAULT_SCAN_H_MAX, help="Maximum h for --scan.")
    parser.add_argument("--h-points", type=int, default=DEFAULT_SCAN_H_POINTS, help="Number of h points for --scan.")
    parser.add_argument("--output", default=DEFAULT_SCAN_OUTPUT, help="Output file stem for --scan.")
    parser.add_argument(
        "--max-states",
        type=int,
        default=DEFAULT_MAX_STATES,
        help="Safety limit for Hilbert-space dimension.",
    )
    parser.add_argument(
        "--add-nearest-neighbor",
        action="store_true",
        help="Also add the old periodic nearest-neighbor sz_i sz_{i+1} term.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.j_file:
        j_file = Path(args.j_file)
    elif args.n != DEFAULT_N or args.mode != DEFAULT_MODE:
        j_file = SCRIPT_DIR / f"J_N{args.n}_mode{args.mode}.mat"
    else:
        j_file = SCRIPT_DIR / DEFAULT_J_FILE
    J = load_j_matrix(j_file)
    n = J.shape[0]
    dim = 1 << n

    if dim > args.max_states:
        raise SystemExit(
            f"N={n} gives Hilbert dimension {dim}, above --max-states={args.max_states}. "
            "Exact diagonalization is only practical for small N."
        )

    print(f"J file: {j_file}", flush=True)
    print(f"N = {n}, dim = {dim}, k = {args.num_energies}", flush=True)

    if args.scan:
        h_values = np.linspace(args.h_min, args.h_max, args.h_points)
        print(
            f"Scanning h from {args.h_min} to {args.h_max} with {args.h_points} points...",
            flush=True,
        )
        energies_per_spin, gaps_per_spin, observables = scan_h_values(
            J,
            h_values,
            args.num_energies,
            add_nearest_neighbor=args.add_nearest_neighbor,
        )
        base_path = output_base_path(args.output)
        csv_path, mat_path = save_scan_results(
            base_path,
            h_values,
            energies_per_spin,
            gaps_per_spin,
            observables,
            j_file,
            n,
        )
        png_path = plot_scan_results(base_path, h_values, energies_per_spin, gaps_per_spin, observables)
        gap01 = gaps_per_spin["gap01_over_N"]
        min_index = int(np.argmin(gap01))
        print(f"Minimum scanned gap01/N = {gap01[min_index]:.16g} at h = {h_values[min_index]:.16g}")
        if "gap20_over_N" in gaps_per_spin:
            gap20 = gaps_per_spin["gap20_over_N"]
            min20_index = int(np.argmin(gap20))
            print(f"Minimum scanned gap20/N = {gap20[min20_index]:.16g} at h = {h_values[min20_index]:.16g}")
        mode_slope = np.abs(np.gradient(observables["mode2"], h_values))
        slope_index = int(np.argmax(mode_slope))
        print(
            f"Largest |d mode2/dh| = {mode_slope[slope_index]:.16g} "
            f"near h = {h_values[slope_index]:.16g}"
        )
        print(f"Saved CSV: {csv_path}")
        print(f"Saved MAT: {mat_path}")
        print(f"Saved plot: {png_path}")
        return

    print(f"h = {args.h}", flush=True)
    print("Computing lowest energies...", flush=True)

    energies = lowest_energies(
        J,
        h=args.h,
        k=args.num_energies,
        add_nearest_neighbor=args.add_nearest_neighbor,
    )

    if args.add_nearest_neighbor:
        print("Included extra periodic nearest-neighbor sz_i sz_{i+1} term.")
    print("Lowest energies:")
    for i, energy in enumerate(energies):
        print(f"E[{i}] = {energy:.16g}")


if __name__ == "__main__":
    main()
