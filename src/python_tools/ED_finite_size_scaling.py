import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

import ED_energy as ed


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NS = [10, 12, 14, 16, 18, 20]


def parse_ns(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finite-size ED scan for mode7 trapped-ion J matrices."
    )
    parser.add_argument("--ns", default=",".join(map(str, DEFAULT_NS)), help="Comma-separated N values.")
    parser.add_argument("--mode", type=int, default=7, help="Phonon mode label used in J_N{N}_mode{mode}.mat.")
    parser.add_argument("--h-min", type=float, default=4.0)
    parser.add_argument("--h-max", type=float, default=9.0)
    parser.add_argument("--h-points", type=int, default=11)
    parser.add_argument("-k", "--k", type=int, default=4)
    parser.add_argument("--output", default="ED_scaling_mode7")
    parser.add_argument("--max-states", type=int, default=ed.DEFAULT_MAX_STATES)
    return parser.parse_args()


def crossing(h_values, y_values, target):
    diff = y_values - target
    for i in range(len(diff) - 1):
        if diff[i] == 0:
            return h_values[i]
        if diff[i] * diff[i + 1] < 0:
            weight = (target - y_values[i]) / (y_values[i + 1] - y_values[i])
            return h_values[i] + weight * (h_values[i + 1] - h_values[i])
    return np.nan


def max_abs_slope(h_values, y_values):
    slope = np.gradient(y_values, h_values)
    idx = int(np.argmax(np.abs(slope)))
    return h_values[idx], slope[idx], y_values[idx]


def scan_one_size(N, mode, h_values, k, max_states, output_stem):
    j_file = SCRIPT_DIR / f"J_N{N}_mode{mode}.mat"
    J = ed.load_j_matrix(j_file)
    dim = 1 << J.shape[0]
    if dim > max_states:
        raise RuntimeError(f"N={N} dimension {dim} exceeds max_states={max_states}")

    print(f"\n=== N={N}: {j_file.name}, dim={dim} ===", flush=True)
    energies, gaps, observables = ed.scan_h_values(J, h_values, k)
    base_path = SCRIPT_DIR / f"{output_stem}_N{N}"
    csv_path, mat_path = ed.save_scan_results(base_path, h_values, energies, gaps, observables, j_file, N)
    png_path = ed.plot_scan_results(base_path, h_values, energies, gaps, observables)

    h_slope, slope, mode2_value = max_abs_slope(h_values, observables["mode2"])
    mode2_cross = crossing(h_values, observables["mode2"], 0.5)
    mode_abs_cross = crossing(h_values, observables["mode_abs"], 0.5)
    mx_cross = crossing(h_values, observables["mx"], 0.5)
    return {
        "N": N,
        "dim": dim,
        "mode2_slope_h": h_slope,
        "mode2_slope": slope,
        "mode2_at_slope": mode2_value,
        "mode2_cross_0p5_h": mode2_cross,
        "mode_abs_cross_0p5_h": mode_abs_cross,
        "mx_cross_0p5_h": mx_cross,
        "gap01_min_h": h_values[int(np.argmin(gaps["gap01_over_N"]))],
        "gap01_min": np.min(gaps["gap01_over_N"]),
        "gap20_min_h": h_values[int(np.argmin(gaps["gap20_over_N"]))],
        "gap20_min": np.min(gaps["gap20_over_N"]),
        "csv": str(csv_path),
        "mat": str(mat_path),
        "png": str(png_path),
    }


def save_summary(output_stem, summary):
    keys = [
        "N",
        "dim",
        "mode2_slope_h",
        "mode2_slope",
        "mode2_at_slope",
        "mode2_cross_0p5_h",
        "mode_abs_cross_0p5_h",
        "mx_cross_0p5_h",
        "gap01_min_h",
        "gap01_min",
        "gap20_min_h",
        "gap20_min",
    ]
    rows = [[entry[key] for key in keys] for entry in summary]
    csv_path = SCRIPT_DIR / f"{output_stem}_summary.csv"
    np.savetxt(csv_path, np.asarray(rows, dtype=float), delimiter=",", header=",".join(keys), comments="")

    mat_path = SCRIPT_DIR / f"{output_stem}_summary.mat"
    sio.savemat(mat_path, {key: np.asarray([entry[key] for entry in summary]) for key in keys})
    return csv_path, mat_path


def plot_summary(output_stem, summary):
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = np.asarray([entry["N"] for entry in summary], dtype=float)
    invN = 1 / N
    h_slope = np.asarray([entry["mode2_slope_h"] for entry in summary], dtype=float)
    h_cross = np.asarray([entry["mode2_cross_0p5_h"] for entry in summary], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(N, h_slope, marker="o", label="max |d mode2/dh|")
    axes[0].plot(N, h_cross, marker="s", label="mode2=0.5")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("pseudo-critical h")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(invN, h_slope, marker="o", label="max |d mode2/dh|")
    axes[1].plot(invN, h_cross, marker="s", label="mode2=0.5")
    axes[1].set_xlabel("1/N")
    axes[1].set_ylabel("pseudo-critical h")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    path = SCRIPT_DIR / f"{output_stem}_summary.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    Ns = parse_ns(args.ns)
    h_values = np.linspace(args.h_min, args.h_max, args.h_points)
    summary = []
    for N in Ns:
        summary.append(scan_one_size(N, args.mode, h_values, args.k, args.max_states, args.output))

    csv_path, mat_path = save_summary(args.output, summary)
    png_path = plot_summary(args.output, summary)

    print("\nFinite-size summary:")
    for entry in summary:
        print(
            f"N={entry['N']}: h_slope={entry['mode2_slope_h']:.8g}, "
            f"h_mode2_0.5={entry['mode2_cross_0p5_h']:.8g}, "
            f"h_modeabs_0.5={entry['mode_abs_cross_0p5_h']:.8g}, "
            f"gap20_min={entry['gap20_min']:.8g}"
        )
    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved summary MAT: {mat_path}")
    print(f"Saved summary plot: {png_path}")


if __name__ == "__main__":
    main()
