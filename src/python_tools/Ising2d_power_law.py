"""Generate 2D trapped-ion power-law coupling matrices.

The coupling is

    J_ij = J0 * (mean nearest-neighbor distance / r_ij)**alpha

with zero diagonal. This script generates the `J0`/power-law coupling data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from equilibrium_positions import equilibrium_positions


def _loadmat(path: Path, wanted: set[str] | None = None) -> dict[str, np.ndarray]:
    try:
        return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except Exception:
        from mat5_corr_tools import read_mat5_arrays

        return read_mat5_arrays(path, wanted=wanted)


def power_law_coupling_from_positions(r: np.ndarray, *, j0: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    """Return the normalized power-law coupling matrix from ion positions."""

    r = np.asarray(r, dtype=float)
    dist = np.sqrt(np.sum((r[np.newaxis, :, :] - r[:, np.newaxis, :]) ** 2, axis=2))
    np.fill_diagonal(dist, np.max(dist) * 10)
    nearest = np.min(dist, axis=1)
    inv_dist = 1 / dist
    np.fill_diagonal(inv_dist, 0)
    return j0 * (inv_dist * np.mean(nearest)) ** alpha


def generate_positions(
    n_ions: int,
    *,
    trap_freq_mhz: tuple[float, float, float] = (0.60, 2.164, 0.144),
    seed: int | None = None,
    convergence_threshold: float = 0.1,
    iterate_steps: int = 100,
) -> np.ndarray:
    """Generate and z-sort equilibrium positions for the 2D crystal."""

    if seed is not None:
        np.random.seed(seed)
    potential = [[], *(2 * np.pi * np.array(trap_freq_mhz)) ** 2]
    r = equilibrium_positions(n_ions, potential, mass_amu=171, charge_e=1, method="cooling")
    while True:
        r2 = equilibrium_positions(
            n_ions,
            potential,
            mass_amu=171,
            charge_e=1,
            method="cooling",
            r0=r,
        )
        shift = float(np.max(np.abs(r - r2)))
        r = r2
        print(f"position_shift {shift:.12g}")
        if shift < convergence_threshold:
            break
    r = equilibrium_positions(
        n_ions,
        potential,
        mass_amu=171,
        charge_e=1,
        method="iterate",
        args={"step": iterate_steps},
        r0=r,
    )
    return np.asarray(r)[np.argsort(np.asarray(r)[:, 2]), :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N", type=int, help="number of ions")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--j0", type=float, default=1.0)
    parser.add_argument("--trap-freq-mhz", type=float, nargs=3, default=(0.60, 2.164, 0.144))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--positions-mat", type=Path, help="reuse positions from an existing MAT file")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.positions_mat:
        data = _loadmat(args.positions_mat, wanted={"r"})
        r = np.asarray(data["r"], dtype=float)
        if r.shape[0] != args.N:
            raise ValueError(f"{args.positions_mat} contains {r.shape[0]} ions, not N={args.N}")
    else:
        r = generate_positions(args.N, trap_freq_mhz=tuple(args.trap_freq_mhz), seed=args.seed)

    coupling = power_law_coupling_from_positions(r, j0=args.j0, alpha=args.alpha)
    output = args.output or Path(f"Ising2d_N{args.N}_alpha{args.alpha}.mat")
    savemat(output, {"N": args.N, "r": r, "J": coupling})
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
