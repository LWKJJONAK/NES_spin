"""Generate trapped-ion spin-spin couplings for a selected transverse mode.

The mode index follows the manuscript convention: mode 1 is the highest-frequency
transverse mode, so mode 7 is selected with `omega_k[-7]`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from equilibrium_positions import equilibrium_positions
from ising_coupling_coefficients import ising_coupling


def _loadmat(path: Path, wanted: set[str] | None = None) -> dict[str, np.ndarray]:
    try:
        return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except Exception:
        from mat5_corr_tools import read_mat5_arrays

        return read_mat5_arrays(path, wanted=wanted)


def compute_transverse_modes(
    n_ions: int,
    *,
    trap_freq_mhz: tuple[float, float, float] = (0.69, 2.140, 0.167),
    seed: int | None = None,
    cooling_repeats: int = 5,
    iterate_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute z-sorted positions and transverse normal modes."""

    if seed is not None:
        np.random.seed(seed)
    potential = [[], *(2 * np.pi * np.array(trap_freq_mhz)) ** 2]
    r = equilibrium_positions(n_ions, potential, method="cooling")
    for _ in range(cooling_repeats):
        r = equilibrium_positions(n_ions, potential, method="cooling", r0=r)
    r, omega_k, b_jk = equilibrium_positions(
        n_ions,
        potential,
        method="iterate",
        args={"step": iterate_steps},
        r0=r,
        collective_mode=True,
    )
    transverse_population = np.sum(b_jk[1::3, :] ** 2, axis=0)
    transverse = transverse_population > 0.99
    if int(np.sum(transverse)) != n_ions:
        raise RuntimeError(f"Expected {n_ions} transverse modes, found {int(np.sum(transverse))}")
    omega_k = omega_k[transverse]
    b_jk = b_jk[1::3, transverse]
    order = np.argsort(r[:, 2])
    return r[order, :], omega_k, b_jk[order, :]


def coupling_for_mode(
    omega_k: np.ndarray,
    b_jk: np.ndarray,
    *,
    mode: int = 7,
    detuning_mhz: float = 0.00075,
    rabi_mhz: float = 0.010,
) -> np.ndarray:
    """Return the coupling matrix for one transverse mode."""

    omega_k = np.asarray(omega_k, dtype=float).ravel()
    b_jk = np.asarray(b_jk, dtype=float)
    mu = omega_k[-mode] + 2 * np.pi * detuning_mhz
    omega_rabi = 2 * np.pi * rabi_mhz
    return -ising_coupling(omega_k, b_jk, mu, omega_rabi, jac=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N", type=int, help="number of ions")
    parser.add_argument("--mode", type=int, default=7, help="mode counted from highest frequency")
    parser.add_argument("--detuning-mhz", type=float, default=0.00075)
    parser.add_argument("--rabi-mhz", type=float, default=0.010)
    parser.add_argument("--trap-freq-mhz", type=float, nargs=3, default=(0.69, 2.140, 0.167))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--modes-mat", type=Path, help="reuse r, omega_k, b_jk from an existing MAT file")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.modes_mat:
        data = _loadmat(args.modes_mat, wanted={"r", "omega_k", "b_jk"})
        r = np.asarray(data["r"], dtype=float)
        omega_k = np.asarray(data["omega_k"], dtype=float).ravel()
        b_jk = np.asarray(data["b_jk"], dtype=float)
        if r.shape[0] != args.N:
            raise ValueError(f"{args.modes_mat} contains {r.shape[0]} ions, not N={args.N}")
    else:
        r, omega_k, b_jk = compute_transverse_modes(
            args.N,
            trap_freq_mhz=tuple(args.trap_freq_mhz),
            seed=args.seed,
        )

    coupling = coupling_for_mode(
        omega_k,
        b_jk,
        mode=args.mode,
        detuning_mhz=args.detuning_mhz,
        rabi_mhz=args.rabi_mhz,
    )
    output = args.output or Path(f"J_N{args.N}_mode{args.mode}.mat")
    savemat(output, {"J": coupling})
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
