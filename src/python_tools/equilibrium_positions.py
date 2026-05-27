"""Equilibrium positions and collective modes for trapped-ion crystals.

Coordinates are in micrometers. Trap frequencies are supplied with the 2*pi
factor included and in MHz-compatible units, matching the coupling
generation scripts.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


Potential = tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]


def _decode_potential_single(v: float | Potential) -> Potential:
    if isinstance(v, (float, int, np.floating, np.integer)):
        value = float(v)
        return (
            lambda x: 0.5 * value * x**2,
            lambda x: value * x,
            lambda x: value * np.eye(x.size),
        )
    return v


def _decode_potential_all(v: Iterable[object]) -> Potential:
    values = list(v)
    if len(values) == 1:
        return _decode_potential_single(values[0])
    vx = _decode_potential_single(values[1])
    vy = _decode_potential_single(values[2])
    vz = _decode_potential_single(values[3])

    def fun(r: np.ndarray) -> np.ndarray:
        coords = np.reshape(r, (-1, 3))
        return vx[0](coords[:, 0]) + vy[0](coords[:, 1]) + vz[0](coords[:, 2])

    def jac(r: np.ndarray) -> np.ndarray:
        coords = np.reshape(r, (-1, 3))
        return np.stack((vx[1](coords[:, 0]), vy[1](coords[:, 1]), vz[1](coords[:, 2])), axis=-1)

    def hess(r: np.ndarray) -> np.ndarray:
        coords = np.reshape(r, (-1, 3))
        n_ions = coords.shape[0]
        matrix = np.zeros((n_ions, 3, n_ions, 3))
        matrix[:, 0, :, 0] = vx[2](coords[:, 0])
        matrix[:, 1, :, 1] = vy[2](coords[:, 1])
        matrix[:, 2, :, 2] = vz[2](coords[:, 2])
        return matrix.reshape((n_ions * 3, n_ions * 3))

    return fun, jac, hess


def _coulomb_value(r: np.ndarray, charge: np.ndarray) -> float:
    coords = np.reshape(r, (-1, 3))
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1)
    invdist = 1 / dist
    np.fill_diagonal(invdist, 0)
    charge_pair = charge.reshape((-1, 1)) * charge.reshape((1, -1))
    return float(0.5 * np.sum(invdist * charge_pair))


def _coulomb_grad(r: np.ndarray, charge: np.ndarray) -> np.ndarray:
    coords = np.reshape(r, (-1, 3))
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1)
    invdist = 1 / dist
    np.fill_diagonal(invdist, 0)
    charge_pair = charge.reshape((-1, 1)) * charge.reshape((1, -1))
    return -np.sum(diff * (invdist**3 * charge_pair)[:, :, np.newaxis], axis=1).ravel()


def _coulomb_hess(r: np.ndarray, charge: np.ndarray) -> np.ndarray:
    coords = np.reshape(r, (-1, 3))
    n_ions = coords.shape[0]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1)
    invdist = 1 / dist
    np.fill_diagonal(invdist, 0)
    charge_pair = charge.reshape((-1, 1)) * charge.reshape((1, -1))
    hess = -3 * diff[:, :, :, np.newaxis] * diff[:, :, np.newaxis, :] * (
        invdist**5 * charge_pair
    )[:, :, np.newaxis, np.newaxis]
    hess[:, :, range(3), range(3)] += (invdist**3 * charge_pair)[:, :, np.newaxis]
    hess[range(n_ions), range(n_ions), :, :] = -np.sum(hess, axis=1)
    return np.swapaxes(hess, 1, 2).reshape((n_ions * 3, n_ions * 3))


def equilibrium_positions(
    n_ions: int,
    potential: Iterable[object] | None = None,
    *,
    mass_amu: float = 171,
    charge_e: float = 1,
    r0: np.ndarray | None = None,
    method: str = "minimize",
    args: dict[str, float | int] | None = None,
    collective_mode: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ion equilibrium positions and optionally normal modes."""

    if potential is None:
        raise ValueError("A trap potential must be supplied.")
    if args is None:
        args = {}
    mass = np.full(n_ions, float(mass_amu))
    charge = np.full(n_ions, float(charge_e))
    coef = 8.9876e9 * 1.6022e-19**2 / (mass[0] * 1.6605e-27 * 1e-6)
    trap_fun, trap_jac, trap_hess = _decode_potential_all(potential)

    def fun(flat_r: np.ndarray) -> float:
        trap = np.sum(trap_fun(flat_r) * (charge / charge[0]))
        return float(trap + _coulomb_value(flat_r, charge) * coef)

    def jac(flat_r: np.ndarray) -> np.ndarray:
        trap = trap_jac(flat_r).reshape((-1, 3)) * (charge / charge[0]).reshape((-1, 1))
        return trap.ravel() + _coulomb_grad(flat_r, charge) * coef

    def hess(flat_r: np.ndarray) -> np.ndarray:
        trap = trap_hess(flat_r).reshape((n_ions, 3, n_ions, 3))
        trap[range(n_ions), :, range(n_ions), :] *= (charge / charge[0]).reshape((-1, 1, 1))
        return trap.reshape((n_ions * 3, n_ions * 3)) + _coulomb_hess(flat_r, charge) * coef

    if r0 is None:
        r0 = np.random.rand(n_ions * 3)
    else:
        r0 = np.asarray(r0, dtype=float).reshape(n_ions * 3)

    if method == "minimize":
        from scipy.optimize import minimize

        options = {key: args[key] for key in args.keys() & {"xtol", "maxiter"}}
        sol = minimize(fun, r0, method="Newton-CG", jac=jac, hess=hess, options=options)
        r = sol.x.reshape((n_ions, 3))
    elif method == "cooling":
        from scipy.integrate import solve_ivp

        gamma = float(args.get("gamma", 0.1))
        total_time = float(args.get("T", 100))
        v0 = np.asarray(args.get("v0", np.zeros((n_ions, 3))), dtype=float)

        def rhs(_t: float, x: np.ndarray) -> np.ndarray:
            return np.concatenate(
                (
                    x[3 * n_ions :],
                    -jac(x[: 3 * n_ions]) * mass[0] / np.repeat(mass, 3) - gamma * x[3 * n_ions :],
                )
            )

        sol = solve_ivp(
            rhs,
            (0, total_time),
            np.concatenate((r0.ravel(), v0.ravel())),
            method="RK45",
            t_eval=np.array([0, total_time]),
        )
        r = sol.y[: 3 * n_ions, -1].reshape((n_ions, 3))
    elif method == "iterate":
        alpha = float(args.get("alpha", 0.1))
        steps = int(args.get("step", 100))
        flat_r = r0.copy()
        for _ in range(steps):
            flat_r = flat_r - alpha * np.linalg.solve(hess(flat_r), jac(flat_r))
        r = flat_r.reshape((n_ions, 3))
    else:
        raise ValueError(f"Unsupported method: {method}")

    if not collective_mode:
        return r
    from scipy.linalg import eigh

    hessian = hess(r.ravel())
    mass_matrix = np.diag(np.repeat(mass, 3)) / mass[0]
    omega2, modes = eigh(hessian, mass_matrix)
    omega_k = np.sqrt(omega2)
    modes = modes / np.sqrt(np.sum(modes**2, axis=0, keepdims=True))
    return r, omega_k, modes
