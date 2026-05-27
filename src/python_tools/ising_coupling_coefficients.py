"""Spin-spin Ising couplings from transverse phonon modes."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array


def ising_coupling(
    omega_k: np.ndarray,
    b_jk: np.ndarray,
    mu: float,
    omega_j: float | complex | np.ndarray,
    *,
    laser_wavelength_nm: float = 411,
    ion_mass_amu: float = 171,
    keepdiag: bool = False,
    jac: bool = True,
) -> np.ndarray | tuple[np.ndarray, coo_array, np.ndarray]:
    """Return the Ising coupling matrix in kHz with the 2*pi factor kept."""

    omega_k = np.asarray(omega_k, dtype=float).ravel()
    b_jk = np.asarray(b_jk, dtype=float)
    eta_k = (
        2
        * 2
        * np.pi
        / laser_wavelength_nm
        * 1e9
        * np.sqrt(6.62607e-34 / 2 / np.pi / 2 / ion_mass_amu / 1.6605e-27 / omega_k / 1e6)
    )
    mode = np.sum(
        b_jk[:, np.newaxis, :]
        * b_jk[np.newaxis, :, :]
        * (eta_k**2 * omega_k / (mu**2 - omega_k**2)).reshape((1, 1, -1)),
        axis=2,
    )
    if not keepdiag:
        np.fill_diagonal(mode, 0)

    omega_j = np.reshape([omega_j], (-1, 1))
    coupling = mode * np.real(omega_j * omega_j.T.conj()) * 1e3
    if not jac:
        return coupling

    indi = np.repeat(np.arange(mode.size), 2)
    indj = np.array([[i, j] for i in range(mode.shape[0]) for j in range(mode.shape[1])]).ravel()
    data = np.stack((mode * omega_j.T.conj(), mode * omega_j.conj()), axis=2).ravel() * 1e3
    jac_omega = coo_array((data, (indi, indj)), shape=(mode.size, mode.shape[0])).tocsr()
    jac_mu = np.sum(
        b_jk[:, np.newaxis, :]
        * b_jk[np.newaxis, :, :]
        * (eta_k**2 * omega_k / (mu**2 - omega_k**2) ** 2 * -2 * mu).reshape((1, 1, -1)),
        axis=2,
    )
    if not keepdiag:
        np.fill_diagonal(jac_mu, 0)
    jac_mu = (jac_mu * np.real(omega_j * omega_j.T.conj()) * 1e3).ravel()
    return coupling, jac_omega, jac_mu


# Compatibility alias for scripts that use the original capitalized name.
Ising_coupling = ising_coupling
