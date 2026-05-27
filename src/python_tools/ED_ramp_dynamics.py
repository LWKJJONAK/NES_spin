"""Exact time evolution for a transverse-field Ising ramp.

This script computes finite-size trapped-ion ramp dynamics: starting from a
large transverse field, evolve under

    H(t) = sum_ij J_ij Z_i Z_j - h(t) sum_i X_i,

with the Guo et al. exponential ramp shape h(t) = h0 exp(-t / tau), then
project the final state onto eigenstates of the final static Hamiltonian.
The default h0 convention uses a Kac-like scale J0 from the input J matrix,
so the command line can mirror the experimental statement B0 > 50 J0 while
remaining dimensionless for the matrices in this repository.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import LinearOperator, eigsh, expm_multiply, lobpcg

try:
    from scipy.sparse.linalg._eigen import arpack as scipy_arpack

    # The local SciPy/NumPy combination can expose arpack_int as a dtype object
    # whose repr recurses inside NumPy. ARPACK accepts the scalar type directly.
    scipy_arpack.arpack_int = np.int32
except Exception:
    pass


SCRIPT_DIR = Path(__file__).resolve().parent
DENSE_LIMIT = 4096


def align8(offset: int) -> int:
    return (offset + 7) & ~7


def read_mat_tag(data: bytes, offset: int):
    raw = struct.unpack_from("<I", data, offset)[0]
    if raw >> 16:
        dtype = raw & 0xFFFF
        nbytes = raw >> 16
        return dtype, nbytes, offset + 4, True
    dtype, nbytes = struct.unpack_from("<II", data, offset)
    return dtype, nbytes, offset + 8, False


def read_v5_real_double_matrix(path, variable="J"):
    """Small MATLAB-v5 reader for simple real double matrices.

    This reader handles the simple uncompressed MAT-v5 files generated for the
    J matrices when scipy.io.loadmat cannot parse a particular file.
    """
    data = Path(path).read_bytes()
    if len(data) < 136 or data[124:128] not in (b"\x00\x01IM", b"IM\x00\x01"):
        raise ValueError(f"{path} does not look like a MATLAB v5 file")

    offset = 128
    mi_matrix, matrix_nbytes, offset, small = read_mat_tag(data, offset)
    if small or mi_matrix != 14:
        raise ValueError(f"{path} does not contain an uncompressed MATLAB matrix")
    matrix_end = offset + matrix_nbytes

    dtype, nbytes, offset, small = read_mat_tag(data, offset)
    if dtype != 6 or nbytes != 8:
        raise ValueError(f"{path} has an unsupported array-flags block")
    offset = offset + (4 if small else align8(nbytes) - nbytes + nbytes)

    dtype, nbytes, offset, small = read_mat_tag(data, offset)
    if dtype != 5 or small:
        raise ValueError(f"{path} has unsupported dimensions metadata")
    dims = struct.unpack_from("<" + "i" * (nbytes // 4), data, offset)
    offset = align8(offset + nbytes)
    if len(dims) != 2:
        raise ValueError(f"{path} contains a {len(dims)}D array, expected 2D")

    dtype, nbytes, offset, small = read_mat_tag(data, offset)
    if dtype != 1:
        raise ValueError(f"{path} has an unsupported variable-name block")
    name = data[offset : offset + nbytes].decode("latin1")
    if small:
        offset += 4
    else:
        offset = align8(offset + nbytes)
    if variable is not None and name != variable:
        raise KeyError(f"{path} contains variable {name!r}, not {variable!r}")

    dtype, nbytes, offset, small = read_mat_tag(data, offset)
    if dtype != 9 or small:
        raise ValueError(f"{path} data block is not a real double matrix")
    expected = int(np.prod(dims)) * 8
    if nbytes != expected:
        raise ValueError(f"{path} data block has {nbytes} bytes, expected {expected}")
    if offset + nbytes > matrix_end:
        raise ValueError(f"{path} matrix data exceed the declared MAT element")

    return np.frombuffer(data, dtype="<f8", count=expected // 8, offset=offset).reshape(dims, order="F").copy()


def load_j_matrix(path):
    path = Path(path)
    try:
        data = sio.loadmat(path)
        if "J" not in data:
            raise KeyError(f"{path} does not contain a variable named 'J'")
        J = np.asarray(data["J"], dtype=float)
    except Exception:
        J = np.asarray(read_v5_real_double_matrix(path, "J"), dtype=float)

    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"J must be a square matrix, got shape {J.shape}")
    return 0.5 * (J + J.T)


def spin_signs(states, site):
    bits = ((states >> np.uint64(site)) & np.uint64(1)).astype(np.int8)
    return 1 - 2 * bits


def ising_diagonal(J):
    n = J.shape[0]
    dim = 1 << n
    states = np.arange(dim, dtype=np.uint64)
    spins = [spin_signs(states, i).astype(float) for i in range(n)]
    diag = np.full(dim, np.trace(J), dtype=float)

    for i in range(1, n):
        zi = spins[i]
        for j in range(i):
            coeff = J[i, j] + J[j, i]
            if coeff != 0:
                diag += coeff * zi * spins[j]
    return diag


def flip_indices(n):
    dim = 1 << n
    states = np.arange(dim, dtype=np.uint64)
    return [states ^ np.uint64(1 << i) for i in range(n)]


def hamiltonian_operator(diag, h, flips, dtype=np.complex128):
    dim = diag.size
    dtype = np.dtype(dtype)

    def matvec(v):
        v = np.asarray(v).reshape(-1)
        out = diag * v
        if h != 0:
            for flipped in flips:
                out = out - h * v[flipped]
        return out

    return LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec, dtype=dtype)


def dense_hamiltonian(diag, h, flips):
    dim = diag.size
    eye = np.eye(dim)
    op = hamiltonian_operator(diag, h, flips, dtype=np.float64)
    return np.column_stack([op.matvec(eye[:, i]) for i in range(dim)])


def lowest_eigensystem(diag, h, flips, k):
    dim = diag.size
    if k < 1:
        raise ValueError("k must be positive")
    if k > dim:
        raise ValueError(f"k={k} exceeds Hilbert dimension {dim}")

    if h == 0:
        order = np.argsort(diag, kind="stable")[:k]
        vecs = np.zeros((dim, k), dtype=float)
        vecs[order, np.arange(k)] = 1.0
        return diag[order], vecs

    if dim <= DENSE_LIMIT or k >= dim:
        vals, vecs = np.linalg.eigh(dense_hamiltonian(diag, h, flips))
        return vals[:k], vecs[:, :k]

    op = hamiltonian_operator(diag, h, flips, dtype=np.float64)
    try:
        vals, vecs = eigsh(op, k=k, which="SA")
    except Exception as exc:
        print(f"eigsh failed ({exc}); falling back to lobpcg.", flush=True)
        rng = np.random.default_rng(1234)
        X = rng.normal(size=(dim, k))
        vals, vecs = lobpcg(op, X, largest=False, tol=1e-8, maxiter=300)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def product_x_state(n):
    dim = 1 << n
    return np.full(dim, 1 / np.sqrt(dim), dtype=np.complex128)


def exponential_ramp(t, h0, tau):
    return h0 * np.exp(-t / tau)


def evolve_exponential_ramp(diag, flips, psi0, h0, tau, total_time, n_steps, progress=False):
    if tau <= 0:
        raise ValueError("tau must be positive")
    if total_time < 0:
        raise ValueError("total_time must be non-negative")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    psi = np.asarray(psi0, dtype=np.complex128).copy()
    if total_time == 0:
        return psi

    dt = total_time / n_steps
    trace_h = np.sum(diag)
    for step in range(n_steps):
        midpoint = (step + 0.5) * dt
        h = exponential_ramp(midpoint, h0, tau)
        H = hamiltonian_operator(diag, h, flips, dtype=np.complex128)

        def matvec(v, H=H):
            v = np.asarray(v).reshape(-1)
            return (-1j * dt) * H.matvec(v)

        def rmatvec(v, H=H):
            v = np.asarray(v).reshape(-1)
            return (1j * dt) * H.matvec(v)

        A = LinearOperator(H.shape, matvec=matvec, rmatvec=rmatvec, dtype=np.complex128)
        psi = expm_multiply(A, psi, traceA=(-1j * dt) * trace_h)
        norm = np.linalg.norm(psi)
        if norm == 0 or not np.isfinite(norm):
            raise FloatingPointError(f"state norm became invalid at step {step}")
        psi /= norm

        if progress and (step == 0 or step + 1 == n_steps or (step + 1) % max(1, n_steps // 10) == 0):
            print(f"step {step + 1}/{n_steps}: h={h:.8g}, norm={norm:.12g}", flush=True)
    return psi


def apply_global_x_rotation(psi, n, theta):
    """Apply exp(i theta sum_i X_i) to a full state vector."""
    out = np.asarray(psi, dtype=np.complex128).copy()
    c = np.cos(theta)
    s = 1j * np.sin(theta)
    dim = out.size

    for site in range(n):
        half_block = 1 << site
        block = half_block << 1
        for start in range(0, dim, block):
            lo = slice(start, start + half_block)
            hi = slice(start + half_block, start + block)
            a = out[lo].copy()
            b = out[hi].copy()
            out[lo] = c * a + s * b
            out[hi] = s * a + c * b
    return out


def apply_split_step(psi, diag, flips, h, dt):
    """Second-order step for H = diag - h sum_i X_i at fixed h."""
    del flips  # The split X rotation uses the bit layout directly.
    n = int(np.log2(diag.size))
    phase = np.exp(-0.5j * dt * diag)
    out = phase * psi
    out = apply_global_x_rotation(out, n, h * dt)
    out = phase * out
    return out / np.linalg.norm(out)


def evolve_exponential_ramp_split(diag, flips, psi0, h0, tau, total_time, n_steps, progress=False):
    """Piecewise-midpoint Strang splitting for the exponential ramp."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if total_time < 0:
        raise ValueError("total_time must be non-negative")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    psi = np.asarray(psi0, dtype=np.complex128).copy()
    if total_time == 0:
        return psi

    dt = total_time / n_steps
    for step in range(n_steps):
        midpoint = (step + 0.5) * dt
        h = exponential_ramp(midpoint, h0, tau)
        psi = apply_split_step(psi, diag, flips, h, dt)

        if progress and (step == 0 or step + 1 == n_steps or (step + 1) % max(1, n_steps // 10) == 0):
            print(f"step {step + 1}/{n_steps}: h={h:.8g}, norm={np.linalg.norm(psi):.12g}", flush=True)
    return psi


def kac_scale(J, kind="abs"):
    n = J.shape[0]
    offdiag = J[~np.eye(n, dtype=bool)]
    if kind == "abs":
        return np.sum(np.abs(offdiag)) / n
    if kind == "signed":
        return np.sum(offdiag) / n
    if kind == "rms":
        return np.sqrt(np.sum(offdiag**2) / n)
    raise ValueError(f"unsupported J0 kind: {kind}")


def run_ramp(
    J,
    h0,
    tau,
    total_time,
    n_steps,
    num_eigs,
    initial="product-x",
    project_h=None,
    evolution_method="split",
    progress=False,
):
    n = J.shape[0]
    diag = ising_diagonal(J)
    flips = flip_indices(n)
    final_h = exponential_ramp(total_time, h0, tau)
    projection_h = final_h if project_h is None else project_h

    if initial == "product-x":
        psi0 = product_x_state(n)
    elif initial == "exact-ground":
        _, vecs0 = lowest_eigensystem(diag, h0, flips, 1)
        psi0 = vecs0[:, 0].astype(np.complex128)
    else:
        raise ValueError("initial must be 'product-x' or 'exact-ground'")

    if evolution_method == "split":
        psi = evolve_exponential_ramp_split(diag, flips, psi0, h0, tau, total_time, n_steps, progress=progress)
    elif evolution_method == "krylov":
        psi = evolve_exponential_ramp(diag, flips, psi0, h0, tau, total_time, n_steps, progress=progress)
    else:
        raise ValueError("evolution_method must be 'split' or 'krylov'")
    energies, eigvecs = lowest_eigensystem(diag, projection_h, flips, num_eigs)
    amplitudes = eigvecs.conj().T @ psi
    populations = np.abs(amplitudes) ** 2
    cumulative = np.cumsum(populations)
    expected_energy = np.vdot(psi, hamiltonian_operator(diag, projection_h, flips).matvec(psi)).real

    return {
        "N": n,
        "h0": h0,
        "tau": tau,
        "total_time": total_time,
        "final_h": final_h,
        "projection_h": projection_h,
        "state_norm": float(np.linalg.norm(psi)),
        "evolution_method": evolution_method,
        "energies": energies,
        "excitations": energies - energies[0],
        "amplitudes": amplitudes,
        "populations": populations,
        "cumulative_populations": cumulative,
        "outside_population": max(0.0, 1.0 - float(np.sum(populations))),
        "expected_energy": expected_energy,
        "final_state": psi,
    }


def write_population_csv(path, result):
    path = Path(path)
    table = np.column_stack(
        [
            np.arange(result["energies"].size),
            result["energies"],
            result["excitations"],
            result["populations"],
            result["cumulative_populations"],
        ]
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# N={result['N']}\n")
        f.write(f"# h0={result['h0']:.16g}\n")
        f.write(f"# tau={result['tau']:.16g}\n")
        f.write(f"# total_time={result['total_time']:.16g}\n")
        f.write(f"# final_h={result['final_h']:.16g}\n")
        f.write(f"# projection_h={result['projection_h']:.16g}\n")
        f.write(f"# state_norm={result['state_norm']:.16g}\n")
        f.write(f"# expected_energy={result['expected_energy']:.16g}\n")
        f.write(f"# outside_population={result['outside_population']:.16g}\n")
        f.write("state,energy,excitation,population,cumulative_population\n")
        np.savetxt(f, table, delimiter=",", fmt=["%d", "%.16e", "%.16e", "%.16e", "%.16e"])
    return path


def write_mat(path, result, save_state=False):
    data = {
        key: result[key]
        for key in [
            "N",
            "h0",
            "tau",
            "total_time",
            "final_h",
            "projection_h",
            "state_norm",
            "energies",
            "excitations",
            "amplitudes",
            "populations",
            "cumulative_populations",
            "outside_population",
            "expected_energy",
        ]
    }
    if save_state:
        data["final_state"] = result["final_state"]
    sio.savemat(path, data)
    return Path(path)


def parse_args():
    parser = argparse.ArgumentParser(description="ED real-time ramp dynamics and final eigenstate populations.")
    parser.add_argument("j_file", nargs="?", default=str(SCRIPT_DIR / "J_N20_mode7.mat"))
    parser.add_argument("--num-eigs", "-k", type=int, default=16, help="Number of final eigenstates to project onto.")
    parser.add_argument("--h0", type=float, default=None, help="Initial transverse field in the same units as J.")
    parser.add_argument("--h0-over-j0", type=float, default=50.0, help="Used when --h0 is omitted.")
    parser.add_argument("--j0-kind", choices=["abs", "signed", "rms"], default="abs")
    parser.add_argument("--tau", type=float, default=1.0, help="Dimensionless ramp time constant.")
    parser.add_argument(
        "--total-time",
        type=float,
        default=None,
        help="Total dimensionless ramp time. Default: 5.1 * tau, matching the Guo et al. T > 5 tau convention.",
    )
    parser.add_argument("--n-steps", type=int, default=200, help="Piecewise-constant midpoint evolution steps.")
    parser.add_argument("--project-h", type=float, default=None, help="Projection Hamiltonian field. Default: h(T).")
    parser.add_argument("--initial", choices=["product-x", "exact-ground"], default="product-x")
    parser.add_argument("--evolution-method", choices=["split", "krylov"], default="split")
    parser.add_argument("--output", default=None, help="CSV output path.")
    parser.add_argument("--mat-output", default=None, help="MAT output path.")
    parser.add_argument("--save-state", action="store_true", help="Also store the final state vector in the MAT output.")
    parser.add_argument("--max-states", type=int, default=1 << 22)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    J = load_j_matrix(args.j_file)
    n = J.shape[0]
    dim = 1 << n
    if dim > args.max_states:
        raise SystemExit(
            f"N={n} gives Hilbert dimension {dim}, above --max-states={args.max_states}. "
            "Use a smaller N or raise --max-states deliberately."
        )

    j0 = kac_scale(J, args.j0_kind)
    h0 = args.h0 if args.h0 is not None else args.h0_over_j0 * j0
    total_time = args.total_time if args.total_time is not None else 5.1 * args.tau

    print(f"J file: {args.j_file}")
    print(f"N={n}, dim={dim}, J0({args.j0_kind})={j0:.16g}")
    print(f"h0={h0:.16g}, tau={args.tau:.16g}, total_time={total_time:.16g}")
    print(f"initial={args.initial}, num_eigs={args.num_eigs}")

    result = run_ramp(
        J,
        h0=h0,
        tau=args.tau,
        total_time=total_time,
        n_steps=args.n_steps,
        num_eigs=args.num_eigs,
        initial=args.initial,
        project_h=args.project_h,
        evolution_method=args.evolution_method,
        progress=args.progress,
    )

    output = Path(args.output) if args.output else SCRIPT_DIR / "ED_ramp_populations.csv"
    mat_output = Path(args.mat_output) if args.mat_output else output.with_suffix(".mat")
    write_population_csv(output, result)
    write_mat(mat_output, result, save_state=args.save_state)

    print(f"final_h={result['final_h']:.16g}, projection_h={result['projection_h']:.16g}")
    print(f"state_norm={result['state_norm']:.16g}")
    print(f"expected final energy={result['expected_energy']:.16g}")
    print(f"population outside reported eigenstates={result['outside_population']:.16g}")
    print("Lowest reported populations:")
    for idx, (energy, population) in enumerate(zip(result["energies"], result["populations"])):
        print(f"  {idx:3d}  E={energy:.12g}  pop={population:.12g}")
    print(f"Saved CSV: {output}")
    print(f"Saved MAT: {mat_output}")


if __name__ == "__main__":
    main()
