"""Convert MAT coupling data with variable `J` into a C++ `Jmat*.h` include."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_j_matrix(path: Path) -> np.ndarray:
    try:
        return np.asarray(loadmat(path)["J"], dtype=float)
    except Exception:
        python_tools = Path(__file__).resolve().parents[1] / "python_tools"
        sys.path.insert(0, str(python_tools))
        from mat5_corr_tools import read_mat5_arrays

        return np.asarray(read_mat5_arrays(path, wanted={"J"})["J"], dtype=float)


def format_jmat_header(j_matrix: np.ndarray) -> str:
    n_ions = j_matrix.shape[0]
    lines = [f"RealScalar Jmat[{n_ions}][{n_ions}] = {{"]
    for i in range(n_ions):
        row = ",".join(repr(float(j_matrix[i, j])) for j in range(n_ions))
        suffix = "," if i != n_ions - 1 else "};"
        lines.append("{" + row + "}" + suffix)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N", type=int)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()

    path = args.input or Path(f"J_N{args.N}_mode7.mat")
    j_matrix = load_j_matrix(path)
    if j_matrix.shape != (args.N, args.N):
        raise ValueError(f"{path} contains shape {j_matrix.shape}, expected {(args.N, args.N)}")
    print(format_jmat_header(j_matrix))


if __name__ == "__main__":
    main()
