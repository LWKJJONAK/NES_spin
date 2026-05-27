# Reproduction Workflow

This document describes the code workflow for building and running the NES-Spin
implementation. The repository contains source code, configuration notes,
coupling-generation utilities, and small examples.

## 1. Environment

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate nes-spin-repro
```

For pip-based Python environments:

```bash
python3 -m pip install -r requirements.txt
```

Commands below use `python3` from the active environment. If the environment
exposes Python as `python`, either command name can be used; the C++ build also
accepts `PYTHON=python ./build.sh opt`.

The C++ executable requires:

- C++23 compiler
- OpenMP
- Eigen3 headers
- MATIO library
- Python with NumPy and SciPy

The build scripts search the active conda environment, Homebrew prefixes, and
standard Linux include/library paths for Eigen and MATIO.

If headers/libraries are not in standard locations, set:

```bash
export EIGEN_INC=/path/to/eigen3
export MATIO_INC=/path/to/matio/include
export MATIO_LIB=/path/to/matio/lib
```

## 2. Build Targets

Core implementation:

- `src/rbm_cpp/rbm.cc`: natural-excited-states training and correlation output.
- `src/rbm_cpp/minresqlp.h`: MINRES-QLP solver.
- `src/rbm_cpp/build.sh`: compile-time target builder.
- `src/rbm_cpp/load_jmat.py`: MAT-to-C++ coupling-header converter.

The binary size is compile-time fixed by:

- `-DNUM_SPINS=N`
- `-DNUM_HIDDEN_UNITS=M`
- `-DNUM_EIGENSTATES=K`
- `-DJMAT_INC="JmatN.h"` for trapped-ion runs

For optimization runs, use this sequence:

1. Decide the compile-time system size and RBM dimensions: `N`, `M`, and `K`.
2. Add or edit the corresponding `"N M K"` entry in the `PARAMS` array in
   `src/rbm_cpp/build.sh`.
3. Compile the optimized binary from `src/rbm_cpp/`:

```bash
cd src/rbm_cpp
./build.sh opt
```

For `N > 16`, the build writes:

```text
rbm-matfree-N{N}-M{M}-K{K}-opt
```

For `N <= 16`, the build writes:

```text
rbm-N{N}-M{M}-K{K}-opt
```

If `Jmat{N}.h` is absent, `build.sh` generates a mode-7 trapped-ion coupling
MAT file and converts it to `Jmat{N}.h`. To use a custom coupling, create
`Jmat{N}.h` before running the build.

## 3. Coupling Matrix Workflows

External pairwise Ising couplings enter the C++ code through a generated header
selected by `-DJMAT_INC`. The header must define:

```cpp
RealScalar Jmat[N][N]
```

The runtime model option that uses this matrix is `--model trappedion`.

For coupling to a selected transverse phonon mode:

```bash
python3 src/python_tools/generate_trapped_ion_coupling.py 20 \
  --mode 7 --output src/rbm_cpp/J_N20_mode7.mat

cd src/rbm_cpp
python3 load_jmat.py 20 --input J_N20_mode7.mat > Jmat20.h
```

For a two-dimensional power-law coupling:

```bash
python3 src/python_tools/Ising2d_power_law.py 20 \
  --alpha 1.0 --output src/rbm_cpp/power_law_N20_alpha1.0.mat

cd src/rbm_cpp
python3 load_jmat.py 20 --input power_law_N20_alpha1.0.mat > Jmat20.h
```

After the header is in place, add the matching `"N M K"` entry to `PARAMS`,
compile with `./build.sh opt`, and run the generated executable with
`--model trappedion`. The separate runtime option `--model longrange` is the
built-in one-dimensional long-range model in `rbm.cc`; it does not read a
MAT/header coupling matrix.

## 4. Runtime Parameters

Important command-line options:

| Option | Meaning | Code default |
|---|---|---|
| `--model` | Hamiltonian model: `ising`, `periodicising`, `hs`, `xxz`, `longrange`, `trappedion` | required |
| `-h`, `--h` | external field | required by parser |
| `-n`, `--niters` | stochastic-reconfiguration iterations | required |
| `-m`, `--mc-steps-percpu` | Monte Carlo samples per CPU worker | required |
| `-c`, `--ncpus` | CPU workers / OpenMP threads | required |
| `--burn-in-steps` | burn-in sweeps | `64` |
| `--tol` | stopping tolerance for per-spin energies | `1e-6` |
| `--lrate-init` | initial learning rate | `0.1` |
| `--lrate-decay` | learning-rate decay | `0.99` |
| `--lrate-cutoff` | learning-rate floor | `0.01` |
| `--reglam-init` | initial SR regularization | `0.01` |
| `--reglam-decay` | regularization decay | `0.9` |
| `--reglam-cutoff` | regularization floor | `0.001` |
| `--minres-rtol` | MINRES-QLP tolerance | `1e-10` |
| `--minres-maxit` | MINRES-QLP max iterations | unlimited unless specified |
| `--seed` | random seed | random-device seed if absent |
| `--load-rbm` | load a previous RBM checkpoint | empty |
| `--dump-rbm` | dump final RBM checkpoint with this prefix | empty |
| `--correlation-data` | output MAT file for correlation/coor data | empty; no correlation file written |

Method summaries are in:

- `configs/rbm_architecture.yaml`
- `configs/minres_qlp.yaml`
- `configs/curriculum_schedule.yaml`
- `docs/METHOD_TO_FILE_MAP.md`

## 5. Saving And Continuing RBM Training

Use `--dump-rbm PREFIX` to save the final RBM parameters from a run. The C++
code appends the compile-time system size to the prefix, so:

```bash
--dump-rbm ion_h1
```

with an `N=20`, `M=80`, `K=2` binary writes:

```text
ion_h1N20M80K2.mat
```

The checkpoint contains MATLAB variables `a`, `b`, and `w`. To continue
training from that saved RBM, start another run with `--load-rbm FILE` and,
optionally, a new `--dump-rbm PREFIX`:

```bash
./rbm-matfree-N20-M80-K2-opt \
  -c 8 --model trappedion -h 1 -n 100 -m 400 \
  --load-rbm ion_h1N20M80K2.mat \
  --dump-rbm ion_h1_continue
```

The loading binary must have the same compile-time `N`, `M`, and `K` as the
checkpoint.

## 6. Minimal Example

The short trapped-ion example is:

```bash
cd examples/minimal
./run_minimal_trapped_ion.sh
```

It generates an `N=20` mode-7 coupling inside `examples/minimal/build/`,
compiles an `N=20`, `M=80`, `K=3` binary, runs three iterations, and writes
`minimal_corr.mat`.

## 7. Verification Commands

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src/python_tools/tests/test_coupling_generation.py
bash -n src/rbm_cpp/build.sh examples/minimal/run_minimal_trapped_ion.sh
```

The first command checks the Python coupling utilities. The second checks the
shell entry points. The minimal example exercises the C++ compile/run path.
