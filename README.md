# NES-Spin Code

This repository provides a C++ and Python implementation of the
natural-excited-states (NES) method for long-range quantum spin systems and
trapped-ion Hamiltonians.

It includes the optimization executable, coupling-generation utilities,
configuration references, and runnable examples for building compile-time
targets, generating model couplings, running optimization jobs, and writing
checkpoints or correlation outputs.

## Repository Layout

- `src/rbm_cpp/`: C++23 optimization implementation, MINRES-QLP solver, helper
  headers, and build script.
- `src/python_tools/`: Python utilities for exact diagonalization, trapped-ion
  coupling generation, power-law coupling generation, and MAT-file conversion.
- `configs/`: concise descriptions of the RBM architecture, curriculum
  schedule, and MINRES-QLP options.
- `examples/minimal/`: short trapped-ion compile/run example.
- `docs/`: code structure and reproduction guide.
- `environment.yml`, `requirements.txt`: Python and library requirements.

## Environment

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate nes-spin-repro
```

For pip-based Python environments:

```bash
python3 -m pip install -r requirements.txt
```

The C++ code requires:

- a C++23 compiler with OpenMP support
- Eigen3 headers
- MATIO library
- Python 3 with NumPy and SciPy

The build scripts search the active conda environment, Homebrew prefixes, and
standard Linux include/library paths for Eigen and MATIO.

On many Linux systems:

```bash
sudo apt-get install g++ libeigen3-dev libmatio-dev python3 python3-scipy
```

On macOS/Homebrew systems:

```bash
brew install eigen matio gcc
```

Apple clang does not provide OpenMP by default. If using Homebrew GCC, build
with a command such as:

```bash
cd src/rbm_cpp
CXX=g++-14 PYTHON=python3 ./build.sh opt
```

If Eigen or MATIO is installed outside standard compiler search paths, set:

```bash
export EIGEN_INC=/path/to/eigen3
export MATIO_INC=/path/to/matio/include
export MATIO_LIB=/path/to/matio/lib
```

## Build And Run

The executable is specialized at compile time by `N`, `M`, and `K`.
Set the desired triples in the `PARAMS` array in `src/rbm_cpp/build.sh`.

```bash
PARAMS=( "20 80 2" )
```

Then compile:

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

If a `Jmat{N}.h` header is absent, `build.sh` generates a mode-7 trapped-ion
coupling MAT file and converts it to the C++ header. To use a custom coupling,
generate or provide a MAT file containing variable `J`, convert it to
`Jmat{N}.h`, and build after the header is in place.

Run the generated executable with command-line options for the calculation. For
example:

```bash
time ./rbm-matfree-N20-M80-K2-opt \
  -c 8 --model trappedion -h 1 -n 100 -m 400 --tol 1e-8
```

On Linux systems, `taskset` can be added for CPU binding:

```bash
time taskset -c 0-7 ./rbm-matfree-N20-M80-K2-opt \
  -c 8 --model trappedion -h 1 -n 100 -m 400 --tol 1e-8
```

## Minimal Example

The minimal trapped-ion example generates the required `N=20` mode-7 coupling
inside its local build directory, compiles a short-run binary, and runs three
optimization iterations:

```bash
cd examples/minimal
./run_minimal_trapped_ion.sh
```

The example prints the random seed and three energy estimates per iteration.
It also writes `minimal_corr.mat` as an example of the correlation-output path.

## Coupling Generation

External pairwise Ising couplings enter the C++ code through `Jmat{N}.h`, which
must define:

```cpp
RealScalar Jmat[N][N]
```

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

After conversion, compile the matching `N`, `M`, and `K` target and run with
`--model trappedion`. In this workflow, `--model trappedion` denotes the
pairwise Ising Hamiltonian using the compiled `Jmat` matrix. The separate
runtime option `--model longrange` is the built-in one-dimensional long-range
model in `rbm.cc`.

## Checkpoints

RBM checkpoints are written with:

```bash
--dump-rbm PREFIX
```

which produces `PREFIXN{N}M{M}K{K}.mat` containing `a`, `b`, and `w`. Continue
from a checkpoint with:

```bash
--load-rbm FILE
```

The loading binary must use the same compile-time `N`, `M`, and `K` as the
checkpoint.

## Tests

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src/python_tools/tests/test_coupling_generation.py
```

This test covers the coupling-generation utilities. The minimal example covers
the C++ compile/run path.

## Citation

@unpublished{Ma2025Solving,
  title = {Solving Excited States for Long-Range Interacting Trapped Ions with Neural Networks},
  author = {Ma, Yixuan and Liu, Chang and Li, Weikang and Zhang, Shun-Yao and Duan, L.-M. and Wu, Yukai and Deng, Dong-Ling},
  year = 2025,
  month = jun,
  eprint = {2506.08594},
  archiveprefix = {arXiv}
}
