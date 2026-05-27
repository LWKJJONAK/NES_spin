# Package Manifest

## Core Implementation

- `src/rbm_cpp/rbm.cc`: natural-excited-states variational Monte Carlo implementation.
- `src/rbm_cpp/minresqlp.h`: MINRES-QLP Krylov solver.
- `src/rbm_cpp/build.sh`: build script for compile-time `N`, `M`, and `K` targets.
- `src/rbm_cpp/load_jmat.py`: helper for converting MAT files containing `J` into C++ include files.
- `src/rbm_cpp/test-minresqlp.cc`: minimal MINRES-QLP smoke test.
- `src/rbm_cpp/magic_enum/`, `src/rbm_cpp/argparse.h`, `src/rbm_cpp/pcg_random.h`,
  `src/rbm_cpp/pcg_extras.h`, `src/rbm_cpp/except.h`: header dependencies.

The build and example workflows write generated coupling files, including
`J_N*_mode*.mat`, `power_law_*.mat`, and `Jmat*.h`, in their working
directories. These generated files are covered by `.gitignore`.

## Python Tools

- `src/python_tools/generate_trapped_ion_coupling.py`: transverse phonon-mode coupling generator.
- `src/python_tools/Ising2d_power_law.py`: two-dimensional power-law coupling generator.
- `src/python_tools/equilibrium_positions.py`: trapped-ion equilibrium positions and normal modes.
- `src/python_tools/ising_coupling_coefficients.py`: Ising coupling coefficients from transverse modes.
- `src/python_tools/load_jmat.py`: MAT-to-C++ coupling-header converter.
- `src/python_tools/ED_energy.py`, `src/python_tools/ED_periodic_ising.py`,
  `src/python_tools/ED_finite_size_scaling.py`, `src/python_tools/ED_ramp_dynamics.py`:
  exact-diagonalization utilities.
- `src/python_tools/tests/test_coupling_generation.py`: Python coupling tests.

## Configuration And Methods

- `configs/minres_qlp.yaml`
- `configs/curriculum_schedule.yaml`
- `configs/rbm_architecture.yaml`
- `docs/METHOD_TO_FILE_MAP.md`
- `docs/REPRODUCTION_WORKFLOW.md`

## Examples

- `examples/minimal/run_minimal_trapped_ion.sh`
- `examples/minimal/config_minimal_trapped_ion.yaml`
- `examples/minimal/README.md`

The minimal example generates its coupling files in `examples/minimal/build/`
before compiling.

## Package Metadata

- `README.md`
- `.gitignore`
- `LICENSE`
- `CITATION.cff`
- `requirements.txt`
- `environment.yml`
