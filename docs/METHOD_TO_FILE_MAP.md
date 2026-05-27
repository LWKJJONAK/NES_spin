# Method And Code Map

This map links the numerical methods to the files that implement them.

## Natural-Excited-States Optimization

- Main implementation: `src/rbm_cpp/rbm.cc`
- RBM architecture summary: `configs/rbm_architecture.yaml`
- Compile-time target selection: `src/rbm_cpp/build.sh`
- Runtime options: `docs/REPRODUCTION_WORKFLOW.md`

## Stochastic Reconfiguration

- MINRES-QLP solver: `src/rbm_cpp/minresqlp.h`
- Solver configuration summary: `configs/minres_qlp.yaml`
- Matrix-free covariance-vector products: `src/rbm_cpp/rbm.cc`

## Hamiltonian Models

- Transverse-field Ising: `src/rbm_cpp/rbm.cc`
- Periodic Ising: `src/rbm_cpp/rbm.cc`
- Haldane-Shastry: `src/rbm_cpp/rbm.cc`
- XXZ reference model: `src/rbm_cpp/rbm.cc`
- Built-in one-dimensional long-range model: `src/rbm_cpp/rbm.cc`
- External pairwise Ising coupling through `Jmat`: `src/rbm_cpp/rbm.cc`

## Coupling Generation

- Trapped-ion equilibrium positions and normal modes:
  `src/python_tools/equilibrium_positions.py`
- Transverse phonon-mode coupling:
  `src/python_tools/generate_trapped_ion_coupling.py`
- Two-dimensional power-law coupling:
  `src/python_tools/Ising2d_power_law.py`
- MAT-to-C++ header conversion:
  `src/rbm_cpp/load_jmat.py`

## Exact-Diagonalization Utilities

- Static energy scans: `src/python_tools/ED_energy.py`
- Periodic Ising utilities: `src/python_tools/ED_periodic_ising.py`
- Finite-size scaling helper: `src/python_tools/ED_finite_size_scaling.py`
- Ramp dynamics helper: `src/python_tools/ED_ramp_dynamics.py`

## Examples And Tests

- Minimal trapped-ion compile/run example:
  `examples/minimal/run_minimal_trapped_ion.sh`
- Minimal example parameter summary:
  `examples/minimal/config_minimal_trapped_ion.yaml`
- Python coupling tests:
  `src/python_tools/tests/test_coupling_generation.py`
