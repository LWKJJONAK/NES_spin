# Minimal Trapped-Ion Example

This directory contains a short compile/run example for users with Eigen, MATIO,
OpenMP, and a C++23 compiler.

Run:

```bash
./run_minimal_trapped_ion.sh
```

The script:

1. Generates an `N=20` mode-7 trapped-ion coupling in `build/`.
2. Converts the generated MAT file to a C++ `Jmat20.h` header.
3. Compiles the C++ code for `N=20`, `M=80`, `K=3`.
4. Runs three trapped-ion optimization iterations.

Expected behavior:

- the executable prints the random seed;
- each iteration prints three energy estimates per spin;
- `minimal_corr.mat` is written at the end.

The corresponding parameter summary is in `config_minimal_trapped_ion.yaml`.
