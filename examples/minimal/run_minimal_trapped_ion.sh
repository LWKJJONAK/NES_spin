#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
SRC_DIR="$PACKAGE_ROOT/src/rbm_cpp"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"

cd "$SRC_DIR"

has_matio_lib() {
  compgen -G "$1/libmatio.*" > /dev/null
}

if [[ -z "${EIGEN_INC:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/include/eigen3" ]]; then
    EIGEN_INC="$CONDA_PREFIX/include/eigen3"
  elif [[ -d /usr/include/eigen3 ]]; then
    EIGEN_INC=/usr/include/eigen3
  elif [[ -d /usr/local/include/eigen3 ]]; then
    EIGEN_INC=/usr/local/include/eigen3
  elif [[ -d /opt/homebrew/include/eigen3 ]]; then
    EIGEN_INC=/opt/homebrew/include/eigen3
  else
    EIGEN_INC=/usr/include/eigen3
  fi
fi

choose_cxx() {
  for candidate in g++-14 g++-13 c++; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if [[ "$(uname -s)" == "Darwin" ]]; then
      compiler_file=$(file "$(command -v "$candidate")" 2>/dev/null || true)
      machine_arch=$(uname -m)
      if [[ "$machine_arch" == "arm64" && "$compiler_file" == *"x86_64"* && "$compiler_file" != *"arm64"* ]]; then
        continue
      fi
      if [[ "$machine_arch" == "x86_64" && "$compiler_file" == *"arm64"* && "$compiler_file" != *"x86_64"* ]]; then
        continue
      fi
    fi
    echo "$candidate"
    return
  done
  echo c++
}

if [[ -z "${CXX:-}" ]]; then
  CXX=$(choose_cxx)
fi

if [[ -z "${MATIO_INC:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/include/matio.h" ]]; then
    MATIO_INC="$CONDA_PREFIX/include"
  elif [[ -f /usr/local/include/matio.h ]]; then
    MATIO_INC=/usr/local/include
  elif [[ -f /opt/homebrew/include/matio.h ]]; then
    MATIO_INC=/opt/homebrew/include
  else
    MATIO_INC=/usr/include
  fi
fi

if [[ -z "${MATIO_LIB:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]] && has_matio_lib "$CONDA_PREFIX/lib"; then
    MATIO_LIB="$CONDA_PREFIX/lib"
  elif [[ -d /usr/local/lib ]] && has_matio_lib /usr/local/lib; then
    MATIO_LIB=/usr/local/lib
  elif [[ -d /opt/homebrew/lib ]] && has_matio_lib /opt/homebrew/lib; then
    MATIO_LIB=/opt/homebrew/lib
  elif [[ -d /usr/lib ]] && has_matio_lib /usr/lib; then
    MATIO_LIB=/usr/lib
  else
    MATIO_LIB=/usr/lib
  fi
fi

COUPLING_MAT="$BUILD_DIR/J_N20_mode7.mat"
COUPLING_HEADER="$BUILD_DIR/Jmat20.h"

if [[ ! -f "$COUPLING_HEADER" ]]; then
  if [[ -n "${PYTHON:-}" ]]; then
    python_cmd=$PYTHON
  elif command -v python3 >/dev/null 2>&1; then
    python_cmd=python3
  else
    python_cmd=python
  fi
  "$python_cmd" "$PACKAGE_ROOT/src/python_tools/generate_trapped_ion_coupling.py" 20 \
    --mode 7 --output "$COUPLING_MAT"
  "$python_cmd" "$SRC_DIR/load_jmat.py" 20 --input "$COUPLING_MAT" > "$COUPLING_HEADER"
fi

"$CXX" -DNOGPU -std=c++23 -Wall -Wno-uninitialized -Wno-maybe-uninitialized \
  -DNDEBUG -O3 -march=native -mtune=native -Wno-unused-variable \
  -I"${EIGEN_INC}" -I"${MATIO_INC}" -L"${MATIO_LIB}" -fopenmp \
  -DNUM_SPINS=20 -DNUM_HIDDEN_UNITS=80 -DNUM_EIGENSTATES=3 \
  -DMATFREE=1 -DGITHASH="\"nes-spin-code\"" -DJMAT_INC="\"$COUPLING_HEADER\"" \
  rbm.cc -lmatio -o "$BUILD_DIR/rbm-matfree-N20-M80-K3-opt"

cd "$SCRIPT_DIR"
"$BUILD_DIR/rbm-matfree-N20-M80-K3-opt" \
  --model TrappedIon \
  --h 2 \
  --niters 3 \
  --mc-steps-percpu 32 \
  --ncpus 1 \
  --burn-in-steps 16 \
  --lrate-init 0.05 \
  --lrate-decay 0.99 \
  --lrate-cutoff 0.01 \
  --reglam-init 50 \
  --reglam-decay 0.9 \
  --reglam-cutoff 0.001 \
  --minres-rtol 1e-10 \
  --minres-maxit 1000 \
  --seed 12345 \
  --correlation-data minimal_corr.mat

test -f minimal_corr.mat
echo "Wrote minimal_corr.mat"
