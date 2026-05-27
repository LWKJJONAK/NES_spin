#!/bin/bash
set -euo pipefail

CUDA_INC=${CUDA_INC:=/opt/cuda/include}

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

if [[ -z "${MATIO_INC:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/include/matio.h" ]]; then
	MATIO_INC="$CONDA_PREFIX/include"
    elif [[ -f /usr/include/matio.h ]]; then
	MATIO_INC=/usr/include
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
if [[ -z "${PYTHON:-}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
	PYTHON=python3
    else
	PYTHON=python
    fi
fi

OPT=0
NVGPU=0
AMDGPU=0
INTEL=0
buildty="debug"

BUILD_KIND=${1:-}

if [[ $BUILD_KIND == "opt" ]]; then
    OPT=1
    buildty="opt"
fi

if [[ $BUILD_KIND == "nvgpu" ]]; then
    NVGPU=1
    buildty="nvgpu"
fi

if [[ $BUILD_KIND == "amdgpu" ]]; then
    AMDGPU=1
    buildty="amdgpu"
fi

if [[ $BUILD_KIND == "intel" ]]; then
    INTEL=1
    buildty="intel"
fi

OUTDIR=$PWD
cd "$(dirname "$0")"
GITHASH=$(git show -s --format="%h (%ci)" HEAD 2>/dev/null || echo "nes-spin-code")
COMMONOPTS="-std=c++23 -Wall -Wno-uninitialized -I${EIGEN_INC} -I${MATIO_INC}"
LINKLIBS="-L${MATIO_LIB} -lmatio"
DBGOPTS="-DDBG=1 -DDEBUG -g -fsanitize=address"
NDBGOPTS="-DNDEBUG -O3 -march=native -mtune=native -Wno-unused-variable"
if (( $AMDGPU )); then
    COMMONOPTS+=" -Wno-unused-but-set-variable -Wno-unused-result"
else
    COMMONOPTS+=" -Wno-maybe-uninitialized"
fi
if (( $OPT )) || (( $INTEL )) || (( $NVGPU )) || (( $AMDGPU )); then
    COMMONOPTS+=" $NDBGOPTS"
else
    COMMONOPTS+=" $DBGOPTS"
fi
if (( $NVGPU )); then
    LINKLIBS+=" -lcublas -lcusolver -lcurand"
elif (( $AMDGPU )); then
    LINKLIBS+=" -lhipblas -lhipsolver -lhiprand"
fi
CXX_BIN=${CXX:-c++}
NVCC="nvcc -forward-unknown-to-host-compiler $COMMONOPTS -Wno-unknown-pragmas -diag-suppress 68 -march=native -fopenmp -arch sm_80"
HIPCC="hipcc $COMMONOPTS -D__HIPCC__ -march=native -fopenmp"
CPU_CXX="$CXX_BIN -DNOGPU $COMMONOPTS -march=native -fopenmp"
ICXX="icpx -DNOGPU $COMMONOPTS -xhost -fiopenmp -Wno-tautological-constant-compare -Wno-unused-but-set-variable"

if (( $NVGPU )); then
    COMPILE_CMD=$NVCC
elif (( $AMDGPU )); then
    COMPILE_CMD=$HIPCC
elif (( $INTEL )); then
    COMPILE_CMD=$ICXX
else
    COMPILE_CMD=$CPU_CXX
fi

TESTS="test-minresqlp"

for i in $TESTS; do
    $COMPILE_CMD -DGITHASH="\"$GITHASH\"" $i.cc  -o $OUTDIR/$i-$buildty
done

# Each entry is "N M K". Set the desired compile-time sizes here before
# building; the runtime command must use the matching binary.
PARAMS=( "20 80 2" )

TRAPPED_ION_MODE=${TRAPPED_ION_MODE:=7}

for params in "${PARAMS[@]}"; do
    N=$(echo $params | cut -f1 -d' ')
    M=$(echo $params | cut -f2 -d' ')
    K=$(echo $params | cut -f3 -d' ')
    PARAMDEF="-DNUM_SPINS=$N -DNUM_HIDDEN_UNITS=$M -DNUM_EIGENSTATES=$K"
    if [[ ! -f "Jmat${N}.h" ]]; then
	MATFILE="J_N${N}_mode${TRAPPED_ION_MODE}.mat"
	if [[ ! -f "$MATFILE" ]]; then
	    $PYTHON ../python_tools/generate_trapped_ion_coupling.py "$N" \
		    --mode "$TRAPPED_ION_MODE" --output "$MATFILE"
	fi
	$PYTHON load_jmat.py "$N" --input "$MATFILE" > "Jmat${N}.h"
    fi
    if (( $N <= 16 )); then
	$COMPILE_CMD $PARAMDEF -DGITHASH="\"$GITHASH\"" -DJMAT_INC="\"Jmat${N}.h\"" \
		     rbm.cc $LINKLIBS -o $OUTDIR/rbm-N$N-M$M-K$K-$buildty
    else
	$COMPILE_CMD $PARAMDEF -DMATFREE=1 -DGITHASH="\"$GITHASH\"" -DJMAT_INC="\"Jmat${N}.h\"" \
		     rbm.cc $LINKLIBS -o $OUTDIR/rbm-matfree-N$N-M$M-K$K-$buildty
    fi
done
