#!/bin/bash
# Build OpenCilk from source into a local prefix under $OPENCILK_ROOT.
#
# Usage:
#   ./evaluation/scripts/build_opencilk.sh [jobs]
#
# Override the install location with the OPENCILK_ROOT environment variable
# (defaults to $HOME/opencilk).
#
# Result:
#   $OPENCILK_ROOT/install/bin/clang   (OpenCilk C compiler)
#   $OPENCILK_ROOT/install/bin/clang++ (OpenCilk C++ compiler)

set -euo pipefail

OPENCILK_ROOT="${OPENCILK_ROOT:-$HOME/opencilk}"
OPENCILK_TAG="${OPENCILK_TAG:-opencilk/v3.0}"
JOBS="${1:-32}"
SRC_DIR="$OPENCILK_ROOT/src"
BUILD_DIR="$OPENCILK_ROOT/build"
INSTALL_PREFIX="$OPENCILK_ROOT/install"
INFRA_DIR="$OPENCILK_ROOT/infrastructure"

# OpenCilk must be bootstrapped with GCC/Clang, not NVIDIA HPC SDK (nvc/nvc++).
# When the nvidia module is loaded, CMake may pick nvc and the build fails.
resolve_host_compiler() {
    local cc="${CC:-}"
    local cxx="${CXX:-}"

    if [[ -z "$cc" ]]; then
        if command -v gcc >/dev/null 2>&1; then
            cc="$(command -v gcc)"
        else
            echo "Error: gcc not found. Load gcc-toolset or set CC/CXX." >&2
            exit 1
        fi
    fi
    if [[ -z "$cxx" ]]; then
        if command -v g++ >/dev/null 2>&1; then
            cxx="$(command -v g++)"
        else
            cxx="${cc/cc/c++}"
        fi
    fi

    case "$cc" in
        *nvc|*nvc++)
            echo "Error: CC points to NVIDIA HPC compiler ($cc)." >&2
            echo "Use GCC for bootstrapping OpenCilk, e.g.:" >&2
            echo "  module unload nvidia nv-hpcx" >&2
            echo "  CC=gcc CXX=g++ $0 $JOBS" >&2
            exit 1
            ;;
    esac

    export CC="$cc"
    export CXX="$cxx"
}

build_dir_uses_nvc() {
    [[ -f "$BUILD_DIR/CMakeCache.txt" ]] \
        && grep -q '/compilers/bin/nvc' "$BUILD_DIR/CMakeCache.txt"
}

opencilk_source_ready() {
    [[ -d "$SRC_DIR/.git" ]] \
        && [[ -f "$SRC_DIR/llvm/CMakeLists.txt" ]] \
        && [[ -d "$SRC_DIR/cheetah/.git" ]] \
        && [[ -d "$SRC_DIR/cilktools/.git" ]]
}

resolve_host_compiler

if [[ -x "$INSTALL_PREFIX/bin/clang" ]]; then
    echo "OpenCilk already installed at $INSTALL_PREFIX"
    "$INSTALL_PREFIX/bin/clang" --version | head -3
    exit 0
fi

if type module >/dev/null 2>&1 && module list 2>&1 | grep -q nvidia; then
    echo "Note: nvidia module is loaded. Using CC=$CC CXX=$CXX for OpenCilk bootstrap."
fi

mkdir -p "$OPENCILK_ROOT"

if [[ ! -d "$INFRA_DIR/.git" ]]; then
    echo "Cloning OpenCilk infrastructure ($OPENCILK_TAG)..."
    git clone -b "$OPENCILK_TAG" --depth 1 https://github.com/OpenCilk/infrastructure "$INFRA_DIR"
fi

if ! opencilk_source_ready; then
    if [[ -e "$SRC_DIR" ]]; then
        echo "Removing incomplete OpenCilk source at $SRC_DIR..."
        rm -rf "$SRC_DIR"
    fi
    echo "Fetching OpenCilk source..."
    "$INFRA_DIR/tools/get" -t "$OPENCILK_TAG" "$SRC_DIR"
fi

if build_dir_uses_nvc; then
    echo "Removing stale OpenCilk build configured with nvc..."
    rm -rf "$BUILD_DIR"
fi

echo "Building OpenCilk with $JOBS jobs (CC=$CC, CXX=$CXX)..."
"$INFRA_DIR/tools/build" "$SRC_DIR" "$BUILD_DIR" "$JOBS"

echo "Installing OpenCilk to $INSTALL_PREFIX..."
cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" -P "$BUILD_DIR/cmake_install.cmake"

echo "Done."
"$INSTALL_PREFIX/bin/clang" --version | head -5
