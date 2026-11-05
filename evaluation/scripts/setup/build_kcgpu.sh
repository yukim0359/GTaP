#!/bin/bash
# Build the GH200-ported KCGPU fork used by GTaP k-clique evaluation.
#
# Usage:
#   ./evaluation/scripts/setup/build_kcgpu.sh
#
# Env overrides:
#   KCGPU_DIR       Checkout / build tree (default: <gtap>/../KCGPU)
#   KCGPU_REPO      Git URL if cloning (default: https://github.com/yukim0359/KCGPU.git)
#   KCGPU_REF       Branch / tag / commit to checkout when cloning or updating
#                   (default: main)
#   KCGPU_CUDA_ARCH NVCC arch number (default: 90)
#   KCGPU_PROFILE   0 = e2e binary (default), 1 = SM/warp profile binary
#   KCGPU_UPDATE     Set to 1 to git fetch + checkout KCGPU_REF before build
#
# Always runs make -B (no "already built" skip).
#
# Result:
#   $KCGPU_DIR/build/exe/src/main.cu.exe
#
# Graphs (SNAP MTX under $KCGPU_DIR/graphs/) are not fetched here; point
# GRAPH_DIR at an existing tree (e.g. KCGPU_before/graphs) when running.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
GTAP_ROOT=$(cd "$EVAL_DIR/.." && pwd)
WORK_ROOT=$(cd "$GTAP_ROOT/.." && pwd)

KCGPU_DIR="${KCGPU_DIR:-$WORK_ROOT/KCGPU}"
KCGPU_REPO="${KCGPU_REPO:-https://github.com/yukim0359/KCGPU.git}"
KCGPU_REF="${KCGPU_REF:-main}"
KCGPU_CUDA_ARCH="${KCGPU_CUDA_ARCH:-90}"
KCGPU_PROFILE="${KCGPU_PROFILE:-0}"
KCGPU_UPDATE="${KCGPU_UPDATE:-0}"

EXE="$KCGPU_DIR/build/exe/src/main.cu.exe"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "Error: nvcc not found. Load a CUDA toolkit module (or set PATH) first." >&2
    exit 1
fi

if [[ ! -d "$KCGPU_DIR/.git" ]]; then
    if [[ -e "$KCGPU_DIR" ]]; then
        echo "Error: $KCGPU_DIR exists but is not a git checkout." >&2
        echo "Remove it or set KCGPU_DIR to an empty/missing path to clone." >&2
        exit 1
    fi
    echo "Cloning $KCGPU_REPO ($KCGPU_REF) into $KCGPU_DIR ..."
    git clone --branch "$KCGPU_REF" --single-branch "$KCGPU_REPO" "$KCGPU_DIR" ||
        git clone "$KCGPU_REPO" "$KCGPU_DIR"
    if [[ "$(git -C "$KCGPU_DIR" rev-parse --abbrev-ref HEAD)" != "$KCGPU_REF" ]]; then
        git -C "$KCGPU_DIR" checkout "$KCGPU_REF"
    fi
elif [[ "$KCGPU_UPDATE" == "1" ]]; then
    echo "Updating $KCGPU_DIR to $KCGPU_REF ..."
    git -C "$KCGPU_DIR" fetch --tags origin
    git -C "$KCGPU_DIR" checkout "$KCGPU_REF"
    if git -C "$KCGPU_DIR" rev-parse --verify -q "origin/$KCGPU_REF" >/dev/null; then
        git -C "$KCGPU_DIR" merge --ff-only "origin/$KCGPU_REF"
    fi
fi

if [[ ! -f "$KCGPU_DIR/Makefile" ]]; then
    echo "Error: no Makefile in $KCGPU_DIR" >&2
    exit 1
fi

echo "Building KCGPU in $KCGPU_DIR (CUDA_ARCH=$KCGPU_CUDA_ARCH KCGPU_PROFILE=$KCGPU_PROFILE)"
make -C "$KCGPU_DIR" CUDA_ARCH="$KCGPU_CUDA_ARCH" KCGPU_PROFILE="$KCGPU_PROFILE" -B

if [[ ! -x "$EXE" ]]; then
    echo "Error: expected binary missing after build: $EXE" >&2
    exit 1
fi

echo "Done: $EXE"
if [[ ! -d "$KCGPU_DIR/graphs" ]]; then
    echo "Note: $KCGPU_DIR/graphs not found. Evaluation scripts need SNAP MTX via GRAPH_DIR."
fi
