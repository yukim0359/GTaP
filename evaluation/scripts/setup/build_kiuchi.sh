#!/bin/bash
# Build the GH200-ported Kiuchi et al. fine-grained fork-join tree
# (DynaSOAr + FibOnlyOne / NQCutoff / CilkSort) used by GTaP microbenchmarks.
#
# Usage:
#   ./evaluation/scripts/setup/build_kiuchi.sh
#
# Env overrides:
#   DYNASOAR_ROOT   Checkout / build tree
#                   (default: <gtap>/../fine-grained-fork-join)
#   KIUCHI_REPO     Git URL if cloning
#                   (default: https://github.com/yukim0359/fine-grained-fork-join.git)
#   KIUCHI_REF      Branch / tag / commit when cloning or updating (default: master)
#   KIUCHI_TARGETS  Make targets (default: paper subset)
#   KIUCHI_UPDATE    Set to 1 to git fetch + checkout KIUCHI_REF before build
#
# Always runs make -B (no "already built" skip). Clone uses --recurse-submodules
# so yukim0359/dynasoar (GH200 allocator fixes) is fetched automatically.
#
# Result (defaults):
#   $DYNASOAR_ROOT/bin/FibOnlyOne
#   $DYNASOAR_ROOT/bin/NQCutoff
#   $DYNASOAR_ROOT/bin/CilkSort

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
GTAP_ROOT=$(cd "$EVAL_DIR/.." && pwd)
WORK_ROOT=$(cd "$GTAP_ROOT/.." && pwd)

DYNASOAR_ROOT="${DYNASOAR_ROOT:-$WORK_ROOT/fine-grained-fork-join}"
KIUCHI_REPO="${KIUCHI_REPO:-https://github.com/yukim0359/fine-grained-fork-join.git}"
KIUCHI_REF="${KIUCHI_REF:-master}"
KIUCHI_TARGETS="${KIUCHI_TARGETS:-bin/FibOnlyOne bin/NQCutoff bin/CilkSort}"
KIUCHI_UPDATE="${KIUCHI_UPDATE:-0}"

# shellcheck disable=SC2206
TARGETS=($KIUCHI_TARGETS)

if ! command -v nvcc >/dev/null 2>&1; then
    echo "Error: nvcc not found. Load a CUDA toolkit module (or set PATH) first." >&2
    exit 1
fi

if [[ ! -d "$DYNASOAR_ROOT/.git" ]]; then
    if [[ -e "$DYNASOAR_ROOT" ]]; then
        echo "Error: $DYNASOAR_ROOT exists but is not a git checkout." >&2
        echo "Remove it or set DYNASOAR_ROOT to an empty/missing path to clone." >&2
        exit 1
    fi
    echo "Cloning $KIUCHI_REPO ($KIUCHI_REF) into $DYNASOAR_ROOT ..."
    if ! git clone --recurse-submodules --branch "$KIUCHI_REF" --single-branch \
            "$KIUCHI_REPO" "$DYNASOAR_ROOT"; then
        git clone --recurse-submodules "$KIUCHI_REPO" "$DYNASOAR_ROOT"
        git -C "$DYNASOAR_ROOT" checkout "$KIUCHI_REF"
        git -C "$DYNASOAR_ROOT" submodule update --init --recursive
    fi
elif [[ "$KIUCHI_UPDATE" == "1" ]]; then
    echo "Updating $DYNASOAR_ROOT to $KIUCHI_REF ..."
    git -C "$DYNASOAR_ROOT" fetch --tags origin
    git -C "$DYNASOAR_ROOT" checkout "$KIUCHI_REF"
    if git -C "$DYNASOAR_ROOT" rev-parse --verify -q "origin/$KIUCHI_REF" >/dev/null; then
        git -C "$DYNASOAR_ROOT" merge --ff-only "origin/$KIUCHI_REF"
    fi
    git -C "$DYNASOAR_ROOT" submodule update --init --recursive
fi

if [[ ! -f "$DYNASOAR_ROOT/Makefile" ]]; then
    echo "Error: no Makefile in $DYNASOAR_ROOT" >&2
    exit 1
fi

if [[ ! -f "$DYNASOAR_ROOT/dynasoar/dynasoar.h" ]]; then
    echo "Initializing dynasoar submodule ..."
    git -C "$DYNASOAR_ROOT" submodule update --init --recursive
fi

if [[ ! -f "$DYNASOAR_ROOT/dynasoar/dynasoar.h" ]]; then
    echo "Error: dynasoar submodule missing under $DYNASOAR_ROOT/dynasoar" >&2
    exit 1
fi

echo "Building Kiuchi targets in $DYNASOAR_ROOT: ${TARGETS[*]}"
# Fork Makefile hardcodes sm_90; paper binaries only (not make all).
make -C "$DYNASOAR_ROOT" -B "${TARGETS[@]}"

missing=0
for t in "${TARGETS[@]}"; do
    # Accept either bin/Foo or Foo
    if [[ "$t" == bin/* ]]; then
        exe="$DYNASOAR_ROOT/$t"
    else
        exe="$DYNASOAR_ROOT/bin/$t"
    fi
    if [[ ! -x "$exe" ]]; then
        echo "Error: expected binary missing after build: $exe" >&2
        missing=1
    else
        echo "  $exe"
    fi
done
if [[ "$missing" != "0" ]]; then
    exit 1
fi

echo "Done."
