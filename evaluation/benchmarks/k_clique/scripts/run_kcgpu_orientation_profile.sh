#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

KCGPU_VARIANT=orientation \
KCGPU_Q="${KCGPU_Q:-o1b}" \
KCGPU_ORIENT="${KCGPU_ORIENT:-degree}" \
exec "$SCRIPT_DIR/run_kcgpu_profile.sh" "$@"
