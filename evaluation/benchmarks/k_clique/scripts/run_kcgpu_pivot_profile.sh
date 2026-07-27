#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

KCGPU_VARIANT=pivot \
KCGPU_Q="${KCGPU_Q:-p1b}" \
exec "$SCRIPT_DIR/run_kcgpu_profile.sh" "$@"
