#!/bin/bash
# Paper Fig.7 (GTaP): as-Skitter, k=9, pivoting, with GTAP_ENABLE_PROFILING=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K_CLIQUE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$K_CLIQUE_DIR" || exit 1

GTAP_ROOT="$(cd "$K_CLIQUE_DIR/../../.." && pwd)"
KCGPU_DIR="${KCGPU_DIR:-$GTAP_ROOT/../KCGPU}"
GRAPH_DIR="${GRAPH_DIR:-$KCGPU_DIR/graphs}"
GRAPH="${GRAPH:-$GRAPH_DIR/as-Skitter/as-Skitter.mtx}"
PROF=profile

if [ ! -f "$GRAPH" ]; then
    echo "ERROR: graph not found: $GRAPH" >&2
    exit 1
fi

read -r HEAVY SECOND THIRD _ORIENT_MTP PIVOT_MTP < <(
    python3 scripts/common/load_gtap_graph_k_config.py --values as-Skitter 9
)
make -B gtap_pivot GTAP_ENABLE_PROFILING=1 \
  GTAP_HEAVY_CANDIDATES="$HEAVY" \
  GTAP_SECOND_HEAVY_CANDIDATES="$SECOND" \
  GTAP_THIRD_HEAVY_CANDIDATES="$THIRD"

mkdir -p "$PROF"
# Binary writes ./profile/pivot/<graph>-<k>.
GTAP_MAX_TASKS_PER_WARP="$PIVOT_MTP" ./bin/gtap_pivot "$GRAPH" 9

python3 - <<'PY'
import json
from pathlib import Path
p = Path("profile/pivot/as-Skitter-9/profile.json")
with p.open(encoding="utf-8") as stream:
    profile = json.load(stream)
summary = profile["task_execution"]
print(f"as-Skitter k9 intervals={summary['intervals']['recorded_count']}")
PY
