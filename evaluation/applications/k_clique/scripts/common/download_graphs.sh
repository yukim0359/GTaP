#!/bin/bash
# Fetch the paper k-clique graphs and write Matrix Market files.
#
# Downloads SNAP undirected edgelists (publicly reachable) and converts them
# to pattern-symmetric MTX with contiguous 1..n IDs — the same layout as the
# SuiteSparse Matrix Collection SNAP/* matrices used in the evaluation.
# (Direct SuiteSparse MM tarball URLs often return HTTP 403 from clusters.)
#
# Usage (from evaluation/):
#   make -C applications/k_clique graphs
#
# Env:
#   KCGPU_DIR   Default checkout (../KCGPU from gtap root)
#   GRAPH_DIR   Output tree (default: $KCGPU_DIR/graphs)
#   GRAPHS      Space-separated names (default: com-DBLP as-Skitter com-Orkut)
#
# Already-present .mtx files are left untouched.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
K_CLIQUE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
EVAL_DIR=$(cd "$K_CLIQUE_DIR/../.." && pwd)
GTAP_ROOT=$(cd "$EVAL_DIR/.." && pwd)
WORK_ROOT=$(cd "$GTAP_ROOT/.." && pwd)

KCGPU_DIR="${KCGPU_DIR:-$WORK_ROOT/KCGPU}"
GRAPH_DIR="${GRAPH_DIR:-$KCGPU_DIR/graphs}"
GRAPHS="${GRAPHS:-com-DBLP as-Skitter com-Orkut}"
CONVERT="$SCRIPT_DIR/edgelist_to_symmetric_mtx.py"

# Paper graphs only (§6.4.2).
declare -A SNAP_URL=(
    ["com-DBLP"]="https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz"
    ["as-Skitter"]="https://snap.stanford.edu/data/as-skitter.txt.gz"
    ["com-Orkut"]="https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz"
)

if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required" >&2
    exit 1
fi
if [[ ! -f "$CONVERT" ]]; then
    echo "Error: converter missing: $CONVERT" >&2
    exit 1
fi

mkdir -p "$GRAPH_DIR"
echo "GRAPH_DIR=$GRAPH_DIR"

download_one() {
    local name="$1"
    local url="${SNAP_URL[$name]:-}"
    local dest_dir="$GRAPH_DIR/$name"
    local mtx="$dest_dir/$name.mtx"
    local gz="$dest_dir/$name.ungraph.txt.gz"
    local txt="$dest_dir/$name.ungraph.txt"

    if [[ -z "$url" ]]; then
        echo "Error: unknown graph '$name' (known: ${!SNAP_URL[*]})" >&2
        return 1
    fi

    if [[ -f "$mtx" ]]; then
        echo "Skip $name (exists: $mtx)"
        return 0
    fi

    mkdir -p "$dest_dir"

    if [[ ! -f "$txt" ]]; then
        if [[ ! -f "$gz" ]]; then
            echo "Downloading $name ..."
            curl -fL --retry 3 --retry-delay 2 -o "$gz.partial" "$url"
            mv "$gz.partial" "$gz"
        fi
        echo "Decompressing $name ..."
        gzip -dc "$gz" >"$txt.partial"
        mv "$txt.partial" "$txt"
    fi

    echo "Converting $name -> Matrix Market ..."
    python3 "$CONVERT" "$txt" "$mtx.partial"
    mv "$mtx.partial" "$mtx"
    echo "Done: $mtx"
}

# shellcheck disable=SC2086
for g in $GRAPHS; do
    download_one "$g"
done

echo "All requested graphs ready under $GRAPH_DIR"
