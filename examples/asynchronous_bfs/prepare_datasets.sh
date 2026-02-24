#!/bin/bash
#
# Downloads and prepares graph datasets for the asynchronous BFS example.
# Each dataset is downloaded as a tar.gz from Dropbox, extracted, and the
# pre-built binary CSR file is placed in the datasets/ directory.
#
# Usage:
#   ./prepare_datasets.sh              # download all four datasets
#   ./prepare_datasets.sh road_usa     # download a specific dataset
#
# Output directory: <script_dir>/datasets/
#
# After running this script, launch the BFS binary with:
#   ./bin/asynchronous_bfs datasets/soc-LiveJournal1_di.csr  0
#   ./bin/asynchronous_bfs datasets/hollywood-2009_ud.csr    0
#   ./bin/asynchronous_bfs datasets/indochina-2004_di.csr    40
#   ./bin/asynchronous_bfs datasets/road_usa_ud.csr          0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$SCRIPT_DIR/datasets"
mkdir -p "$DATASETS_DIR"

# --------------------------------------------------------------------------
# Dataset definitions
#   Each line: <name> <dropbox-url> <csr-suffix (di|ud)>
# --------------------------------------------------------------------------
declare -a NAMES=(
    "soc-LiveJournal1"
    "hollywood-2009"
    "indochina-2004"
    "road_usa"
)
declare -A URLS=(
    ["soc-LiveJournal1"]="https://www.dropbox.com/s/qgk292rfzhhhazq/soc-LiveJournal1.tar.gz"
    ["hollywood-2009"]="https://www.dropbox.com/s/2evu3xb1bwte4kt/hollywood-2009.tar.gz"
    ["indochina-2004"]="https://www.dropbox.com/s/vk0gyfqcgmih6pw/indochina-2004.tar.gz"
    ["road_usa"]="https://www.dropbox.com/s/5mo1f04ygogluw4/road_usa.tar.gz"
)
declare -A SUFFIXES=(
    ["soc-LiveJournal1"]="di"
    ["hollywood-2009"]="ud"
    ["indochina-2004"]="di"
    ["road_usa"]="ud"
)

# --------------------------------------------------------------------------
# Filter to requested dataset(s)
# --------------------------------------------------------------------------
if [ $# -ge 1 ]; then
    NAMES=("$@")
fi

# --------------------------------------------------------------------------
# Helper: download and extract one dataset
# --------------------------------------------------------------------------
prepare_one() {
    local name="$1"
    local url="${URLS[$name]}"
    local suffix="${SUFFIXES[$name]}"
    local csr_name="${name}_${suffix}.csr"
    local dest="$DATASETS_DIR/$csr_name"

    if [ -f "$dest" ]; then
        echo "[$name] Already present: $dest"
        return 0
    fi

    local tarball="$DATASETS_DIR/${name}.tar.gz"

    # Download
    if [ ! -f "$tarball" ]; then
        echo "[$name] Downloading from $url ..."
        wget --quiet --show-progress -O "$tarball" "${url}?dl=1"
    else
        echo "[$name] Archive already downloaded: $tarball"
    fi

    # Extract into a temporary subdirectory to avoid collisions
    local tmpdir
    tmpdir=$(mktemp -d "$DATASETS_DIR/${name}_extract_XXXXXX")
    echo "[$name] Extracting ..."
    tar xzf "$tarball" -C "$tmpdir"

    # Locate the .csr file anywhere inside the extracted tree
    local csr_found
    csr_found=$(find "$tmpdir" -name "$csr_name" | head -n 1)

    if [ -z "$csr_found" ]; then
        echo "[$name] ERROR: $csr_name not found inside archive." >&2
        rm -rf "$tmpdir"
        return 1
    fi

    mv "$csr_found" "$dest"
    rm -rf "$tmpdir"

    # Keep the tarball so re-running is fast; remove it to save space:
    # rm -f "$tarball"

    echo "[$name] Ready: $dest"
}

# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
failed=0
for name in "${NAMES[@]}"; do
    if [ -z "${URLS[$name]+_}" ]; then
        echo "Unknown dataset: $name" >&2
        echo "Available: ${!URLS[*]}" >&2
        failed=1
        continue
    fi
    prepare_one "$name" || failed=1
done

if [ "$failed" -ne 0 ]; then
    echo "One or more datasets failed to prepare." >&2
    exit 1
fi

echo ""
echo "All requested datasets are ready in: $DATASETS_DIR"
