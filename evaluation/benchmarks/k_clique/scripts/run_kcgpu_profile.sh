#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
K_CLIQUE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
GTAP_ROOT=$(cd "$K_CLIQUE_DIR/../../.." && pwd)
WORK_ROOT=$(cd "$GTAP_ROOT/.." && pwd)
KCGPU_DIR=${KCGPU_DIR:-"$WORK_ROOT/KCGPU"}
GRAPH_DIR=${GRAPH_DIR:-"$KCGPU_DIR/graphs"}

DEVICE=${DEVICE:-0}
KCGPU_CUDA_ARCH=${KCGPU_CUDA_ARCH:-90}
KCGPU_PROCESS=${KCGPU_PROCESS:-edge}
KCGPU_ELEMENT=${KCGPU_ELEMENT:-bw}
KCGPU_VARIANT=${KCGPU_VARIANT:-}
if [ -z "$KCGPU_VARIANT" ]; then
    case "${KCGPU_Q:-}" in
        p*) KCGPU_VARIANT=pivot ;;
        *) KCGPU_VARIANT=orientation ;;
    esac
fi
case "$KCGPU_VARIANT" in
    orientation)
        KCGPU_Q=${KCGPU_Q:-o1b}
        KCGPU_APP_NAME=kcgpu_orientation
        KCGPU_ORIENT=${KCGPU_ORIENT:-degree}
        ;;
    pivot)
        KCGPU_Q=${KCGPU_Q:-p1b}
        KCGPU_APP_NAME=kcgpu_pivot
        KCGPU_ORIENT=${KCGPU_ORIENT:-degen}
        ;;
    *)
        echo "ERROR: KCGPU_VARIANT must be orientation or pivot" >&2
        exit 1
        ;;
esac
KCGPU_ALLOC=${KCGPU_ALLOC:-gpu}
KCGPU_SORT=${KCGPU_SORT:-1}

GRAPH_NAME=${GRAPH_NAME:-DBLP}
K=${K:-7}

if [ -z "${GRAPH_PATH:-}" ]; then
    case "$GRAPH_NAME" in
        DBLP|com-DBLP)
            GRAPH_PATH="$GRAPH_DIR/com-DBLP/com-DBLP.mtx"
            ;;
        Orkut|com-Orkut)
            GRAPH_PATH="$GRAPH_DIR/com-Orkut/com-Orkut.mtx"
            ;;
        Skitter|as-Skitter)
            GRAPH_PATH="$GRAPH_DIR/as-Skitter/as-Skitter.mtx"
            ;;
        LiveJournal|com-LiveJournal)
            GRAPH_PATH="$GRAPH_DIR/com-LiveJournal/com-LiveJournal.mtx"
            ;;
        Facebook|ego-Facebook)
            GRAPH_PATH="$GRAPH_DIR/ego-Facebook/ego-Facebook.mtx"
            ;;
        *)
            candidate="$GRAPH_DIR/$GRAPH_NAME/$GRAPH_NAME.mtx"
            if [ -f "$candidate" ]; then
                GRAPH_PATH="$candidate"
            else
                echo "ERROR: unknown GRAPH_NAME=$GRAPH_NAME and GRAPH_PATH was not set." >&2
                echo "Known names: DBLP, Orkut, as-Skitter, LiveJournal, Facebook" >&2
                echo "Or pass GRAPH_PATH=/path/to/graph.mtx explicitly." >&2
                exit 1
            fi
            ;;
    esac
fi
if [ ! -f "$GRAPH_PATH" ]; then
    echo "ERROR: graph file not found: $GRAPH_PATH" >&2
    exit 1
fi

PROFILE_ROOT=${PROFILE_ROOT:-"$K_CLIQUE_DIR/profile/kcgpu"}
IMG_ROOT=${IMG_ROOT:-"$K_CLIQUE_DIR/img/kcgpu"}
LOG_DIR=${LOG_DIR:-"$K_CLIQUE_DIR/logs/kcgpu_profile"}
FIG_FORMAT=${FIG_FORMAT:-pdf}
MAX_WORKERS=${MAX_WORKERS:-30}

mkdir -p "$PROFILE_ROOT" "$IMG_ROOT" "$LOG_DIR"

if [ -n "${CUDA_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
fi

kcgpu_exe="$KCGPU_DIR/build/exe/src/main.cu.exe"
if [ "${KCGPU_SKIP_BUILD:-0}" != "1" ]; then
    echo "Building KCGPU with KCGPU_PROFILE=1 (CUDA_ARCH=$KCGPU_CUDA_ARCH)"
    make_args=(CUDA_ARCH="$KCGPU_CUDA_ARCH" KCGPU_PROFILE=1)
    if [ -n "${KCPROFILE_MAX_SAMPLES:-}" ]; then
        make_args+=(KCPROFILE_MAX_SAMPLES="$KCPROFILE_MAX_SAMPLES")
        echo "  KCPROFILE_MAX_SAMPLES=$KCPROFILE_MAX_SAMPLES"
    fi
    make_target=()
    if [ "${KCGPU_FORCE_BUILD:-0}" = "1" ]; then
        make_target=(-B)
    fi
    if ! make -C "$KCGPU_DIR" "${make_args[@]}" "${make_target[@]}"; then
        echo "ERROR: KCGPU profile build failed." >&2
        exit 1
    fi
else
    echo "Skipping KCGPU build (KCGPU_SKIP_BUILD=1)"
fi
if [ ! -x "$kcgpu_exe" ]; then
    echo "ERROR: KCGPU profile binary not found: $kcgpu_exe" >&2
    exit 1
fi
profile_markers=$(
    strings "$kcgpu_exe" 2>/dev/null |
        grep -c "KCGPU profile exported:" ||
        true
)
if [ "${profile_markers:-0}" -eq 0 ]; then
    echo "ERROR: KCGPU binary was built without profile support." >&2
    exit 1
fi
if [ "${KCGPU_BUILD_ONLY:-0}" = "1" ]; then
    echo "KCGPU profile binary ready: $kcgpu_exe"
    exit 0
fi

tag="_${GRAPH_NAME}_k${K}_${KCGPU_PROCESS}_${KCGPU_ORIENT}_${KCGPU_Q}"
export KCGPU_PROFILE_DIR="$PROFILE_ROOT"
export KCGPU_PROFILE_TAG="${tag#_}"
timeline_csv="$PROFILE_ROOT/${KCGPU_APP_NAME}_warp_timeline_working${tag}.csv"
stats_csv="$PROFILE_ROOT/${KCGPU_APP_NAME}_warp_statistics_working${tag}.csv"
sm_csv="$PROFILE_ROOT/${KCGPU_APP_NAME}_sm_balance${tag}.csv"

kcgpu_args=(
    ./build/exe/src/main.cu.exe
    -g "$GRAPH_PATH"
    -d "$DEVICE"
    -m kc
    -o "$KCGPU_ORIENT"
    -a "$KCGPU_ALLOC"
    -k "$K"
    -p "$KCGPU_PROCESS"
    -e "$KCGPU_ELEMENT"
    -q "$KCGPU_Q"
)
if [ "$KCGPU_SORT" = "1" ]; then
    kcgpu_args+=(-s 1)
fi

log_file="$LOG_DIR/${GRAPH_NAME}_k${K}_${KCGPU_PROCESS}_${KCGPU_ORIENT}_${KCGPU_Q}.log"
run_start_epoch=$(date +%s)
if [ "${KCGPU_FIGURES_ONLY:-0}" = "1" ]; then
    echo "Skipping KCGPU run (KCGPU_FIGURES_ONLY=1); using existing profile CSVs"
    if [ ! -f "$timeline_csv" ]; then
        echo "ERROR: profile CSV not found: $timeline_csv" >&2
        exit 1
    fi
else
    echo "Running KCGPU profile: graph=$GRAPH_NAME k=$K orient=$KCGPU_ORIENT q=$KCGPU_Q"
    set +e
    kcgpu_out=$(
        cd "$KCGPU_DIR" &&
        "${kcgpu_args[@]}" 2>&1
    )
    kcgpu_exit=$?
    set -e
    printf "%s\n" "$kcgpu_out" | tee "$log_file"
    echo "KCGPU exit code: $kcgpu_exit"
    if [ "$kcgpu_exit" -ne 0 ]; then
        echo "ERROR: KCGPU profile run failed; not generating figures from stale CSVs." >&2
        exit "$kcgpu_exit"
    fi

    if [ ! -f "$timeline_csv" ]; then
        echo "ERROR: profile CSV was not created: $timeline_csv" >&2
        echo "Hint: rebuild with 'make -C \"$KCGPU_DIR\" KCGPU_PROFILE=1 -B' and rerun." >&2
        exit 1
    fi
    timeline_mtime=$(stat -c %Y "$timeline_csv")
    if [ "$timeline_mtime" -lt "$run_start_epoch" ]; then
        echo "ERROR: profile CSV was not refreshed by this run: $timeline_csv" >&2
        exit 1
    fi
fi

echo "Generating SM working timeline (max_sms=$MAX_WORKERS)"
python3 "$K_CLIQUE_DIR/../../scripts/plot_kcgpu_sm_timeline_paper.py" \
    --profile-dir "$PROFILE_ROOT" \
    --variant "$KCGPU_VARIANT" \
    --profile-tag "$tag" \
    --max-sms "$MAX_WORKERS" \
    --output "$IMG_ROOT/${KCGPU_APP_NAME}${tag}_sm_timeline.${FIG_FORMAT}"

echo "Profile CSV: $timeline_csv"
echo "SM balance CSV: $sm_csv"
echo "Figures: $IMG_ROOT"
