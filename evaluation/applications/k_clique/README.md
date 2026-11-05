# k-clique

GTaP thread-mode implementations of orientation-based and pivot-based
k-clique counting, evaluated against KCGPU.

## Setup

Requires the GTaP compiler, CUDA toolkit, KCGPU, and Python 3.
Set `CUDA_PATH`, `CUDA_ARCH`, and `CXX_BIN` according to the artifact setup.

Run the following commands from this directory:

```sh
export KCGPU_DIR=/path/to/KCGPU   # optional; default is <gtap>/../KCGPU
export GRAPH_DIR=/path/to/graphs  # optional; default is $KCGPU_DIR/graphs
make setup
```

`make setup` builds the pivot binomial table (`ncr`) and downloads SNAP
edgelists into contiguous-ID Matrix Market files.

```text
com-DBLP/com-DBLP.mtx
as-Skitter/as-Skitter.mtx
com-Orkut/com-Orkut.mtx
```

## Run

Run both experiments:

```sh
mkdir -p k579/results
bash scripts/k579/run_orientation_k579_e2e.sh
bash scripts/k579/run_pivot_k579_e2e.sh
```

Each script builds the required binaries and runs all three graphs with
k = 5, 7, and 9, using 20 repetitions by default. Graph-specific GTaP
settings are defined in `data/gtap_graph_k_config.csv`.

For a shorter check:

```sh
GRAPHS=DBLP K_VALUES=5 REPEATS=1 \
  bash scripts/k579/run_pivot_k579_e2e.sh
```

Build the paper table from the per-run CSVs:

```sh
bash scripts/k579/summarize_k579_e2e_combined.sh
```

## Results

CSV results and summaries are written to `k579/results/`; individual run
logs are written to `k579/logs/`.

Check that `gtap_exit` and `kcgpu_exit` are zero and `match` is `YES`.
The CSV files also report execution times for both implementations.
