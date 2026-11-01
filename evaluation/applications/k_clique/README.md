# k-clique

GTaP thread-mode implementations of orientation-based and pivot-based
k-clique counting, evaluated against KCGPU.

## Setup

Requires the GTaP compiler, CUDA toolkit, KCGPU, and Python 3.
Set `CUDA_PATH`, `CUDA_ARCH`, and `CXX_BIN` according to the artifact setup.

Run the following commands from this directory:

```sh
export KCGPU_DIR=/path/to/KCGPU
export KCGPU_ROOT="$KCGPU_DIR"
export GRAPH_DIR=/path/to/graphs
```

The graph directory must contain these Matrix Market files:

```text
com-DBLP/com-DBLP.mtx
as-Skitter/as-Skitter.mtx
com-Orkut/com-Orkut.mtx
```

## Run

Generate the binomial-coefficient table required by pivot counting:

```sh
make ncr
```

Run both experiments:

```sh
mkdir -p k579/results
bash scripts/k579/run_kcgpu_orientation_k579_graph_tuned.sh
bash scripts/k579/run_kcgpu_pivot_k579_graph_tuned.sh
```

Each script builds the required binaries and runs all three graphs with
k = 5, 7, and 9, using 20 repetitions by default. Graph-specific GTaP
settings are defined in `data/gtap_graph_k_config.csv`.

For a shorter check:

```sh
GRAPHS=DBLP K_VALUES=5 REPEATS=1 \
  bash scripts/k579/run_kcgpu_pivot_k579_graph_tuned.sh
```

## Results

CSV results and summaries are written to `k579/results/`; individual run
logs are written to `k579/logs/`.

Check that `gtap_exit` and `kcgpu_exit` are zero and `match` is `YES`.
The CSV files also report execution times for both implementations.
