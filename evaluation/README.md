# Evaluation

Performance experiments for the [GTaP paper](https://arxiv.org/abs/2604.05982).
The default settings target one NVIDIA GH200 node with 72 Grace CPU cores
and CUDA Toolkit 12.9. Run the commands below from `evaluation/`.

## Setup

Build GTaP's Clang and LLVM OpenMP first; see the
[repository setup](../README.md#getting-started).

Then run the following preparatory steps.

```sh
export CUDA_PATH=/path/to/cuda
export CUDA_ARCH=sm_90
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pandas matplotlib

./scripts/setup/build_opencilk.sh
./scripts/setup/build_kiuchi.sh
./scripts/setup/build_kcgpu.sh
make -C applications/k_clique setup
```

The setup scripts install OpenCilk under `$HOME/opencilk/install`
and clone the modified Kiuchi and
KCGPU repositories alongside the GTaP checkout. Their headers document
location and revision overrides (`KCGPU_DIR`, `DYNASOAR_ROOT`, and so on).

## Run and plot

```sh
./eval.sh list          # available experiments and plot targets
./eval.sh run fib       # build and measure one experiment
./eval.sh run paper     # alternatively, run all performance experiments

# After the required experiments finish:
./eval.sh plot micro    # regenerate the runtime-comparison figure
./eval.sh plot paper    # regenerate all performance figures/tables
```

Run measurements on a GPU node. The experiment scripts build the required
binaries and use their paper settings by default; `run paper` executes the
full set sequentially. `plot` reads saved CSVs / profile dumps without
rerunning measurements.

## Experiments

| Paper experiment | `run` targets | `plot` target |
| --- | --- | --- |
| [Thread/block modes](mode-comparison/binary_tree/) (§6.1) | `tree-depth`, `tree-compute` | `tree` |
| [Heterogeneous tree / DAQ](mode-comparison/hetero_tree/) (§6.2) | `hetero` | `hetero` |
| [Runtime comparison](microbenchmarks/runtime-comparison/) (§6.3.1) | `fib`, `nq`, `cilksort` | `micro` |
| [DAQ applicability](microbenchmarks/daq/) (§6.3.2) | `daq-fib`, `daq-nq`, `daq-cilksort` | `daq` |
| [FMM theta sweep](applications/fmm/scripts/dtt_theta/) (§6.4.1) | `fmm-theta` | `fmm-theta` |
| [FMM phase breakdown](applications/fmm/scripts/stack/) (§6.4.1) | `fmm-stack` | `fmm-stack` |
| [k-clique timings](applications/k_clique/) (§6.4.2) | `kclique-orientation`, `kclique-pivot` | `kclique` |
| [k-clique pivot timelines](applications/k_clique/scripts/profile/) (Fig. 7) | `kclique-timeline-gtap`, `kclique-timeline-kcgpu` | `kclique-timeline-gtap`, `kclique-timeline-kcgpu` |
| [Scheduler comparison](scheduler-comparison/) (§6.5) | `thread-fib`, `sm-fib` | `thread-sm-fib` |

## Results

- Most CSVs are written beside the corresponding experiment scripts.
  FMM's theta sweep writes `applications/fmm/fmm_dtt_theta_results.csv`;
  its phase breakdown writes CSVs under `applications/fmm/csv/`.
- k-clique writes under `applications/k_clique/k579/`: per-run and summary
  CSVs in `results/`, logs in `logs/`, and the generated table in
  `latex/k579_benchmark_tables.tex`.
  Run `./eval.sh plot kclique` after both k-clique experiments finish.
- Fig. 7 timelines need profile-instrumented runs, then
  `./eval.sh plot kclique-timeline-gtap` /
  `./eval.sh plot kclique-timeline-kcgpu`.
- Performance figures are written to `img/` by scripts in `scripts/plot/`.

PBS users can use `submit` instead of `run` after adjusting the `.pbs` files
for their cluster.
