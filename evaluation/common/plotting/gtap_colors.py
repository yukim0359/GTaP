"""Shared GTaP color mapping for evaluation plots."""

# GTaP benchmarks: fixed role-color mapping
COL_GTAP_THREAD = "#1f77b4"  # GTaP (Thread)
COL_GTAP_BLOCK  = "#d62728"  # GTaP (Block)
COL_OMP         = "#ff7f0e"  # CPU OpenMP
COL_CILK        = "#9467bd"  # CPU Cilk (OpenCilk)
COL_SEQ         = "#2ca02c"  # CPU Sequential
COL_KIUCHI      = "#e377c2"  # Kiuchi et al. (fine-grained-fork-join)
COL_GPU_NQUEENS = "#17becf"  # Standalone GPU N-Queens solver (Ya et al.)
COL_GPU_THRUST  = "#8c564b"  # GPU Thrust sort (CUDA standard library)

# Legend labels (fib / nq / cilksort comparison plots).
LABEL_GTAP_THREAD = "GTaP (Thread-Level Workers)"
LABEL_KIUCHI = "Kiuchi et al."
LABEL_CPU_OMP = "CPU OpenMP"
LABEL_CPU_CILK = "CPU OpenCilk"
LABEL_RATIO_KIUCHI = r"Kiuchi et al. / GTaP"
LABEL_RATIO_OMP = r"CPU OpenMP / GTaP"
LABEL_RATIO_CILK = r"CPU OpenCilk / GTaP"

ROLE_COLORS = {
    "gtap_thread": COL_GTAP_THREAD,
    "gtap_block": COL_GTAP_BLOCK,
    "omp": COL_OMP,
    "cilk": COL_CILK,
    "seq": COL_SEQ,
    "kiuchi": COL_KIUCHI,
    "gpu_nqueens": COL_GPU_NQUEENS,
    "gpu_thrust": COL_GPU_THRUST,
}
