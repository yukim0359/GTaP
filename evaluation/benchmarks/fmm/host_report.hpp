#pragma once

#include "host_types.hpp"

#include <cstdio>

static void print_run_report(const FmmRun &run) {
  const int n = run.options.n;
  const Real theta = run.options.theta;
  const int ncells = run.host.ncells;
  const int nleaf = run.host.nleaf;
  const FmmDttStats &dtt = run.dtt;
  const FmmTimings &t = run.timings;
  const FmmValidation &v = run.validation;

  std::printf("=== exa-style FMM3D run summary ===\n");
  std::printf("Problem: N=%d  Nodes=%d  Leaves=%d  theta=%.2f  NCRIT=%d  kernel=Laplace exa-style P=%d NTERM=%d  precision=%s\n",
              n, ncells, nleaf, theta, NCRIT, EXA_P, EXA_NTERM,
              FMM3D_SINGLE ? "single" : "double");
  std::printf("M2L Cnm constant table: %d entries, %.1f KiB (re+im, P^4=%d)\n",
              M2L_CNM_SIZE,
              static_cast<double>(M2L_CNM_SIZE) * 2.0 * sizeof(Real) / 1024.0,
              EXA_P * EXA_P * EXA_P * EXA_P);
  std::printf("Sph prefactor constant table: %d entries, %.1f KiB (4*P^2=%d)\n",
              EXA_NSPH,
              static_cast<double>(EXA_NSPH) * sizeof(Real) / 1024.0,
              4 * EXA_P * EXA_P);
  std::printf("FMM constant memory total: %.1f KiB / 64 KiB\n",
              static_cast<double>(FMM_CONST_BYTES) / 1024.0);
#if FMM3D_PARTITIONED_FUSED_DTT
  std::printf("Interactions: (partitioned fused DTT; list counters not updated during traversal)\n");
#else
  std::printf("Interactions: M2L=%lld  P2P=%lld\n", dtt.m2l_total_ll, dtt.p2p_total_ll);
#endif
  std::printf("Validation: phi_checksum=%.9e  acc_checksum=%.9e  nonzero_phi=%d/%d  nonzero_acc=%d/%d\n",
              v.checksum, v.acc_checksum, v.nonzero_phi, n, v.nonzero_acc, n);
  if (v.direct_validated) {
    std::printf("Direct check: N=%d  phi_rel_l2=%.9e  acc_rel_l2=%.9e  direct_time=%.3f ms\n",
                n, v.phi_rel_l2, v.acc_rel_l2, t.ms_direct);
  } else {
    std::printf("Direct check: skipped (N=%d > FMM3D_DIRECT_VALIDATE_N=%d)\n",
                n, DIRECT_VALIDATE_N);
  }
  if (v.sample_direct_validated) {
    std::printf("Sample direct check: samples=%d  phi_rel_l2=%.9e  acc_rel_l2=%.9e  "
                "phi_rel_linf=%.9e  acc_rel_linf=%.9e  "
                "phi_p95=%.9e  acc_p95=%.9e  phi_p99=%.9e  acc_p99=%.9e  "
                "sample_direct_time=%.3f ms\n",
                run.options.direct_sample_n, v.sample_phi_rel_l2, v.sample_acc_rel_l2,
                v.sample_phi_rel_linf, v.sample_acc_rel_linf,
                v.sample_phi_rel_p95, v.sample_acc_rel_p95,
                v.sample_phi_rel_p99, v.sample_acc_rel_p99, t.ms_sample_direct);
  } else {
    std::printf("Sample direct check: disabled (FMM3D_DIRECT_SAMPLE_N=0)\n");
  }

  std::printf("\n=== Pipeline timeline (ms) ===\n");
#if FMM3D_GPU_TREE_BUILD
  std::printf("  [GPU+CPU]     build adaptive tree, Morton sort assisted: %.3f\n", t.ms_tree_build);
#else
  std::printf("  [CPU]         build adaptive tree: %.3f\n", t.ms_tree_build);
#endif
#if !FMM3D_GPU_UPWARD
  std::printf("  [CPU]         upward pass (P2M/M2M): %.3f\n", t.ms_upward);
#endif
  std::printf("  [CPU]         build traversal metadata lists: %.3f\n", t.ms_metadata);
  std::printf("  [H2D+bind]    upload static arrays and bind symbols: %.3f\n", t.ms_h2d_bind);
#if FMM3D_GPU_UPWARD
  std::printf("  [GPU]         upward pass (P2M/M2M), level-by-level: %.3f\n", t.ms_upward);
#endif
  std::printf("  [DTT init]    fixed-cap buffers + runtime setup: %.3f\n", t.ms_dtt_init);
#if FMM3D_WORKLIST_DTT
  std::printf("  [GPU]         DTT worklist setup: %.3f\n", t.ms_dtt_fill_setup);
  std::printf("  [GPU]         DTT traversal core: %.3f\n", t.ms_dtt_core);
  std::printf("  [GPU]         DTT post/check: %.3f\n", t.ms_dtt_fill_post);
#elif FMM3D_ENABLE_GTAP_DTT
  std::printf("  [GPU]         DTT setup: %.3f\n", t.ms_dtt_fill_setup);
#if FMM3D_PARTITIONED_FUSED_DTT
  std::printf("  [GPU/GTaP]    DTT exafmm-style range split + fused M2L/P2P: %.3f\n",
              t.ms_dtt_construct_handoff);
#elif FMM3D_GTAP_FUSED
  std::printf("  [GPU/GTaP]    DTT traversal + fused M2L/P2P: %.3f\n", t.ms_dtt_construct_handoff);
#else
  std::printf("  [GPU/GTaP]    DTT traversal core: %.3f\n", t.ms_dtt_fill_kernel);
#endif
  std::printf("  [GPU]         DTT post/check: %.3f\n", t.ms_dtt_fill_post);
#else
#if FMM3D_SERIAL_HOST_DTT
  std::printf("  [CPU/Serial]  DTT reset/counts: %.3f\n",
              dtt.host_dtt_breakdown.ms_reset_counts);
  std::printf("  [CPU/Serial]  DTT traversal core: %.3f\n", t.ms_dtt_core);
  std::printf("  [CPU/Serial]  DTT list stats: %.3f\n", dtt.host_dtt_breakdown.ms_stats);
#else
  std::printf("  [CPU/OpenMP]  DTT reset/counts: %.3f\n",
              dtt.host_dtt_breakdown.ms_reset_counts);
  std::printf("  [CPU/OpenMP]  DTT traversal core: %.3f\n", t.ms_dtt_core);
  std::printf("  [CPU/OpenMP]  DTT list stats: %.3f\n", dtt.host_dtt_breakdown.ms_stats);
#endif
  std::printf("  [H2D]         DTT list counts: %.3f\n",
              dtt.host_dtt_breakdown.ms_h2d_counts);
  std::printf("  [H2D]         DTT list payloads: %.3f\n",
              dtt.host_dtt_breakdown.ms_h2d_lists);
#endif
  std::printf("  [GPU]         M2L kernel: %.3f\n", t.ms_m2l);
#if FMM3D_GPU_L2L
  std::printf("  [GPU]         L2L downward pass, level-by-level: %.3f\n", t.ms_l2l);
#else
  std::printf("  [D2H]         copy tree for L2L: %.3f\n", t.ms_l2l_tree_d2h);
  std::printf("  [CPU]         L2L downward pass: %.3f\n", t.ms_l2l);
  std::printf("  [H2D]         copy tree after L2L: %.3f\n", t.ms_l2l_tree_h2d);
#endif
  std::printf("  [GPU]         L2P + P2P eval kernel: %.3f\n", t.ms_eval);
  std::printf("  [D2H]         copy phi/acc result arrays: %.3f\n", t.ms_d2h);
  std::printf("\n=== DTT comparison focus ===\n");
  std::printf("DTT core traversal: %.3f ms\n", t.ms_dtt_core);
  std::printf("DTT list construction (steady-state): %.3f ms\n", t.ms_dtt_construct);
#if !FMM3D_WORKLIST_DTT && !FMM3D_ENABLE_GTAP_DTT
  std::printf("DTT GPU handoff (H2D lists): %.3f ms\n",
              dtt.host_dtt_breakdown.ms_h2d_counts + dtt.host_dtt_breakdown.ms_h2d_lists);
#endif
  std::printf("DTT list construction + GPU handoff: %.3f ms\n",
              t.ms_dtt_construct_handoff);
  std::printf("DTT init: %.3f ms\n", t.ms_dtt_init);
  std::printf("DTT + init: %.3f ms\n", t.ms_dtt_construct_handoff + t.ms_dtt_init);
#if FMM3D_ENABLE_GTAP_DTT
  std::printf("  (GTaP runtime subset: %.3f ms)\n", t.ms_gtap_init);
#elif FMM3D_WORKLIST_DTT
  std::printf("  (worklist pair-buffer subset: %.3f ms)\n", t.ms_worklist_pair_buffer_alloc);
#else
  std::printf("  (host list-buffer subset: %.3f ms)\n", t.ms_host_list_buffer_alloc);
#endif
  std::printf("DTT fixed-cap metrics: m2l_cap=%d  p2p_cap=%d  max_m2l=%d  max_p2p=%d  overflow=%d\n",
              DTT_M2L_CAP, DTT_P2P_CAP, dtt.max_m2l_count, dtt.max_p2p_count, dtt.overflow);
#if FMM3D_M2L_COMPACT_TARGETS
  std::printf("M2L target cells: %d/%d\n", dtt.m2l_target_count, ncells);
#endif
#if FMM3D_WORKLIST_DTT
  std::printf("DTT worklist metrics: pair_cap=%d  iterations=%d  max_frontier=%d  overflow=%d\n",
              run.device.worklist_pair_cap, dtt.worklist_iterations, dtt.worklist_max_frontier, dtt.worklist_overflow);
  std::printf("DTT mode: GPU worklist fixed-cap lists (level-synchronous)\n");
#elif FMM3D_ENABLE_GTAP_DTT
#if FMM3D_PARTITIONED_FUSED_DTT
  std::printf("DTT mode: GTaP partitioned fused traversal (exafmm-style dualTreeTraversal + child-range TraverseRange; no atomics on L/phi)\n");
#elif FMM3D_GTAP_FUSED
  std::printf("DTT mode: GTaP fused traversal (M2L/P2P during DTT; M2L kernel skipped)\n");
#else
  std::printf("DTT mode: GTaP task traversal, GPU fixed-cap lists\n");
#endif
#else
#if FMM3D_SERIAL_HOST_DTT
  std::printf("DTT mode: CPU serial host fixed-cap lists, then H2D to GPU\n");
#else
  std::printf("DTT mode: CPU/OpenMP host fixed-cap lists, then H2D to GPU\n");
#endif
#endif
  std::printf("\n=== Totals ===\n");
  std::printf("Execution time: %.3f ms\n", t.ms_execution);
}
