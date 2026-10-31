#pragma once

#include "fmm_types.cuh"
#include "host_options.hpp"

#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

struct FmmHostState {
  HostBodies hb;
  std::vector<TreeNode> htree;
  DeviceBodies sorted_device_bodies;
  std::vector<int> leaf_idx;
  std::vector<int> body_cell;
  std::vector<int> cell_depth;
  std::vector<int> l2l_level_offsets;
  std::vector<int> l2l_cells;
  std::vector<int> upward_level_offsets;
  std::vector<int> upward_cells;
  int ncells = 0;
  int nleaf = 0;
  int max_cell_depth = 0;
  uint64_t body_seed = 0;
};

struct FmmDeviceBuffers {
  TreeNode *d_tree = nullptr;
  Real *d_bx = nullptr;
  Real *d_by = nullptr;
  Real *d_bz = nullptr;
  Real *d_bq = nullptr;
  Real *d_phi = nullptr;
  Real *d_ax = nullptr;
  Real *d_ay = nullptr;
  Real *d_az = nullptr;
  int *d_leaf_idx = nullptr;
  int *d_body_cell = nullptr;
  int *d_l2l_cells = nullptr;
  int *d_upward_cells = nullptr;
  int *d_m2l_count = nullptr;
  int *d_p2p_count = nullptr;
  int *d_m2l_src = nullptr;
  int *d_p2p_src = nullptr;
  int *d_m2l_targets = nullptr;
  int *d_dtt_overflow = nullptr;
  int *d_child_lo = nullptr;
  int *d_child_hi = nullptr;
  DttPair *d_pairs_a = nullptr;
  DttPair *d_pairs_b = nullptr;
  int *d_worklist_out_count = nullptr;
  int *d_worklist_overflow = nullptr;
  int worklist_pair_cap = 0;
  size_t pair_cap_size = 0;
  cudaEvent_t ev_gpu_start = nullptr;
  cudaEvent_t ev_gpu_end = nullptr;
};

struct FmmDttStats {
  int overflow = 0;
  int max_m2l_count = 0;
  int max_p2p_count = 0;
  long long m2l_total_ll = 0;
  long long p2p_total_ll = 0;
  int m2l_target_count = 0;
  std::vector<int> h_m2l_targets;
  int worklist_iterations = 0;
  int worklist_max_frontier = 0;
  int worklist_overflow = 0;
  HostFixedLists host_fixed_lists;
  HostDttBreakdown host_dtt_breakdown;
};

struct FmmTimings {
  std::chrono::high_resolution_clock::time_point run_start{};
  double ms_tree_build = 0.0;
  double ms_upward = 0.0;
  double ms_gtap_init = 0.0;
  double ms_dtt = 0.0;
  [[maybe_unused]] double ms_dtt_fill_setup = 0.0;
  [[maybe_unused]] double ms_dtt_fill_kernel = 0.0;
  [[maybe_unused]] double ms_dtt_fill_post = 0.0;
  [[maybe_unused]] double ms_dtt_core = 0.0;
  [[maybe_unused]] double ms_dtt_construct = 0.0;
  [[maybe_unused]] double ms_dtt_construct_handoff = 0.0;
  double ms_dtt_init = 0.0;
  [[maybe_unused]] double ms_host_list_buffer_alloc = 0.0;
  [[maybe_unused]] double ms_worklist_pair_buffer_alloc = 0.0;
  double ms_metadata = 0.0;
  double ms_h2d_bind = 0.0;
  double ms_m2l = 0.0;
  double ms_l2l = 0.0;
  [[maybe_unused]] double ms_l2l_tree_d2h = 0.0;
  [[maybe_unused]] double ms_l2l_tree_h2d = 0.0;
  double ms_eval = 0.0;
  double ms_d2h = 0.0;
  double ms_execution = 0.0;
  double ms_sample_direct = 0.0;
  double ms_direct = 0.0;
};

struct FmmValidation {
  std::vector<Real> hphi;
  std::vector<Real> hax;
  std::vector<Real> hay;
  std::vector<Real> haz;
  double checksum = 0.0;
  double acc_checksum = 0.0;
  int nonzero_phi = 0;
  int nonzero_acc = 0;
  bool sample_direct_validated = false;
  double sample_phi_rel_l2 = 0.0;
  double sample_acc_rel_l2 = 0.0;
  double sample_phi_rel_linf = 0.0;
  double sample_acc_rel_linf = 0.0;
  double sample_phi_rel_p95 = 0.0;
  double sample_acc_rel_p95 = 0.0;
  double sample_phi_rel_p99 = 0.0;
  double sample_acc_rel_p99 = 0.0;
  bool direct_validated = false;
  double phi_rel_l2 = 0.0;
  double acc_rel_l2 = 0.0;
};

struct FmmRun {
  FmmOptions options;
  FmmHostState host;
  FmmDeviceBuffers device;
  FmmDttStats dtt;
  FmmTimings timings;
  FmmValidation validation;
};
