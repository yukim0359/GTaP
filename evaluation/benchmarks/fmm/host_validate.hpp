#pragma once

#include "host_cuda.hpp"
#include "host_timing.hpp"
#include "host_types.hpp"
#include "numerics/validate.cuh"

#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

static bool validate_results(FmmRun &run) {
  const int n = run.options.n;
  const int direct_sample_n = run.options.direct_sample_n;
  FmmValidation &v = run.validation;
  FmmTimings &t = run.timings;

  for (int i = 0; i < n; ++i) {
    v.checksum += v.hphi[i];
    v.acc_checksum += static_cast<double>(v.hax[i]) + v.hay[i] + v.haz[i];
    if (v.hphi[i] != Real(0)) v.nonzero_phi++;
    if (v.hax[i] != Real(0) || v.hay[i] != Real(0) || v.haz[i] != Real(0)) v.nonzero_acc++;
  }

  if (direct_sample_n > 0) {
    const std::chrono::high_resolution_clock::time_point t_sample_direct_start =
        std::chrono::high_resolution_clock::now();
    std::vector<int> sample_idx;
    sample_idx.reserve(direct_sample_n);
    if (direct_sample_n == n) {
      for (int i = 0; i < n; ++i) sample_idx.push_back(i);
    } else {
      std::mt19937_64 rng(static_cast<uint64_t>(DIRECT_SAMPLE_SEED));
      std::uniform_int_distribution<int> dist(0, n - 1);
      while (static_cast<int>(sample_idx.size()) < direct_sample_n) {
        sample_idx.push_back(dist(rng));
        std::sort(sample_idx.begin(), sample_idx.end());
        sample_idx.erase(std::unique(sample_idx.begin(), sample_idx.end()),
                         sample_idx.end());
      }
    }

    int *d_sample_idx = nullptr;
    double *d_phi_diff2 = nullptr, *d_phi_norm2 = nullptr;
    double *d_acc_diff2 = nullptr, *d_acc_norm2 = nullptr;
    double *d_phi_rel_abs = nullptr, *d_acc_rel_abs = nullptr;
    const size_t sample_bytes = sizeof(double) * sample_idx.size();
    CUDA_CHECK(cudaMalloc(&d_sample_idx, sizeof(int) * sample_idx.size()));
    CUDA_CHECK(cudaMalloc(&d_phi_diff2, sample_bytes));
    CUDA_CHECK(cudaMalloc(&d_phi_norm2, sample_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc_diff2, sample_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc_norm2, sample_bytes));
    CUDA_CHECK(cudaMalloc(&d_phi_rel_abs, sample_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc_rel_abs, sample_bytes));
    CUDA_CHECK(cudaMemcpy(d_sample_idx, sample_idx.data(),
                          sizeof(int) * sample_idx.size(), cudaMemcpyHostToDevice));
    fmm3d_direct_sample_validation_kernel<<<static_cast<int>(sample_idx.size()),
                                            DIRECT_SAMPLE_BLOCK_SIZE,
                                            sizeof(double) * 4 * DIRECT_SAMPLE_BLOCK_SIZE>>>(
        d_sample_idx, static_cast<int>(sample_idx.size()),
        d_phi_diff2, d_phi_norm2, d_acc_diff2, d_acc_norm2,
        d_phi_rel_abs, d_acc_rel_abs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> phi_diff2(sample_idx.size()), phi_norm2(sample_idx.size());
    std::vector<double> acc_diff2(sample_idx.size()), acc_norm2(sample_idx.size());
    std::vector<double> phi_rel_abs(sample_idx.size()), acc_rel_abs(sample_idx.size());
    CUDA_CHECK(cudaMemcpy(phi_diff2.data(), d_phi_diff2, sample_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(phi_norm2.data(), d_phi_norm2, sample_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(acc_diff2.data(), d_acc_diff2, sample_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(acc_norm2.data(), d_acc_norm2, sample_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(phi_rel_abs.data(), d_phi_rel_abs, sample_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(acc_rel_abs.data(), d_acc_rel_abs, sample_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_sample_idx);
    cudaFree(d_phi_diff2);
    cudaFree(d_phi_norm2);
    cudaFree(d_acc_diff2);
    cudaFree(d_acc_norm2);
    cudaFree(d_phi_rel_abs);
    cudaFree(d_acc_rel_abs);

    double phi_diff2_sum = 0.0, phi_norm2_sum = 0.0;
    double acc_diff2_sum = 0.0, acc_norm2_sum = 0.0;
    for (size_t i = 0; i < sample_idx.size(); ++i) {
      phi_diff2_sum += phi_diff2[i];
      phi_norm2_sum += phi_norm2[i];
      acc_diff2_sum += acc_diff2[i];
      acc_norm2_sum += acc_norm2[i];
      v.sample_phi_rel_linf = std::max(v.sample_phi_rel_linf, phi_rel_abs[i]);
      v.sample_acc_rel_linf = std::max(v.sample_acc_rel_linf, acc_rel_abs[i]);
    }
    std::sort(phi_rel_abs.begin(), phi_rel_abs.end());
    std::sort(acc_rel_abs.begin(), acc_rel_abs.end());
    const auto percentile = [](const std::vector<double> &vec, double p) {
      if (vec.empty()) return 0.0;
      const size_t idx = std::min(vec.size() - 1,
                                  static_cast<size_t>(std::ceil(p * vec.size())) - 1);
      return vec[idx];
    };
    v.sample_phi_rel_l2 = std::sqrt(phi_diff2_sum / std::max(phi_norm2_sum, 1.0e-300));
    v.sample_acc_rel_l2 = std::sqrt(acc_diff2_sum / std::max(acc_norm2_sum, 1.0e-300));
    v.sample_phi_rel_p95 = percentile(phi_rel_abs, 0.95);
    v.sample_acc_rel_p95 = percentile(acc_rel_abs, 0.95);
    v.sample_phi_rel_p99 = percentile(phi_rel_abs, 0.99);
    v.sample_acc_rel_p99 = percentile(acc_rel_abs, 0.99);
    const std::chrono::high_resolution_clock::time_point t_sample_direct_end =
        std::chrono::high_resolution_clock::now();
    t.ms_sample_direct = h_elapsed_ms(t_sample_direct_start, t_sample_direct_end);
    v.sample_direct_validated = true;
  }

  if (DIRECT_VALIDATE_N > 0 && n <= DIRECT_VALIDATE_N) {
    const std::chrono::high_resolution_clock::time_point t_direct_start =
        std::chrono::high_resolution_clock::now();
    double phi_diff2 = 0.0;
    double phi_norm2 = 0.0;
    double acc_diff2 = 0.0;
    double acc_norm2 = 0.0;
    const HostBodies &hb = run.host.hb;
    #pragma omp parallel for reduction(+:phi_diff2,phi_norm2,acc_diff2,acc_norm2)
    for (int i = 0; i < n; ++i) {
      const double xi = hb.x[i];
      const double yi = hb.y[i];
      const double zi = hb.z[i];
      double phi_ref = 0.0;
      double ax_ref = 0.0;
      double ay_ref = 0.0;
      double az_ref = 0.0;
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        const double dx = xi - static_cast<double>(hb.x[j]);
        const double dy = yi - static_cast<double>(hb.y[j]);
        const double dz = zi - static_cast<double>(hb.z[j]);
        const double r2 = dx * dx + dy * dy + dz * dz + static_cast<double>(H_EPS2);
        const double inv_r = 1.0 / std::sqrt(r2);
        const double q_over_r = static_cast<double>(hb.q[j]) * inv_r;
        phi_ref += q_over_r;
        const double inv_r3_q = q_over_r / r2;
        ax_ref -= dx * inv_r3_q;
        ay_ref -= dy * inv_r3_q;
        az_ref -= dz * inv_r3_q;
      }
      const double dphi = static_cast<double>(v.hphi[i]) - phi_ref;
      const double dax = static_cast<double>(v.hax[i]) - ax_ref;
      const double day = static_cast<double>(v.hay[i]) - ay_ref;
      const double daz = static_cast<double>(v.haz[i]) - az_ref;
      phi_diff2 += dphi * dphi;
      phi_norm2 += phi_ref * phi_ref;
      acc_diff2 += dax * dax + day * day + daz * daz;
      acc_norm2 += ax_ref * ax_ref + ay_ref * ay_ref + az_ref * az_ref;
    }
    const std::chrono::high_resolution_clock::time_point t_direct_end =
        std::chrono::high_resolution_clock::now();
    t.ms_direct = h_elapsed_ms(t_direct_start, t_direct_end);
    v.phi_rel_l2 = std::sqrt(phi_diff2 / std::max(phi_norm2, 1.0e-300));
    v.acc_rel_l2 = std::sqrt(acc_diff2 / std::max(acc_norm2, 1.0e-300));
    v.direct_validated = true;
  }
  return true;
}
