#pragma once

#include "../device_symbols.cuh"
#include "../fmm_types.cuh"

__global__ void fmm3d_direct_sample_validation_kernel(
    const int *sample_idx, int nsample,
    double *phi_diff2, double *phi_norm2,
    double *acc_diff2, double *acc_norm2,
    double *phi_rel_abs, double *acc_rel_abs) {
  const int sid = blockIdx.x;
  if (sid >= nsample) return;
  const int i = sample_idx[sid];
  if (i < 0 || i >= g_n) return;

  const double xi = static_cast<double>(g_bx[i]);
  const double yi = static_cast<double>(g_by[i]);
  const double zi = static_cast<double>(g_bz[i]);
  double phi = 0.0;
  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;

  for (int j = threadIdx.x; j < g_n; j += blockDim.x) {
    if (j == i) continue;
    const double dx = xi - static_cast<double>(g_bx[j]);
    const double dy = yi - static_cast<double>(g_by[j]);
    const double dz = zi - static_cast<double>(g_bz[j]);
    const double r2 = dx * dx + dy * dy + dz * dz + static_cast<double>(H_EPS2);
    const double inv_r = 1.0 / sqrt(r2);
    const double q_over_r = static_cast<double>(g_bq[j]) * inv_r;
    phi += q_over_r;
    const double inv_r3_q = q_over_r / r2;
    ax -= dx * inv_r3_q;
    ay -= dy * inv_r3_q;
    az -= dz * inv_r3_q;
  }

  extern __shared__ double smem_direct_sample[];
  double *sphi = smem_direct_sample;
  double *sax = sphi + blockDim.x;
  double *say = sax + blockDim.x;
  double *saz = say + blockDim.x;
  sphi[threadIdx.x] = phi;
  sax[threadIdx.x] = ax;
  say[threadIdx.x] = ay;
  saz[threadIdx.x] = az;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) {
      sphi[threadIdx.x] += sphi[threadIdx.x + offset];
      sax[threadIdx.x] += sax[threadIdx.x + offset];
      say[threadIdx.x] += say[threadIdx.x + offset];
      saz[threadIdx.x] += saz[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const double ref_phi = sphi[0];
    const double ref_ax = sax[0];
    const double ref_ay = say[0];
    const double ref_az = saz[0];
    const double dphi = static_cast<double>(g_phi[i]) - ref_phi;
    const double dax = static_cast<double>(g_ax[i]) - ref_ax;
    const double day = static_cast<double>(g_ay[i]) - ref_ay;
    const double daz = static_cast<double>(g_az[i]) - ref_az;
    const double acc_ref2 = ref_ax * ref_ax + ref_ay * ref_ay + ref_az * ref_az;
    const double acc_err2 = dax * dax + day * day + daz * daz;
    phi_diff2[sid] = dphi * dphi;
    phi_norm2[sid] = ref_phi * ref_phi;
    acc_diff2[sid] = acc_err2;
    acc_norm2[sid] = acc_ref2;
    phi_rel_abs[sid] = fabs(dphi) / fmax(fabs(ref_phi), 1.0e-300);
    acc_rel_abs[sid] = sqrt(acc_err2) / fmax(sqrt(acc_ref2), 1.0e-300);
  }
}
