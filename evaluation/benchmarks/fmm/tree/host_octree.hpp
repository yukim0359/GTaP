#pragma once

#include "../fmm_types.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static void h_print_leaf_depth_stats(const std::vector<TreeNode> &tree,
                                     const std::vector<int> &cell_depth,
                                     const std::vector<int> &leaf_idx,
                                     int n_bodies) {
  const int nleaf = static_cast<int>(leaf_idx.size());
  const int morton_cap = std::min(MORTON_BITS, MAX_DEPTH);
  if (nleaf == 0) {
    std::printf("Tree depth: no leaves (morton_cap=%d)\n", morton_cap);
    return;
  }

  int max_all_depth = 0;
  for (int d : cell_depth) max_all_depth = std::max(max_all_depth, d);

  int max_leaf_depth = 0;
  int leaves_at_cap = 0;
  int leaves_nbody_gt_ncrit = 0;
  std::vector<int> hist(static_cast<size_t>(morton_cap) + 1u, 0);
  for (int li = 0; li < nleaf; ++li) {
    const int ci = leaf_idx[li];
    const int d = cell_depth[ci];
    max_leaf_depth = std::max(max_leaf_depth, d);
    if (d >= 0 && d <= morton_cap) hist[static_cast<size_t>(d)]++;
    if (d >= morton_cap) leaves_at_cap++;
    if (tree[ci].nbody > NCRIT) leaves_nbody_gt_ncrit++;
  }

  std::printf("Tree depth: max_all=%d max_leaf=%d morton_cap=%d (MORTON_BITS=%d)\n",
              max_all_depth, max_leaf_depth, morton_cap, MORTON_BITS);
  std::printf("Leaf depth: at_cap=%d/%d (%.1f%%)  nbody>NCRIT=%d/%d (%.1f%%)  avg_nbody/leaf=%.1f\n",
              leaves_at_cap, nleaf, 100.0 * static_cast<double>(leaves_at_cap) / nleaf,
              leaves_nbody_gt_ncrit, nleaf,
              100.0 * static_cast<double>(leaves_nbody_gt_ncrit) / nleaf,
              static_cast<double>(n_bodies) / nleaf);
  std::printf("Leaf depth histogram:\n");
  for (int d = 0; d <= morton_cap; ++d) {
    if (hist[static_cast<size_t>(d)] == 0) continue;
    std::printf("  d=%2d: %8d (%5.1f%%)\n", d, hist[static_cast<size_t>(d)],
                100.0 * static_cast<double>(hist[static_cast<size_t>(d)]) / nleaf);
  }
}

// ---------- Host octree build ----------

static uint64_t h_body_rng_seed(int n) {
#if FMM3D_BODY_SEED_PER_N
  return static_cast<uint64_t>(FMM3D_BODY_SEED) + static_cast<uint64_t>(n);
#else
  return static_cast<uint64_t>(FMM3D_BODY_SEED);
#endif
}

static void h_generate_bodies(HostBodies &bodies, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<Real> pos(Real(0.05), Real(0.95));
  std::uniform_real_distribution<Real> mass(Real(0.5), Real(2));
  const int n = bodies.size();
  for (int i = 0; i < n; ++i) {
    bodies.x[i] = pos(rng);
    bodies.y[i] = pos(rng);
    bodies.z[i] = pos(rng);
    bodies.q[i] = mass(rng);
  }
}

static void h_make_child_node(std::vector<TreeNode> &tree, int parent_idx, int oct) {
  const TreeNode &p = tree[parent_idx];
  TreeNode c{};
  c.r = p.r * Real(0.5);
  c.cx = p.cx + ((oct & 1) ? c.r : -c.r);
  c.cy = p.cy + ((oct & 2) ? c.r : -c.r);
  c.cz = p.cz + ((oct & 4) ? c.r : -c.r);
  c.parent = parent_idx;
  for (int k = 0; k < 8; ++k) c.child[k] = -1;
  c.ibody = 0;
  c.nbody = 0;
  c.is_leaf = 1;
  h_zero_multipole_coeffs(c);
  tree.push_back(c);
}

static void h_subdivide(std::vector<TreeNode> &tree,
                        std::vector<int> &perm,
                        const HostBodies &bodies,
                        int node_idx, int lo, int hi, int depth) {
  const int n = hi - lo;
  if (n <= NCRIT || depth >= MAX_DEPTH) {
    tree[node_idx].is_leaf = 1;
    tree[node_idx].ibody = lo;
    tree[node_idx].nbody = n;
    return;
  }

  const Real cx = tree[node_idx].cx;
  const Real cy = tree[node_idx].cy;
  const Real cz = tree[node_idx].cz;
  int count[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = lo; i < hi; ++i) {
    const int pi = perm[i];
    int oct = int(bodies.x[pi] >= cx) | (int(bodies.y[pi] >= cy) << 1) |
              (int(bodies.z[pi] >= cz) << 2);
    count[oct]++;
  }

  int offs[8];
  {
    int run = lo;
    for (int oct = 0; oct < 8; ++oct) {
      offs[oct] = run;
      run += count[oct];
    }
  }
  std::vector<int> tmp(n);
  int local_off[8];
  for (int oct = 0; oct < 8; ++oct) local_off[oct] = offs[oct] - lo;
  for (int i = lo; i < hi; ++i) {
    const int pi = perm[i];
    int oct = int(bodies.x[pi] >= cx) | (int(bodies.y[pi] >= cy) << 1) |
              (int(bodies.z[pi] >= cz) << 2);
    tmp[local_off[oct]++] = perm[i];
  }
  for (int i = 0; i < n; ++i) perm[lo + i] = tmp[i];

  tree[node_idx].is_leaf = 0;
  int child_idx_local[8];
  for (int oct = 0; oct < 8; ++oct) child_idx_local[oct] = -1;
  for (int oct = 0; oct < 8; ++oct) {
    if (count[oct] == 0) {
      tree[node_idx].child[oct] = -1;
      continue;
    }
    int child_idx = -1;
    #pragma omp critical(fmm3d_tree_push_node)
    {
      h_make_child_node(tree, node_idx, oct);
      child_idx = static_cast<int>(tree.size()) - 1;
    }
    tree[node_idx].child[oct] = child_idx;
    child_idx_local[oct] = child_idx;
  }

  for (int oct = 0; oct < 8; ++oct) {
    if (child_idx_local[oct] == -1) continue;
    int cidx = child_idx_local[oct];
    int clo = offs[oct];
    int chi = offs[oct] + count[oct];
    #pragma omp task default(none) firstprivate(cidx, clo, chi, depth, oct) \
                     shared(tree, perm, bodies, count) \
                     if((depth < HOST_DTT_TASK_DEPTH) && (count[oct] >= HOST_DTT_TASK_MIN_N))
    {
      h_subdivide(tree, perm, bodies, cidx, clo, chi, depth + 1);
    }
  }
  #pragma omp taskwait
  tree[node_idx].ibody = lo;
  tree[node_idx].nbody = n;
}

[[maybe_unused]] static std::vector<TreeNode> h_build_tree(HostBodies &bodies_inout) {
  const int n = static_cast<int>(bodies_inout.size());
  std::vector<TreeNode> tree;
  tree.reserve(std::max(2, 2 * n));
  TreeNode root{};
  root.cx = Real(0.5);
  root.cy = Real(0.5);
  root.cz = Real(0.5);
  root.r = Real(0.5);
  root.parent = -1;
  for (int k = 0; k < 8; ++k) root.child[k] = -1;
  root.ibody = 0;
  root.nbody = n;
  root.is_leaf = 1;
  h_zero_multipole_coeffs(root);
  tree.push_back(root);

  std::vector<int> perm(n);
  for (int i = 0; i < n; ++i) perm[i] = i;
  #pragma omp parallel
  {
    #pragma omp single nowait
    h_subdivide(tree, perm, bodies_inout, 0, 0, n, 0);
  }

  HostBodies sorted(n);
  for (int i = 0; i < n; ++i) {
    const int src = perm[i];
    sorted.x[i] = bodies_inout.x[src];
    sorted.y[i] = bodies_inout.y[src];
    sorted.z[i] = bodies_inout.z[src];
    sorted.q[i] = bodies_inout.q[src];
  }
  bodies_inout = std::move(sorted);
  return tree;
}

static inline int h_morton_octant_at(uint64_t key, int depth, int bits) {
  const int shift = 3 * (bits - depth);
  return static_cast<int>((key >> shift) & 7ull);
}

static void h_subdivide_morton_sorted(std::vector<TreeNode> &tree,
                                      const std::vector<uint64_t> &keys,
                                      int node_idx, int lo, int hi,
                                      int depth, int bits) {
  const int n = hi - lo;
  TreeNode &node = tree[node_idx];
  node.ibody = lo;
  node.nbody = n;
  if (n <= NCRIT || depth >= bits || depth >= MAX_DEPTH) {
    node.is_leaf = 1;
    return;
  }

  int begin[8];
  int end[8];
  for (int oct = 0; oct < 8; ++oct) {
    begin[oct] = hi;
    end[oct] = hi;
  }
  int pos = lo;
  int nonempty = 0;
  while (pos < hi) {
    const int oct = h_morton_octant_at(keys[pos], depth + 1, bits);
    const int b = pos;
    while (pos < hi && h_morton_octant_at(keys[pos], depth + 1, bits) == oct) {
      ++pos;
    }
    begin[oct] = b;
    end[oct] = pos;
    ++nonempty;
  }

  if (nonempty <= 1 && depth + 1 >= bits) {
    node.is_leaf = 1;
    return;
  }

  node.is_leaf = 0;
  for (int oct = 0; oct < 8; ++oct) {
    node.child[oct] = -1;
    if (begin[oct] == hi) continue;
    h_make_child_node(tree, node_idx, oct);
    const int child_idx = static_cast<int>(tree.size()) - 1;
    tree[node_idx].child[oct] = child_idx;
  }
  for (int oct = 0; oct < 8; ++oct) {
    const int child_idx = tree[node_idx].child[oct];
    if (child_idx == -1) continue;
    h_subdivide_morton_sorted(tree, keys, child_idx, begin[oct], end[oct],
                              depth + 1, bits);
  }
}
