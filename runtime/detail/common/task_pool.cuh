#pragma once

namespace gtap::detail {

// High bit is the ring lap. The task id occupies the low 31 bits, so one
// relaxed store publishes both and a reuse sees this id or an older slot.
__device__ __forceinline__ int pack_task_id_slot(int id, int ticket, int pool_size) {
    const unsigned ring_lap =
        static_cast<unsigned>(ticket) / static_cast<unsigned>(pool_size);
    return static_cast<int>(
        (static_cast<unsigned>(id) & 0x7fffffffu) | ((ring_lap & 1u) << 31));
}

__device__ __forceinline__ bool ring_lap_matches(int packed, int ticket, int pool_size) {
    const unsigned ring_lap =
        static_cast<unsigned>(ticket) / static_cast<unsigned>(pool_size);
    return (static_cast<unsigned>(packed) >> 31) == (ring_lap & 1u);
}

__device__ __forceinline__ int unpack_task_id_slot(int packed) {
    return static_cast<int>(static_cast<unsigned>(packed) & 0x7fffffffu);
}

__host__ __device__ __forceinline__ int max_safe_release_ticket(
    int alloc_pos,
    int pool_size
) {
    // Borrows below pool_size take a static tid and never read the ring, so the
    // first recycle lap (tickets pool_size .. 2*pool_size-1) is always safe.
    const int covered = alloc_pos > pool_size ? alloc_pos : pool_size;
    return pool_size - 1 + covered;
}

}  // namespace gtap::detail
