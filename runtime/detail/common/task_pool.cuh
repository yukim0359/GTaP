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

}  // namespace gtap::detail
