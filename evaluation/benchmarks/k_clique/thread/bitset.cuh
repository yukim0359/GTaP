#pragma once

#include "device_state.cuh"

__device__ __forceinline__ bool is_bit_set(const unsigned long long* bits, int idx) {
    return (bits[idx >> 6] >> (idx & 63)) & 1ULL;
}

__device__ __forceinline__ void clear_bits(unsigned long long* bits, int words) {
    for (int w = 0; w < words; ++w) bits[w] = 0ULL;
}

__device__ __forceinline__ void init_full_bitset(unsigned long long* bits, int len) {
    int words = words_for_len(len);
    for (int w = 0; w < words; ++w) bits[w] = ~0ULL;
    int rem = len & 63;
    if (rem != 0) bits[words - 1] = (1ULL << rem) - 1ULL;
}

__device__ __forceinline__ int popcount_bits(const unsigned long long* bits, int words) {
    int total = 0;
    for (int w = 0; w < words; ++w) total += __popcll(bits[w]);
    return total;
}

__device__ __forceinline__ unsigned long long valid_bits_mask_for_word(int word, int words, int cand_count) {
    if (word + 1 != words) return ~0ULL;
    int rem = cand_count & 63;
    return rem == 0 ? ~0ULL : ((1ULL << rem) - 1ULL);
}

__device__ __forceinline__ int next_set_bit_from(
        const unsigned long long* bits, int words, int start, int* out_word) {
    int word = start >> 6;
    int bit = start & 63;
    if (word >= words) return -1;

    unsigned long long mask = bits[word] & (~0ULL << bit);
    while (mask == 0ULL) {
        ++word;
        if (word >= words) return -1;
        mask = bits[word];
    }

    *out_word = word;
    return (word << 6) + (__ffsll((long long)mask) - 1);
}
