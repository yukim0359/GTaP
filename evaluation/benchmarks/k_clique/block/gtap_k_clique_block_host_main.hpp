#pragma once

#include <cuda_runtime.h>
#include <time.h>
#include <vector>

#include "../k_clique_gpu_orient.cuh"

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err__));        \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

#include "../k_clique_host_graph.hpp"
#include "../k_clique_ncr_host.hpp"

struct GtapKCliqueParsedArgs {
    bool file_mode = false;
    int n = 128;
    int k = 7;
    int edge_probability_percent = 20;
    int validate = 1;
    unsigned seed = 42;
    HostGraph graph;
    const char* graph_path = nullptr;
};

struct GtapKCliqueDeviceBuffers {
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    int* d_edge_src = nullptr;
    int* d_cand_scratch = nullptr;
    unsigned long long* d_bit_scratch = nullptr;
    unsigned long long* d_encode_scratch = nullptr;
    int* d_encode_slot_stack = nullptr;
    unsigned long long* d_recurse_tree_stats = nullptr;

    size_t cand_scratch_ints = 0;
    size_t bit_scratch_words = 0;
    size_t encode_scratch_words = 0;
    int num_workers = 0;
};

struct GtapKCliqueRunResult {
    double preprocess_ms = 0.0;
    double count_phase_ms = 0.0;
    double kernel_ms = 0.0;
    double cpu_ms = 0.0;
    unsigned long long gpu_count = 0ULL;
    unsigned long long cpu_count = 0ULL;
    int scratch_overflow = 0;
#ifdef GTAP_K_STATS
    int candidate_highwater = 0;
    unsigned long long recurse_encoded_calls = 0ULL;
    unsigned long long recurse_encoded_roots = 0ULL;
    unsigned long long recurse_max_encoded_tree_calls = 0ULL;
    unsigned long long task_max_cycles[K_PROFILE_KINDS] = {0ULL};
    int task_max_info[K_PROFILE_KINDS * K_PROFILE_FIELDS] = {0};
#endif
};

struct GtapKCliqueHostHooks {
    bool set_cuda_stack_limit = false;
    const char* profile_name = nullptr;
    bool (*on_count_phase_begin)(GtapKCliqueDeviceBuffers& buffers) = nullptr;
    cudaError_t (*bind_extras)() = nullptr;
    void (*free_extras)() = nullptr;
    void (*print_variant_tuning)() = nullptr;
};

static void gtap_k_clique_free_device_buffers(GtapKCliqueDeviceBuffers& buffers) {
    cudaFree(buffers.d_row_ptr);
    cudaFree(buffers.d_col_idx);
    cudaFree(buffers.d_edge_src);
    cudaFree(buffers.d_cand_scratch);
    cudaFree(buffers.d_bit_scratch);
    cudaFree(buffers.d_encode_scratch);
    cudaFree(buffers.d_encode_slot_stack);
    cudaFree(buffers.d_recurse_tree_stats);
    buffers = GtapKCliqueDeviceBuffers{};
}

static bool gtap_k_clique_parse_args(int argc, char** argv, GtapKCliqueParsedArgs& args) {
    args.file_mode = (argc >= 2 && !is_integer_string(argv[1]));
    if (args.file_mode) {
        args.graph_path = argv[1];
        if (!load_graph_file(argv[1], args.graph)) return false;
        args.n = args.graph.n;
        if (argc >= 3 && is_integer_string(argv[2])) {
            args.k = parse_int_arg(argv, argc, 2, 7);
            args.validate = parse_int_arg(argv, argc, argc >= 5 ? 4 : 3, 0);
        } else {
            args.k = 7;
            args.validate = parse_int_arg(argv, argc, 2, 0);
        }
    } else {
        args.n = parse_int_arg(argv, argc, 1, 128);
        args.k = parse_int_arg(argv, argc, 2, 7);
        args.edge_probability_percent = parse_int_arg(argv, argc, 3, 20);
        args.seed = (unsigned)parse_int_arg(argv, argc, 5, 42);
        args.validate = parse_int_arg(argv, argc, 6, 1);
        if (args.edge_probability_percent < 0 || args.edge_probability_percent > 100) {
            fprintf(stderr, "edge_probability_percent must be in [0, 100], got %d\n",
                    args.edge_probability_percent);
            return false;
        }
        args.graph = generate_random_graph(args.n, args.edge_probability_percent, args.seed);
    }

    if (args.n < 1) {
        fprintf(stderr, "n must be positive, got %d\n", args.n);
        return false;
    }
    if (args.k < 3 || args.k > args.n) {
        fprintf(stderr, "k must be in [3, n], got k=%d\n", args.k);
        return false;
    }
    if (args.k - 3 > GTAP_K_BIT_BUFFER_COUNT) {
        fprintf(stderr,
                "GTAP_K_BIT_BUFFER_COUNT=%d is too small for k=%d; need at least %d\n",
                GTAP_K_BIT_BUFFER_COUNT,
                args.k,
                args.k - 3);
        return false;
    }
    return true;
}

static bool gtap_k_clique_preprocess(
    const GtapKCliqueParsedArgs& args,
    const GtapKCliqueHostHooks& hooks,
    HostGraph& exec_graph,
    GtapOrientMode& orient_mode,
    GtapKCliqueDeviceBuffers& buffers,
    double& preprocess_ms) {
    timespec preprocess_start, preprocess_stop;
    clock_gettime(CLOCK_MONOTONIC, &preprocess_start);

    orient_mode = parse_gtap_orient_from_env(GTAP_DEFAULT_ORIENT);

    size_t num_oriented_edges = 0;
    CUDA_CHECK(gtap_gpu_orient_csr(
        args.graph.row_ptr.data(),
        args.graph.col_idx.data(),
        args.graph.n,
        args.graph.undirected_edges,
        orient_mode,
        &buffers.d_row_ptr,
        &buffers.d_col_idx,
        &buffers.d_edge_src,
        &num_oriented_edges,
        args.validate ? &exec_graph : nullptr));
    if (!args.validate) {
        exec_graph.n = args.graph.n;
        exec_graph.undirected_edges = args.graph.undirected_edges;
        exec_graph.oriented = true;
        exec_graph.col_idx.resize(num_oriented_edges);
    }

    // Block runtime: one scratch set per CUDA block (blockIdx.x).
    buffers.num_workers = GTAP_PIVOT_BLOCK_GRID_SIZE;
    const int words_per_worker = (GTAP_K_MAX_CANDIDATES + 63) >> 6;
    buffers.cand_scratch_ints =
        (size_t)buffers.num_workers * (size_t)GTAP_K_MAX_CANDIDATES;
    buffers.bit_scratch_words =
        (size_t)buffers.num_workers * (size_t)GTAP_K_BIT_BUFFER_COUNT * (size_t)words_per_worker;
    buffers.encode_scratch_words =
        ((size_t)GTAP_K_ENCODE_SLOTS + (size_t)buffers.num_workers) *
        (size_t)GTAP_K_MAX_CANDIDATES *
        (size_t)words_per_worker;

    std::vector<int> encode_slot_stack((size_t)GTAP_K_ENCODE_SLOTS);
    for (int i = 0; i < GTAP_K_ENCODE_SLOTS; ++i) {
        encode_slot_stack[(size_t)i] = i;
    }

    CUDA_CHECK(cudaMalloc(&buffers.d_cand_scratch, buffers.cand_scratch_ints * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &buffers.d_bit_scratch, buffers.bit_scratch_words * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(
        &buffers.d_encode_scratch, buffers.encode_scratch_words * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(
        &buffers.d_encode_slot_stack, encode_slot_stack.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(
        buffers.d_encode_slot_stack,
        encode_slot_stack.data(),
        encode_slot_stack.size() * sizeof(int),
        cudaMemcpyHostToDevice));
    if (hooks.set_cuda_stack_limit) {
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, GTAP_K_CUDA_STACK_SIZE));
    }
#ifdef GTAP_K_STATS
    CUDA_CHECK(cudaMalloc(
        &buffers.d_recurse_tree_stats,
        (size_t)buffers.num_workers * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(
        buffers.d_recurse_tree_stats,
        0,
        (size_t)buffers.num_workers * sizeof(unsigned long long)));
#else
    (void)buffers.d_recurse_tree_stats;
#endif

    clock_gettime(CLOCK_MONOTONIC, &preprocess_stop);
    preprocess_ms = elapsed_ms(preprocess_start, preprocess_stop);
    return true;
}

static bool gtap_k_clique_run_count_phase(
    const GtapKCliqueParsedArgs& args,
    const GtapKCliqueHostHooks& hooks,
    GtapKCliqueDeviceBuffers& buffers,
    int num_exec_edges,
    GtapKCliqueRunResult& result) {
    timespec count_phase_start, count_phase_stop;
    clock_gettime(CLOCK_MONOTONIC, &count_phase_start);

    if (hooks.on_count_phase_begin != nullptr && !hooks.on_count_phase_begin(buffers)) {
        if (hooks.free_extras != nullptr) hooks.free_extras();
        gtap_k_clique_free_device_buffers(buffers);
        return false;
    }

    {
        const size_t task_stride = gtap_host_task_data_stride();
        const size_t max_tasks_global =
            (size_t)GTAP_PIVOT_BLOCK_MAX_TASKS_PER_BLOCK *
            (size_t)GTAP_PIVOT_BLOCK_GRID_SIZE;
        const size_t task_data_bytes = task_stride * max_tasks_global;
        const size_t header_bytes = (size_t)40 * max_tasks_global;
        const double pool_mib = task_data_bytes / (1024.0 * 1024.0);
        fprintf(stderr,
                "GTAP block runtime pool: MAX_TASKS_PER_BLOCK=%d GRID=%d "
                "task_stride=%zu max_tasks_global=%zu task_data~%.1f MiB "
                "headers~%.1f MiB\n",
                GTAP_PIVOT_BLOCK_MAX_TASKS_PER_BLOCK,
                GTAP_PIVOT_BLOCK_GRID_SIZE,
                task_stride,
                max_tasks_global,
                pool_mib,
                header_bytes / (1024.0 * 1024.0));
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            fprintf(stderr,
                    "GPU mem before gtap_initialize: free %.1f / total %.1f GiB\n",
                    free_bytes / (1024.0 * 1024.0 * 1024.0),
                    total_bytes / (1024.0 * 1024.0 * 1024.0));
        }
        fprintf(stderr,
                "gtap_initialize: allocating + zeroing ~%.1f GiB (may take 1-2 min)...\n",
                (pool_mib + header_bytes / (1024.0 * 1024.0) + 128.0) / 1024.0);
        fflush(stderr);
    }

    timespec init_start, init_stop;
    clock_gettime(CLOCK_MONOTONIC, &init_start);
    gtap_block_config config{
        .grid_size = GTAP_PIVOT_BLOCK_GRID_SIZE,
        .max_tasks_per_block = GTAP_PIVOT_BLOCK_MAX_TASKS_PER_BLOCK,
    };
    cudaError_t err = gtap_initialize(config);
    clock_gettime(CLOCK_MONOTONIC, &init_stop);
    fprintf(stderr, "gtap_initialize done in %.1f s\n",
            elapsed_ms(init_start, init_stop) / 1000.0);
    fflush(stderr);
    if (err != cudaSuccess) {
        fprintf(stderr, "gtap_initialize failed: %s\n", cudaGetErrorString(err));
        if (hooks.free_extras != nullptr) hooks.free_extras();
        gtap_k_clique_free_device_buffers(buffers);
        return false;
    }

    const int encode_slot_top_init = GTAP_K_ENCODE_SLOTS;
    unsigned long long zero = 0ULL;
    int scratch_overflow = 0;
    int encode_slot_lock_init = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(g_answer, &zero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_scratch_overflow, &scratch_overflow, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_encode_slot_lock, &encode_slot_lock_init, sizeof(int)));
#ifdef GTAP_K_STATS
    int candidate_highwater = 0;
    unsigned long long profile_cycles_zero[K_PROFILE_KINDS] = {0ULL};
    int profile_info_zero[K_PROFILE_KINDS * K_PROFILE_FIELDS] = {0};
    unsigned long long recurse_zero = 0ULL;
    CUDA_CHECK(cudaMemcpyToSymbol(g_candidate_highwater, &candidate_highwater, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        g_task_max_cycles,
        profile_cycles_zero,
        sizeof(profile_cycles_zero)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        g_task_max_info,
        profile_info_zero,
        sizeof(profile_info_zero)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_recurse_encoded_calls, &recurse_zero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_recurse_encoded_roots, &recurse_zero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        g_recurse_max_encoded_tree_calls, &recurse_zero, sizeof(unsigned long long)));
#endif
    CUDA_CHECK(cudaMemcpyToSymbol(g_encode_slot_top, &encode_slot_top_init, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_num_vertices, &args.n, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_num_edges, &num_exec_edges, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_clique_k, &args.k, sizeof(int)));
    err = bind_graph(
        buffers.d_row_ptr,
        buffers.d_col_idx,
        buffers.d_edge_src,
        buffers.d_cand_scratch,
        buffers.d_bit_scratch,
        buffers.d_encode_scratch,
        buffers.d_encode_slot_stack);
    if (err != cudaSuccess) {
        fprintf(stderr, "bind_graph failed: %s\n", cudaGetErrorString(err));
        if (hooks.free_extras != nullptr) hooks.free_extras();
        gtap_k_clique_free_device_buffers(buffers);
        gtap_finalize();
        return false;
    }
    if (hooks.bind_extras != nullptr) {
        err = hooks.bind_extras();
        if (err != cudaSuccess) {
            fprintf(stderr, "bind_extras failed: %s\n", cudaGetErrorString(err));
            if (hooks.free_extras != nullptr) hooks.free_extras();
            gtap_k_clique_free_device_buffers(buffers);
            gtap_finalize();
            return false;
        }
    }
#ifdef GTAP_K_STATS
    err = bind_recurse_stats(buffers.d_recurse_tree_stats);
    if (err != cudaSuccess) {
        fprintf(stderr, "bind_recurse_stats failed: %s\n", cudaGetErrorString(err));
        if (hooks.free_extras != nullptr) hooks.free_extras();
        gtap_k_clique_free_device_buffers(buffers);
        gtap_finalize();
        return false;
    }
#endif

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    fprintf(stderr, "launching exec_kernel_k<<<GRID=%d, BLOCK=%d>>> ...\n",
            GTAP_PIVOT_BLOCK_GRID_SIZE, GTAP_BLOCK_SIZE);
    fflush(stderr);
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(gtap_launch(exec_kernel_k));
    CUDA_CHECK(gtap_synchronize());
    CUDA_CHECK(cudaGetLastError());
    fprintf(stderr, "exec_kernel_k finished\n");
    fflush(stderr);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    clock_gettime(CLOCK_MONOTONIC, &count_phase_stop);

    result.count_phase_ms = elapsed_ms(count_phase_start, count_phase_stop);
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    result.kernel_ms = kernel_ms;
    CUDA_CHECK(cudaMemcpyFromSymbol(&result.gpu_count, g_answer, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        &result.scratch_overflow, g_scratch_overflow, sizeof(int)));
#ifdef GTAP_K_STATS
    CUDA_CHECK(cudaMemcpyFromSymbol(
        &result.candidate_highwater, g_candidate_highwater, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        result.task_max_cycles,
        g_task_max_cycles,
        sizeof(result.task_max_cycles)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        result.task_max_info,
        g_task_max_info,
        sizeof(result.task_max_info)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        &result.recurse_encoded_calls, g_recurse_encoded_calls, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        &result.recurse_encoded_roots, g_recurse_encoded_roots, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        &result.recurse_max_encoded_tree_calls,
        g_recurse_max_encoded_tree_calls,
        sizeof(unsigned long long)));
    std::vector<unsigned long long> h_recurse_tree_encoded((size_t)buffers.num_workers, 0ULL);
    CUDA_CHECK(cudaMemcpy(
        h_recurse_tree_encoded.data(),
        buffers.d_recurse_tree_stats,
        (size_t)buffers.num_workers * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    for (int worker = 0; worker < buffers.num_workers; ++worker) {
        if (h_recurse_tree_encoded[(size_t)worker] > result.recurse_max_encoded_tree_calls) {
            result.recurse_max_encoded_tree_calls = h_recurse_tree_encoded[(size_t)worker];
        }
    }
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return true;
}

static void gtap_k_clique_print_results(
    const GtapKCliqueParsedArgs& args,
    const GtapKCliqueHostHooks& hooks,
    const HostGraph& exec_graph,
    GtapOrientMode orient_mode,
    const GtapKCliqueDeviceBuffers& buffers,
    const GtapKCliqueRunResult& result) {
    if (args.file_mode) {
        printf("k-clique graph file: %s n=%d k=%d edges=%zu mode=%s-oriented (GTAP block)\n",
               args.graph_path, args.n, args.k, args.graph.undirected_edges,
               gtap_orient_mode_name(orient_mode));
    } else {
        printf("k-clique random graph: n=%d k=%d p=%d%% seed=%u edges=%zu mode=%s-oriented (GTAP block)\n",
               args.n, args.k, args.edge_probability_percent, args.seed,
               args.graph.undirected_edges, gtap_orient_mode_name(orient_mode));
    }
    printf("oriented_edges: %zu\n", exec_graph.col_idx.size());
    printf("GTAP_BLOCK_SIZE: %d\n", GTAP_BLOCK_SIZE);
    printf("GTAP_GRID_SIZE: %d\n", GTAP_PIVOT_BLOCK_GRID_SIZE);
    printf("GTAP_K_RANGE_CUTOFF: %d\n", GTAP_K_RANGE_CUTOFF);
    printf("GTAP_K_MAX_CANDIDATES: %d\n", GTAP_K_MAX_CANDIDATES);
    printf("GTAP_K_BIT_BUFFER_COUNT: %d\n", GTAP_K_BIT_BUFFER_COUNT);
    printf("GTAP_K_CUDA_STACK_SIZE: %d\n", GTAP_K_CUDA_STACK_SIZE);
    if (hooks.print_variant_tuning != nullptr) {
        hooks.print_variant_tuning();
    }
    printf("scratch blocks: %d\n", buffers.num_workers);
    printf("candidate scratch: %zu ints\n", buffers.cand_scratch_ints);
    printf("bit scratch: %zu uint64 words\n", buffers.bit_scratch_words);
    printf("encode slots: %d\n", GTAP_K_ENCODE_SLOTS);
    printf("encode scratch: %zu uint64 words (slots + per-block)\n", buffers.encode_scratch_words);
#ifdef GTAP_K_STATS
    printf("GTaP max candidate length: %d / %d\n",
           result.candidate_highwater, GTAP_K_MAX_CANDIDATES);
#endif
    printf("GTaP count: %llu\n", result.gpu_count);
    if (args.validate) {
        printf("CPU count:  %llu\n", result.cpu_count);
        printf("Validation: %s\n",
               (!result.scratch_overflow && result.gpu_count == result.cpu_count)
                   ? "PASSED"
                   : "FAILED");
    } else {
        printf("Validation: %s\n",
               result.scratch_overflow ? "FAILED_SCRATCH_OVERFLOW" : "SKIPPED");
    }
    printf("GTaP count phase time: %.3f ms\n", result.count_phase_ms);
    printf("GTaP kernel time: %.3f ms\n", result.kernel_ms);
    printf("GTaP preprocess+transfer time: %.3f ms\n", result.preprocess_ms);
    if (args.validate) printf("CPU execution time:  %.3f ms\n", result.cpu_ms);

#ifdef GTAP_ENABLE_PROFILING
    if (hooks.profile_name != nullptr) {
        char profile_directory[512];
        snprintf(profile_directory, sizeof(profile_directory), "./profile/%s",
                 hooks.profile_name);
        gtap_export_profile({
            .output_directory = profile_directory,
            .overwrite = true,
        });
    }
#endif
}

static int gtap_k_clique_host_main(int argc, char** argv, const GtapKCliqueHostHooks& hooks) {
    GtapKCliqueParsedArgs args;
    if (!gtap_k_clique_parse_args(argc, argv, args)) {
        return 1;
    }

    HostGraph exec_graph;
    GtapOrientMode orient_mode = GTAP_ORIENT_DEGEN;
    GtapKCliqueDeviceBuffers buffers;
    GtapKCliqueRunResult result;

    if (!gtap_k_clique_preprocess(args, hooks, exec_graph, orient_mode, buffers, result.preprocess_ms)) {
        return 1;
    }

    const int num_exec_edges = (int)exec_graph.col_idx.size();
    if (!gtap_k_clique_run_count_phase(args, hooks, buffers, num_exec_edges, result)) {
        return 1;
    }

    if (args.validate) {
        timespec cpu_start, cpu_stop;
        clock_gettime(CLOCK_MONOTONIC, &cpu_start);
        result.cpu_count = cpu_count_k_oriented(exec_graph, args.k);
        clock_gettime(CLOCK_MONOTONIC, &cpu_stop);
        result.cpu_ms = elapsed_ms(cpu_start, cpu_stop);
    }

    gtap_k_clique_print_results(args, hooks, exec_graph, orient_mode, buffers, result);

    if (hooks.free_extras != nullptr) hooks.free_extras();
    gtap_k_clique_free_device_buffers(buffers);
    gtap_finalize();
    return (!result.scratch_overflow && (!args.validate || result.gpu_count == result.cpu_count))
               ? 0
               : 2;
}
