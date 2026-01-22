#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#ifdef _WIN32
#include <direct.h>
#endif

#ifdef PROFILE

static inline void ensure_directory_exists(const char* path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        if (errno == ENOENT) {
            // Directory doesn't exist, create it
            #ifdef _WIN32
                _mkdir(path);
            #else
                mkdir(path, 0755);
            #endif
        }
    }
}

static inline void save_block_having_task_time_to_csv(long long* having_task_time_data, int* indices, long long min_time, const char* app_name) {
    ensure_directory_exists("./profile");
    char timeline_path[256];
    snprintf(timeline_path, sizeof(timeline_path), "./profile/%s_block_timeline_having_task.csv", app_name);
    FILE* timeline_file = fopen(timeline_path, "w");
    if (!timeline_file) {
        printf("Error: Could not create %s_block_timeline_having_task.csv file\n", app_name);
        return;
    }
    fprintf(timeline_file, "block_id,time_index,timestamp_ns,relative_time_ms,state,state_description\n");
    int totalBlocks = GTAP_GRID_SIZE;
    for (int block = 0; block < totalBlocks; block++) {
        int samples = indices[block];
        if (samples == 0) continue;
        struct TimeEntry { long long timestamp; int index; int state; };
        struct TimeEntry* entries = (struct TimeEntry*)malloc(sizeof(struct TimeEntry) * samples);
        int entry_count = 0;
        for (int i = 0; i < samples; i++) {
            long long time = having_task_time_data[block * MAX_PROFILE_DATA + i];
            if (time > 0) {
                entries[entry_count].timestamp = time;
                entries[entry_count].index = i;
                entries[entry_count].state = (i % 2 == 0) ? 1 : 0; // even:start(having), odd:end
                entry_count++;
            }
        }
        for (int i = 0; i < entry_count - 1; i++) {
            for (int j = i + 1; j < entry_count; j++) {
                if (entries[i].timestamp > entries[j].timestamp) {
                    struct TimeEntry temp = entries[i];
                    entries[i] = entries[j];
                    entries[j] = temp;
                }
            }
        }
        for (int i = 0; i < entry_count; i++) {
            double relative_time_ms = (entries[i].timestamp - min_time) / 1000000.0;
            const char* state_desc = (entries[i].state == 1) ? "Having" : "NotHaving";
            fprintf(timeline_file, "%d,%d,%lld,%.6f,%d,%s\n",
                    block, entries[i].index, entries[i].timestamp,
                    relative_time_ms, entries[i].state, state_desc);
        }
        free(entries);
    }
    fclose(timeline_file);
    printf("\nBlock timeline saved to: %s_block_timeline_having_task.csv\n", app_name);

    char stats_path[256];
    snprintf(stats_path, sizeof(stats_path), "./profile/%s_block_statistics_having_task.csv", app_name);
    FILE* stats_file = fopen(stats_path, "w");
    if (!stats_file) {
        printf("Error: Could not create %s_block_statistics_having_task.csv file\n", app_name);
        return;
    }
    fprintf(stats_file, "block_id,total_samples,first_activity_ms,last_activity_ms,duration_ms\n");
    for (int block = 0; block < totalBlocks; block++) {
        int samples = indices[block];
        if (samples == 0) {
            fprintf(stats_file, "%d,0,0.0,0.0,0.0\n", block);
            continue;
        }
        int having = 0, not_having = 0;
        long long first_time = LLONG_MAX, last_time = 0;
        for (int i = 0; i < samples; i++) {
            long long time = having_task_time_data[block * MAX_PROFILE_DATA + i];
            if (time > 0) {
                if (time < first_time) first_time = time;
                if (time > last_time) last_time = time;
                if (i % 2 == 0) having++; else not_having++;
            }
        }
        double first_activity_ms = (first_time - min_time) / 1000000.0;
        double last_activity_ms = (last_time - min_time) / 1000000.0;
        double duration_ms = (last_time - first_time) / 1000000.0;
        fprintf(stats_file, "%d,%d,%.6f,%.6f,%.6f\n",
                block, having, first_activity_ms, last_activity_ms, duration_ms);
    }
    fclose(stats_file);
    printf("Block statistics saved to: %s_block_statistics_having_task.csv\n", app_name);
}

static inline void visualize_having_task_time(const char* app_name) {
    printf("\n=== Block Having-Task Time Visualization ===\n");
    int totalBlocks = GTAP_GRID_SIZE;
    int* d_indices;
    cudaMalloc(&d_indices, sizeof(int) * totalBlocks);
    get_final_having_task_time_indices<<<totalBlocks, 1>>>(d_indices);
    cudaDeviceSynchronize();

    int* indices = (int*)malloc(sizeof(int) * totalBlocks);
    cudaMemcpy(indices, d_indices, sizeof(int) * totalBlocks, cudaMemcpyDeviceToHost);

    long long* data = (long long*)malloc(sizeof(long long) * totalBlocks * MAX_PROFILE_DATA);
    if (!data) {
        printf("Error: Memory allocation failed for having_task_time_data\n");
        free(indices);
        cudaFree(d_indices);
        return;
    }
    cudaError_t st = get_having_task_time_data(data);
    if (st != cudaSuccess) {
        printf("Error getting having-task time data: %s\n", cudaGetErrorString(st));
        free(indices);
        free(data);
        cudaFree(d_indices);
        return;
    }

    long long min_time = LLONG_MAX, max_time = 0;
    int total_samples = 0;
    for (int block = 0; block < totalBlocks; block++) {
        for (int i = 0; i < indices[block]; i++) {
            long long t = data[block * MAX_PROFILE_DATA + i];
            if (t > 0) {
                if (t < min_time) min_time = t;
                if (t > max_time) max_time = t;
                total_samples++;
            }
        }
    }
    if (total_samples == 0) {
        printf("No having-task time data recorded.\n");
        free(indices);
        free(data);
        cudaFree(d_indices);
        return;
    }

    printf("Total samples: %d\n", total_samples);
    printf("Time range: %.3f ms to %.3f ms\n", min_time / 1000000.0, max_time / 1000000.0);
    printf("Duration: %.3f ms\n", (max_time - min_time) / 1000000.0);

    int max_blocks_to_show = (totalBlocks < 10) ? totalBlocks : 10;
    int timeline_width = 80;
    printf("\nTimeline visualization (first %d blocks):\n", max_blocks_to_show);
    printf("Block |");
    for (int i = 0; i < timeline_width; i++) printf((i % 10 == 0) ? "|" : "-");
    printf("\n");
    for (int block = 0; block < max_blocks_to_show; block++) {
        printf("%5d |", block);
        char timeline[81];
        for (int i = 0; i < timeline_width; i++) timeline[i] = ' ';
        timeline[timeline_width] = '\0';
        for (int i = 0; i < indices[block]; i++) {
            long long t = data[block * MAX_PROFILE_DATA + i];
            if (t > 0) {
                int pos = (int)((t - min_time) * timeline_width / (max_time - min_time));
                if (pos >= 0 && pos < timeline_width) timeline[pos] = (i % 2 == 0) ? 'H' : 'N';
            }
        }
        printf("%s\n", timeline);
    }

    save_block_having_task_time_to_csv(data, indices, min_time, app_name);
    free(indices);
    free(data);
    cudaFree(d_indices);
}

static inline void save_block_working_time_to_csv(long long* working_time_data, int* indices, int* counts_data, long long min_time, const char* app_name) {
    ensure_directory_exists("./profile");
    char timeline_path[256];
    snprintf(timeline_path, sizeof(timeline_path), "./profile/%s_block_timeline_working.csv", app_name);
    FILE* timeline_file = fopen(timeline_path, "w");
    if (!timeline_file) {
        printf("Error: Could not create %s_block_timeline_working.csv file\n", app_name);
        return;
    }
    fprintf(timeline_file, "block_id,time_index,timestamp_ns,relative_time_ms,state,state_description,tasks_in_batch\n");
    int totalBlocks = GTAP_GRID_SIZE;
    for (int block = 0; block < totalBlocks; block++) {
        int samples = indices[block];
        if (samples == 0) continue;
        struct TimeEntry { long long timestamp; int index; int state; int count; };
        struct TimeEntry* entries = (struct TimeEntry*)malloc(sizeof(struct TimeEntry) * samples);
        int entry_count = 0;
        for (int i = 0; i < samples; i++) {
            long long time = working_time_data[block * MAX_PROFILE_DATA + i];
            if (time > 0) {
                entries[entry_count].timestamp = time;
                entries[entry_count].index = i;
                entries[entry_count].state = (i % 2 == 0) ? 1 : 0; // even:pre, odd:post
                entries[entry_count].count = counts_data[block * MAX_PROFILE_DATA + i];
                entry_count++;
            }
        }
        for (int i = 0; i < entry_count - 1; i++) {
            for (int j = i + 1; j < entry_count; j++) {
                if (entries[i].timestamp > entries[j].timestamp) {
                    struct TimeEntry temp = entries[i];
                    entries[i] = entries[j];
                    entries[j] = temp;
                }
            }
        }
        for (int i = 0; i < entry_count; i++) {
            double relative_time_ms = (entries[i].timestamp - min_time) / 1000000.0;
            const char* state_desc = (entries[i].state == 1) ? "Working" : "NotWorking";
            fprintf(timeline_file, "%d,%d,%lld,%.6f,%d,%s,%d\n",
                    block, entries[i].index, entries[i].timestamp,
                    relative_time_ms, entries[i].state, state_desc, entries[i].count);
        }
        free(entries);
    }
    fclose(timeline_file);
    printf("\nBlock timeline saved to: %s_block_timeline_working.csv\n", app_name);

    char stats_path2[256];
    snprintf(stats_path2, sizeof(stats_path2), "./profile/%s_block_statistics_working.csv", app_name);
    FILE* stats_file = fopen(stats_path2, "w");
    if (!stats_file) {
        printf("Error: Could not create %s_block_statistics_working.csv file\n", app_name);
        return;
    }
    fprintf(stats_file, "block_id,total_samples,first_activity_ms,last_activity_ms,duration_ms,total_tasks\n");
    for (int block = 0; block < totalBlocks; block++) {
        int samples = indices[block];
        if (samples == 0) {
            fprintf(stats_file, "%d,0,0.0,0.0,0.0,0\n", block);
            continue;
        }
        int pre = 0, post = 0, total_tasks = 0;
        long long first_time = LLONG_MAX, last_time = 0;
        for (int i = 0; i < samples; i++) {
            long long time = working_time_data[block * MAX_PROFILE_DATA + i];
            if (time > 0) {
                if (time < first_time) first_time = time;
                if (time > last_time) last_time = time;
                if (i % 2 == 0) pre++; else { post++; total_tasks += counts_data[block * MAX_PROFILE_DATA + i]; }
            }
        }
        double first_activity_ms = (first_time - min_time) / 1000000.0;
        double last_activity_ms = (last_time - min_time) / 1000000.0;
        double duration_ms = (last_time - first_time) / 1000000.0;
        fprintf(stats_file, "%d,%d,%.6f,%.6f,%.6f,%d\n",
                block, pre, first_activity_ms, last_activity_ms, duration_ms, total_tasks);
    }
    fclose(stats_file);
    printf("Block statistics saved to: %s_block_statistics_working.csv\n", app_name);
}

static inline void visualize_working_time(const char* app_name) {
    printf("\n=== Block Working Time Visualization ===\n");
    int totalBlocks = GTAP_GRID_SIZE;
    int* d_indices;
    cudaMalloc(&d_indices, sizeof(int) * totalBlocks);
    get_final_working_time_indices<<<totalBlocks, 1>>>(d_indices);
    cudaDeviceSynchronize();

    int* indices = (int*)malloc(sizeof(int) * totalBlocks);
    cudaMemcpy(indices, d_indices, sizeof(int) * totalBlocks, cudaMemcpyDeviceToHost);

    long long* working_time_data = (long long*)malloc(sizeof(long long) * totalBlocks * MAX_PROFILE_DATA);
    int* counts_data = (int*)malloc(sizeof(int) * totalBlocks * MAX_PROFILE_DATA);
    if (!working_time_data || !counts_data) {
        printf("Error: Memory allocation failed for working data\n");
        free(indices);
        free(working_time_data);
        free(counts_data);
        cudaFree(d_indices);
        return;
    }
    cudaError_t st1 = get_working_time_data(working_time_data);
    // Block runtime doesn't have tasks_processed_count, so initialize to 0
    memset(counts_data, 0, sizeof(int) * totalBlocks * MAX_PROFILE_DATA);
    if (st1 != cudaSuccess) {
        printf("Error getting working data: %s\n", cudaGetErrorString(st1));
        free(indices);
        free(working_time_data);
        free(counts_data);
        cudaFree(d_indices);
        return;
    }

    long long min_time = LLONG_MAX, max_time = 0;
    int total_samples = 0;
    for (int block = 0; block < totalBlocks; block++) {
        for (int i = 0; i < indices[block]; i++) {
            long long t = working_time_data[block * MAX_PROFILE_DATA + i];
            if (t > 0) {
                if (t < min_time) min_time = t;
                if (t > max_time) max_time = t;
                total_samples++;
            }
        }
    }
    if (total_samples == 0) {
        printf("No working time data recorded.\n");
        free(indices);
        free(working_time_data);
        free(counts_data);
        cudaFree(d_indices);
        return;
    }

    printf("Total samples: %d\n", total_samples);
    printf("Time range: %.3f ms to %.3f ms\n", min_time / 1000000.0, max_time / 1000000.0);
    printf("Duration: %.3f ms\n", (max_time - min_time) / 1000000.0);

    int max_blocks_to_show = (totalBlocks < 10) ? totalBlocks : 10;
    int timeline_width = 80;
    printf("\nTimeline visualization (first %d blocks):\n", max_blocks_to_show);
    printf("Block |");
    for (int i = 0; i < timeline_width; i++) printf((i % 10 == 0) ? "|" : "-");
    printf("\n");
    for (int block = 0; block < max_blocks_to_show; block++) {
        printf("%5d |", block);
        char timeline[81];
        for (int i = 0; i < timeline_width; i++) timeline[i] = ' ';
        timeline[timeline_width] = '\0';
        for (int i = 0; i < indices[block]; i++) {
            long long t = working_time_data[block * MAX_PROFILE_DATA + i];
            if (t > 0) {
                int pos = (int)((t - min_time) * timeline_width / (max_time - min_time));
                if (pos >= 0 && pos < timeline_width) timeline[pos] = (i % 2 == 0) ? 'W' : 'N'; // Working / NotWorking
            }
        }
        printf("%s\n", timeline);
    }

    save_block_working_time_to_csv(working_time_data, indices, counts_data, min_time, app_name);
    free(indices);
    free(working_time_data);
    free(counts_data);
    cudaFree(d_indices);
}

void visualize_profile(const char* app_name) {
    visualize_having_task_time(app_name);
    visualize_working_time(app_name);
}

#endif
