#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

static uint64_t rng_state = 123456789ULL;

static uint64_t rng_next() {
    rng_state = rng_state * 1103515245ULL + 12345ULL;
    return rng_state;
}

static int rng_int(int min, int max) {
    uint64_t range = (uint64_t)max - (uint64_t)min + 1;
    return min + (int)(rng_next() % range);
}

static int compare_int(const void* a, const void* b) {
    int x = *(const int*)a;
    int y = *(const int*)b;
    return (x > y) - (x < y);
}

static int sorted_filename(const char* output_file, char* out, size_t out_size) {
    const char* slash = strrchr(output_file, '/');
    const char* dot = strrchr(output_file, '.');
    int written;
    if (dot && (!slash || dot > slash)) {
        int prefix = (int)(dot - output_file);
        written = snprintf(out, out_size, "%.*s.sorted%s", prefix, output_file, dot);
    } else {
        written = snprintf(out, out_size, "%s.sorted", output_file);
    }
    if (written < 0 || (size_t)written >= out_size) {
        fprintf(stderr, "Error: Sorted output path is too long for %s\n", output_file);
        return -1;
    }
    return 0;
}

// Element count, then the ints.
static void save_array(const char* filename, int* data, size_t n) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
        return;
    }

    fwrite(&n, sizeof(size_t), 1, fp);
    fwrite(data, sizeof(int), n, fp);

    fclose(fp);
    printf("Saved %zu elements to %s\n", n, filename);
}

static int* load_array(const char* filename, size_t* n) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for reading\n", filename);
        return NULL;
    }

    if (fread(n, sizeof(size_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Cannot read size from %s\n", filename);
        fclose(fp);
        return NULL;
    }

    int* data = (int*)malloc(sizeof(int) * (*n));
    if (!data) {
        fprintf(stderr, "Error: Cannot allocate memory for %zu elements\n", *n);
        fclose(fp);
        return NULL;
    }

    if (fread(data, sizeof(int), *n, fp) != *n) {
        fprintf(stderr, "Error: Cannot read data from %s\n", filename);
        free(data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    printf("Loaded %zu elements from %s\n", *n, filename);
    return data;
}

static void save_array_text(const char* filename, int* data, size_t n) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
        return;
    }

    fprintf(fp, "%zu\n", n);
    for (size_t i = 0; i < n; i++) {
        fprintf(fp, "%d\n", data[i]);
    }

    fclose(fp);
    printf("Saved %zu elements to %s (text format)\n", n, filename);
}

static void save_array_as(const char* filename, int* data, size_t n, int text_mode) {
    if (text_mode) {
        save_array_text(filename, data, n);
    } else {
        save_array(filename, data, n);
    }
}

static void print_stats(int* data, size_t n) {
    if (n == 0) return;

    int min = data[0], max = data[0];
    long long sum = 0;

    for (size_t i = 0; i < n; i++) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
        sum += data[i];
    }

    printf("Array statistics:\n");
    printf("  Size: %zu\n", n);
    printf("  Min: %d\n", min);
    printf("  Max: %d\n", max);
    printf("  Average: %.2f\n", (double)sum / n);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <size> [output_file] [--text] [--load <input_file>]\n", argv[0]);
        printf("  size: Number of elements to generate\n");
        printf("  output_file: Output file name (default: data.bin)\n");
        printf("  --text: Save in text format instead of binary\n");
        printf("  --load: Load existing array from file\n");
        printf("  A sorted copy is also written when generating (data.bin -> data.sorted.bin)\n");
        printf("\nExamples:\n");
        printf("  %s 1000000                    # Generate 1M elements, save to data.bin\n", argv[0]);
        printf("  %s 1000000 test.bin           # Generate 1M elements, save to test.bin\n", argv[0]);
        printf("  %s 1000000 test.bin --text    # Generate 1M elements, save as text\n", argv[0]);
        printf("  %s 0 --load data.bin          # Load existing array from data.bin\n", argv[0]);
        return 1;
    }

    size_t n = (size_t)atoll(argv[1]);
    const char* output_file = (argc > 2) ? argv[2] : "data.bin";
    int text_mode = 0;
    const char* load_file = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--text") == 0) {
            text_mode = 1;
        } else if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            load_file = argv[i + 1];
            i++;
        }
    }

    int* data = NULL;
    int generated = 0;

    if (load_file) {
        data = load_array(load_file, &n);
        if (!data) {
            return 1;
        }
        print_stats(data, n);
    } else if (n > 0) {
        data = (int*)malloc(sizeof(int) * n);
        if (!data) {
            fprintf(stderr, "Error: Cannot allocate memory for %zu elements\n", n);
            return 1;
        }

        printf("Generating %zu random integers...\n", n);

        for (size_t i = 0; i < n; i++) {
            data[i] = rng_int(-2147483648, 2147483647);
        }

        print_stats(data, n);
        generated = 1;
    } else if (!load_file) {
        fprintf(stderr, "Error: Invalid size %zu\n", n);
        return 1;
    }

    save_array_as(output_file, data, n, text_mode);

    // Saved above, so sorting in place leaves the unsorted file unchanged.
    if (generated) {
        char sorted_file[4096];
        if (sorted_filename(output_file, sorted_file, sizeof(sorted_file)) != 0) {
            free(data);
            return 1;
        }
        printf("Sorting %zu elements...\n", n);
        qsort(data, n, sizeof(int), compare_int);
        save_array_as(sorted_file, data, n, text_mode);
    }

    free(data);
    return 0;
}
