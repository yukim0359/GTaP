BENCHMARK_NAME := binary_tree

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJ_DIR   ?= /work/gc64/c64099/gtap

CUDA_PATH  ?= /work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9/cuda
CUDA_ARCH  ?= sm_90

CXX_BIN    ?= $(PROJ_DIR)/clang-gtap/build/bin/clang++
GT_INC     ?= $(PROJ_DIR)/runtime

# ---- Inputs / Outputs ----
SRC_GTAP_THREAD := $(SCRIPT_DIR)/gtap_thread_$(BENCHMARK_NAME).cu
SRC_GTAP_BLOCK  := $(SCRIPT_DIR)/gtap_block_$(BENCHMARK_NAME).cu
SRC_OMP         := $(SCRIPT_DIR)/omp_$(BENCHMARK_NAME).cpp
SRC_GTAP_THREAD_SEQMEM := $(SCRIPT_DIR)/gtap_thread_$(BENCHMARK_NAME)_seqmem.cu
SRC_GTAP_BLOCK_SEQMEM  := $(SCRIPT_DIR)/gtap_block_$(BENCHMARK_NAME)_seqmem.cu
SRC_OMP_SEQMEM         := $(SCRIPT_DIR)/omp_$(BENCHMARK_NAME)_seqmem.cpp
SRC_OMP_CUDA    := $(SCRIPT_DIR)/omp_cuda_$(BENCHMARK_NAME).cu
SRC_SEQ         := $(SCRIPT_DIR)/seq_$(BENCHMARK_NAME).c

OUT_DIR       := $(SCRIPT_DIR)/bin
OUT_GTAP_THREAD := $(OUT_DIR)/gtap_thread_$(BENCHMARK_NAME)
OUT_GTAP_BLOCK  := $(OUT_DIR)/gtap_block_$(BENCHMARK_NAME)
OUT_OMP       := $(OUT_DIR)/omp_$(BENCHMARK_NAME)
OUT_GTAP_THREAD_SEQMEM := $(OUT_DIR)/gtap_thread_$(BENCHMARK_NAME)_seqmem
OUT_GTAP_BLOCK_SEQMEM  := $(OUT_DIR)/gtap_block_$(BENCHMARK_NAME)_seqmem
OUT_OMP_SEQMEM         := $(OUT_DIR)/omp_$(BENCHMARK_NAME)_seqmem
OUT_OMP_CUDA  := $(OUT_DIR)/omp_cuda_$(BENCHMARK_NAME)
OUT_SEQ       := $(OUT_DIR)/seq_$(BENCHMARK_NAME)

# ---- GTAP build settings ----
GTAP_GRID_SIZE           ?= 1000
GTAP_BLOCK_SIZE          ?= 64
GTAP_MAX_TASKS_PER_WARP  ?= 100000
GTAP_MAX_TASKS_PER_BLOCK ?= 100000
GTAP_MAX_CHILD_TASKS     ?= 2
GTAP_NUM_QUEUES          ?= 1
GTAP_CFLAGS              ?= -DGTAP_TERMINATE_ON_FIRST_TASK_FINISH

# ---- OpenMP build settings ----
CC_OMP           ?= $(CXX_BIN)
OMP_RESOURCE_DIR ?= $(PROJ_DIR)/clang-gtap/build/lib/clang/21
OMP_INC          := $(OMP_RESOURCE_DIR)/include
OMP_LIB_DIR      ?= $(PROJ_DIR)/clang-gtap/build/lib
CFLAGS_OMP       ?= -O3 -Wall -Wextra -fopenmp -I"$(OMP_INC)"
LDFLAGS_OMP      ?= -L"$(OMP_LIB_DIR)" -Wl,-rpath,"$(OMP_LIB_DIR)" -lomp

# ---- OpenMP CUDA build settings ----
OMP_CUDA_RESOURCE_DIR ?= $(PROJ_DIR)/clang-gtap/build/lib/clang/21
OMP_CUDA_INC          := $(OMP_CUDA_RESOURCE_DIR)/include
OMP_CUDA_LIB_DIR      ?= $(PROJ_DIR)/clang-gtap/build/lib
CFLAGS_OMP_CUDA       ?= -O3 -Wall -Wextra -fopenmp -I"$(OMP_CUDA_INC)"
LDFLAGS_OMP_CUDA      ?= -L"$(OMP_CUDA_LIB_DIR)" -Wl,-rpath,"$(OMP_CUDA_LIB_DIR)" -lomp

# ---- Sequential build settings ----
CC_SEQ           ?= $(CXX_BIN)
CFLAGS_SEQ       ?= -O3 -Wall -Wextra

.PHONY: all gtap_thread gtap_block omp omp_cuda seq clean print-vars

all: gtap_thread gtap_block omp omp_cuda 

print-vars:
	@echo "SCRIPT_DIR=$(SCRIPT_DIR)"
	@echo "PROJ_DIR=$(PROJ_DIR)"
	@echo "CXX_BIN=$(CXX_BIN)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "CUDA_ARCH=$(CUDA_ARCH)"
	@echo "GT_INC=$(GT_INC)"
	@echo "OUT_DIR=$(OUT_DIR)"

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

# ---- Build: gtap_$(BENCHMARK_NAME).cu ----
gtap_thread: $(OUT_GTAP_THREAD)

$(OUT_GTAP_THREAD): $(SRC_GTAP_THREAD) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" -DGTAP_MAX_CHILD_TASKS="$(GTAP_MAX_CHILD_TASKS)" \
	  -DGTAP_NUM_QUEUES="$(GTAP_NUM_QUEUES)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_GTAP_THREAD)" -o "$(OUT_GTAP_THREAD)"


gtap_block: $(OUT_GTAP_BLOCK)

$(OUT_GTAP_BLOCK): $(SRC_GTAP_BLOCK) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_BLOCK="$(GTAP_MAX_TASKS_PER_BLOCK)" -DGTAP_MAX_CHILD_TASKS="$(GTAP_MAX_CHILD_TASKS)" \
	  -DGTAP_NUM_QUEUES="$(GTAP_NUM_QUEUES)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_GTAP_BLOCK)" -o "$(OUT_GTAP_BLOCK)"


# ---- Build: omp_$(BENCHMARK_NAME).cpp ----
omp: $(OUT_OMP)

$(OUT_OMP): $(SRC_OMP) | $(OUT_DIR)
	"$(CXX_BIN)" $(CFLAGS_OMP) $(LDFLAGS_OMP) "$(SRC_OMP)" -o "$(OUT_OMP)"

# ---- Build: omp_cuda_$(BENCHMARK_NAME).cu ----
omp_cuda: $(OUT_OMP_CUDA)

$(OUT_OMP_CUDA): $(SRC_OMP_CUDA) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  $(CFLAGS_OMP_CUDA) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  $(LDFLAGS_OMP_CUDA) \
	  "$(SRC_OMP_CUDA)" -o "$(OUT_OMP_CUDA)"

# # ---- Build: seq_$(BENCHMARK_NAME).c ----
# seq: $(OUT_SEQ)

# $(OUT_SEQ): $(SRC_SEQ) | $(OUT_DIR)
# 	"$(CC_SEQ)" $(CFLAGS_SEQ) "$(SRC_SEQ)" -o "$(OUT_SEQ)"

clean:
	rm -f $(OUT_GTAP_THREAD) $(OUT_GTAP_BLOCK) $(OUT_OMP) $(OUT_OMP_CUDA) $(OUT_SEQ) \
	  $(OUT_GTAP_THREAD_SEQMEM) $(OUT_GTAP_BLOCK_SEQMEM) $(OUT_OMP_SEQMEM)
