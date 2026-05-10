BENCHMARK_NAME := mergesort

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJ_DIR   ?= $(abspath $(SCRIPT_DIR)/../../..)

CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

CLANG_BIN  ?= $(PROJ_DIR)/clang-gtap/build/bin/clang
CXX_BIN    ?= $(PROJ_DIR)/clang-gtap/build/bin/clang++
GT_INC     ?= $(PROJ_DIR)/runtime

# ---- Inputs / Outputs ----
SRC_GTAP   := $(SCRIPT_DIR)/gtap_$(BENCHMARK_NAME).cu
SRC_OMP    := $(SCRIPT_DIR)/omp_$(BENCHMARK_NAME).cpp
SRC_SEQ    := $(SCRIPT_DIR)/seq_$(BENCHMARK_NAME).cpp

OUT_DIR    := $(SCRIPT_DIR)/bin
OUT_GTAP   := $(OUT_DIR)/gtap_$(BENCHMARK_NAME)
OUT_OMP    := $(OUT_DIR)/omp_$(BENCHMARK_NAME)
OUT_SEQ    := $(OUT_DIR)/seq_$(BENCHMARK_NAME)

# ---- GTAP build settings ----
GTAP_GRID_SIZE          ?= 1000
GTAP_BLOCK_SIZE         ?= 32
GTAP_MAX_TASKS_PER_WARP ?= 80000
GTAP_NUM_QUEUES         ?= 1
GTAP_CFLAGS             ?= -DGTAP_TERMINATE_ON_FIRST_TASK_FINISH

# ---- OpenMP build settings ----
CXX_OMP          ?= $(CXX_BIN)
OMP_RESOURCE_DIR ?= $(PROJ_DIR)/clang-gtap/build/lib/clang/21
OMP_INC          := $(OMP_RESOURCE_DIR)/include
OMP_LIB_DIR      ?= $(PROJ_DIR)/clang-gtap/build/lib
CXXFLAGS_OMP     ?= -O3 -Wall -Wextra -fopenmp -I"$(OMP_INC)"
LDFLAGS_OMP      ?= -L"$(OMP_LIB_DIR)" -Wl,-rpath,"$(OMP_LIB_DIR)" -lomp

# ---- Sequential build settings ----
CXX_SEQ          ?= $(CXX_BIN)
CXXFLAGS_SEQ     ?= -O3 -Wall -Wextra

.PHONY: all gtap omp seq clean print-vars

all: gtap omp seq

print-vars:
	@echo "SCRIPT_DIR=$(SCRIPT_DIR)"
	@echo "PROJ_DIR=$(PROJ_DIR)"
	@echo "CLANG_BIN=$(CLANG_BIN)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "CUDA_ARCH=$(CUDA_ARCH)"
	@echo "GT_INC=$(GT_INC)"
	@echo "OUT_DIR=$(OUT_DIR)"

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

# ---- Build: gtap_$(BENCHMARK_NAME).cu ----
gtap: $(OUT_GTAP)

$(OUT_GTAP): $(SRC_GTAP) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  -DGTAP_NUM_QUEUES="$(GTAP_NUM_QUEUES)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_GTAP)" -o "$(OUT_GTAP)"

# ---- Build: omp_$(BENCHMARK_NAME).c ----
omp: $(OUT_OMP)

$(OUT_OMP): $(SRC_OMP) | $(OUT_DIR)
	"$(CXX_OMP)" $(CXXFLAGS_OMP) $(LDFLAGS_OMP) "$(SRC_OMP)" -o "$(OUT_OMP)"

# ---- Build: seq_$(BENCHMARK_NAME).c ----
seq: $(OUT_SEQ)

$(OUT_SEQ): $(SRC_SEQ) | $(OUT_DIR)
	"$(CXX_SEQ)" $(CXXFLAGS_SEQ) "$(SRC_SEQ)" -o "$(OUT_SEQ)"

clean:
	rm -f $(OUT_GTAP) $(OUT_OMP) $(OUT_SEQ)
