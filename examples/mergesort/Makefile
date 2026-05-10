# GTaP example: Mergesort
GTAP_ROOT  ?= ../..
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CLANG_BIN  ?= $(GTAP_ROOT)/clang-gtap/build/bin/clang++
GT_INC     ?= $(GTAP_ROOT)/runtime
CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

SRC     := $(SCRIPT_DIR)/mergesort.cu
OUT_DIR := $(SCRIPT_DIR)/bin
OUT     := $(OUT_DIR)/mergesort

GTAP_GRID_SIZE          ?= 1000
GTAP_BLOCK_SIZE         ?= 32
GTAP_MAX_TASKS_PER_WARP ?= 80000
GTAP_MAX_CHILD_TASKS    ?= 2
GTAP_NUM_QUEUES         ?= 1
GTAP_CFLAGS             ?=

.PHONY: all clean
all: $(OUT)

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

$(OUT): $(SRC) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE=$(GTAP_GRID_SIZE) -DGTAP_BLOCK_SIZE=$(GTAP_BLOCK_SIZE) \
	  -DGTAP_MAX_TASKS_PER_WARP=$(GTAP_MAX_TASKS_PER_WARP) \
	  -DGTAP_MAX_CHILD_TASKS=$(GTAP_MAX_CHILD_TASKS) \
	  -DGTAP_NUM_QUEUES=$(GTAP_NUM_QUEUES) \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC)" -o "$(OUT)"

clean:
	rm -f $(OUT)
