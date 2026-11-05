# GTaP example: SpMV (thread and block variants)
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SCRIPT_DIR)/../common.mk

SRC_THREAD := $(SCRIPT_DIR)/spmv_thread.cu
SRC_BLOCK  := $(SCRIPT_DIR)/spmv_block.cu
OUT_THREAD := $(OUT_DIR)/spmv_thread
OUT_BLOCK  := $(OUT_DIR)/spmv_block

GTAP_CFLAGS ?=

.PHONY: all thread block clean
all: thread block

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

# ---- Thread variant ----
thread: $(OUT_THREAD)

$(OUT_THREAD): $(SRC_THREAD) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  $(GTAP_CUDA_LIBS) \
	  "$(SRC_THREAD)" -o "$(OUT_THREAD)"

# ---- Block variant ----
block: $(OUT_BLOCK)

$(OUT_BLOCK): $(SRC_BLOCK) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  $(GTAP_CUDA_LIBS) \
	  "$(SRC_BLOCK)" -o "$(OUT_BLOCK)"

clean:
	rm -f "$(OUT_THREAD)" "$(OUT_BLOCK)"
