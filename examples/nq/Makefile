# GTaP example: N-Queens
GTAP_ROOT  ?= ../..
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CLANG_BIN  ?= $(GTAP_ROOT)/clang-gtap/build/bin/clang
GT_INC     ?= $(GTAP_ROOT)/runtime
CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

SRC     := $(SCRIPT_DIR)/nq.cu
OUT_DIR := $(SCRIPT_DIR)/bin
OUT     := $(OUT_DIR)/nq

GTAP_CFLAGS ?= -fgtap-no-taskwait

.PHONY: all clean
all: $(OUT)

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

$(OUT): $(SRC) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC)" -o "$(OUT)"

clean:
	rm -f $(OUT)
