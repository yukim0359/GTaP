# GTaP example: N-Queens
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SCRIPT_DIR)/../common.mk

SRC := $(SCRIPT_DIR)/nq.cu
OUT := $(OUT_DIR)/nq

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
	  $(GTAP_CUDA_LIBS) \
	  "$(SRC)" -o "$(OUT)"

clean:
	rm -f $(OUT)
