# GTaP example: Cilksort
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SCRIPT_DIR)/../common.mk

SRC := $(SCRIPT_DIR)/cilksort.cu
OUT := $(OUT_DIR)/cilksort

GTAP_CFLAGS ?=

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
