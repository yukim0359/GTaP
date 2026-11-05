# Shared toolchain for example Makefiles.
# The including Makefile must set SCRIPT_DIR (its own directory) first.

COMMON_MK_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
GTAP_ROOT     ?= $(abspath $(COMMON_MK_DIR)/..)

CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

CLANG_BIN  ?= $(GTAP_ROOT)/clang-gtap/build/bin/clang++
GT_INC     ?= $(GTAP_ROOT)/runtime
OUT_DIR    ?= $(SCRIPT_DIR)/bin

GTAP_CUDA_LIBS := -L"$(CUDA_PATH)/lib64" -lcudart
