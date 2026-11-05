# Shared toolchain for evaluation Makefiles.
# The including Makefile must set SCRIPT_DIR (its own directory) first.

COMMON_MK_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJ_DIR      ?= $(abspath $(COMMON_MK_DIR)/..)

CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

CLANG_BIN  ?= $(PROJ_DIR)/clang-gtap/build/bin/clang
CXX_BIN    ?= $(PROJ_DIR)/clang-gtap/build/bin/clang++
GT_INC     ?= $(PROJ_DIR)/runtime
OUT_DIR    ?= $(SCRIPT_DIR)/bin

CC_OMP           ?= $(CLANG_BIN)
CXX_OMP          ?= $(CXX_BIN)
OMP_RESOURCE_DIR ?= $(PROJ_DIR)/clang-gtap/build/lib/clang/21
OMP_INC          := $(OMP_RESOURCE_DIR)/include
OMP_LIB_DIR      ?= $(PROJ_DIR)/clang-gtap/build/runtimes/runtimes-bins/openmp/runtime/src
CFLAGS_OMP       ?= -O3 -Wall -Wextra -fopenmp -I"$(OMP_INC)"
CXXFLAGS_OMP     ?= -O3 -Wall -Wextra -fopenmp -I"$(OMP_INC)"
# DT_RPATH so the linked libomp is used even if LD_LIBRARY_PATH has another libomp.so.
LDFLAGS_OMP      ?= -L"$(OMP_LIB_DIR)" -Wl,-rpath,"$(OMP_LIB_DIR)" -Wl,--disable-new-dtags -lomp

OPENCILK_ROOT    ?= $(HOME)/opencilk
OPENCILK_PREFIX  ?= $(OPENCILK_ROOT)/install
CC_CILK          ?= $(OPENCILK_PREFIX)/bin/clang
CXX_CILK         ?= $(OPENCILK_PREFIX)/bin/clang++
CFLAGS_CILK      ?= -O3 -Wall -Wextra -fopencilk
LDFLAGS_CILK     ?= -fopencilk

GTAP_CUDA_LIBS := -L"$(CUDA_PATH)/lib64" -lcudart
