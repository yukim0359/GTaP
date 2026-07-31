GTAP_ROOT  := ../..
CLANG_BIN  ?= $(GTAP_ROOT)/clang-gtap/build/bin/clang++
GT_INC     ?= $(GTAP_ROOT)/runtime
CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

OUT_DIR := bin
TESTS := fib_thread fib_block binary_tree_thread binary_tree_block \
	multiple_taskwait_queue
BINS := $(addprefix $(OUT_DIR)/,$(TESTS))

COMMON_FLAGS := -O2 -x cuda --cuda-path="$(CUDA_PATH)" \
	--cuda-gpu-arch="$(CUDA_ARCH)" -I"$(GT_INC)" \
	-L"$(CUDA_PATH)/lib64" -lcudart

.PHONY: all check clean
all: $(BINS)

$(OUT_DIR):
	mkdir -p "$@"

$(OUT_DIR)/fib_thread: fib_thread.cu | $(OUT_DIR)
	"$(CLANG_BIN)" $(COMMON_FLAGS) "$<" -o "$@"

$(OUT_DIR)/fib_block: fib_block.cu | $(OUT_DIR)
	"$(CLANG_BIN)" $(COMMON_FLAGS) -DGTAP_BLOCK_SIZE=32 "$<" -o "$@"

$(OUT_DIR)/binary_tree_thread: binary_tree_thread.cu | $(OUT_DIR)
	"$(CLANG_BIN)" $(COMMON_FLAGS) "$<" -o "$@"

$(OUT_DIR)/binary_tree_block: binary_tree_block.cu | $(OUT_DIR)
	"$(CLANG_BIN)" $(COMMON_FLAGS) -DGTAP_BLOCK_SIZE=32 "$<" -o "$@"

$(OUT_DIR)/multiple_taskwait_queue: multiple_taskwait_queue.cu | $(OUT_DIR)
	"$(CLANG_BIN)" $(COMMON_FLAGS) "$<" -o "$@"

check: all
	@echo "[ RUN      ] fib_thread"
	@$(OUT_DIR)/fib_thread 10
	@echo "[       OK ] fib_thread"
	@echo "[ RUN      ] fib_block"
	@$(OUT_DIR)/fib_block 10
	@echo "[       OK ] fib_block"
	@echo "[ RUN      ] binary_tree_thread"
	@$(OUT_DIR)/binary_tree_thread 6
	@echo "[       OK ] binary_tree_thread"
	@echo "[ RUN      ] binary_tree_block"
	@$(OUT_DIR)/binary_tree_block 6
	@echo "[       OK ] binary_tree_block"
	@echo "[ RUN      ] multiple_taskwait_queue"
	@$(OUT_DIR)/multiple_taskwait_queue
	@echo "[       OK ] multiple_taskwait_queue"

clean:
	rm -f $(BINS)
