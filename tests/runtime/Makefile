GTAP_ROOT  := ../..
CLANG_BIN  ?= $(GTAP_ROOT)/clang-gtap/build/bin/clang++
GT_INC     ?= $(GTAP_ROOT)/runtime
CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

OUT_DIR := bin
TESTS := fib_thread fib_block binary_tree_thread binary_tree_block \
	multiple_taskwait_queue invalid_queue_thread
BINS := $(addprefix $(OUT_DIR)/,$(TESTS))
DEPS := $(addsuffix .d,$(BINS))

COMMON_FLAGS := -O3 -x cuda --cuda-path="$(CUDA_PATH)" \
	--cuda-gpu-arch="$(CUDA_ARCH)" -I"$(GT_INC)" \
	-L"$(CUDA_PATH)/lib64" -lcudart

.PHONY: all check clean
all: $(BINS)

$(OUT_DIR):
	mkdir -p "$@"

# A missing .d is an out-of-date prerequisite, so the first build records headers.
# The compile refreshes the .d; .PRECIOUS keeps make from deleting that side effect.
$(OUT_DIR)/%: %.cu $(OUT_DIR)/%.d | $(OUT_DIR)
	"$(CLANG_BIN)" $(COMMON_FLAGS) -MMD -MP -MT "$@" -MF "$(OUT_DIR)/$*.d" "$<" -o "$@"

$(OUT_DIR)/%.d: ;
.PRECIOUS: $(OUT_DIR)/%.d

-include $(DEPS)

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
	@echo "[ RUN      ] invalid_queue_thread"
	@$(OUT_DIR)/invalid_queue_thread
	@echo "[       OK ] invalid_queue_thread"

clean:
	rm -f $(BINS) $(DEPS)
