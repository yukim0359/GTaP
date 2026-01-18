#pragma once

#include "gtap_runtime_block.cuh"
#include "../common/gtap_macro_common.cuh"

#define TASK_DEF(func_name, TaskType) \
__device__ void func_name(TaskType* self, int self_tid, TaskContext* __ctx)

#define TASK_BEGIN() \
    int __dsl_child_count = 0; \
    switch ((int)load_L2(&d_task_headers[self_tid].state)) { \
        case 0:

#define TASK_SPAWN(child, func_ptr, child_task_kind) \
    do { \
        __gtap_spawn_task(__ctx, self_tid, &__dsl_child_count, (void(*)(void*, int, TaskContext*))func_ptr, (child), (child_task_kind)); \
    } while (0)

#define TASK_JOIN(task_kind_after_join) \
    do { \
        __gtap_set_state_for_join(self_tid, __dsl_child_count, __LINE__, (task_kind_after_join)); \
        return; case __LINE__: ; \
        __dsl_child_count = 0; \
    } while (0)

#define TASK_FINISH() \
    do { \
        __gtap_finish_task(self_tid, __ctx); \
        return; \
    } while (0)

#define TASK_END() \
        default: break; \
    }

#define TASK_CHILD_ID(i) (load_L2(&d_task_headers[self_tid].child_ids[(i)]))
#define TASK_CHILD_RESULT(i) (load_task_result(TASK_CHILD_ID((i))))

#define TASK_ENQUEUE_INITIAL(TaskType, func_ptr, initial) \
    do { \
        __gtap_push_initial_task<TaskType>((void(*)(void*, int, TaskContext*))func_ptr, (initial)); \
    } while (0)
