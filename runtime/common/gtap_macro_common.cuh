template <typename T> struct __task_false { static const bool value = false; };

// Type-erased task data access - uses d_tasks (char* byte array) instead of fixed Task array
template <typename U>
__device__ __forceinline__ void set_task_result_impl(U*, int tid, int value, decltype(&U::result)* = nullptr) {
    // Get task data pointer from d_tasks (type-erased byte array)
    U* __typed = __gtap_get_task_data<U>(tid);
    store_L2(&__typed->result, value);
}

template <typename U>
__device__ __forceinline__ void set_task_result_impl(U*, int, int, ...) {
    static_assert(__task_false<U>::value, "Task has no member named 'result'");
}

template <typename U>
__device__ __forceinline__ int load_task_result_impl(U*, int tid, decltype(&U::result)* = nullptr) {
    // Get task data pointer from d_tasks (type-erased byte array)
    U* __typed = __gtap_get_task_data<U>(tid);
    return load_L2(&__typed->result);
}

template <typename U>
__device__ __forceinline__ int load_task_result_impl(U*, int, ...) {
    static_assert(__task_false<U>::value, "Task has no member named 'result'");
    return 0;
}

#define set_task_result(tid, value) ((void)set_task_result_impl((Task*)0, (tid), (value), (decltype(&Task::result)*)0))
#define load_task_result(tid) load_task_result_impl((Task*)0, (tid), (decltype(&Task::result)*)0)

/* Which is better, macro or function? */
// __device__ __forceinline__ void set_task_result(int tid, int value) {
//     set_task_result_impl((Task*)0, tid, value, (decltype(&Task::result)*)0);
// }

// __device__ __forceinline__ int load_task_result(int tid) {
//     return load_task_result_impl((Task*)0, tid, (decltype(&Task::result)*)0);
// }

// Generic i32 field accessors for arbitrary Task fields
// unused for now and they might have bugs
template <typename U>
__device__ __forceinline__ void set_task_field_i32_impl(int tid, int U::*field_ptr, int value) {
    // Get task data pointer from d_tasks (type-erased byte array)
    U* __typed = __gtap_get_task_data<U>(tid);
    store_L2(&(__typed->*field_ptr), value);
}

template <typename U>
__device__ __forceinline__ int load_task_field_i32_impl(int tid, int U::*field_ptr) {
    // Get task data pointer from d_tasks (type-erased byte array)
    U* __typed = __gtap_get_task_data<U>(tid);
    return load_L2(&(__typed->*field_ptr));
}

#define set_task_field_i32(tid, member, value) ((void)set_task_field_i32_impl((tid), &Task::member, (value)))
#define load_task_field_i32(tid, member) load_task_field_i32_impl((tid), &Task::member)
