#pragma once

namespace gtap::detail {

enum TerminationMode {
    TERMINATE_ON_ALL_TASKS_FINISH,  // default
    TERMINATE_ON_FIRST_TASK_FINISH  // finish when first task finishes
};

}  // namespace gtap::detail
