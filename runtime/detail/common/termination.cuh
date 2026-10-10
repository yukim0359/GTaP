#pragma once

namespace gtap::detail {

enum TerminationMode {
    TERMINATE_ON_ALL_TASKS_FINISH,  // default
    TERMINATE_ON_ROOT_TASK_FINISH   // finish when the root task finishes
};

}  // namespace gtap::detail
