#pragma once

#include <cstddef>
#include <set>
#include <string>

#include "rt_header.h"

typedef enum _RTThreadSched {
    RT_SCHED_OTHER,
    RT_SCHED_RR,
    RT_SCHED_FIFO,
    RT_SCHED_MAX,
} RTThreadSched;

// The field order follows the layout inferred from the current AArch64
// RTThreadPool/RTExecutor objects: size, priority, CPU set, name, policy.
class RTThreadOptions {
 public:
    std::size_t stackSize = 0;
    INT32 priority = 0;
    std::set<INT32> selectedCpus;
    std::string namePrefix;
    RTThreadSched schedPolicy = RT_SCHED_OTHER;
};
