#pragma once

#include <functional>
#include <memory>
#include <string>

#include "RTThreadOptions.h"
#include "rt_header.h"

typedef enum _RTThreadPoolMode {
    RT_THREAD_POOL_RANDOM_MODE,
    RT_THREAD_POOL_ASSIGN_MODE,
} RTThreadPoolMode;

class RTThreadPool {
 public:
    explicit RTThreadPool(INT32 numThreads);
    RTThreadPool(const RTThreadPool &) = delete;
    RTThreadPool &operator=(const RTThreadPool &) = delete;
    RTThreadPool(const std::string &namePrefix, INT32 numThreads,
                 RTThreadPoolMode mode = RT_THREAD_POOL_RANDOM_MODE);
    RTThreadPool(const RTThreadOptions &threadOptions,
                 const std::string &namePrefix, INT32 numThreads,
                 RTThreadPoolMode mode = RT_THREAD_POOL_RANDOM_MODE);
    ~RTThreadPool();

    void startWorkers();
    void schedule(std::function<void()> callback, INT32 lockThreadId = 0);
    INT32 getNumThreads() const;
    const RTThreadOptions &getThreadOptions() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

namespace internal {
std::string createThreadName(const std::string &prefix, INT32 threadId);
}
