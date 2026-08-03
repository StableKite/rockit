#include "RTExecutor.h"

#include <algorithm>
#include <thread>
#include <utility>

RTTaskQueue::~RTTaskQueue() = default;
RTExecutor::~RTExecutor() = default;

RTThreadPoolExecutor::RTThreadPoolExecutor(INT32 numThreads)
    : mThreadPool("mediapipe", numThreads, RT_THREAD_POOL_RANDOM_MODE) {
    start();
}

RTThreadPoolExecutor::RTThreadPoolExecutor(const RTThreadOptions &threadOptions,
                                           INT32 numThreads,
                                           RTThreadPoolMode mode)
    : mThreadPool(threadOptions,
                  threadOptions.namePrefix.empty() ? "mediapipe" : threadOptions.namePrefix,
                  numThreads,
                  mode) {
    start();
}

RTThreadPoolExecutor::~RTThreadPoolExecutor() = default;

void RTThreadPoolExecutor::start() {
    mStackSize = mThreadPool.getThreadOptions().stackSize;
    mThreadPool.startWorkers();
}

void RTThreadPoolExecutor::schedule(std::function<void()> task, INT32 threadId) {
    mThreadPool.schedule(std::move(task), threadId);
}

INT32 RTThreadPoolExecutor::getNumThreads() const {
    return mThreadPool.getNumThreads();
}

std::size_t RTThreadPoolExecutor::getStackSize() const {
    return mStackSize;
}

RTExecutor *RTThreadPoolExecutor::create(RtMetaData *extendOptions) {
    INT32 count = 4;
    const char *name = nullptr;
    if (extendOptions) {
        INT32 configured = count;
        if (extendOptions->findInt32("exec_thread_num", &configured) && configured > 0) {
            count = configured;
        }
        (void)extendOptions->findCString("exec_name", &name);
    }
    RTThreadOptions options;
    if (name)
        options.namePrefix = name;
    return new RTThreadPoolExecutor(options, count, RT_THREAD_POOL_RANDOM_MODE);
}

RTSingleExecutor::RTSingleExecutor(const RTThreadOptions &threadOptions)
    : mThreadPool(threadOptions, "singleExecutor", 1, RT_THREAD_POOL_RANDOM_MODE) {
    mThreadPool.startWorkers();
}

RTSingleExecutor::~RTSingleExecutor() = default;

void RTSingleExecutor::schedule(std::function<void()> task, INT32 threadId) {
    mThreadPool.schedule(std::move(task), threadId);
}

INT32 RTSingleExecutor::getNumThreads() const {
    return 1;
}

RTExecutor *RTSingleExecutor::create(void *) {
    return new RTSingleExecutor(RTThreadOptions{});
}
