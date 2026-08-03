#pragma once

#include <functional>

#include "RTThreadOptions.h"
#include "RTThreadPool.h"
#include "rt_metadata.h"

class RTTaskQueue {
 public:
    virtual ~RTTaskQueue();
    virtual void runNextTask() = 0;
};

class RTExecutor {
 public:
    virtual ~RTExecutor();
    virtual void addTask(RTTaskQueue *taskQueue, INT32 threadId = 0) {
        schedule([taskQueue] {
            if (taskQueue) taskQueue->runNextTask();
        }, threadId);
    }
    virtual void schedule(std::function<void()> task, INT32 threadId = 0) = 0;
    virtual INT32 getNumThreads() const = 0;
};

class RTThreadPoolExecutor : public RTExecutor {
 public:
    static RTExecutor *create(RtMetaData *extendOptions);
    explicit RTThreadPoolExecutor(INT32 numThreads);
    explicit RTThreadPoolExecutor(const RTThreadOptions &threadOptions,
                                  INT32 numThreads,
                                  RTThreadPoolMode mode = RT_THREAD_POOL_RANDOM_MODE);
    ~RTThreadPoolExecutor() override;
    void schedule(std::function<void()> task, INT32 threadId = 0) override;
    INT32 getNumThreads() const override;
    std::size_t getStackSize() const;

 private:
    void start();
    RTThreadPool mThreadPool;
    std::size_t mStackSize = 0;
};

class RTSingleExecutor : public RTExecutor {
 public:
    static RTExecutor *create(void *extendOptions);
    explicit RTSingleExecutor(const RTThreadOptions &threadOptions);
    ~RTSingleExecutor() override;
    void schedule(std::function<void()> task, INT32 threadId = 0) override;
    INT32 getNumThreads() const override;

 private:
    RTThreadPool mThreadPool;
};
