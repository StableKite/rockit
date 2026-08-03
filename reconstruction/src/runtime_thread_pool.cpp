#include "RTThreadPool.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace {
constexpr INT32 kMaxThreads = 32;

void set_current_thread_name(const std::string &name) {
#if defined(__linux__)
    (void)pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
#else
    (void)name;
#endif
}
} // namespace

struct RTThreadPool::Impl {
    struct WorkerQueue {
        std::mutex mutex;
        std::condition_variable condition;
        std::deque<std::function<void()>> tasks;
    };

    RTThreadOptions options;
    std::string namePrefix;
    INT32 numThreads = 1;
    RTThreadPoolMode mode = RT_THREAD_POOL_RANDOM_MODE;

    mutable std::mutex lifecycleMutex;
    bool started = false;
    std::atomic<bool> stopping{false};

    std::mutex randomMutex;
    std::condition_variable randomCondition;
    std::deque<std::function<void()>> randomTasks;

    std::vector<std::unique_ptr<WorkerQueue>> assigned;
    std::vector<std::thread> threads;

    Impl(const RTThreadOptions &threadOptions,
         std::string prefix,
         INT32 count,
         RTThreadPoolMode poolMode)
        : options(threadOptions), namePrefix(std::move(prefix)), mode(poolMode) {
        if (count <= 0)
            count = 1;
        numThreads = std::min(count, kMaxThreads);
        if (namePrefix.empty())
            namePrefix = "mediapipe";
        assigned.reserve(static_cast<std::size_t>(numThreads));
        for (INT32 i = 0; i < numThreads; ++i) {
            assigned.push_back(std::make_unique<WorkerQueue>());
        }
    }

    bool shouldStopRandomLocked() const {
        return stopping.load(std::memory_order_acquire) && randomTasks.empty();
    }

    void randomWorker(INT32 id) {
        set_current_thread_name(internal::createThreadName(namePrefix, id));
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(randomMutex);
                randomCondition.wait(lock, [this] {
                    return stopping.load(std::memory_order_acquire) || !randomTasks.empty();
                });
                if (shouldStopRandomLocked())
                    return;
                task = std::move(randomTasks.front());
                randomTasks.pop_front();
            }
            if (task)
                task();
        }
    }

    void assignedWorker(INT32 id) {
        set_current_thread_name(internal::createThreadName(namePrefix, id));
        WorkerQueue &queue = *assigned[static_cast<std::size_t>(id)];
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue.mutex);
                queue.condition.wait(lock, [this, &queue] {
                    return stopping.load(std::memory_order_acquire) || !queue.tasks.empty();
                });
                if (stopping.load(std::memory_order_acquire) && queue.tasks.empty())
                    return;
                task = std::move(queue.tasks.front());
                queue.tasks.pop_front();
            }
            if (task)
                task();
        }
    }

    void start() {
        std::lock_guard<std::mutex> lock(lifecycleMutex);
        if (started)
            return;
        stopping.store(false, std::memory_order_release);
        threads.reserve(static_cast<std::size_t>(numThreads));
        for (INT32 i = 0; i < numThreads; ++i) {
            if (mode == RT_THREAD_POOL_ASSIGN_MODE) {
                threads.emplace_back([this, i] {
                    assignedWorker(i);
                });
            } else {
                threads.emplace_back([this, i] {
                    randomWorker(i);
                });
            }
        }
        started = true;
    }

    void schedule(std::function<void()> task, INT32 requestedThread) {
        if (!task)
            return;

        // Keep the lifecycle lock until the task is visible in its queue. This
        // closes the race where shutdown could set stopping and notify all
        // workers after the check but before the enqueue, leaving an accepted
        // task stranded with no worker to consume it.
        std::unique_lock<std::mutex> lifecycleLock(lifecycleMutex);
        if (stopping.load(std::memory_order_acquire)) {
            throw std::runtime_error("RTThreadPool is stopping");
        }

        if (mode == RT_THREAD_POOL_ASSIGN_MODE) {
            INT32 normalized = requestedThread % numThreads;
            if (normalized < 0)
                normalized += numThreads;
            WorkerQueue &queue = *assigned[static_cast<std::size_t>(normalized)];
            {
                std::lock_guard<std::mutex> lock(queue.mutex);
                queue.tasks.emplace_back(std::move(task));
            }
            lifecycleLock.unlock();
            queue.condition.notify_one();
        } else {
            {
                std::lock_guard<std::mutex> lock(randomMutex);
                randomTasks.emplace_back(std::move(task));
            }
            lifecycleLock.unlock();
            randomCondition.notify_one();
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(lifecycleMutex);
            stopping.store(true, std::memory_order_release);
            if (!started)
                return;
        }

        randomCondition.notify_all();
        for (auto &queue : assigned)
            queue->condition.notify_all();
        for (auto &thread : threads) {
            if (thread.joinable())
                thread.join();
        }

        std::lock_guard<std::mutex> lock(lifecycleMutex);
        threads.clear();
        started = false;
    }
};

RTThreadPool::RTThreadPool(INT32 numThreads)
    : RTThreadPool(std::string("mediapipe"), numThreads, RT_THREAD_POOL_RANDOM_MODE) {}

RTThreadPool::RTThreadPool(const std::string &namePrefix, INT32 numThreads, RTThreadPoolMode mode)
    : mImpl(std::make_unique<Impl>(RTThreadOptions{}, namePrefix, numThreads, mode)) {}

RTThreadPool::RTThreadPool(const RTThreadOptions &threadOptions,
                           const std::string &namePrefix,
                           INT32 numThreads,
                           RTThreadPoolMode mode)
    : mImpl(std::make_unique<Impl>(threadOptions,
                                   namePrefix.empty() ? threadOptions.namePrefix : namePrefix,
                                   numThreads,
                                   mode)) {}

RTThreadPool::~RTThreadPool() {
    mImpl->shutdown();
}

void RTThreadPool::startWorkers() {
    mImpl->start();
}

void RTThreadPool::schedule(std::function<void()> callback, INT32 lockThreadId) {
    mImpl->schedule(std::move(callback), lockThreadId);
}

INT32 RTThreadPool::getNumThreads() const {
    return mImpl->numThreads;
}

const RTThreadOptions &RTThreadPool::getThreadOptions() const {
    return mImpl->options;
}

namespace internal {
std::string createThreadName(const std::string &prefix, INT32 threadId) {
    std::string value = prefix + "/" + std::to_string(threadId);
    if (value.size() > 15)
        value.resize(15);
    return value;
}
} // namespace internal
