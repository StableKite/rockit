#include "RTScheduler.h"
#include "RTTaskNode.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <utility>

namespace {
template <class Predicate>
RT_RET wait_for_predicate(std::condition_variable &condition,
                          std::unique_lock<std::mutex> &lock,
                          INT64 timeoutUs,
                          Predicate predicate) {
    if (predicate())
        return RT_OK;
    if (timeoutUs < 0) {
        condition.wait(lock, predicate);
        return RT_OK;
    }
    if (!condition.wait_for(lock, std::chrono::microseconds(timeoutUs), predicate))
        return RT_ERR_TIMEOUT;
    return RT_OK;
}
} // namespace

struct RTScheduler::Impl {
    enum class State {
        NotStarted,
        Running,
        Paused,
        Cancelling,
        Terminated,
    };

    explicit Impl(RTTaskGraph *owner) : graph(owner) {}

    RTTaskGraph *graph = nullptr;
    mutable std::mutex mutex;
    std::condition_variable condition;
    State state = State::NotStarted;
    bool eos = false;
    bool hasError = false;
    bool queuesRunning = false;
    bool graphInputThrottled = false;
    UINT64 observedGeneration = 0;
    std::size_t activeTasks = 0;
    std::set<RTTaskNode *> openedNodes;
    RTExecutor *defaultExecutor = nullptr;
    std::map<RTTaskNode *, RTExecutor *> executors;

    RTExecutor *executorFor(RTTaskNode *node) {
        auto it = executors.find(node);
        return it == executors.end() ? defaultExecutor : it->second;
    }

    void taskCompleted() {
        std::lock_guard<std::mutex> lock(mutex);
        if (activeTasks != 0)
            --activeTasks;
        if (eos && activeTasks == 0 && state == State::Cancelling)
            state = State::Terminated;
        condition.notify_all();
    }
};

RTScheduler::RTScheduler(RTTaskGraph *ctx) : mImpl(std::make_unique<Impl>(ctx)) {}
RTScheduler::~RTScheduler() = default;

void RTScheduler::setDefaultExecutor(RTExecutor *executor) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->defaultExecutor = executor;
}

RT_RET RTScheduler::setExecutor(RTTaskNode *node, RTExecutor *executor) {
    if (!node || !executor)
        return RT_ERR_NULL_PTR;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->executors[node] = executor;
    return RT_OK;
}

void RTScheduler::scheduleNode(RTTaskNode *node, RTTaskNodeContext *context) {
    RTExecutor *executor = nullptr;
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        if (!node || mImpl->state != Impl::State::Running || !mImpl->queuesRunning) {
            return;
        }
        executor = mImpl->executorFor(node);
        if (!executor)
            return;
        ++mImpl->activeTasks;
    }

    if (!context)
        context = node->getDefaultNodeContext();
    executor->schedule([impl = mImpl.get(), node, context] {
        const RT_RET result = context ? node->processNode(context) : RT_ERR_NULL_PTR;
        if (result != RT_OK && result != RT_ERR_NO_BUFFER && result != RT_ERR_RETRY &&
            result != RT_ERR_SUSPEND) {
            std::lock_guard<std::mutex> lock(impl->mutex);
            impl->hasError = true;
        }
        impl->taskCompleted();
    });
}

void RTScheduler::scheduleUnthrottledNodes(const std::vector<RTTaskNode *> &nodesToSchedule) {
    for (RTTaskNode *node : nodesToSchedule)
        scheduleNode(node, nullptr);
}

void RTScheduler::scheduleSuspendThrottledNodes(const std::vector<RTTaskNode *> &nodes) {
    scheduleUnthrottledNodes(nodes);
}

RT_RET RTScheduler::waitUntilDone(INT64 timeoutUs) {
    std::unique_lock<std::mutex> lock(mImpl->mutex);
    return wait_for_predicate(mImpl->condition, lock, timeoutUs, [this] {
        return mImpl->state == Impl::State::Terminated;
    });
}

RT_RET RTScheduler::waitUntilIdle(INT64 timeoutUs) {
    std::unique_lock<std::mutex> lock(mImpl->mutex);
    return wait_for_predicate(mImpl->condition, lock, timeoutUs, [this] {
        return mImpl->activeTasks == 0;
    });
}

RT_RET RTScheduler::waitUntilEos(INT64 timeoutUs) {
    std::unique_lock<std::mutex> lock(mImpl->mutex);
    return wait_for_predicate(mImpl->condition, lock, timeoutUs, [this] {
        return mImpl->eos;
    });
}

RT_RET RTScheduler::cleanupAfterRun() {
    reset();
    return RT_OK;
}

RT_RET RTScheduler::applicationThreadAwait(const std::function<bool()> &wakeupCondition,
                                           INT64 timeoutUs) {
    if (!wakeupCondition)
        return RT_ERR_NULL_PTR;
    std::unique_lock<std::mutex> lock(mImpl->mutex);
    return wait_for_predicate(mImpl->condition, lock, timeoutUs, wakeupCondition);
}

void RTScheduler::queueIdleStateChanged(bool idle) {
    if (idle)
        handleIdle();
}

void RTScheduler::addNodeToOpenedQueue(RTTaskNode *node) {
    if (!node)
        return;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->openedNodes.insert(node);
}

void RTScheduler::removeNodeFromOpenedQueue(RTTaskNode *node) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->openedNodes.erase(node);
    mImpl->condition.notify_all();
}

RT_BOOL RTScheduler::checkScheduleDone() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return (mImpl->activeTasks == 0 && mImpl->openedNodes.empty()) ? RT_TRUE : RT_FALSE;
}

RT_RET RTScheduler::tryToScheduleIdleNode() {
    return RT_OK;
}

void RTScheduler::handleIdle() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->condition.notify_all();
}

bool RTScheduler::isIdle() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->activeTasks == 0;
}

RT_BOOL RTScheduler::isRunning() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->state == Impl::State::Running ? RT_TRUE : RT_FALSE;
}

void RTScheduler::start() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->state = Impl::State::Running;
    mImpl->queuesRunning = true;
    mImpl->eos = false;
    mImpl->hasError = false;
    mImpl->condition.notify_all();
}

void RTScheduler::flush() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->observedGeneration = 0;
    mImpl->condition.notify_all();
}

void RTScheduler::reset() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->state = Impl::State::NotStarted;
    mImpl->queuesRunning = false;
    mImpl->eos = false;
    mImpl->hasError = false;
    mImpl->graphInputThrottled = false;
    mImpl->observedGeneration = 0;
    mImpl->openedNodes.clear();
    mImpl->condition.notify_all();
}

void RTScheduler::quit() {
    stop();
}

void RTScheduler::reachEos() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->eos = true;
    if (mImpl->state == Impl::State::Cancelling && mImpl->activeTasks == 0)
        mImpl->state = Impl::State::Terminated;
    mImpl->condition.notify_all();
}

void RTScheduler::pause() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->state == Impl::State::Running)
        mImpl->state = Impl::State::Paused;
    mImpl->queuesRunning = false;
    mImpl->condition.notify_all();
}

void RTScheduler::resume() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->state == Impl::State::Paused)
        mImpl->state = Impl::State::Running;
    mImpl->queuesRunning = true;
    mImpl->condition.notify_all();
}

void RTScheduler::stop() {
    std::unique_lock<std::mutex> lock(mImpl->mutex);
    if (mImpl->state == Impl::State::Terminated)
        return;
    mImpl->state = Impl::State::Cancelling;
    mImpl->queuesRunning = false;
    mImpl->condition.wait(lock, [this] {
        return mImpl->activeTasks == 0;
    });
    mImpl->state = Impl::State::Terminated;
    mImpl->condition.notify_all();
}

void RTScheduler::pauseAndWait() {
    pause();
    (void)waitForUntilPaused();
}

void RTScheduler::setQueuesRunning(bool running) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->queuesRunning = running;
    mImpl->condition.notify_all();
}

void RTScheduler::submitWaitingTasksOnQueues() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->condition.notify_all();
}

void RTScheduler::notifyHasError(RT_BOOL hasError) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->hasError = hasError != RT_FALSE;
    mImpl->condition.notify_all();
}

void RTScheduler::notifySchedulerPaused() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->condition.notify_all();
}

void RTScheduler::throttledGraphInputStream() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->graphInputThrottled = true;
}

void RTScheduler::unthrottledGraphInputStream() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->graphInputThrottled = false;
    mImpl->condition.notify_all();
}

RT_RET RTScheduler::waitUntilGraphInputStreamUnthrottled(void *, INT64 timeoutUs) {
    std::unique_lock<std::mutex> lock(mImpl->mutex);
    return wait_for_predicate(mImpl->condition, lock, timeoutUs, [this] {
        return !mImpl->graphInputThrottled;
    });
}

void RTScheduler::emittedObservedOutput() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    ++mImpl->observedGeneration;
    mImpl->condition.notify_all();
}

RT_RET RTScheduler::waitForObservedOutput(INT64 timeoutUs) {
    std::unique_lock<std::mutex> lock(mImpl->mutex);
    const UINT64 initial = mImpl->observedGeneration;
    return wait_for_predicate(mImpl->condition, lock, timeoutUs, [this, initial] {
        return mImpl->observedGeneration != initial;
    });
}

RT_RET RTScheduler::waitForUntilPaused() {
    std::unique_lock<std::mutex> lock(mImpl->mutex);
    return wait_for_predicate(mImpl->condition, lock, -1, [this] {
        return mImpl->state == Impl::State::Paused || mImpl->state == Impl::State::Terminated;
    });
}
