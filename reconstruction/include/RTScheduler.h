#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "RTExecutor.h"
#include "rt_header.h"

class RTTaskNode;
class RTTaskNodeContext;
class RTTaskGraph;

class RTScheduler {
 public:
    RTScheduler(const RTScheduler &) = delete;
    RTScheduler &operator=(const RTScheduler &) = delete;
    explicit RTScheduler(RTTaskGraph *ctx);
    ~RTScheduler();

    void setDefaultExecutor(RTExecutor *executor);
    RT_RET setExecutor(RTTaskNode *node, RTExecutor *executor);
    void scheduleNode(RTTaskNode *node, RTTaskNodeContext *nodeContext);
    void scheduleUnthrottledNodes(const std::vector<RTTaskNode *> &nodesToSchedule);
    void scheduleSuspendThrottledNodes(const std::vector<RTTaskNode *> &nodes);

    RT_RET waitUntilDone(INT64 timeoutUs = -1);
    RT_RET waitUntilIdle(INT64 timeoutUs = -1);
    RT_RET waitUntilEos(INT64 timeoutUs = -1);
    RT_RET cleanupAfterRun();
    RT_RET applicationThreadAwait(const std::function<bool()> &wakeupCondition,
                                  INT64 timeoutUs = -1);

    void queueIdleStateChanged(bool idle);
    void addNodeToOpenedQueue(RTTaskNode *node);
    void removeNodeFromOpenedQueue(RTTaskNode *node);
    RT_BOOL checkScheduleDone();
    RT_RET tryToScheduleIdleNode();
    void handleIdle();
    bool isIdle();
    RT_BOOL isRunning();
    void start();
    void flush();
    void reset();
    void quit();
    void reachEos();
    void pause();
    void resume();
    void stop();
    void pauseAndWait();
    void setQueuesRunning(bool running);
    void submitWaitingTasksOnQueues();
    void notifyHasError(RT_BOOL hasError);
    void notifySchedulerPaused();
    void throttledGraphInputStream();
    void unthrottledGraphInputStream();
    RT_RET waitUntilGraphInputStreamUnthrottled(void *secondaryMutex,
                                                INT64 timeoutUs);
    void emittedObservedOutput();
    RT_RET waitForObservedOutput(INT64 timeoutUs = -1);
    RT_RET waitForUntilPaused();

 private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};
