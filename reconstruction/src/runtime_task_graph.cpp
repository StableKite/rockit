#include "RTTaskGraph.h"

#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "RTExecutor.h"
#include "RTScheduler.h"
#include "RTTaskNode.h"
#include "RTTaskNodeFactory.h"

namespace {
std::atomic<UINT64> g_next_graph_uid{1};
}

struct RTTaskGraph::Impl {
    explicit Impl(UINT64 graphUid, std::string graphName)
        : uid(graphUid), name(std::move(graphName)),
          ownedExecutor(std::make_unique<RTThreadPoolExecutor>(2)),
          scheduler(std::make_unique<RTScheduler>(nullptr)) {
        scheduler->setDefaultExecutor(ownedExecutor.get());
    }

    UINT64 uid;
    std::string name;
    std::mutex mutex;
    bool prepared = false;
    bool released = false;
    RTGraphIOStreamMode ioMode = WAIT_TILL_NOT_FULL;
    std::unique_ptr<RTExecutor> ownedExecutor;
    RTExecutor *externalExecutor = nullptr;
    std::unique_ptr<RTScheduler> scheduler;
    RTCBHandle nextCallback = 1;
    RTLinkHandle nextLink = 1;
    std::map<RTCBHandle, std::pair<std::string, std::function<RT_RET(RTMediaBuffer *)>>> callbacks;
    std::map<INT32, RTCBHandle> legacyCallbacks;
    std::map<RTLinkHandle, RTTaskGraph *> downGraphs;
    INT32 nextNodeId = 1;
    std::vector<std::unique_ptr<RTTaskNode>> ownedNodes;
    std::set<std::pair<RTTaskNode *, RTTaskNode *>> nodeLinks;
};

RTTaskGraph::RTTaskGraph(const char *tagName)
    : RTTaskGraph(g_next_graph_uid.fetch_add(1), tagName) {}

RTTaskGraph::RTTaskGraph(UINT64 uid, const char *tagName)
    : mImpl(std::make_unique<Impl>(uid, tagName ? tagName : "")) {}

RTTaskGraph::~RTTaskGraph() = default;

UINT64 RTTaskGraph::getUid() {
    return mImpl->uid;
}
std::string RTTaskGraph::getName() {
    return mImpl->name;
}

RT_RET RTTaskGraph::autoBuild(const char *, RT_BOOL) {
    return RT_ERR_UNSUPPORT;
}

RT_RET RTTaskGraph::prepare(RtMetaData *params) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->released)
        return RT_ERR_BAD;
    mImpl->scheduler->reset();
    for (auto &node : mImpl->ownedNodes) {
        RT_RET result = node->prepareForRun(
            params,
            [scheduler = mImpl->scheduler.get(), raw = node.get()] {
                scheduler->addNodeToOpenedQueue(raw);
            },
            [scheduler = mImpl->scheduler.get(), raw = node.get()](RTTaskNodeContext *context) {
                scheduler->scheduleNode(raw, context);
            });
        if (result != RT_OK)
            return result;
    }
    mImpl->prepared = true;
    return RT_OK;
}

RT_RET RTTaskGraph::start() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->released || !mImpl->prepared)
        return RT_ERR_INIT;
    mImpl->scheduler->start();
    for (auto &node : mImpl->ownedNodes) {
        mImpl->scheduler->scheduleNode(node.get(), node->getDefaultNodeContext());
    }
    return RT_OK;
}

RT_RET RTTaskGraph::stop(RT_BOOL block) {
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        if (mImpl->released)
            return RT_ERR_BAD;
        mImpl->scheduler->stop();
    }
    return block ? mImpl->scheduler->waitUntilDone(-1) : RT_OK;
}

RT_RET RTTaskGraph::resume() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->released)
        return RT_ERR_BAD;
    mImpl->scheduler->resume();
    return RT_OK;
}

RT_RET RTTaskGraph::pause(RT_BOOL block) {
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        if (mImpl->released)
            return RT_ERR_BAD;
        mImpl->scheduler->pause();
    }
    return block ? mImpl->scheduler->waitForUntilPaused() : RT_OK;
}

RT_RET RTTaskGraph::flush() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->released)
        return RT_ERR_BAD;
    mImpl->scheduler->flush();
    return RT_OK;
}

RT_RET RTTaskGraph::reset() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->released)
        return RT_ERR_BAD;
    mImpl->scheduler->reset();
    mImpl->prepared = false;
    return RT_OK;
}

RT_RET RTTaskGraph::release() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->released)
        return RT_OK;
    mImpl->scheduler->stop();
    mImpl->callbacks.clear();
    mImpl->legacyCallbacks.clear();
    mImpl->downGraphs.clear();
    for (auto &node : mImpl->ownedNodes)
        (void)node->cleanupAfterRun();
    mImpl->nodeLinks.clear();
    mImpl->ownedNodes.clear();
    mImpl->prepared = false;
    mImpl->released = true;
    return RT_OK;
}

RT_RET RTTaskGraph::invoke(INT32 cmd, RtMetaData *params) {
    switch (cmd) {
    case GRAPH_CMD_PREPARE:
        return prepare(params);
    case GRAPH_CMD_START:
        return start();
    case GRAPH_CMD_STOP:
        return stop(RT_TRUE);
    case GRAPH_CMD_RESUME:
        return resume();
    case GRAPH_CMD_PAUSE:
        return pause(RT_TRUE);
    case GRAPH_CMD_FLUSH:
        return flush();
    default:
        return RT_ERR_UNSUPPORT;
    }
}

RT_RET RTTaskGraph::waitForObservedOutput(INT64 timeoutUs) {
    return mImpl->scheduler->waitForObservedOutput(timeoutUs);
}
RT_RET RTTaskGraph::waitUntilDone(INT64 timeoutUs) {
    return mImpl->scheduler->waitUntilDone(timeoutUs);
}
RT_RET RTTaskGraph::waitUntilEos(INT64 timeoutUs) {
    return mImpl->scheduler->waitUntilEos(timeoutUs);
}
RT_RET RTTaskGraph::queryStat(RTTaskGraphStat *) {
    return RT_ERR_UNSUPPORT;
}
RT_RET RTTaskGraph::dump() {
    return RT_OK;
}

RTCBHandle RTTaskGraph::observeOutputStream(const std::string streamName,
                                            std::function<RT_RET(RTMediaBuffer *)> streamCallback) {
    if (!streamCallback)
        return RT_INVALID_CB_HANDLE;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->released)
        return RT_INVALID_CB_HANDLE;
    const RTCBHandle handle = mImpl->nextCallback++;
    mImpl->callbacks.emplace(handle, std::make_pair(streamName, std::move(streamCallback)));
    return handle;
}

RT_RET RTTaskGraph::cancelObserveOutputStream(RTCBHandle handle) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->callbacks.erase(handle) ? RT_OK : RT_ERR_VALUE;
}

RT_RET RTTaskGraph::observeOutputStream(const std::string &streamName,
                                        INT32 streamId,
                                        std::function<RT_RET(RTMediaBuffer *)> streamCallback) {
    const RTCBHandle handle = observeOutputStream(streamName, std::move(streamCallback));
    if (handle == RT_INVALID_CB_HANDLE)
        return RT_ERR_VALUE;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->legacyCallbacks[streamId] = handle;
    return RT_OK;
}

RT_RET RTTaskGraph::cancelObserveOutputStream(INT32 streamId) {
    RTCBHandle handle = RT_INVALID_CB_HANDLE;
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        auto it = mImpl->legacyCallbacks.find(streamId);
        if (it == mImpl->legacyCallbacks.end())
            return RT_ERR_VALUE;
        handle = it->second;
        mImpl->legacyCallbacks.erase(it);
    }
    return cancelObserveOutputStream(handle);
}

RT_RET RTTaskGraph::addSubGraph(const char *) {
    return RT_ERR_UNSUPPORT;
}
RT_RET RTTaskGraph::removeSubGraph(const char *) {
    return RT_ERR_UNSUPPORT;
}
RT_BOOL RTTaskGraph::hasMirror(INT32, RT_BOOL) {
    return RT_FALSE;
}

RT_RET RTTaskGraph::setExternalExecutor(RTExecutor *executor) {
    if (!executor)
        return RT_ERR_NULL_PTR;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->externalExecutor = executor;
    mImpl->scheduler->setDefaultExecutor(executor);
    return RT_OK;
}

RTLinkHandle RTTaskGraph::addDownGraph(RTTaskGraph *graph, std::string, std::string) {
    if (!graph)
        return RT_INVALID_LINK_HANDLE;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    const RTLinkHandle handle = mImpl->nextLink++;
    mImpl->downGraphs[handle] = graph;
    return handle;
}

RT_RET RTTaskGraph::removeDownGraph(RTLinkHandle handle) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->downGraphs.erase(handle) ? RT_OK : RT_ERR_VALUE;
}

RT_RET RTTaskGraph::setGraphOutputStreamAddMode(RTGraphIOStreamMode mode) {
    if (mode < WAIT_TILL_NOT_FULL || mode > DROP_IF_FULL)
        return RT_ERR_VALUE;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->ioMode = mode;
    return RT_OK;
}

RTTaskNode *RTTaskGraph::createNode(std::string nodeConfig, std::string) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->released || nodeConfig.empty())
        return nullptr;
    std::unique_ptr<RTTaskNode> node(RTTaskNodeFactory::instance()->createNode(nodeConfig.c_str()));
    if (!node)
        return nullptr;
    RTTaskNodeInfo info{mImpl->nextNodeId++, nodeConfig};
    if (node->initialize(info.nodeId, &info) != RT_OK)
        return nullptr;
    RTTaskNode *result = node.get();
    mImpl->ownedNodes.push_back(std::move(node));
    return result;
}
RT_RET RTTaskGraph::linkNode(RTTaskNode *src, RTTaskNode *dst) {
    if (!src || !dst || src == dst)
        return RT_ERR_VALUE;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->nodeLinks.emplace(src, dst).second ? RT_OK : RT_ERR_VALUE;
}
RT_RET RTTaskGraph::unlinkNode(RTTaskNode *src, RTTaskNode *dst) {
    if (!src || !dst)
        return RT_ERR_NULL_PTR;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->nodeLinks.erase({src, dst}) ? RT_OK : RT_ERR_VALUE;
}
RT_BOOL RTTaskGraph::hasLinkMode(std::string) {
    return RT_FALSE;
}
RT_RET RTTaskGraph::selectLinkMode(std::string) {
    return RT_ERR_UNSUPPORT;
}
RT_RET RTTaskGraph::clearLinkShips() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->nodeLinks.clear();
    return RT_OK;
}

void RTTaskGraph::sendInterrupt(std::string) {
    mImpl->scheduler->notifyHasError(RT_TRUE);
}

RT_BOOL RTTaskGraph::unthrottleSources() {
    mImpl->scheduler->unthrottledGraphInputStream();
    return RT_TRUE;
}

RT_RET RTTaskGraph::addPacketToInputStream(std::string, INT64, RTMediaBuffer *) {
    return RT_ERR_UNSUPPORT;
}
RTCBHandle RTTaskGraph::addOutputStreamPoller(const std::string streamName) {
    return observeOutputStream(streamName, [](RTMediaBuffer *) {
        return RT_OK;
    });
}
RT_RET RTTaskGraph::removeOutputStreamPoller(RTCBHandle handle) {
    return cancelObserveOutputStream(handle);
}
RT_RET RTTaskGraph::checkInputStreamFull(std::string) {
    return RT_OK;
}
