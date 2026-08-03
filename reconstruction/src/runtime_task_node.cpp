#include "RTTaskNode.h"
#include "RTTaskNodeContext.h"
#include "RTTaskNodeContextManager.h"
#include "RTTaskNodeFactory.h"
#include "RTStreamInfo.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <utility>

RTStreamInfo::RTStreamInfo() : mStreamType("none"), mStreamName("none") {}
RTStreamInfo::~RTStreamInfo() = default;
void RTStreamInfo::setStreamType(std::string value) {
    mStreamType = std::move(value);
}
std::string RTStreamInfo::streamType() {
    return mStreamType;
}
void RTStreamInfo::setStreamName(std::string value) {
    mStreamName = std::move(value);
}
std::string RTStreamInfo::streamName() {
    return mStreamName;
}
void RTStreamInfo::setStreamId(INT32 value) {
    mStreamId = value;
}
INT32 RTStreamInfo::streamId() {
    return mStreamId;
}

namespace {
std::string normalize_stream(std::string value) {
    return value.empty() ? std::string("none") : value;
}
} // namespace

struct RTTaskNodeContext::Impl {
    explicit Impl(std::string name,
                  INT32 id,
                  std::vector<RTStreamInfo *> *in,
                  std::vector<RTStreamInfo *> *out)
        : nodeName(std::move(name)), nodeId(id), inputInfos(in), outputInfos(out) {
        if (inputInfos) {
            for (auto *info : *inputInfos) {
                if (info)
                    inputs.try_emplace(normalize_stream(info->streamType()));
            }
        }
        if (outputInfos) {
            for (auto *info : *outputInfos) {
                if (info)
                    outputs.try_emplace(normalize_stream(info->streamType()));
            }
        }
    }

    mutable std::mutex mutex;
    std::condition_variable outputCondition;
    std::string nodeName;
    INT32 nodeId{-1};
    RtMetaData *options{nullptr};
    bool suspended{false};
    INT32 maxBatch{4};
    RTScheduleMode scheduleMode{RT_SCHEDULE_DEFAULT};
    RTBufferListener *listener{nullptr};
    std::vector<RTStreamInfo *> *inputInfos{nullptr};
    std::vector<RTStreamInfo *> *outputInfos{nullptr};
    std::map<std::string, std::deque<RTMediaBuffer *>> inputs;
    std::map<std::string, std::deque<RTMediaBuffer *>> outputs;
    std::map<std::string, INT32> outputCounts;
    std::map<std::string, RTMediaBufferPool *> outputPools;
    std::set<std::string> interrupts;
};

RTTaskNodeContext::RTTaskNodeContext(std::string nodeName,
                                     INT32 nodeId,
                                     std::vector<RTStreamInfo *> *inputInfos,
                                     std::vector<RTStreamInfo *> *outputInfos)
    : mImpl(std::make_unique<Impl>(std::move(nodeName), nodeId, inputInfos, outputInfos)) {}
RTTaskNodeContext::~RTTaskNodeContext() = default;
RT_RET RTTaskNodeContext::prepareForRun(RtMetaData *options) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->options = options;
    mImpl->suspended = false;
    return RT_OK;
}
RT_RET RTTaskNodeContext::cleanupAfterRun() {
    return flush();
}
const std::string &RTTaskNodeContext::nodeName() const {
    return mImpl->nodeName;
}
INT32 RTTaskNodeContext::nodeId() const {
    return mImpl->nodeId;
}
void RTTaskNodeContext::suspend() {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    mImpl->suspended = true;
}
void RTTaskNodeContext::resume() {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    mImpl->suspended = false;
}
bool RTTaskNodeContext::isSuspend() {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    return mImpl->suspended;
}
RtMetaData *RTTaskNodeContext::options() {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    return mImpl->options;
}
RT_RET RTTaskNodeContext::flush() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    for (auto &entry : mImpl->inputs)
        entry.second.clear();
    for (auto &entry : mImpl->outputs)
        entry.second.clear();
    mImpl->outputCondition.notify_all();
    return RT_OK;
}
void RTTaskNodeContext::sendInterrupt(std::string reason) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->interrupts.insert(std::move(reason));
}
void RTTaskNodeContext::cancelInterrupt(std::string reason) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->interrupts.erase(reason);
}
void RTTaskNodeContext::setOutputBufferListener(RTBufferListener *listener) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->listener = listener;
}
void RTTaskNodeContext::setMaxBatchPrcoessSize(INT32 size) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->maxBatch = std::max<INT32>(1, size);
}
INT32 RTTaskNodeContext::getMaxBatchPrcoessSize() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->maxBatch;
}
RTScheduleMode RTTaskNodeContext::getScheduleMode() {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->scheduleMode;
}
void RTTaskNodeContext::setScheduleMode(RTScheduleMode mode) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->scheduleMode = mode;
}
RTStreamInfo *RTTaskNodeContext::getInputInfo(std::string type) {
    if (!mImpl->inputInfos)
        return nullptr;
    type = normalize_stream(std::move(type));
    for (auto *info : *mImpl->inputInfos)
        if (info && normalize_stream(info->streamType()) == type)
            return info;
    return nullptr;
}
RTStreamInfo *RTTaskNodeContext::getOutputInfo(std::string type) {
    if (!mImpl->outputInfos)
        return nullptr;
    type = normalize_stream(std::move(type));
    for (auto *info : *mImpl->outputInfos)
        if (info && normalize_stream(info->streamType()) == type)
            return info;
    return nullptr;
}
RT_RET RTTaskNodeContext::getPackets(std::list<RTMediaBuffer *> *packets, std::string type) {
    if (!packets)
        return RT_ERR_NULL_PTR;
    packets->clear();
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->suspended)
        return RT_ERR_SUSPEND;
    auto it = mImpl->inputs.find(normalize_stream(std::move(type)));
    if (it == mImpl->inputs.end())
        return RT_ERR_VALUE;
    const INT32 limit = std::max<INT32>(1, mImpl->maxBatch);
    for (INT32 i = 0; i < limit && !it->second.empty(); ++i) {
        packets->push_back(it->second.front());
        it->second.pop_front();
    }
    return packets->empty() ? RT_ERR_NO_BUFFER : RT_OK;
}
RT_BOOL RTTaskNodeContext::hasInputStream(std::string type) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->inputs.count(normalize_stream(std::move(type))) ? RT_TRUE : RT_FALSE;
}
RT_BOOL RTTaskNodeContext::hasOutputStream(std::string type) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->outputs.count(normalize_stream(std::move(type))) ? RT_TRUE : RT_FALSE;
}
INT32 RTTaskNodeContext::inputQueueSize(std::string type) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    auto it = mImpl->inputs.find(normalize_stream(std::move(type)));
    return it == mImpl->inputs.end() ? 0 : static_cast<INT32>(it->second.size());
}
RT_BOOL RTTaskNodeContext::inputIsEmpty(std::string type) {
    return inputQueueSize(std::move(type)) == 0 ? RT_TRUE : RT_FALSE;
}
RT_BOOL RTTaskNodeContext::outputIsEmpty(std::string type) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    auto it = mImpl->outputs.find(normalize_stream(std::move(type)));
    return (it == mImpl->outputs.end() || it->second.empty()) ? RT_TRUE : RT_FALSE;
}
RTMediaBuffer *RTTaskNodeContext::inputHeadBuffer(std::string type) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    auto it = mImpl->inputs.find(normalize_stream(std::move(type)));
    return (it == mImpl->inputs.end() || it->second.empty()) ? nullptr : it->second.front();
}
RTMediaBuffer *RTTaskNodeContext::dequeInputBuffer(std::string type) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    auto it = mImpl->inputs.find(normalize_stream(std::move(type)));
    if (it == mImpl->inputs.end() || it->second.empty())
        return nullptr;
    RTMediaBuffer *value = it->second.front();
    it->second.pop_front();
    return value;
}
RT_RET RTTaskNodeContext::queueInputBuffer(RTMediaBuffer *packet, std::string type) {
    if (!packet)
        return RT_ERR_NULL_PTR;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (mImpl->suspended)
        return RT_ERR_SUSPEND;
    mImpl->inputs[normalize_stream(std::move(type))].push_back(packet);
    return RT_OK;
}
RTMediaBuffer *RTTaskNodeContext::dequeOutputBuffer(RT_BOOL block, UINT32, std::string type) {
    std::unique_lock<std::mutex> lock(mImpl->mutex);
    auto key = normalize_stream(std::move(type));
    auto ready = [&] {
        return !mImpl->outputs[key].empty();
    };
    if (block && !ready())
        mImpl->outputCondition.wait_for(lock, std::chrono::milliseconds(50), ready);
    if (!ready())
        return nullptr;
    auto *value = mImpl->outputs[key].front();
    mImpl->outputs[key].pop_front();
    return value;
}
RT_RET RTTaskNodeContext::queueOutputBuffer(RTMediaBuffer *packet, std::string type) {
    if (!packet)
        return RT_ERR_NULL_PTR;
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        if (mImpl->suspended)
            return RT_ERR_SUSPEND;
        const std::string key = normalize_stream(std::move(type));
        const auto countIt = mImpl->outputCounts.find(key);
        const INT32 limit = countIt == mImpl->outputCounts.end() ? 0 : countIt->second;
        auto &queue = mImpl->outputs[key];
        if (limit > 0 && static_cast<INT32>(queue.size()) >= limit)
            return RT_ERR_LIST_FULL;
        queue.push_back(packet);
    }
    mImpl->outputCondition.notify_one();
    return RT_OK;
}
RT_RET RTTaskNodeContext::getBufferStat(RTTaskNodeStat *stat) {
    return stat ? RT_OK : RT_ERR_NULL_PTR;
}
RT_RET RTTaskNodeContext::reallocOutputBuffers(INT32 size, std::string type) {
    if (size < 0)
        return RT_ERR_VALUE;
    return setOutputBufferCount(size, std::move(type));
}
RT_RET RTTaskNodeContext::attachOutStreamPool(RTMediaBufferPool *pool, std::string type) {
    if (!pool)
        return RT_ERR_NULL_PTR;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->outputPools[normalize_stream(std::move(type))] = pool;
    return RT_OK;
}
RT_RET RTTaskNodeContext::detachOutStreamPool(std::string type) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->outputPools.erase(normalize_stream(std::move(type))) ? RT_OK : RT_ERR_VALUE;
}
RT_RET RTTaskNodeContext::pollOutputBuffer(RT_BOOL block, UINT32, std::string type) {
    std::unique_lock<std::mutex> lock(mImpl->mutex);
    const std::string key = normalize_stream(std::move(type));
    auto ready = [&] {
        return !mImpl->outputs[key].empty() || mImpl->suspended;
    };
    if (block && !ready())
        mImpl->outputCondition.wait_for(lock, std::chrono::milliseconds(50), ready);
    if (mImpl->suspended)
        return RT_ERR_SUSPEND;
    return mImpl->outputs[key].empty() ? RT_ERR_NO_BUFFER : RT_OK;
}
INT32 RTTaskNodeContext::getOutputBufferSize(std::string type) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    auto it = mImpl->outputs.find(normalize_stream(std::move(type)));
    return it == mImpl->outputs.end() ? 0 : static_cast<INT32>(it->second.size());
}
RT_RET RTTaskNodeContext::setOutputBufferCount(INT32 count, std::string type) {
    if (count < 0)
        return RT_ERR_VALUE;
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->outputCounts[normalize_stream(std::move(type))] = count;
    return RT_OK;
}
INT32 RTTaskNodeContext::getOutputBufferCount(std::string type) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    auto it = mImpl->outputCounts.find(normalize_stream(std::move(type)));
    return it == mImpl->outputCounts.end() ? 0 : it->second;
}
RT_RET RTTaskNodeContext::dequeInputBufferByPerStream(std::vector<RTMediaBuffer *> *packets) {
    if (!packets)
        return RT_ERR_NULL_PTR;
    packets->clear();
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    for (auto &entry : mImpl->inputs) {
        if (!entry.second.empty()) {
            packets->push_back(entry.second.front());
            entry.second.pop_front();
        }
    }
    return packets->empty() ? RT_ERR_NO_BUFFER : RT_OK;
}
RT_RET RTTaskNodeContext::dump() {
    return RT_OK;
}

RTTaskNodeContextManager::RTTaskNodeContextManager() = default;
RTTaskNodeContextManager::~RTTaskNodeContextManager() = default;
RT_RET RTTaskNodeContextManager::initialize(std::string name,
                                            INT32 id,
                                            std::vector<RTStreamInfo *> *input,
                                            std::vector<RTStreamInfo *> *output,
                                            bool parallel) {
    mNodeName = std::move(name);
    mNodeId = id;
    mInputInfos = input;
    mOutputInfos = output;
    mParallelRunning = parallel;
    mDefaultContext = std::make_unique<RTTaskNodeContext>(mNodeName, mNodeId, input, output);
    return RT_OK;
}
RT_RET RTTaskNodeContextManager::prepareForRun(RtMetaData *options) {
    mOptions = options;
    return mDefaultContext ? mDefaultContext->prepareForRun(options) : RT_ERR_INIT;
}
RT_RET RTTaskNodeContextManager::cleanupAfterRun() {
    return mDefaultContext ? mDefaultContext->cleanupAfterRun() : RT_OK;
}
RTTaskNodeContext *RTTaskNodeContextManager::getDefaultNodeContext() const {
    return mDefaultContext.get();
}

struct RTTaskNodeBase::Impl {
    enum class Status {
        Uninitialized,
        Prepared,
        Opened,
        Active,
        Closed
    };
    mutable std::mutex mutex;
    INT32 nodeId{-1};
    INT32 componentUid{0};
    INT32 maxInputQueue{0};
    INT32 maxBatch{4};
    Status status{Status::Uninitialized};
    RtMetaData *options{nullptr};
    RTSchedulerQueue *schedulerQueue{nullptr};
    RTTaskNodeInfo info{};
    RTTaskNodeContextManager contextManager;
    std::vector<RTStreamInfo *> inputInfos;
    std::vector<RTStreamInfo *> outputInfos;
    std::vector<INT32> sourceNodes;
    std::vector<RTOutputStreamManager *> streamSources;
    std::function<void()> openedCallback;
    std::function<void(RTTaskNodeContext *)> scheduleCallback;
    std::set<std::string> interrupts;
    bool scheduling{false};
};

RTTaskNodeBase::RTTaskNodeBase() : mImpl(std::make_unique<Impl>()) {}
RTTaskNodeBase::~RTTaskNodeBase() = default;
RT_RET RTTaskNodeBase::initialize(INT32 nodeId, RTTaskNodeInfo *nodeInfo) {
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->nodeId = nodeId;
    mImpl->info.nodeId = nodeId;
    if (nodeInfo)
        mImpl->info = *nodeInfo;
    mImpl->contextManager.initialize(
        mImpl->info.nodeName, nodeId, &mImpl->inputInfos, &mImpl->outputInfos, false);
    mImpl->status = Impl::Status::Uninitialized;
    return RT_OK;
}
RT_RET RTTaskNodeBase::initialize(INT32 id,
                                  RTTaskNodeInfo *info,
                                  RTInputStreamManager *input,
                                  RTOutputStreamManager *output) {
    (void)input;
    (void)output;
    return initialize(id, info);
}
RT_RET RTTaskNodeBase::initialize(INT32 id,
                                  RTTaskNodeInfo *info,
                                  const std::vector<RTInputStreamManager *> &,
                                  const std::vector<RTOutputStreamManager *> &) {
    return initialize(id, info);
}
RT_RET RTTaskNodeBase::prepareForRun(RtMetaData *options,
                                     std::function<void()> opened,
                                     std::function<void(RTTaskNodeContext *)> schedule) {
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        if (mImpl->status == Impl::Status::Active || mImpl->status == Impl::Status::Opened)
            return RT_ERR_BAD;
        mImpl->options = options;
        mImpl->openedCallback = std::move(opened);
        mImpl->scheduleCallback = std::move(schedule);
    }
    RT_RET ret = initSupportOptions();
    if (ret != RT_OK)
        return ret;
    ret = mImpl->contextManager.prepareForRun(options);
    if (ret != RT_OK)
        return ret;
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        mImpl->status = Impl::Status::Prepared;
    }
    auto *context = getDefaultNodeContext();
    ret = open(context);
    if (ret != RT_OK)
        return ret;
    nodeOpened();
    return RT_OK;
}
RT_RET RTTaskNodeBase::cleanupAfterRun() {
    RT_RET closeResult = closeNode();
    RT_RET cleanup = mImpl->contextManager.cleanupAfterRun();
    return closeResult != RT_OK ? closeResult : cleanup;
}
INT32 RTTaskNodeBase::getID() const {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    return mImpl->nodeId;
}
RtMetaData *RTTaskNodeBase::getOptions() {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    return mImpl->options;
}
RT_BOOL RTTaskNodeBase::isSource() const {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    return mImpl->inputInfos.empty() ? RT_TRUE : RT_FALSE;
}
RT_BOOL RTTaskNodeBase::isSink() const {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    return mImpl->outputInfos.empty() ? RT_TRUE : RT_FALSE;
}
RT_BOOL RTTaskNodeBase::hasMirror(RT_BOOL) const {
    return RT_FALSE;
}
INT32 RTTaskNodeBase::numInputStreams() const {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    return static_cast<INT32>(mImpl->inputInfos.size());
}
INT32 RTTaskNodeBase::numOutputStreams() const {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    return static_cast<INT32>(mImpl->outputInfos.size());
}
RT_RET RTTaskNodeBase::processNode(RTTaskNodeContext *context) {
    if (!context)
        return RT_ERR_NULL_PTR;
    if (context->isSuspend())
        return RT_ERR_SUSPEND;
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        if (mImpl->status == Impl::Status::Closed || mImpl->status == Impl::Status::Uninitialized)
            return RT_ERR_BAD;
        mImpl->status = Impl::Status::Active;
    }
    return process(context);
}
RT_RET RTTaskNodeBase::quickProcessNode(RTTaskNodeContext *context) {
    return processNode(context);
}
void RTTaskNodeBase::nodeOpened() {
    std::function<void()> callback;
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        mImpl->status = Impl::Status::Opened;
        callback = mImpl->openedCallback;
    }
    if (callback)
        callback();
}
bool RTTaskNodeBase::closed() {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    return mImpl->status == Impl::Status::Closed;
}
RT_RET RTTaskNodeBase::flush() {
    auto *ctx = getDefaultNodeContext();
    return ctx ? ctx->flush() : RT_OK;
}
RT_RET RTTaskNodeBase::reset() {
    RT_RET r = flush();
    std::lock_guard<std::mutex> l(mImpl->mutex);
    mImpl->status = Impl::Status::Prepared;
    return r;
}
RT_BOOL RTTaskNodeBase::isDone() {
    return closed() ? RT_TRUE : RT_FALSE;
}
RT_RET RTTaskNodeBase::closeNode() {
    RTTaskNodeContext *context = getDefaultNodeContext();
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        if (mImpl->status == Impl::Status::Closed)
            return RT_OK;
        if (mImpl->status == Impl::Status::Uninitialized)
            return RT_ERR_BAD;
    }
    RT_RET ret = close(context);
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->status = Impl::Status::Closed;
    return ret;
}
RT_RET RTTaskNodeBase::closeInputStreams() {
    return RT_OK;
}
RT_RET RTTaskNodeBase::closeOutputStreams() {
    return RT_OK;
}
void RTTaskNodeBase::setMaxInputStreamQueueSize(INT32 size) {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    mImpl->maxInputQueue = std::max<INT32>(0, size);
}
void RTTaskNodeBase::setMaxBatchPrcoessSize(INT32 size) {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    mImpl->maxBatch = std::max<INT32>(1, size);
    if (auto *c = getDefaultNodeContext())
        c->setMaxBatchPrcoessSize(mImpl->maxBatch);
}
void RTTaskNodeBase::setQueueSizeCallbacks(std::function<void(RTInputStreamManager *, bool *)>,
                                           std::function<void(RTInputStreamManager *, bool *)>,
                                           std::function<void(RTInputStreamManager *, bool *)>) {}
RT_RET RTTaskNodeBase::addStreamSource(INT32 streamId, RTOutputStreamManager *source) {
    if (!source)
        return RT_ERR_NULL_PTR;
    std::lock_guard<std::mutex> l(mImpl->mutex);
    mImpl->sourceNodes.push_back(streamId);
    mImpl->streamSources.push_back(source);
    return RT_OK;
}
RT_RET RTTaskNodeBase::addStreamSource(RTOutputStreamManager *source) {
    return addStreamSource(static_cast<INT32>(mImpl->sourceNodes.size()), source);
}
RT_RET RTTaskNodeBase::removeStreamSource(RTOutputStreamManager *source) {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    auto it = std::find(mImpl->streamSources.begin(), mImpl->streamSources.end(), source);
    if (it == mImpl->streamSources.end())
        return RT_ERR_VALUE;
    auto idx = static_cast<std::size_t>(std::distance(mImpl->streamSources.begin(), it));
    mImpl->streamSources.erase(it);
    if (idx < mImpl->sourceNodes.size())
        mImpl->sourceNodes.erase(mImpl->sourceNodes.begin() + idx);
    return RT_OK;
}
const std::vector<INT32> &RTTaskNodeBase::getStreamSource() {
    return mImpl->sourceNodes;
}
void RTTaskNodeBase::checkIfBecameReady() {
    if (mImpl->scheduleCallback)
        mImpl->scheduleCallback(getDefaultNodeContext());
}
RT_BOOL RTTaskNodeBase::tryToBeginScheduling() {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    if (mImpl->scheduling)
        return RT_FALSE;
    mImpl->scheduling = true;
    return RT_TRUE;
}
void RTTaskNodeBase::schedulingLoop() {
    if (mImpl->scheduleCallback)
        mImpl->scheduleCallback(getDefaultNodeContext());
}
void RTTaskNodeBase::endScheduling() {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    mImpl->scheduling = false;
}
RTTaskNodeContext *RTTaskNodeBase::getDefaultNodeContext() const {
    return mImpl->contextManager.getDefaultNodeContext();
}
RTSchedulerQueue *RTTaskNodeBase::getSchedulerQueue() const {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    return mImpl->schedulerQueue;
}
void RTTaskNodeBase::setSchedulerQueue(RTSchedulerQueue *queue) {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    mImpl->schedulerQueue = queue;
}
RT_RET RTTaskNodeBase::sendInterrupt(std::string reason) {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    mImpl->interrupts.insert(reason);
    if (auto *c = getDefaultNodeContext())
        c->sendInterrupt(reason);
    return RT_OK;
}
RT_RET RTTaskNodeBase::cancelInterrupt(std::string reason) {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    mImpl->interrupts.erase(reason);
    if (auto *c = getDefaultNodeContext())
        c->cancelInterrupt(reason);
    return RT_OK;
}
RT_RET RTTaskNodeBase::dump() {
    return RT_OK;
}
RT_RET RTTaskNodeBase::attachOutStreamPool(RTMediaBufferPool *pool, std::string type) {
    auto *c = getDefaultNodeContext();
    return c ? c->attachOutStreamPool(pool, std::move(type)) : RT_ERR_INIT;
}
RT_RET RTTaskNodeBase::detachOutStreamPool(std::string type) {
    auto *c = getDefaultNodeContext();
    return c ? c->detachOutStreamPool(std::move(type)) : RT_ERR_INIT;
}
void RTTaskNodeBase::setComponentUid(INT32 uid) {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    mImpl->componentUid = uid;
}
INT32 RTTaskNodeBase::getComponentUid() {
    std::lock_guard<std::mutex> l(mImpl->mutex);
    return mImpl->componentUid;
}
RT_RET RTTaskNodeBase::invoke(RtMetaData *meta) {
    return invokeInternal(meta);
}
RT_RET RTTaskNodeBase::queryStat(RTTaskNodeStat *stat) {
    return stat ? queryStatInternal(stat) : RT_ERR_NULL_PTR;
}
RT_RET RTTaskNodeBase::queryStatInternal(RTTaskNodeStat *) {
    return RT_ERR_UNSUPPORT;
}
RT_RET RTTaskNodeBase::initSupportOptions() {
    return RT_OK;
}
RT_RET RTTaskNodeBase::invokeInternal(RtMetaData *) {
    return RT_ERR_UNSUPPORT;
}

RTTaskNode::RTTaskNode() = default;
RTTaskNode::~RTTaskNode() = default;
RT_RET RTTaskNode::invoke(RtMetaData *meta) {
    return invokeInternal(meta);
}
RT_RET RTTaskNode::initSupportOptions() {
    return RT_OK;
}
RT_RET RTTaskNode::invokeInternal(RtMetaData *) {
    return RT_ERR_UNSUPPORT;
}

namespace {
std::mutex &factory_mutex() {
    static std::mutex value;
    return value;
}
} // namespace
RTTaskNodeFactory::RTTaskNodeFactory() = default;
RTTaskNodeFactory::~RTTaskNodeFactory() = default;
RTTaskNodeFactory *RTTaskNodeFactory::instance() {
    static RTTaskNodeFactory factory;
    return &factory;
}
RT_RET RTTaskNodeFactory::registerStub(RTNodeStub *stub) {
    if (!stub || !stub->mName || !stub->mCreateObj)
        return RT_ERR_NULL_PTR;
    std::lock_guard<std::mutex> lock(factory_mutex());
    for (auto *item : mStubs)
        if (item && (item->mUid == stub->mUid || std::strcmp(item->mName, stub->mName) == 0))
            return RT_ERR_VALUE;
    mStubs.push_back(stub);
    return RT_OK;
}
RTTaskNode *RTTaskNodeFactory::createNode(RTNodeStub *stub) {
    if (!stub || !stub->mCreateObj)
        return nullptr;
    auto *node = stub->mCreateObj();
    if (node)
        node->setComponentUid(stub->mUid);
    return node;
}
RTTaskNode *RTTaskNodeFactory::createNode(const char *name) {
    if (!name)
        return nullptr;
    std::lock_guard<std::mutex> lock(factory_mutex());
    for (auto *stub : mStubs)
        if (stub && stub->mName && std::strcmp(stub->mName, name) == 0)
            return createNode(stub);
    return nullptr;
}
RTTaskNode *RTTaskNodeFactory::createNode(RTStubUid uid) {
    return createNode(getStubByUid(uid));
}
RTTaskNode *RTTaskNodeFactory::createNode(RTPadCaps *caps) {
    if (!caps)
        return nullptr;
    std::lock_guard<std::mutex> lock(factory_mutex());
    for (auto *stub : mStubs)
        if (stub && (stub->mCapsSink.mBufferType == caps->mBufferType ||
                     stub->mCapsSrc.mBufferType == caps->mBufferType))
            return createNode(stub);
    return nullptr;
}
RTStubUid RTTaskNodeFactory::getUidByName(const char *name) {
    if (!name)
        return kStubNone;
    std::lock_guard<std::mutex> lock(factory_mutex());
    for (auto *stub : mStubs)
        if (stub && stub->mName && std::strcmp(stub->mName, name) == 0)
            return static_cast<RTStubUid>(stub->mUid);
    return kStubNone;
}
RTNodeStub *RTTaskNodeFactory::getStubByUid(RTStubUid uid) {
    std::lock_guard<std::mutex> lock(factory_mutex());
    for (auto *stub : mStubs)
        if (stub && stub->mUid == static_cast<INT32>(uid))
            return stub;
    return nullptr;
}
RTTaskNodeToken::RTTaskNodeToken(const RTNodeStub &stub) : mNodeStub(new RTNodeStub(stub)) {
    (void)RTTaskNodeFactory::instance()->registerStub(mNodeStub);
}
RTTaskNodeToken::~RTTaskNodeToken() = default;
