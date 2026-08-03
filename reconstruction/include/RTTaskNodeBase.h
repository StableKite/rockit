#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "RTStreamInfo.h"
#include "RTTaskNodeContextManager.h"
#include "rt_header.h"
#include "rt_metadata.h"

typedef struct _RTTaskNodeInfo {
    INT32 nodeId;
    std::string nodeName;
} RTTaskNodeInfo;

class RTInputStreamManager;
class RTOutputStreamManager;
class RTSchedulerQueue;
class RTMediaBufferPool;
class RTTaskNodeStat;

class RTTaskNodeBase {
 public:
    RTTaskNodeBase();
    virtual ~RTTaskNodeBase();
    RT_RET prepareForRun(RtMetaData *options,
                         std::function<void()> sourceNodeOpenedCallback,
                         std::function<void(RTTaskNodeContext *)> scheduleCallback);
    RT_RET cleanupAfterRun();
    INT32 getID() const;
    RtMetaData *getOptions();
    RT_BOOL isSource() const;
    RT_BOOL isSink() const;
    RT_BOOL hasMirror(RT_BOOL nodeOnly = RT_FALSE) const;
    INT32 numInputStreams() const;
    INT32 numOutputStreams() const;
    RT_RET processNode(RTTaskNodeContext *context);
    RT_RET quickProcessNode(RTTaskNodeContext *context);
    void nodeOpened();
    bool closed();
    RT_RET flush();
    RT_RET reset();
    RT_BOOL isDone();
    RT_RET closeNode();
    RT_RET closeInputStreams();
    RT_RET closeOutputStreams();
    void setMaxInputStreamQueueSize(INT32 maxQueueSize);
    void setMaxBatchPrcoessSize(INT32 maxBatchSize);
    void setQueueSizeCallbacks(
        std::function<void(RTInputStreamManager *, bool *)> becomesFullCallback,
        std::function<void(RTInputStreamManager *, bool *)> becomesNotFullCallback,
        std::function<void(RTInputStreamManager *, bool *)> becomesEmptyCallback);
    RT_RET initialize(INT32 nodeId, RTTaskNodeInfo *nodeInfo);
    RT_RET initialize(INT32 nodeId, RTTaskNodeInfo *nodeInfo,
                      RTInputStreamManager *inputManager,
                      RTOutputStreamManager *outputManager);
    RT_RET initialize(INT32 nodeId, RTTaskNodeInfo *nodeInfo,
                      const std::vector<RTInputStreamManager *> &inputManagers,
                      const std::vector<RTOutputStreamManager *> &outputManagers);
    RT_RET addStreamSource(INT32 streamId, RTOutputStreamManager *lastOutputManager);
    RT_RET addStreamSource(RTOutputStreamManager *lastOutputManager);
    RT_RET removeStreamSource(RTOutputStreamManager *outputManager);
    const std::vector<INT32> &getStreamSource();
    void checkIfBecameReady();
    RT_BOOL tryToBeginScheduling();
    void schedulingLoop();
    void endScheduling();
    RTTaskNodeContext *getDefaultNodeContext() const;
    RTSchedulerQueue *getSchedulerQueue() const;
    void setSchedulerQueue(RTSchedulerQueue *queue);
    RT_RET sendInterrupt(std::string reason);
    RT_RET cancelInterrupt(std::string reason);
    RT_RET dump();
    RT_RET attachOutStreamPool(RTMediaBufferPool *pool,
                               std::string streamType = "none");
    RT_RET detachOutStreamPool(std::string streamType = "none");
    void setComponentUid(INT32 componentUid);
    INT32 getComponentUid();

    virtual RT_RET open(RTTaskNodeContext *context) = 0;
    virtual RT_RET process(RTTaskNodeContext *context) = 0;
    virtual RT_RET close(RTTaskNodeContext *context) = 0;
    virtual RT_RET invoke(RtMetaData *meta);
    virtual RT_RET queryStat(RTTaskNodeStat *stat);
    virtual RT_RET queryStatInternal(RTTaskNodeStat *nodeStat);

 protected:
    virtual RT_RET initSupportOptions();
    virtual RT_RET invokeInternal(RtMetaData *meta);

 private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};
