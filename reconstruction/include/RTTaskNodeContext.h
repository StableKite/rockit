#pragma once

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "RTStreamInfo.h"
#include "rt_header.h"
#include "rt_metadata.h"

class RTMediaBuffer;
class RTBufferListener;
class RTMediaBufferPool;
class RTTaskNodeStat;

typedef enum _RTScheduleMode {
    RT_SCHEDULE_DEFAULT = 0,
    RT_SCHEDULE_PER_STREAM = 1,
} RTScheduleMode;

class RTTaskNodeContext {
 public:
    explicit RTTaskNodeContext(std::string nodeName, INT32 nodeId,
        std::vector<RTStreamInfo *> *inputInfos,
        std::vector<RTStreamInfo *> *outputInfos);
    ~RTTaskNodeContext();
    RTTaskNodeContext(const RTTaskNodeContext &) = delete;
    RTTaskNodeContext &operator=(const RTTaskNodeContext &) = delete;

    RT_RET prepareForRun(RtMetaData *options);
    RT_RET cleanupAfterRun();
    const std::string &nodeName() const;
    INT32 nodeId() const;
    void suspend();
    void resume();
    RT_RET flush();
    bool isSuspend();
    RtMetaData *options();
    void sendInterrupt(std::string reason);
    void cancelInterrupt(std::string reason);
    void setOutputBufferListener(RTBufferListener *listener);
    void setMaxBatchPrcoessSize(INT32 maxBatchSize);
    INT32 getMaxBatchPrcoessSize();
    RTScheduleMode getScheduleMode();
    void setScheduleMode(RTScheduleMode mode);

    RTStreamInfo *getInputInfo(std::string streamType = "none");
    RTStreamInfo *getOutputInfo(std::string streamType = "none");
    RT_RET getPackets(std::list<RTMediaBuffer *> *packets,
                      std::string streamType = "none");
    RT_BOOL hasInputStream(std::string streamType = "none");
    RT_BOOL hasOutputStream(std::string streamType = "none");
    INT32 inputQueueSize(std::string streamType = "none");
    RT_BOOL inputIsEmpty(std::string streamType = "none");
    RT_BOOL outputIsEmpty(std::string streamType = "none");
    RTMediaBuffer *inputHeadBuffer(std::string streamType = "none");
    RTMediaBuffer *dequeInputBuffer(std::string streamType = "none");
    RT_RET queueInputBuffer(RTMediaBuffer *packet,
                            std::string streamType = "none");
    RTMediaBuffer *dequeOutputBuffer(RT_BOOL block = RT_TRUE, UINT32 size = 0,
                                     std::string streamType = "none");
    RT_RET queueOutputBuffer(RTMediaBuffer *packet,
                             std::string streamType = "none");
    RT_RET getBufferStat(RTTaskNodeStat *stat);
    RT_RET reallocOutputBuffers(INT32 wantSize,
                                std::string streamType = "none");
    RT_RET attachOutStreamPool(RTMediaBufferPool *pool,
                               std::string streamType = "none");
    RT_RET detachOutStreamPool(std::string streamType = "none");
    RT_RET pollOutputBuffer(RT_BOOL block = RT_TRUE, UINT32 size = 0,
                            std::string streamType = "none");
    INT32 getOutputBufferSize(std::string streamType = "none");
    RT_RET setOutputBufferCount(INT32 count,
                                std::string streamType = "none");
    INT32 getOutputBufferCount(std::string streamType = "none");
    RT_RET dequeInputBufferByPerStream(std::vector<RTMediaBuffer *> *packets);
    RT_RET dump();

 private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};
