#pragma once

#include <functional>
#include <memory>
#include <string>

#include "RTGraphCommon.h"
#include "rt_header.h"
#include "rt_metadata.h"

typedef enum ExecutorMode {
    EXECUTOR_THREAD_POOL,
    EXECUTOR_THREAD_LOCAL,
} RTExecutorMode;

typedef enum GraphCmd {
    GRAPH_CMD_PREPARE = 100,
    GRAPH_CMD_START,
    GRAPH_CMD_STOP,
    GRAPH_CMD_RESUME,
    GRAPH_CMD_PAUSE,
    GRAPH_CMD_FLUSH,
    GRAPH_CMD_PRIVATE_CMD,
    GRAPH_CMD_TASK_NODE_PRIVATE_CMD,
    GRAPH_CMD_MAX,
} RTGraphCmd;

class RTExecutor;
class RTScheduler;
class RTTaskNode;
class RTMediaBuffer;
class RTGraphOutputStream;
class RTTaskGraphStat;

class RTTaskGraph {
 public:
    explicit RTTaskGraph(const char *tagName);
    explicit RTTaskGraph(UINT64 uid, const char *tagName);
    virtual ~RTTaskGraph();

    UINT64 getUid();
    std::string getName();
    RT_RET autoBuild(const char *config, RT_BOOL isFileType = RT_TRUE);
    RT_RET prepare(RtMetaData *params = RT_NULL);
    RT_RET start();
    RT_RET stop(RT_BOOL block = RT_TRUE);
    RT_RET resume();
    RT_RET pause(RT_BOOL block = RT_FALSE);
    RT_RET flush();
    RT_RET reset();
    RT_RET release();
    RT_RET invoke(INT32 cmd, RtMetaData *params);
    RT_RET waitForObservedOutput(INT64 timeoutUs = -1);
    RT_RET waitUntilDone(INT64 timeoutUs = -1);
    RT_RET waitUntilEos(INT64 timeoutUs = -1);
    RT_RET queryStat(RTTaskGraphStat *stat);
    RT_RET dump();

    RTCBHandle observeOutputStream(
        const std::string streamName,
        std::function<RT_RET(RTMediaBuffer *)> streamCallback);
    RT_RET cancelObserveOutputStream(RTCBHandle handle);
    RT_RET observeOutputStream(const std::string &streamName, INT32 streamId,
                               std::function<RT_RET(RTMediaBuffer *)> streamCallback);
    RT_RET cancelObserveOutputStream(INT32 streamId);
    RT_RET addSubGraph(const char *graphConfigFile);
    RT_RET removeSubGraph(const char *graphConfigFile);
    RT_BOOL hasMirror(INT32 nodeId, RT_BOOL nodeOnly = RT_FALSE);
    RT_RET setExternalExecutor(RTExecutor *executor);
    RTLinkHandle addDownGraph(RTTaskGraph *graph, std::string srcName,
                              std::string dstName);
    RT_RET removeDownGraph(RTLinkHandle handle);
    RT_RET setGraphOutputStreamAddMode(RTGraphIOStreamMode mode);
    RTTaskNode *createNode(std::string nodeConfig, std::string streamConfig);
    RT_RET linkNode(RTTaskNode *srcNode, RTTaskNode *dstNode);
    RT_RET unlinkNode(RTTaskNode *srcNode, RTTaskNode *dstNode);
    RT_BOOL hasLinkMode(std::string mode);
    RT_RET selectLinkMode(std::string mode);
    RT_RET clearLinkShips();
    void sendInterrupt(std::string reason);
    RT_BOOL unthrottleSources();

 protected:
    RT_RET addPacketToInputStream(std::string streamName, INT64 timeoutUs,
                                  RTMediaBuffer *packet);
    RTCBHandle addOutputStreamPoller(const std::string streamName);
    RT_RET removeOutputStreamPoller(RTCBHandle handle);
    RT_RET checkInputStreamFull(std::string streamName);
    virtual RT_RET onReceiveOutputStreams(RTGraphOutputStream *) { return RT_OK; }

 private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};
