#pragma once

#include <memory>
#include <string>
#include <vector>

#include "RTStreamInfo.h"
#include "RTTaskNodeContext.h"
#include "rt_header.h"
#include "rt_metadata.h"

class RTTaskNodeContextManager {
 public:
    RTTaskNodeContextManager();
    virtual ~RTTaskNodeContextManager();
    RT_RET initialize(std::string nodeName, INT32 nodeId,
                      std::vector<RTStreamInfo *> *inputInfo,
                      std::vector<RTStreamInfo *> *outputInfo,
                      bool isParallelRunning);
    RT_RET prepareForRun(RtMetaData *options);
    RT_RET cleanupAfterRun();
    RTTaskNodeContext *getDefaultNodeContext() const;
 private:
    std::unique_ptr<RTTaskNodeContext> mDefaultContext;
    std::vector<RTStreamInfo *> *mInputInfos{nullptr};
    std::vector<RTStreamInfo *> *mOutputInfos{nullptr};
    bool mParallelRunning{false};
    RtMetaData *mOptions{nullptr};
    std::string mNodeName;
    INT32 mNodeId{-1};
};
