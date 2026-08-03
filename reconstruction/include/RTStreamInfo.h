#pragma once

#include <string>
#include "rt_header.h"

class RTStreamInfo {
 public:
    RTStreamInfo();
    ~RTStreamInfo();
    void setStreamType(std::string streamType);
    std::string streamType();
    void setStreamName(std::string streamName);
    std::string streamName();
    void setStreamId(INT32 streamId);
    INT32 streamId();
 private:
    std::string mStreamType;
    std::string mStreamName;
    INT32 mStreamId{-1};
};
