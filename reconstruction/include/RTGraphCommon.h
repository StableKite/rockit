#pragma once

#include "rt_header.h"

typedef INT64 RTCBHandle;
typedef INT64 RTLinkHandle;

#define RT_INVALID_CB_HANDLE -1
#define RT_INVALID_LINK_HANDLE -1
#define RT_INVALID_STREAM_ID -1
#define RT_INVALID_NODE_ID -1

typedef enum GraphIOStreamMode {
    WAIT_TILL_NOT_FULL,
    ADD_IF_NOT_FULL,
    DROP_IF_FULL,
} RTGraphIOStreamMode;
