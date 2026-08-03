#pragma once

#include "rt_header.h"

// The full vendor metadata container is reconstructed in a later slice.
// These virtual accessors are sufficient for executor construction and allow
// downstream code to provide compatible metadata adapters without coupling to
// this provisional implementation.
class RtMetaData {
 public:
    virtual ~RtMetaData() = default;
    virtual RT_BOOL findInt32(const char *, INT32 *) const { return RT_FALSE; }
    virtual RT_BOOL findCString(const char *, const char **) const { return RT_FALSE; }
};
