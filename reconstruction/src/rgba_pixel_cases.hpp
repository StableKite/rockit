#pragma once

// Shared switch body for packed RGB(A) formats.  The including function must
// provide a `pixel` pointer and return an aggregate with red/green/blue/alpha
// members in that order.
#define ROCKIT_PACKED_RGBA_PIXEL_CASES                                                             \
    case RK_FMT_RGB888:                                                                            \
        return {pixel[0], pixel[1], pixel[2], 255};                                                \
    case RK_FMT_BGR888:                                                                            \
        return {pixel[2], pixel[1], pixel[0], 255};                                                \
    case RK_FMT_RGBA8888:                                                                          \
        return {pixel[0], pixel[1], pixel[2], pixel[3]};                                           \
    case RK_FMT_BGRA8888:                                                                          \
        return {pixel[2], pixel[1], pixel[0], pixel[3]};                                           \
    case RK_FMT_ARGB8888:                                                                          \
        return {pixel[1], pixel[2], pixel[3], pixel[0]};                                           \
    case RK_FMT_ABGR8888:                                                                          \
    case RK_FMT_XBGR8888:                                                                          \
        return {                                                                                   \
            pixel[3], pixel[2], pixel[1], pixel[0]                                                 \
        }
