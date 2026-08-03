#pragma once

#include <atomic>
#include <cstdint>

#include "rk_type.h"
#include "rk_common.h"
#include "rk_comm_video.h"
#include "rk_comm_aio.h"

#include "portable_hash.hpp"

#if defined(_WIN32)
#define ROCKIT_API __declspec(dllexport)
#else
#define ROCKIT_API __attribute__((visibility("default")))
#endif

namespace rockit_local {
bool system_initialized() noexcept;
RK_S32 route_video_frame(const MPP_CHN_S &source, const VIDEO_FRAME_INFO_S &frame, RK_S32 timeout);
RK_S32 route_audio_frame(const MPP_CHN_S &source, const AUDIO_FRAME_S &frame, RK_S32 timeout);
RK_S32 route_audio_stream(const MPP_CHN_S &source, const AUDIO_STREAM_S &stream);
} // namespace rockit_local
