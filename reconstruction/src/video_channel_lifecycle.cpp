#include "rockit_internal.hpp"
#include "local_frame_cpu.hpp"
#include "mpp_video_provider.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rk_comm_vdec.h"
#include "rk_comm_venc.h"
#include "rk_defines.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_vdec.h"
#include "rk_mpi_venc.h"

namespace {

template <typename T> void clear_struct(T *value) {
    if (value)
        std::memset(value, 0, sizeof(T));
}

struct VdecFrameItem {
    VIDEO_FRAME_INFO_S frame{};
    RK_U32 sourceBytes{0};
    bool endOfStream{false};
};

struct VencPacketItem {
    MB_BLK block{MB_INVALID_HANDLE};
    RK_U32 length{0};
    RK_U32 offset{0};
    RK_U64 pts{0};
    bool frameEnd{true};
    bool streamEnd{false};
    bool idr{false};
    RK_U32 userDataLength{0};
};

struct VdecChannelState {
    bool exists{false};
    bool receiving{false};
    VDEC_CHN_ATTR_S attr{};
    VDEC_CHN_PARAM_S param{};
    VIDEO_DISPLAY_MODE_E displayMode{VIDEO_DISPLAY_MODE_PLAYBACK};
    ROTATION_E rotation{ROTATION_0};
    MB_POOL pool{MB_INVALID_POOLID};
    VIDEO_FRAME_INFO_S userPic{};
    bool hasUserPic{false};
    bool userPicEnabled{false};
    RK_U32 recvFrames{0};
    RK_U32 recvBytes{0};
    RK_U32 decodedFrames{0};
    RK_U32 releasedFrames{0};
    bool endOfStream{false};
    RK_U32 inFlightOperations{0};
    std::shared_ptr<rockit_mpp::Decoder> provider;
    std::deque<VdecFrameItem> readyFrames;
    std::unordered_map<MB_BLK, RK_U32> outstandingFrames;
    int fd{-1};
};

struct VencChannelState {
    bool exists{false};
    bool receiving{false};
    VENC_CHN_ATTR_S attr{};
    VENC_CHN_PARAM_S param{};
    VENC_RECV_PIC_PARAM_S recvParam{};
    VENC_SCENE_MODE_E scene{SCENE_0};
    VENC_CHN_BUF_WRAP_S wrap{};
    VENC_CHN_REF_BUF_SHARE_S refShare{};
    VENC_COMBO_ATTR_S combo{};
    ROTATION_E rotation{ROTATION_0};
    MB_POOL pool{MB_INVALID_POOLID};
    bool svc{false};
    bool motionDeblur{false};
    bool motionStaticSwitch{false};
    bool thumbnail{false};
    VENC_CHN thumbnailSource{-1};
    RK_U32 submittedFrames{0};
    RK_U32 emittedStreams{0};
    RK_U32 releasedStreams{0};
    RK_U32 sequence{0};
    bool nextIdr{false};
    RK_U32 inFlightOperations{0};
    std::shared_ptr<rockit_mpp::Encoder> provider;
    std::vector<RK_U8> pendingUserData;
    MB_BLK qpmap{MB_INVALID_HANDLE};
    std::deque<VencPacketItem> readyPackets;
    std::unordered_map<MB_BLK, RK_U32> outstandingPackets;
    std::unordered_map<std::string, std::vector<RK_U8>> options;
    std::unordered_map<RK_U32, VENC_ROI_ATTR_S> roi;
    int fd{-1};
};

struct VideoState {
    std::mutex mutex;
    std::condition_variable changed;
    std::array<VdecChannelState, VDEC_MAX_CHN_NUM> vdec{};
    std::array<VencChannelState, VENC_MAX_CHN_NUM> venc{};
    VDEC_MOD_PARAM_S vdecMod{};
    VENC_PARAM_MOD_S vencMod{};
};

VideoState &video_state() {
    static VideoState state;
    return state;
}

bool valid_vdec(VDEC_CHN chn) {
    return chn >= 0 && chn < VDEC_MAX_CHN_NUM;
}
bool valid_venc(VENC_CHN chn) {
    return chn >= 0 && chn < VENC_MAX_CHN_NUM;
}

RK_S32 require_vdec(VideoState &state, VDEC_CHN chn, VdecChannelState **out) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    auto &channel = state.vdec[static_cast<std::size_t>(chn)];
    if (!channel.exists)
        return RK_ERR_VDEC_UNEXIST;
    if (out)
        *out = &channel;
    return RK_SUCCESS;
}

RK_S32 require_venc(VideoState &state, VENC_CHN chn, VencChannelState **out) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    auto &channel = state.venc[static_cast<std::size_t>(chn)];
    if (!channel.exists)
        return RK_ERR_VENC_UNEXIST;
    if (out)
        *out = &channel;
    return RK_SUCCESS;
}

void release_mb(MB_BLK block);

RK_S32 vdec_pointer_error(VDEC_CHN chn) {
    auto &state = video_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const RK_S32 result = require_vdec(state, chn, nullptr);
    return result == RK_SUCCESS ? RK_ERR_VDEC_NULL_PTR : result;
}

RK_S32 venc_pointer_error(VENC_CHN chn) {
    auto &state = video_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const RK_S32 result = require_venc(state, chn, nullptr);
    return result == RK_SUCCESS ? RK_ERR_VENC_NULL_PTR : result;
}

RK_S32 map_venc_provider_status(rockit_mpp::Status status, bool output_operation = false) noexcept {
    using rockit_mpp::Status;
    switch (status) {
    case Status::Ok:
        return RK_SUCCESS;
    case Status::Again:
    case Status::Timeout:
        return output_operation ? RK_ERR_VENC_BUF_EMPTY : RK_ERR_VENC_BUF_FULL;
    case Status::Unavailable:
        return RK_ERR_VENC_SYS_NOTREADY;
    case Status::Unsupported:
        return RK_ERR_VENC_NOT_SUPPORT;
    case Status::Invalid:
        return RK_ERR_VENC_ILLEGAL_PARAM;
    case Status::NoMemory:
        return RK_ERR_VENC_NOMEM;
    case Status::Corrupt:
    case Status::Failed:
        return RK_FAILURE;
    }
    return RK_FAILURE;
}

RK_S32 map_vdec_provider_status(rockit_mpp::Status status, bool output_operation = false) noexcept {
    using rockit_mpp::Status;
    switch (status) {
    case Status::Ok:
        return RK_SUCCESS;
    case Status::Again:
    case Status::Timeout:
        return output_operation ? RK_ERR_VDEC_BUF_EMPTY : RK_ERR_VDEC_BUF_FULL;
    case Status::Unavailable:
        return RK_ERR_VDEC_SYS_NOTREADY;
    case Status::Unsupported:
        return RK_ERR_VDEC_NOT_SUPPORT;
    case Status::Invalid:
        return RK_ERR_VDEC_ILLEGAL_PARAM;
    case Status::NoMemory:
        return RK_ERR_VDEC_NOMEM;
    case Status::Corrupt:
        return RK_ERR_VDEC_ILLEGAL_PARAM;
    case Status::Failed:
        return RK_FAILURE;
    }
    return RK_FAILURE;
}

bool provider_fallback_allowed(rockit_mpp::Status status) noexcept {
    return !rockit_mpp::policy_is_required() &&
           (status == rockit_mpp::Status::Unavailable || status == rockit_mpp::Status::Unsupported);
}

void release_provider_frames(std::vector<rockit_mpp::DecodedFrame> &frames) noexcept {
    for (auto &frame : frames) {
        if (frame.frame.stVFrame.pMbBlk) {
            release_mb(frame.frame.stVFrame.pMbBlk);
            frame.frame.stVFrame.pMbBlk = MB_INVALID_HANDLE;
        }
    }
    frames.clear();
}

#define ROCKIT_LOCK_VDEC_STATE_CHANNEL(channel_id_)                                               \
    auto &state = video_state();                                                                  \
    std::lock_guard<std::mutex> lock(state.mutex);                                                \
    VdecChannelState *channel = nullptr;                                                          \
    RK_S32 result = require_vdec(state, (channel_id_), &channel);                                 \
    if (result != RK_SUCCESS)                                                                     \
        return result
#define ROCKIT_LOCK_VDEC_SHORT_CHANNEL(channel_id_)                                               \
    auto &s = video_state();                                                                      \
    std::lock_guard<std::mutex> lock(s.mutex);                                                    \
    VdecChannelState *c = nullptr;                                                                \
    RK_S32 r = require_vdec(s, (channel_id_), &c);                                                \
    if (r != RK_SUCCESS)                                                                          \
        return r
#define ROCKIT_LOCK_VENC_STATE_CHANNEL(channel_id_)                                               \
    auto &state = video_state();                                                                  \
    std::lock_guard<std::mutex> lock(state.mutex);                                                \
    VencChannelState *channel = nullptr;                                                          \
    RK_S32 result = require_venc(state, (channel_id_), &channel);                                 \
    if (result != RK_SUCCESS)                                                                     \
        return result
#define ROCKIT_LOCK_VENC_SHORT_CHANNEL(channel_id_)                                               \
    auto &s = video_state();                                                                      \
    std::lock_guard<std::mutex> lock(s.mutex);                                                    \
    VencChannelState *c = nullptr;                                                                \
    RK_S32 r = require_venc(s, (channel_id_), &c);                                                \
    if (r != RK_SUCCESS)                                                                          \
        return r

#define ROCKIT_LOCK_VENC_POINTER_CHANNEL(channel_, pointer_)                                      \
    if (!valid_venc(channel_))                                                                     \
        return RK_ERR_VENC_INVALID_CHNID;                                                          \
    if (!(pointer_))                                                                               \
        return venc_pointer_error(channel_);                                                       \
    ROCKIT_LOCK_VENC_STATE_CHANNEL(channel_)

template <typename T> RK_S32 set_venc_option(VENC_CHN chn, const char *key, const T *value) {
    ROCKIT_LOCK_VENC_POINTER_CHANNEL(chn, value);
    auto &bytes = channel->options[key];
    bytes.resize(sizeof(T));
    std::memcpy(bytes.data(), value, sizeof(T));
    return RK_SUCCESS;
}

template <typename T> RK_S32 get_venc_option(VENC_CHN chn, const char *key, T *value) {
    ROCKIT_LOCK_VENC_POINTER_CHANNEL(chn, value);
    auto it = channel->options.find(key);
    if (it == channel->options.end()) {
        std::memset(value, 0, sizeof(T));
        return RK_SUCCESS;
    }
    if (it->second.size() != sizeof(T))
        return RK_ERR_VENC_ILLEGAL_PARAM;
    std::memcpy(value, it->second.data(), sizeof(T));
    return RK_SUCCESS;
}

std::size_t vdec_capacity(const VdecChannelState &channel) {
    return std::max<std::size_t>(
        1, channel.attr.u32FrameBufCnt == 0 ? 4 : channel.attr.u32FrameBufCnt);
}

std::size_t venc_capacity(const VencChannelState &channel) {
    return std::max<std::size_t>(
        1,
        channel.attr.stVencAttr.u32StreamBufCnt == 0 ? 4 : channel.attr.stVencAttr.u32StreamBufCnt);
}

template <typename Predicate>
bool wait_for(VideoState &state,
              std::unique_lock<std::mutex> &lock,
              RK_S32 timeoutMs,
              Predicate predicate) {
    if (predicate())
        return true;
    if (timeoutMs == 0)
        return false;
    if (timeoutMs < -1)
        return false;
    if (timeoutMs == -1) {
        state.changed.wait(lock, predicate);
        return predicate();
    }
    return state.changed.wait_for(lock, std::chrono::milliseconds(timeoutMs), predicate);
}

void release_mb(MB_BLK block) {
    if (block)
        (void)RK_MPI_MB_ReleaseMB(block);
}

bool retain_mb(MB_BLK block) {
    return block && RK_MPI_MB_AddUserCnt(block) == RK_SUCCESS;
}

void flush_vdec_pending(VdecChannelState &channel) {
    for (const auto &item : channel.readyFrames)
        release_mb(item.frame.stVFrame.pMbBlk);
    channel.readyFrames.clear();
}

void release_vdec_user_pic(VdecChannelState &channel) {
    if (channel.hasUserPic)
        release_mb(channel.userPic.stVFrame.pMbBlk);
    clear_struct(&channel.userPic);
    channel.hasUserPic = false;
    channel.userPicEnabled = false;
}

void flush_venc_pending(VencChannelState &channel) {
    for (const auto &item : channel.readyPackets)
        release_mb(item.block);
    channel.readyPackets.clear();
}

void release_venc_qpmap(VencChannelState &channel) {
    if (channel.qpmap)
        release_mb(channel.qpmap);
    channel.qpmap = MB_INVALID_HANDLE;
}

void reset_vdec_counters(VdecChannelState &channel) {
    channel.recvFrames = 0;
    channel.recvBytes = 0;
    channel.decodedFrames = 0;
    channel.releasedFrames = 0;
    channel.endOfStream = false;
}

void reset_venc_counters(VencChannelState &channel) {
    channel.submittedFrames = 0;
    channel.emittedStreams = 0;
    channel.releasedStreams = 0;
    channel.sequence = 0;
    channel.nextIdr = false;
    channel.pendingUserData.clear();
}

} // namespace

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_CreateChn(VDEC_CHN chn, const VDEC_CHN_ATTR_S *attr) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    if (!attr)
        return RK_ERR_VDEC_NULL_PTR;
    if (attr->enType == RK_VIDEO_ID_Unused)
        return RK_ERR_VDEC_ILLEGAL_PARAM;

    std::shared_ptr<rockit_mpp::Decoder> provider;
    const rockit_mpp::Status provider_status = rockit_mpp::create_decoder(*attr, &provider);
    if (provider_status != rockit_mpp::Status::Ok && !provider_fallback_allowed(provider_status)) {
        return map_vdec_provider_status(provider_status);
    }

    auto &state = video_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto &channel = state.vdec[static_cast<std::size_t>(chn)];
    if (channel.exists)
        return RK_ERR_VDEC_EXIST;
    channel = VdecChannelState{};
    channel.exists = true;
    channel.attr = *attr;
    channel.param.enType = attr->enType;
    channel.fd = 2000 + chn;
    channel.provider = std::move(provider);
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_DestroyChn(VDEC_CHN chn) {
    std::shared_ptr<rockit_mpp::Decoder> provider;
    auto &state = video_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        VdecChannelState *channel = nullptr;
        RK_S32 result = require_vdec(state, chn, &channel);
        if (result != RK_SUCCESS)
            return result;
        if (channel->receiving || channel->inFlightOperations != 0 ||
            !channel->outstandingFrames.empty())
            return RK_ERR_VDEC_BUSY;
        flush_vdec_pending(*channel);
        release_vdec_user_pic(*channel);
        provider = std::move(channel->provider);
        *channel = VdecChannelState{};
        state.changed.notify_all();
    }
    provider.reset();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_GetChnAttr(VDEC_CHN chn, VDEC_CHN_ATTR_S *attr) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    if (!attr)
        return vdec_pointer_error(chn);
    ROCKIT_LOCK_VDEC_STATE_CHANNEL(chn);
    *attr = channel->attr;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_SetChnAttr(VDEC_CHN chn, const VDEC_CHN_ATTR_S *attr) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    if (!attr)
        return vdec_pointer_error(chn);
    auto &state = video_state();
    std::shared_ptr<rockit_mpp::Decoder> old_provider;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        VdecChannelState *channel = nullptr;
        RK_S32 result = require_vdec(state, chn, &channel);
        if (result != RK_SUCCESS)
            return result;
        if (channel->receiving || channel->inFlightOperations != 0 ||
            !channel->readyFrames.empty() || !channel->outstandingFrames.empty())
            return RK_ERR_VDEC_NOT_PERM;
        old_provider = channel->provider;
    }

    std::shared_ptr<rockit_mpp::Decoder> replacement;
    if (old_provider) {
        const rockit_mpp::Status status = rockit_mpp::create_decoder(*attr, &replacement);
        if (status != rockit_mpp::Status::Ok)
            return map_vdec_provider_status(status);
    }

    std::lock_guard<std::mutex> lock(state.mutex);
    VdecChannelState *channel = nullptr;
    RK_S32 result = require_vdec(state, chn, &channel);
    if (result != RK_SUCCESS)
        return result;
    if (channel->receiving || channel->inFlightOperations != 0 || !channel->readyFrames.empty() ||
        !channel->outstandingFrames.empty() || channel->provider != old_provider)
        return RK_ERR_VDEC_NOT_PERM;
    channel->attr = *attr;
    channel->param.enType = attr->enType;
    if (old_provider)
        channel->provider = std::move(replacement);
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_StartRecvStream(VDEC_CHN chn) {
    ROCKIT_LOCK_VDEC_SHORT_CHANNEL(chn);
    c->receiving = true;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_StopRecvStream(VDEC_CHN chn) {
    ROCKIT_LOCK_VDEC_STATE_CHANNEL(chn);
    channel->receiving = false;
    state.changed.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_ResetChn(VDEC_CHN chn) {
    auto &state = video_state();
    std::shared_ptr<rockit_mpp::Decoder> provider;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        VdecChannelState *channel = nullptr;
        RK_S32 result = require_vdec(state, chn, &channel);
        if (result != RK_SUCCESS)
            return result;
        if (!channel->outstandingFrames.empty() || channel->inFlightOperations != 0)
            return RK_ERR_VDEC_BUSY;
        provider = channel->provider;
        if (provider)
            ++channel->inFlightOperations;
    }

    rockit_mpp::Status provider_status = rockit_mpp::Status::Ok;
    if (provider)
        provider_status = rockit_mpp::decoder_reset(provider);

    std::lock_guard<std::mutex> lock(state.mutex);
    VdecChannelState *channel = nullptr;
    RK_S32 result = require_vdec(state, chn, &channel);
    if (result != RK_SUCCESS)
        return result;
    if (provider && channel->inFlightOperations != 0)
        --channel->inFlightOperations;
    if (provider_status != rockit_mpp::Status::Ok) {
        state.changed.notify_all();
        return map_vdec_provider_status(provider_status);
    }
    flush_vdec_pending(*channel);
    reset_vdec_counters(*channel);
    channel->userPicEnabled = false;
    state.changed.notify_all();
    return RK_SUCCESS;
}

#define ROCKIT_VDEC_RELOCK_AFTER_PROVIDER()                                                       \
    lock.lock();                                                                                   \
    result = require_vdec(state, chn, &channel);                                                   \
    if (result != RK_SUCCESS) {                                                                    \
        lock.unlock();                                                                             \
        release_provider_frames(decoded);                                                          \
        return result;                                                                             \
    }                                                                                              \
    if (channel->inFlightOperations != 0)                                                          \
        --channel->inFlightOperations

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_SendStream(VDEC_CHN chn,
                                                    const VDEC_STREAM_S *stream,
                                                    RK_S32 timeoutMs) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    if (!stream)
        return vdec_pointer_error(chn);
    if (timeoutMs < -1)
        return RK_ERR_VDEC_ILLEGAL_PARAM;
    if (stream->u32Len != 0 && !stream->pMbBlk)
        return RK_ERR_VDEC_NULL_PTR;
    if (stream->pMbBlk) {
        const RK_U64 size = RK_MPI_MB_GetSize(stream->pMbBlk);
        const RK_U64 offset = RK_MPI_MB_GetOffset(stream->pMbBlk);
        if (offset > size || stream->u32Len > size - offset)
            return RK_ERR_VDEC_ILLEGAL_PARAM;
    }

    auto &state = video_state();
    std::unique_lock<std::mutex> lock(state.mutex);
    VdecChannelState *channel = nullptr;
    RK_S32 result = require_vdec(state, chn, &channel);
    if (result != RK_SUCCESS)
        return result;
    if (!channel->receiving)
        return RK_ERR_VDEC_NOT_PERM;

    if (!channel->provider) {
        const bool producesFrame =
            stream->pMbBlk && (stream->bEndOfFrame || channel->attr.enMode == VIDEO_MODE_FRAME);
        VIDEO_FRAME_INFO_S routeFrame{};
        bool shouldRoute = false;
        if (producesFrame) {
            const bool ready = wait_for(state, lock, timeoutMs, [&] {
                return !channel->exists || !channel->receiving ||
                       channel->readyFrames.size() < vdec_capacity(*channel);
            });
            if (!ready)
                return RK_ERR_VDEC_BUF_FULL;
            if (!channel->exists)
                return RK_ERR_VDEC_UNEXIST;
            if (!channel->receiving)
                return RK_ERR_VDEC_NOT_PERM;
            if (!retain_mb(stream->pMbBlk))
                return RK_ERR_VDEC_BADADDR;

            VdecFrameItem item{};
            item.frame.stVFrame.pMbBlk = stream->pMbBlk;
            item.frame.stVFrame.u32Width = channel->attr.u32PicWidth;
            item.frame.stVFrame.u32Height = channel->attr.u32PicHeight;
            item.frame.stVFrame.u32VirWidth = channel->attr.u32PicVirWidth
                                                  ? channel->attr.u32PicVirWidth
                                                  : channel->attr.u32PicWidth;
            item.frame.stVFrame.u32VirHeight = channel->attr.u32PicVirHeight
                                                   ? channel->attr.u32PicVirHeight
                                                   : channel->attr.u32PicHeight;
            item.frame.stVFrame.enPixelFormat = RK_FMT_YUV420SP;
            item.frame.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
            item.frame.stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;
            item.frame.stVFrame.pVirAddr[0] = RK_MPI_MB_Handle2VirAddr(stream->pMbBlk);
            item.frame.stVFrame.u64PTS = stream->u64PTS;
            item.sourceBytes = stream->u32Len;
            item.endOfStream = stream->bEndOfStream != RK_FALSE;
            channel->readyFrames.push_back(item);
            routeFrame = item.frame;
            shouldRoute = true;
            ++channel->decodedFrames;
        }

        channel->recvBytes += stream->u32Len;
        if (stream->bEndOfFrame || channel->attr.enMode == VIDEO_MODE_FRAME)
            ++channel->recvFrames;
        if (stream->bEndOfStream)
            channel->endOfStream = true;
        state.changed.notify_all();
        if (!shouldRoute)
            return RK_SUCCESS;
        const MPP_CHN_S sourceChannel{RK_ID_VDEC, 0, chn};
        lock.unlock();
        return rockit_local::route_video_frame(sourceChannel, routeFrame, timeoutMs);
    }

    const bool capacity_ready = wait_for(state, lock, timeoutMs, [&] {
        return !channel->exists || !channel->receiving ||
               channel->readyFrames.size() < vdec_capacity(*channel);
    });
    if (!capacity_ready)
        return RK_ERR_VDEC_BUF_FULL;
    if (!channel->exists)
        return RK_ERR_VDEC_UNEXIST;
    if (!channel->receiving)
        return RK_ERR_VDEC_NOT_PERM;

    std::shared_ptr<rockit_mpp::Decoder> provider = channel->provider;
    ++channel->inFlightOperations;
    lock.unlock();

    std::vector<rockit_mpp::DecodedFrame> decoded;
    const rockit_mpp::Status provider_status =
        rockit_mpp::decoder_send(provider, *stream, timeoutMs, &decoded);

    ROCKIT_VDEC_RELOCK_AFTER_PROVIDER();
    if (channel->provider != provider || !channel->receiving) {
        state.changed.notify_all();
        lock.unlock();
        release_provider_frames(decoded);
        return RK_ERR_VDEC_NOT_PERM;
    }
    if (provider_status != rockit_mpp::Status::Ok && provider_status != rockit_mpp::Status::Again &&
        provider_status != rockit_mpp::Status::Timeout) {
        state.changed.notify_all();
        lock.unlock();
        release_provider_frames(decoded);
        return map_vdec_provider_status(provider_status);
    }

    std::vector<VIDEO_FRAME_INFO_S> route_frames;
    route_frames.reserve(decoded.size());
    bool overflow = false;
    RK_U32 accounted_bytes = stream->u32Len;
    for (auto &output : decoded) {
        if (output.corrupt || !output.frame.stVFrame.pMbBlk) {
            if (output.frame.stVFrame.pMbBlk)
                release_mb(output.frame.stVFrame.pMbBlk);
            overflow = true;
            continue;
        }
        if (channel->readyFrames.size() >= vdec_capacity(*channel)) {
            release_mb(output.frame.stVFrame.pMbBlk);
            overflow = true;
            continue;
        }
        VdecFrameItem item{};
        item.frame = output.frame;
        item.sourceBytes = accounted_bytes;
        accounted_bytes = 0;
        item.endOfStream = output.end_of_stream;
        channel->readyFrames.push_back(item);
        route_frames.push_back(item.frame);
        ++channel->decodedFrames;
        output.frame.stVFrame.pMbBlk = MB_INVALID_HANDLE;
    }
    decoded.clear();

    channel->recvBytes += stream->u32Len;
    if (stream->bEndOfFrame || channel->attr.enMode == VIDEO_MODE_FRAME)
        ++channel->recvFrames;
    if (stream->bEndOfStream)
        channel->endOfStream = true;
    state.changed.notify_all();
    const MPP_CHN_S sourceChannel{RK_ID_VDEC, 0, chn};
    lock.unlock();

    RK_S32 route_result = RK_SUCCESS;
    for (const auto &frame : route_frames) {
        const RK_S32 current = rockit_local::route_video_frame(sourceChannel, frame, timeoutMs);
        if (route_result == RK_SUCCESS && current != RK_SUCCESS)
            route_result = current;
    }
    if (route_result != RK_SUCCESS)
        return route_result;
    return overflow ? RK_ERR_VDEC_BUF_FULL : RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_GetFrame(VDEC_CHN chn,
                                                  VIDEO_FRAME_INFO_S *frame,
                                                  RK_S32 timeoutMs) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    if (!frame)
        return vdec_pointer_error(chn);
    if (timeoutMs < -1)
        return RK_ERR_VDEC_ILLEGAL_PARAM;

    auto &state = video_state();
    std::unique_lock<std::mutex> lock(state.mutex);
    VdecChannelState *channel = nullptr;
    RK_S32 result = require_vdec(state, chn, &channel);
    if (result != RK_SUCCESS)
        return result;

    if (channel->provider && channel->readyFrames.empty() && !channel->userPicEnabled &&
        channel->receiving) {
        std::shared_ptr<rockit_mpp::Decoder> provider = channel->provider;
        ++channel->inFlightOperations;
        lock.unlock();

        std::vector<rockit_mpp::DecodedFrame> decoded;
        const rockit_mpp::Status provider_status =
            rockit_mpp::decoder_drain(provider, timeoutMs, &decoded);

        ROCKIT_VDEC_RELOCK_AFTER_PROVIDER();
        if (channel->provider != provider) {
            state.changed.notify_all();
            lock.unlock();
            release_provider_frames(decoded);
            return RK_ERR_VDEC_UNEXIST;
        }
        if (provider_status != rockit_mpp::Status::Ok &&
            provider_status != rockit_mpp::Status::Again &&
            provider_status != rockit_mpp::Status::Timeout) {
            state.changed.notify_all();
            lock.unlock();
            release_provider_frames(decoded);
            return map_vdec_provider_status(provider_status, true);
        }

        std::vector<VIDEO_FRAME_INFO_S> route_frames;
        route_frames.reserve(decoded.size());
        for (auto &output : decoded) {
            if (output.corrupt || !output.frame.stVFrame.pMbBlk) {
                if (output.frame.stVFrame.pMbBlk)
                    release_mb(output.frame.stVFrame.pMbBlk);
                continue;
            }
            if (channel->readyFrames.size() >= vdec_capacity(*channel)) {
                release_mb(output.frame.stVFrame.pMbBlk);
                continue;
            }
            VdecFrameItem item{};
            item.frame = output.frame;
            item.endOfStream = output.end_of_stream;
            channel->readyFrames.push_back(item);
            route_frames.push_back(item.frame);
            ++channel->decodedFrames;
            output.frame.stVFrame.pMbBlk = MB_INVALID_HANDLE;
        }
        decoded.clear();
        state.changed.notify_all();

        if (!route_frames.empty()) {
            const MPP_CHN_S sourceChannel{RK_ID_VDEC, 0, chn};
            lock.unlock();
            RK_S32 route_result = RK_SUCCESS;
            for (const auto &routed_frame : route_frames) {
                const RK_S32 current =
                    rockit_local::route_video_frame(sourceChannel, routed_frame, timeoutMs);
                if (route_result == RK_SUCCESS && current != RK_SUCCESS)
                    route_result = current;
            }
            lock.lock();
            result = require_vdec(state, chn, &channel);
            if (result != RK_SUCCESS)
                return result;
            if (route_result != RK_SUCCESS)
                return route_result;
        }
    }

    if (!channel->provider) {
        const bool ready = wait_for(state, lock, timeoutMs, [&] {
            return !channel->exists || !channel->readyFrames.empty() || channel->userPicEnabled ||
                   !channel->receiving;
        });
        if (!ready) {
            clear_struct(frame);
            return RK_ERR_VDEC_BUF_EMPTY;
        }
        if (!channel->exists)
            return RK_ERR_VDEC_UNEXIST;
    }

    if (!channel->readyFrames.empty()) {
        VdecFrameItem item = channel->readyFrames.front();
        channel->readyFrames.pop_front();
        if (channel->rotation == ROTATION_0) {
            *frame = item.frame;
        } else {
            rockit_local::FrameTransform transform{};
            transform.rotation = channel->rotation;
            VIDEO_FRAME_INFO_S rotated{};
            if (rockit_local::cpu_transform_frame(item.frame, transform, nullptr, &rotated) !=
                RK_SUCCESS) {
                channel->readyFrames.push_front(item);
                return RK_ERR_VDEC_NOT_SUPPORT;
            }
            release_mb(item.frame.stVFrame.pMbBlk);
            *frame = rotated;
        }
        ++channel->outstandingFrames[frame->stVFrame.pMbBlk];
        channel->recvBytes =
            item.sourceBytes > channel->recvBytes ? 0 : channel->recvBytes - item.sourceBytes;
        state.changed.notify_all();
        return RK_SUCCESS;
    }

    if (channel->userPicEnabled && channel->hasUserPic) {
        if (channel->rotation == ROTATION_0) {
            MB_BLK block = channel->userPic.stVFrame.pMbBlk;
            if (!retain_mb(block))
                return RK_ERR_VDEC_BADADDR;
            *frame = channel->userPic;
        } else {
            rockit_local::FrameTransform transform{};
            transform.rotation = channel->rotation;
            if (rockit_local::cpu_transform_frame(channel->userPic, transform, nullptr, frame) !=
                RK_SUCCESS)
                return RK_ERR_VDEC_NOT_SUPPORT;
        }
        ++channel->outstandingFrames[frame->stVFrame.pMbBlk];
        return RK_SUCCESS;
    }

    clear_struct(frame);
    return RK_ERR_VDEC_BUF_EMPTY;
}

#undef ROCKIT_VDEC_RELOCK_AFTER_PROVIDER

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_ReleaseFrame(VDEC_CHN chn,
                                                      const VIDEO_FRAME_INFO_S *frame) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    if (!frame)
        return vdec_pointer_error(chn);
    ROCKIT_LOCK_VDEC_STATE_CHANNEL(chn);
    MB_BLK block = frame->stVFrame.pMbBlk;
    auto it = channel->outstandingFrames.find(block);
    if (!block || it == channel->outstandingFrames.end())
        return RK_ERR_VDEC_ILLEGAL_PARAM;
    if (--it->second == 0)
        channel->outstandingFrames.erase(it);
    release_mb(block);
    ++channel->releasedFrames;
    state.changed.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_SetRotation(VDEC_CHN, ROTATION_E) {
    return RK_ERR_VDEC_NOT_SUPPORT;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_GetRotation(VDEC_CHN, ROTATION_E *) {
    return RK_ERR_VDEC_NOT_SUPPORT;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_GetModParam(VDEC_MOD_PARAM_S *param) {
    if (!param)
        return RK_ERR_VDEC_NULL_PTR;
    auto &s = video_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    *param = s.vdecMod;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_SetModParam(const VDEC_MOD_PARAM_S *param) {
    if (!param)
        return RK_ERR_VDEC_NULL_PTR;
    if (param->enVdecMBSource < MB_SOURCE_PRIVATE || param->enVdecMBSource >= MB_SOURCE_BUTT)
        return RK_ERR_VDEC_ILLEGAL_PARAM;
    auto &s = video_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.vdecMod = *param;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_SetDisplayMode(VDEC_CHN chn, VIDEO_DISPLAY_MODE_E mode) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    if (mode < VIDEO_DISPLAY_MODE_PREVIEW || mode >= VIDEO_DISPLAY_MODE_BUTT)
        return RK_ERR_VDEC_ILLEGAL_PARAM;
    ROCKIT_LOCK_VDEC_SHORT_CHANNEL(chn);
    c->displayMode = mode;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_GetDisplayMode(VDEC_CHN chn, VIDEO_DISPLAY_MODE_E *mode) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    if (!mode)
        return vdec_pointer_error(chn);
    ROCKIT_LOCK_VDEC_SHORT_CHANNEL(chn);
    *mode = c->displayMode;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_QueryStatus(VDEC_CHN chn, VDEC_CHN_STATUS_S *status) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    if (!status)
        return vdec_pointer_error(chn);
    ROCKIT_LOCK_VDEC_STATE_CHANNEL(chn);
    clear_struct(status);
    status->enType = channel->attr.enType;
    status->bStartRecvStream = channel->receiving ? RK_TRUE : RK_FALSE;
    status->u32RecvStreamFrames = channel->recvFrames;
    status->u32DecodeStreamFrames = channel->decodedFrames;
    status->u32LeftStreamBytes = channel->recvBytes;
    status->u32LeftStreamFrames = static_cast<RK_U32>(channel->readyFrames.size());
    status->u32LeftPics = static_cast<RK_U32>(channel->readyFrames.size());
    const bool swapped = channel->rotation == ROTATION_90 || channel->rotation == ROTATION_270;
    status->u32Width = swapped ? channel->attr.u32PicHeight : channel->attr.u32PicWidth;
    status->u32Height = swapped ? channel->attr.u32PicWidth : channel->attr.u32PicHeight;
    return RK_SUCCESS;
}

#define ROCKIT_VDEC_LOCK_CHANNEL_PTR(channel_, pointer_)                                           \
    if (!valid_vdec(channel_))                                                                     \
        return RK_ERR_VDEC_INVALID_CHNID;                                                          \
    if (!(pointer_))                                                                               \
        return vdec_pointer_error(channel_);                                                       \
    auto &s = video_state();                                                                       \
    std::lock_guard<std::mutex> lock(s.mutex);                                                     \
    VdecChannelState *c = nullptr;                                                                 \
    RK_S32 r = require_vdec(s, channel_, &c);                                                      \
    if (r != RK_SUCCESS)                                                                           \
    return r

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_SetChnParam(VDEC_CHN chn, const VDEC_CHN_PARAM_S *param) {
    ROCKIT_VDEC_LOCK_CHANNEL_PTR(chn, param);
    c->param = *param;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_GetChnParam(VDEC_CHN chn, VDEC_CHN_PARAM_S *param) {
    ROCKIT_VDEC_LOCK_CHANNEL_PTR(chn, param);
    *param = c->param;
    return RK_SUCCESS;
}

#undef ROCKIT_VDEC_LOCK_CHANNEL_PTR

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_GetFd(VDEC_CHN chn) {
    auto &s = video_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    VdecChannelState *c = nullptr;
    RK_S32 r = require_vdec(s, chn, &c);
    return r == RK_SUCCESS ? c->fd : r;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_CloseFd(VDEC_CHN chn) {
    ROCKIT_LOCK_VDEC_SHORT_CHANNEL(chn);
    c->fd = -1;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_AttachMbPool(VDEC_CHN chn, MB_POOL pool) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    if (pool == MB_INVALID_POOLID)
        return RK_ERR_VDEC_ILLEGAL_PARAM;
    ROCKIT_LOCK_VDEC_SHORT_CHANNEL(chn);
    c->pool = pool;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_DetachMbPool(VDEC_CHN chn) {
    ROCKIT_LOCK_VDEC_SHORT_CHANNEL(chn);
    c->pool = MB_INVALID_POOLID;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_SetUserPic(VDEC_CHN chn, const VIDEO_FRAME_INFO_S *pic) {
    if (!valid_vdec(chn))
        return RK_ERR_VDEC_INVALID_CHNID;
    if (!pic)
        return vdec_pointer_error(chn);
    {
        auto &state = video_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        VdecChannelState *channel = nullptr;
        RK_S32 result = require_vdec(state, chn, &channel);
        if (result != RK_SUCCESS)
            return result;
    }
    if (!pic->stVFrame.pMbBlk)
        return RK_ERR_VDEC_ILLEGAL_PARAM;
    if (!retain_mb(pic->stVFrame.pMbBlk))
        return RK_ERR_VDEC_BADADDR;
    auto &state = video_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    VdecChannelState *channel = nullptr;
    RK_S32 result = require_vdec(state, chn, &channel);
    if (result != RK_SUCCESS) {
        release_mb(pic->stVFrame.pMbBlk);
        return result;
    }
    release_vdec_user_pic(*channel);
    channel->userPic = *pic;
    channel->hasUserPic = true;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_EnableUserPic(VDEC_CHN chn, RK_BOOL) {
    ROCKIT_LOCK_VDEC_STATE_CHANNEL(chn);
    if (!channel->hasUserPic)
        return RK_ERR_VDEC_NOT_CONFIG;
    channel->userPicEnabled = true;
    state.changed.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VDEC_DisableUserPic(VDEC_CHN chn) {
    ROCKIT_LOCK_VDEC_STATE_CHANNEL(chn);
    channel->userPicEnabled = false;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SetModParam(const VENC_PARAM_MOD_S *param) {
    if (!param)
        return RK_ERR_VENC_ILLEGAL_PARAM;
    auto &s = video_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (const auto &channel : s.venc)
        if (channel.exists)
            return RK_ERR_VENC_NOT_PERM;
    if (param->enVencModType < MODTYPE_VENC || param->enVencModType >= MODTYPE_BUTT)
        return RK_ERR_VENC_NOT_SUPPORT;
    s.vencMod = *param;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetModParam(VENC_PARAM_MOD_S *param) {
    if (!param)
        return RK_ERR_VENC_ILLEGAL_PARAM;
    if (param->enVencModType < MODTYPE_VENC || param->enVencModType >= MODTYPE_BUTT)
        return RK_ERR_VENC_NOT_SUPPORT;
    auto &s = video_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    *param = s.vencMod;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_CreateChn(VENC_CHN chn, const VENC_CHN_ATTR_S *attr) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (!attr)
        return RK_ERR_VENC_NULL_PTR;
    if (attr->stVencAttr.enType == RK_VIDEO_ID_Unused)
        return RK_ERR_VENC_ILLEGAL_PARAM;

    std::shared_ptr<rockit_mpp::Encoder> provider;
    const rockit_mpp::Status provider_status = rockit_mpp::create_encoder(*attr, &provider);
    if (provider_status != rockit_mpp::Status::Ok && !provider_fallback_allowed(provider_status)) {
        return map_venc_provider_status(provider_status);
    }

    auto &state = video_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto &channel = state.venc[static_cast<std::size_t>(chn)];
    if (channel.exists)
        return RK_ERR_VENC_EXIST;
    channel = VencChannelState{};
    channel.exists = true;
    channel.attr = *attr;
    channel.recvParam.s32RecvPicNum = -1;
    channel.fd = 3000 + chn;
    channel.provider = std::move(provider);
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_DestroyChn(VENC_CHN chn) {
    std::shared_ptr<rockit_mpp::Encoder> provider;
    auto &state = video_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        VencChannelState *channel = nullptr;
        RK_S32 result = require_venc(state, chn, &channel);
        if (result != RK_SUCCESS)
            return result;
        if (channel->receiving || channel->inFlightOperations != 0 ||
            !channel->outstandingPackets.empty())
            return RK_ERR_VENC_BUSY;
        flush_venc_pending(*channel);
        release_venc_qpmap(*channel);
        provider = std::move(channel->provider);
        *channel = VencChannelState{};
        state.changed.notify_all();
    }
    provider.reset();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_ResetChn(VENC_CHN chn) {
    auto &state = video_state();
    std::shared_ptr<rockit_mpp::Encoder> provider;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        VencChannelState *channel = nullptr;
        RK_S32 result = require_venc(state, chn, &channel);
        if (result != RK_SUCCESS)
            return result;
        if (!channel->outstandingPackets.empty() || channel->inFlightOperations != 0)
            return RK_ERR_VENC_BUSY;
        provider = channel->provider;
        if (provider)
            ++channel->inFlightOperations;
    }

    rockit_mpp::Status provider_status = rockit_mpp::Status::Ok;
    if (provider)
        provider_status = rockit_mpp::encoder_reset(provider);

    std::lock_guard<std::mutex> lock(state.mutex);
    VencChannelState *channel = nullptr;
    RK_S32 result = require_venc(state, chn, &channel);
    if (result != RK_SUCCESS)
        return result;
    if (provider && channel->inFlightOperations != 0)
        --channel->inFlightOperations;
    if (provider_status != rockit_mpp::Status::Ok) {
        state.changed.notify_all();
        return map_venc_provider_status(provider_status);
    }
    flush_venc_pending(*channel);
    reset_venc_counters(*channel);
    state.changed.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SetSceneMode(VENC_CHN chn, const VENC_SCENE_MODE_E mode) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (mode < SCENE_0 || mode >= SCENE_BUTT)
        return RK_ERR_VENC_ILLEGAL_PARAM;
    ROCKIT_LOCK_VENC_SHORT_CHANNEL(chn);
    c->scene = mode;
    return RK_SUCCESS;
}

#define ROCKIT_VENC_LOCK_CHANNEL_PTR(channel_, pointer_)                                           \
    if (!valid_venc(channel_))                                                                     \
        return RK_ERR_VENC_INVALID_CHNID;                                                          \
    if (!(pointer_))                                                                               \
        return venc_pointer_error(channel_);                                                       \
    auto &s = video_state();                                                                       \
    std::lock_guard<std::mutex> lock(s.mutex);                                                     \
    VencChannelState *c = nullptr;                                                                 \
    RK_S32 r = require_venc(s, channel_, &c);                                                      \
    if (r != RK_SUCCESS)                                                                           \
    return r

#define ROCKIT_VENC_LOCK_EXISTING_CHANNEL_PTR(channel_, pointer_)                                  \
    if (!(pointer_))                                                                               \
        return RK_ERR_VENC_NULL_PTR;                                                               \
    auto &s = video_state();                                                                       \
    std::lock_guard<std::mutex> lock(s.mutex);                                                     \
    VencChannelState *c = nullptr;                                                                 \
    RK_S32 r = require_venc(s, channel_, &c);                                                      \
    if (r != RK_SUCCESS)                                                                           \
    return r

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetSceneMode(VENC_CHN chn, VENC_SCENE_MODE_E *mode) {
    ROCKIT_VENC_LOCK_CHANNEL_PTR(chn, mode);
    *mode = c->scene;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_StartRecvFrame(VENC_CHN chn,
                                                        const VENC_RECV_PIC_PARAM_S *param) {
    ROCKIT_VENC_LOCK_CHANNEL_PTR(chn, param);
    if (param->s32RecvPicNum == 0 || param->s32RecvPicNum < -1)
        return RK_ERR_VENC_ILLEGAL_PARAM;
    c->recvParam = *param;
    c->receiving = true;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_StopRecvFrame(VENC_CHN chn) {
    auto &state = video_state();
    std::shared_ptr<rockit_mpp::Encoder> provider;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        VencChannelState *channel = nullptr;
        RK_S32 result = require_venc(state, chn, &channel);
        if (result != RK_SUCCESS)
            return result;
        if (!channel->receiving)
            return RK_SUCCESS;
        channel->receiving = false;
        provider = channel->provider;
        if (provider)
            ++channel->inFlightOperations;
        state.changed.notify_all();
    }
    if (!provider)
        return RK_SUCCESS;

    rockit_mpp::EncodedPacket eos_packet{};
    const rockit_mpp::Status status = rockit_mpp::encoder_send_eos(provider, 1000, &eos_packet);

    std::lock_guard<std::mutex> lock(state.mutex);
    VencChannelState *channel = nullptr;
    RK_S32 result = require_venc(state, chn, &channel);
    if (result != RK_SUCCESS) {
        if (eos_packet.block)
            release_mb(eos_packet.block);
        return result;
    }
    if (channel->inFlightOperations != 0)
        --channel->inFlightOperations;
    if (channel->provider != provider) {
        if (eos_packet.block)
            release_mb(eos_packet.block);
        state.changed.notify_all();
        return RK_ERR_VENC_UNEXIST;
    }
    if (status == rockit_mpp::Status::Ok && eos_packet.block) {
        if (channel->readyPackets.size() >= venc_capacity(*channel)) {
            release_mb(eos_packet.block);
            state.changed.notify_all();
            return RK_ERR_VENC_BUF_FULL;
        }
        VencPacketItem packet{};
        packet.block = eos_packet.block;
        packet.length = eos_packet.length;
        packet.offset = eos_packet.offset;
        packet.pts = eos_packet.pts;
        packet.frameEnd = eos_packet.frame_end;
        packet.streamEnd = true;
        packet.idr = eos_packet.idr;
        channel->readyPackets.push_back(packet);
    } else if (status != rockit_mpp::Status::Again && status != rockit_mpp::Status::Timeout &&
               status != rockit_mpp::Status::Ok) {
        if (eos_packet.block)
            release_mb(eos_packet.block);
        state.changed.notify_all();
        return map_venc_provider_status(status);
    }
    state.changed.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_QueryStatus(VENC_CHN chn, VENC_CHN_STATUS_S *status) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (!status)
        return venc_pointer_error(chn);
    ROCKIT_LOCK_VENC_STATE_CHANNEL(chn);
    clear_struct(status);
    status->u32LeftPics = static_cast<RK_U32>(channel->readyPackets.size());
    status->u32LeftStreamFrames = static_cast<RK_U32>(channel->readyPackets.size());
    status->u32CurPacks = channel->readyPackets.empty() ? 0 : 1;
    for (const auto &packet : channel->readyPackets)
        status->u32LeftStreamBytes += packet.length;
    if (channel->receiving && channel->recvParam.s32RecvPicNum > 0) {
        RK_U32 limit = static_cast<RK_U32>(channel->recvParam.s32RecvPicNum);
        RK_U32 used = std::min<RK_U32>(channel->submittedFrames, limit);
        status->u32LeftRecvPics = limit - used;
        status->u32LeftEncPics = static_cast<RK_U32>(channel->readyPackets.size());
    }
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_EnableSvc(VENC_CHN, RK_BOOL) {
    return RK_FAILURE;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_EnableMotionDeblur(VENC_CHN, RK_BOOL) {
    return RK_FAILURE;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_EnableMotionStaticSwitch(VENC_CHN chn, RK_BOOL enable) {
    ROCKIT_LOCK_VENC_SHORT_CHANNEL(chn);
    c->motionStaticSwitch = enable != RK_FALSE;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SetChnAttr(VENC_CHN chn, const VENC_CHN_ATTR_S *attr) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (!attr)
        return venc_pointer_error(chn);
    auto &state = video_state();
    std::shared_ptr<rockit_mpp::Encoder> old_provider;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        VencChannelState *channel = nullptr;
        RK_S32 result = require_venc(state, chn, &channel);
        if (result != RK_SUCCESS)
            return result;
        if (channel->receiving || channel->inFlightOperations != 0 ||
            !channel->readyPackets.empty() || !channel->outstandingPackets.empty())
            return RK_ERR_VENC_NOT_PERM;
        old_provider = channel->provider;
    }

    std::shared_ptr<rockit_mpp::Encoder> replacement;
    if (old_provider) {
        const rockit_mpp::Status status = rockit_mpp::create_encoder(*attr, &replacement);
        if (status != rockit_mpp::Status::Ok)
            return map_venc_provider_status(status);
    }

    std::lock_guard<std::mutex> lock(state.mutex);
    VencChannelState *channel = nullptr;
    RK_S32 result = require_venc(state, chn, &channel);
    if (result != RK_SUCCESS)
        return result;
    if (channel->receiving || channel->inFlightOperations != 0 || !channel->readyPackets.empty() ||
        !channel->outstandingPackets.empty() || channel->provider != old_provider)
        return RK_ERR_VENC_NOT_PERM;
    channel->attr = *attr;
    if (old_provider)
        channel->provider = std::move(replacement);
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetChnAttr(VENC_CHN chn, VENC_CHN_ATTR_S *attr) {
    ROCKIT_VENC_LOCK_CHANNEL_PTR(chn, attr);
    *attr = c->attr;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SetChnBufWrapAttr(VENC_CHN, const VENC_CHN_BUF_WRAP_S *) {
    return RK_FAILURE;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetChnBufWrapAttr(VENC_CHN chn,
                                                           VENC_CHN_BUF_WRAP_S *attr) {
    ROCKIT_VENC_LOCK_EXISTING_CHANNEL_PTR(chn, attr);
    *attr = c->wrap;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SetChnRefBufShareAttr(VENC_CHN,
                                                               const VENC_CHN_REF_BUF_SHARE_S *) {
    return RK_FAILURE;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetChnRefBufShareAttr(VENC_CHN chn,
                                                               VENC_CHN_REF_BUF_SHARE_S *attr) {
    ROCKIT_VENC_LOCK_EXISTING_CHANNEL_PTR(chn, attr);
    *attr = c->refShare;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SetComboAttr(VENC_CHN, VENC_COMBO_ATTR_S *) {
    return RK_FAILURE;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetComboAttr(VENC_CHN chn, VENC_COMBO_ATTR_S *attr) {
    ROCKIT_VENC_LOCK_EXISTING_CHANNEL_PTR(chn, attr);
    *attr = c->combo;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_EnableThumbnail(VENC_CHN chn) {
    ROCKIT_LOCK_VENC_SHORT_CHANNEL(chn);
    c->thumbnail = true;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_ThumbnailBind(VENC_CHN, VENC_CHN) {
    return RK_FAILURE;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_ThumbnailRequest(VENC_CHN chn) {
    ROCKIT_LOCK_VENC_SHORT_CHANNEL(chn);
    return c->thumbnail ? RK_SUCCESS : RK_ERR_VENC_NOT_CONFIG;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SetChnParam(VENC_CHN chn, const VENC_CHN_PARAM_S *param) {
    ROCKIT_VENC_LOCK_CHANNEL_PTR(chn, param);
    c->param = *param;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetChnParam(VENC_CHN chn, VENC_CHN_PARAM_S *param) {
    ROCKIT_VENC_LOCK_CHANNEL_PTR(chn, param);
    *param = c->param;
    return RK_SUCCESS;
}

#undef ROCKIT_VENC_LOCK_EXISTING_CHANNEL_PTR
#undef ROCKIT_VENC_LOCK_CHANNEL_PTR

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SendFrame(VENC_CHN chn,
                                                   const VIDEO_FRAME_INFO_S *frame,
                                                   RK_S32 timeoutMs) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (!frame)
        return venc_pointer_error(chn);
    if (timeoutMs < -1)
        return RK_ERR_VENC_ILLEGAL_PARAM;
    if (!frame->stVFrame.pMbBlk)
        return RK_ERR_VENC_NULL_PTR;

    auto &state = video_state();
    std::unique_lock<std::mutex> lock(state.mutex);
    VencChannelState *channel = nullptr;
    RK_S32 result = require_venc(state, chn, &channel);
    if (result != RK_SUCCESS)
        return result;
    if (!channel->receiving)
        return RK_ERR_VENC_NOT_PERM;
    if (channel->recvParam.s32RecvPicNum > 0 &&
        channel->submittedFrames >= static_cast<RK_U32>(channel->recvParam.s32RecvPicNum))
        return RK_ERR_VENC_BUF_FULL;

    const bool ready = wait_for(state, lock, timeoutMs, [&] {
        return !channel->exists || !channel->receiving ||
               channel->readyPackets.size() < venc_capacity(*channel);
    });
    if (!ready)
        return RK_ERR_VENC_BUF_FULL;
    if (!channel->exists)
        return RK_ERR_VENC_UNEXIST;
    if (!channel->receiving)
        return RK_ERR_VENC_NOT_PERM;

    if (!channel->provider) {
        if (!retain_mb(frame->stVFrame.pMbBlk))
            return RK_ERR_VENC_NOBUF;
        VencPacketItem packet{};
        packet.block = frame->stVFrame.pMbBlk;
        RK_U64 length = RK_MPI_MB_GetLength(packet.block);
        if (length == 0)
            length = RK_MPI_MB_GetSize(packet.block);
        packet.length = static_cast<RK_U32>(std::min<RK_U64>(length, UINT32_MAX));
        packet.offset = RK_MPI_MB_GetOffset(packet.block);
        packet.pts = frame->stVFrame.u64PTS;
        packet.idr = channel->nextIdr || channel->submittedFrames == 0;
        packet.userDataLength = static_cast<RK_U32>(channel->pendingUserData.size());
        channel->nextIdr = false;
        channel->pendingUserData.clear();
        channel->readyPackets.push_back(packet);
        ++channel->submittedFrames;
        state.changed.notify_all();
        return RK_SUCCESS;
    }

    if (!channel->pendingUserData.empty())
        return RK_ERR_VENC_NOT_SUPPORT;
    std::shared_ptr<rockit_mpp::Encoder> provider = channel->provider;
    const bool request_idr = channel->nextIdr || channel->submittedFrames == 0;
    ++channel->inFlightOperations;
    lock.unlock();

    rockit_mpp::Status provider_status = rockit_mpp::Status::Ok;
    if (request_idr)
        provider_status = rockit_mpp::encoder_request_idr(provider);
    rockit_mpp::EncodedPacket encoded{};
    if (provider_status == rockit_mpp::Status::Ok) {
        provider_status = rockit_mpp::encoder_send(provider, *frame, timeoutMs, &encoded);
    }

    lock.lock();
    result = require_venc(state, chn, &channel);
    if (result != RK_SUCCESS) {
        lock.unlock();
        if (encoded.block)
            release_mb(encoded.block);
        return result;
    }
    if (channel->inFlightOperations != 0)
        --channel->inFlightOperations;
    if (channel->provider != provider || !channel->receiving) {
        state.changed.notify_all();
        lock.unlock();
        if (encoded.block)
            release_mb(encoded.block);
        return RK_ERR_VENC_NOT_PERM;
    }
    if (provider_status != rockit_mpp::Status::Ok || !encoded.block) {
        state.changed.notify_all();
        lock.unlock();
        if (encoded.block)
            release_mb(encoded.block);
        return map_venc_provider_status(provider_status);
    }

    VencPacketItem packet{};
    packet.block = encoded.block;
    packet.length = encoded.length;
    packet.offset = encoded.offset;
    packet.pts = encoded.pts;
    packet.frameEnd = encoded.frame_end;
    packet.streamEnd = encoded.stream_end;
    packet.idr = encoded.idr || request_idr;
    channel->nextIdr = false;
    channel->readyPackets.push_back(packet);
    ++channel->submittedFrames;
    state.changed.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SendFrameEx(VENC_CHN chn,
                                                     const USER_FRAME_INFO_S *frame,
                                                     RK_S32 timeoutMs) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (!frame)
        return venc_pointer_error(chn);
    return RK_MPI_VENC_SendFrame(chn, &frame->stUserFrame, timeoutMs);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_RequestIDR(VENC_CHN chn, RK_BOOL) {
    auto &state = video_state();
    std::shared_ptr<rockit_mpp::Encoder> provider;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        VencChannelState *channel = nullptr;
        RK_S32 result = require_venc(state, chn, &channel);
        if (result != RK_SUCCESS)
            return result;
        channel->nextIdr = true;
        provider = channel->provider;
        if (provider)
            ++channel->inFlightOperations;
    }
    if (!provider)
        return RK_SUCCESS;

    const rockit_mpp::Status status = rockit_mpp::encoder_request_idr(provider);
    std::lock_guard<std::mutex> lock(state.mutex);
    VencChannelState *channel = nullptr;
    RK_S32 result = require_venc(state, chn, &channel);
    if (result != RK_SUCCESS)
        return result;
    if (channel->inFlightOperations != 0)
        --channel->inFlightOperations;
    state.changed.notify_all();
    return map_venc_provider_status(status);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_InsertUserData(VENC_CHN chn, RK_U8 *data, RK_U32 length) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (length != 0 && !data)
        return RK_ERR_VENC_NULL_PTR;
    ROCKIT_LOCK_VENC_STATE_CHANNEL(chn);
    if (length == 0)
        channel->pendingUserData.clear();
    else
        channel->pendingUserData.assign(data, data + length);
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SetQpmap(VENC_CHN chn, const MB_BLK block) {
    {
        ROCKIT_LOCK_VENC_POINTER_CHANNEL(chn, block);
    }
    if (!retain_mb(block))
        return RK_ERR_VENC_NOBUF;
    auto &state = video_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    VencChannelState *channel = nullptr;
    RK_S32 result = require_venc(state, chn, &channel);
    if (result != RK_SUCCESS) {
        release_mb(block);
        return result;
    }
    release_venc_qpmap(*channel);
    channel->qpmap = block;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetQpmap(VENC_CHN chn, MB_BLK *block) {
    ROCKIT_LOCK_VENC_POINTER_CHANNEL(chn, block);
    *block = channel->qpmap;
    return channel->qpmap ? RK_SUCCESS : RK_ERR_VENC_NOT_CONFIG;
}

#undef ROCKIT_LOCK_VENC_POINTER_CHANNEL

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetStream(VENC_CHN chn,
                                                   VENC_STREAM_S *stream,
                                                   RK_S32 timeoutMs) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (!stream || !stream->pstPack)
        return venc_pointer_error(chn);
    if (timeoutMs < -1)
        return RK_ERR_VENC_ILLEGAL_PARAM;
    VENC_PACK_S *pack = stream->pstPack;
    auto &state = video_state();
    std::unique_lock<std::mutex> lock(state.mutex);
    VencChannelState *channel = nullptr;
    RK_S32 result = require_venc(state, chn, &channel);
    if (result != RK_SUCCESS)
        return result;
    const bool ready = wait_for(state, lock, timeoutMs, [&] {
        return !channel->exists || !channel->readyPackets.empty() || !channel->receiving;
    });
    if (!ready || channel->readyPackets.empty()) {
        clear_struct(pack);
        stream->u32PackCount = 0;
        return RK_ERR_VENC_BUF_EMPTY;
    }
    if (!channel->exists)
        return RK_ERR_VENC_UNEXIST;

    VencPacketItem item = channel->readyPackets.front();
    channel->readyPackets.pop_front();
    clear_struct(stream);
    clear_struct(pack);
    stream->pstPack = pack;
    stream->u32PackCount = 1;
    stream->u32Seq = channel->sequence++;
    pack->pMbBlk = item.block;
    pack->u32Len = item.length;
    pack->u64PTS = item.pts;
    pack->bFrameEnd = item.frameEnd ? RK_TRUE : RK_FALSE;
    pack->bStreamEnd = item.streamEnd ? RK_TRUE : RK_FALSE;
    pack->u32Offset = item.offset;
    pack->u32DataNum = 1;
    pack->stPackInfo[0].u32PackOffset = item.offset;
    pack->stPackInfo[0].u32PackLength = item.length;
    if (channel->attr.stVencAttr.enType == RK_VIDEO_ID_HEVC)
        pack->DataType.enH265EType = item.idr ? H265E_NALU_IDRSLICE : H265E_NALU_PSLICE;
    else
        pack->DataType.enH264EType = item.idr ? H264E_NALU_IDRSLICE : H264E_NALU_PSLICE;
    ++channel->outstandingPackets[item.block];
    ++channel->emittedStreams;
    state.changed.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_ReleaseStream(VENC_CHN chn, VENC_STREAM_S *stream) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (!stream || !stream->pstPack)
        return venc_pointer_error(chn);
    ROCKIT_LOCK_VENC_STATE_CHANNEL(chn);
    MB_BLK block = stream->pstPack->pMbBlk;
    auto it = channel->outstandingPackets.find(block);
    if (!block || it == channel->outstandingPackets.end())
        return RK_ERR_VENC_ILLEGAL_PARAM;
    if (--it->second == 0)
        channel->outstandingPackets.erase(it);
    release_mb(block);
    ++channel->releasedStreams;
    stream->pstPack->pMbBlk = MB_INVALID_HANDLE;
    stream->u32PackCount = 0;
    state.changed.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetFd(VENC_CHN chn) {
    auto &s = video_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    VencChannelState *c = nullptr;
    RK_S32 r = require_venc(s, chn, &c);
    return r == RK_SUCCESS ? c->fd : r;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_CloseFd(VENC_CHN chn) {
    ROCKIT_LOCK_VENC_SHORT_CHANNEL(chn);
    c->fd = -1;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SetChnRotation(VENC_CHN chn, ROTATION_E rotation) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (rotation < ROTATION_0 || rotation >= ROTATION_BUTT)
        return RK_ERR_VENC_ILLEGAL_PARAM;
    ROCKIT_LOCK_VENC_SHORT_CHANNEL(chn);
    c->rotation = rotation;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetChnRotation(VENC_CHN chn, ROTATION_E *rotation) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (!rotation)
        return venc_pointer_error(chn);
    ROCKIT_LOCK_VENC_SHORT_CHANNEL(chn);
    *rotation = c->rotation;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_AttachMbPool(VENC_CHN chn, MB_POOL pool) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (pool == MB_INVALID_POOLID)
        return RK_ERR_VENC_ILLEGAL_PARAM;
    ROCKIT_LOCK_VENC_SHORT_CHANNEL(chn);
    c->pool = pool;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_DetachMbPool(VENC_CHN chn) {
    ROCKIT_LOCK_VENC_SHORT_CHANNEL(chn);
    c->pool = MB_INVALID_POOLID;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_SetRoiAttr(VENC_CHN chn, const VENC_ROI_ATTR_S *value) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (!value)
        return venc_pointer_error(chn);
    if (value->u32Index >= 8)
        return RK_ERR_VENC_ILLEGAL_PARAM;
    ROCKIT_LOCK_VENC_STATE_CHANNEL(chn);
    channel->roi[value->u32Index] = *value;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_GetRoiAttr(VENC_CHN chn,
                                                    RK_U32 index,
                                                    VENC_ROI_ATTR_S *value) {
    if (!valid_venc(chn))
        return RK_ERR_VENC_INVALID_CHNID;
    if (!value)
        return venc_pointer_error(chn);
    if (index >= 8)
        return RK_ERR_VENC_ILLEGAL_PARAM;
    ROCKIT_LOCK_VENC_STATE_CHANNEL(chn);
    auto it = channel->roi.find(index);
    if (it == channel->roi.end()) {
        std::memset(value, 0, sizeof(*value));
        value->u32Index = index;
        return RK_SUCCESS;
    }
    *value = it->second;
    return RK_SUCCESS;
}

#undef ROCKIT_LOCK_VENC_SHORT_CHANNEL
#undef ROCKIT_LOCK_VENC_STATE_CHANNEL
#undef ROCKIT_LOCK_VDEC_SHORT_CHANNEL
#undef ROCKIT_LOCK_VDEC_STATE_CHANNEL

#define VENC_OPTION_PAIR(Name, Type)                                                               \
    extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_Set##Name(VENC_CHN chn, const Type *value) {          \
        return set_venc_option(chn, #Name, value);                                                 \
    }                                                                                              \
    extern "C" ROCKIT_API RK_S32 RK_MPI_VENC_Get##Name(VENC_CHN chn, Type *value) {                \
        return get_venc_option(chn, #Name, value);                                                 \
    }

VENC_OPTION_PAIR(RcParam, VENC_RC_PARAM_S)
VENC_OPTION_PAIR(RcAdvParam, VENC_RC_ADVPARAM_S)
VENC_OPTION_PAIR(FrameLostStrategy, VENC_FRAMELOST_S)
VENC_OPTION_PAIR(SuperFrameStrategy, VENC_SUPERFRAME_CFG_S)
VENC_OPTION_PAIR(IntraRefresh, VENC_INTRA_REFRESH_S)
VENC_OPTION_PAIR(HierarchicalQp, VENC_HIERARCHICAL_QP_S)
VENC_OPTION_PAIR(CuPrediction, VENC_CU_PREDICTION_S)
VENC_OPTION_PAIR(SkipBias, VENC_SKIP_BIAS_S)
VENC_OPTION_PAIR(DeBreathEffect, VENC_DEBREATHEFFECT_S)
VENC_OPTION_PAIR(JpegParam, VENC_JPEG_PARAM_S)
VENC_OPTION_PAIR(SliceSplit, VENC_SLICE_SPLIT_S)
VENC_OPTION_PAIR(H264IntraPred, VENC_H264_INTRA_PRED_S)
VENC_OPTION_PAIR(H264Trans, VENC_H264_TRANS_S)
VENC_OPTION_PAIR(H264Entropy, VENC_H264_ENTROPY_S)
VENC_OPTION_PAIR(H264Dblk, VENC_H264_DBLK_S)
VENC_OPTION_PAIR(H264Vui, VENC_H264_VUI_S)
VENC_OPTION_PAIR(H265Trans, VENC_H265_TRANS_S)
VENC_OPTION_PAIR(H265Entropy, VENC_H265_ENTROPY_S)
VENC_OPTION_PAIR(H265Dblk, VENC_H265_DBLK_S)
VENC_OPTION_PAIR(H265Sao, VENC_H265_SAO_S)
VENC_OPTION_PAIR(H265PredUnit, VENC_H265_PU_S)
VENC_OPTION_PAIR(H265Vui, VENC_H265_VUI_S)
VENC_OPTION_PAIR(RefParam, VENC_REF_PARAM_S)
VENC_OPTION_PAIR(MjpegParam, VENC_MJPEG_PARAM_S)

#undef VENC_OPTION_PAIR
