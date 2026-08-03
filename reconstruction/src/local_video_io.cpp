#include "rockit_internal.hpp"
#include "local_frame_cpu.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rk_comm_aiisp.h"
#include "rk_comm_dis.h"
#include "rk_comm_swcac.h"
#include "rk_defines.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_vi.h"
#include "rk_mpi_vo.h"
#include "rk_mpi_vpss.h"

namespace {

template <typename T> void zero_struct(T *value) {
    if (value)
        std::memset(value, 0, sizeof(T));
}

void release_mb(MB_BLK block) {
    if (block != MB_INVALID_HANDLE)
        (void)RK_MPI_MB_ReleaseMB(block);
}

bool retain_mb(MB_BLK block) {
    return block != MB_INVALID_HANDLE && RK_MPI_MB_AddUserCnt(block) > 0;
}

MB_BLK frame_block(const VIDEO_FRAME_INFO_S &frame) {
    return frame.stVFrame.pMbBlk;
}

struct FrameItem {
    VIDEO_FRAME_INFO_S frame{};
};

void release_queue(std::deque<FrameItem> &queue) {
    for (const auto &item : queue)
        release_mb(frame_block(item.frame));
    queue.clear();
}

void release_outstanding(std::unordered_map<MB_BLK, RK_U32> &outstanding) {
    for (const auto &[block, count] : outstanding) {
        for (RK_U32 i = 0; i < count; ++i)
            release_mb(block);
    }
    outstanding.clear();
}

bool queue_frame(std::deque<FrameItem> &queue, const VIDEO_FRAME_INFO_S &frame) {
    const MB_BLK block = frame_block(frame);
    if (!retain_mb(block))
        return false;
    queue.push_back(FrameItem{frame});
    return true;
}

template <typename Predicate>
bool wait_until(std::condition_variable &cv,
                std::unique_lock<std::mutex> &lock,
                RK_S32 timeoutMs,
                Predicate predicate) {
    if (predicate())
        return true;
    if (timeoutMs == 0 || timeoutMs < -1)
        return false;
    if (timeoutMs == -1) {
        cv.wait(lock, predicate);
        return predicate();
    }
    return cv.wait_for(lock, std::chrono::milliseconds(timeoutMs), predicate);
}

RECT_S resolve_crop(const RECT_S &crop, bool ratioCoordinates, RK_U32 width, RK_U32 height) {
    if (!ratioCoordinates) {
        return crop;
    }
    RECT_S resolved{};
    resolved.s32X = static_cast<RK_S32>((static_cast<RK_S64>(crop.s32X) * width) / 1000);
    resolved.s32Y = static_cast<RK_S32>((static_cast<RK_S64>(crop.s32Y) * height) / 1000);
    resolved.u32Width = static_cast<RK_U32>((static_cast<RK_U64>(crop.u32Width) * width) / 1000U);
    resolved.u32Height =
        static_cast<RK_U32>((static_cast<RK_U64>(crop.u32Height) * height) / 1000U);
    return resolved;
}

// ---------------- VI ----------------

struct ViChannelState {
    bool configured{false};
    bool enabled{false};
    bool paused{false};
    bool extended{false};
    VI_CHN_ATTR_S attr{};
    VI_CHN_BUF_WRAP_S wrap{};
    RK_BOOL freeze{RK_FALSE};
    MB_POOL pool{MB_INVALID_POOLID};
    VI_USERPIC_ATTR_S userPic{};
    bool hasUserPic{false};
    bool userPicEnabled{false};
    VI_CROP_INFO_S eptz{};
    VI_ISP_MIRROR_FLIP_S mirrorFlip{};
    DIS_CONFIG_S disConfig{};
    DIS_ATTR_S disAttr{};
    SWCAC_CONFIG_S swcac{};
    RK_CODEC_ID_E codec{RK_VIDEO_ID_Unused};
    std::vector<RK_U8> edid;
    RK_U32 edidPad{0};
    RK_U32 edidStartBlock{0};
    VI_EVENT_CALL_BACK_S callback{};
    RK_U32 sequence{0};
    RK_U32 deliveredFrames{0};
    std::unordered_map<MB_BLK, RK_U32> outstandingFrames;
    std::unordered_map<MB_BLK, RK_U32> outstandingStreams;
    int fd{-1};
};

rockit_local::FrameTransform vi_cpu_transform(const ViChannelState &channel,
                                              const VIDEO_FRAME_INFO_S &frame) {
    rockit_local::FrameTransform transform{};
    transform.outputWidth =
        channel.attr.stSize.u32Width ? channel.attr.stSize.u32Width : frame.stVFrame.u32Width;
    transform.outputHeight =
        channel.attr.stSize.u32Height ? channel.attr.stSize.u32Height : frame.stVFrame.u32Height;
    if (channel.eptz.bEnable == RK_TRUE) {
        transform.cropEnabled = RK_TRUE;
        transform.crop = resolve_crop(channel.eptz.stCropRect,
                                      channel.eptz.enCropCoordinate == VI_CROP_RATIO_COOR,
                                      frame.stVFrame.u32Width,
                                      frame.stVFrame.u32Height);
    }
    transform.mirror = channel.mirrorFlip.mirror ? RK_TRUE : RK_FALSE;
    transform.flip = channel.mirrorFlip.flip ? RK_TRUE : RK_FALSE;
    return transform;
}

struct ViPipeState {
    bool started{false};
    std::array<ViChannelState, VI_MAX_CHN_NUM> channels{};
};

struct ViDeviceState {
    bool configured{false};
    bool enabled{false};
    VI_DEV_ATTR_S attr{};
    VI_DEV_BIND_PIPE_S binding{};
};

struct ViState {
    std::mutex mutex;
    std::condition_variable changed;
    std::array<ViDeviceState, VI_MAX_DEV_NUM> devices{};
    std::array<ViPipeState, VI_MAX_PIPE_NUM> pipes{};
};

ViState &vi_state() {
    static ViState state;
    return state;
}

bool valid_vi_dev(VI_DEV dev) {
    return dev >= 0 && dev < VI_MAX_DEV_NUM;
}
bool valid_vi_pipe(VI_PIPE pipe) {
    return pipe >= 0 && pipe < VI_MAX_PIPE_NUM;
}
bool valid_vi_chn(VI_CHN chn) {
    return chn >= 0 && chn < VI_MAX_CHN_NUM;
}

RK_S32 require_vi_channel(ViState &state, VI_PIPE pipe, VI_CHN chn, ViChannelState **out) {
    if (!valid_vi_pipe(pipe))
        return RK_ERR_VI_INVALID_PIPEID;
    if (!valid_vi_chn(chn))
        return RK_ERR_VI_INVALID_CHNID;
    auto &channel =
        state.pipes[static_cast<std::size_t>(pipe)].channels[static_cast<std::size_t>(chn)];
    if (!channel.configured)
        return RK_ERR_VI_UNEXIST;
    if (out)
        *out = &channel;
    return RK_SUCCESS;
}

void release_vi_user_pic(ViChannelState &channel) {
    if (channel.hasUserPic && channel.userPic.enUsrPicMode == VI_USERPIC_MODE_PIC)
        release_mb(channel.userPic.unUsrPic.stUsrPicFrm.stVFrame.pMbBlk);
    zero_struct(&channel.userPic);
    channel.hasUserPic = false;
    channel.userPicEnabled = false;
}

// ---------------- VPSS ----------------

struct VpssChannelState {
    bool configured{false};
    bool enabled{false};
    VPSS_CHN_ATTR_S attr{};
    VPSS_CROP_INFO_S crop{};
    ROTATION_E rotation{ROTATION_0};
    VPSS_ROTATION_EX_ATTR_S rotationEx{};
    MB_POOL pool{MB_INVALID_POOLID};
    bool bufferShare{false};
    std::deque<FrameItem> ready;
    std::unordered_map<MB_BLK, RK_U32> outstanding;
    VIDEO_FRAME_INFO_S latestFrame{};
    bool hasLatestFrame{false};
    int fd{-1};
};

struct VpssGroupState {
    bool started{false};
    VPSS_GRP_ATTR_S attr{};
    VPSS_CROP_INFO_S crop{};
    ROTATION_E rotation{ROTATION_0};
    RK_BOOL mirror{RK_FALSE};
    RK_BOOL flip{RK_FALSE};
    AIISP_ATTR_S aiisp{};
    bool backupFrame{false};
    bool userFrameRateCtrl{false};
    RK_U32 delay{0};
    VIDEO_PROC_DEV_TYPE_E procDev{VIDEO_PROC_DEV_RGA};
    std::deque<FrameItem> groupReady;
    std::unordered_map<MB_BLK, RK_U32> groupOutstanding;
    std::array<VpssChannelState, VPSS_MAX_CHN_NUM> channels{};
};

rockit_local::FrameTransform vpss_group_cpu_transform(const VpssGroupState &group,
                                                      const VIDEO_FRAME_INFO_S &frame) {
    rockit_local::FrameTransform transform{};
    if (group.crop.bEnable == RK_TRUE) {
        transform.cropEnabled = RK_TRUE;
        transform.crop = resolve_crop(group.crop.stCropRect,
                                      group.crop.enCropCoordinate == VPSS_CROP_RATIO_COOR,
                                      frame.stVFrame.u32Width,
                                      frame.stVFrame.u32Height);
    }
    transform.mirror = group.mirror;
    transform.flip = group.flip;
    transform.rotation = group.rotation;
    return transform;
}

rockit_local::FrameTransform
vpss_channel_cpu_transform(const VpssChannelState &channel, RK_U32 inputWidth, RK_U32 inputHeight) {
    rockit_local::FrameTransform transform{};
    if (channel.crop.bEnable == RK_TRUE) {
        transform.cropEnabled = RK_TRUE;
        transform.crop = resolve_crop(channel.crop.stCropRect,
                                      channel.crop.enCropCoordinate == VPSS_CROP_RATIO_COOR,
                                      inputWidth,
                                      inputHeight);
    }
    transform.outputWidth = channel.attr.u32Width;
    transform.outputHeight = channel.attr.u32Height;
    transform.mirror = channel.attr.bMirror;
    transform.flip = channel.attr.bFlip;
    transform.rotation = channel.rotation;
    return transform;
}

struct VpssState {
    std::mutex mutex;
    std::condition_variable changed;
    std::unordered_map<VPSS_GRP, std::unique_ptr<VpssGroupState>> groups;
    VPSS_MOD_PARAM_S modParam{};
};

VpssState &vpss_state() {
    static VpssState state;
    return state;
}

bool valid_vpss_group(VPSS_GRP group) {
    return group >= 0 && group < VPSS_MAX_GRP_NUM;
}
bool valid_vpss_chn(VPSS_CHN chn) {
    return chn >= 0 && chn < VPSS_MAX_CHN_NUM;
}

RK_S32 require_vpss_group(VpssState &state, VPSS_GRP group, VpssGroupState **out) {
    if (!valid_vpss_group(group))
        return RK_ERR_VPSS_INVALID_DEVID;
    auto it = state.groups.find(group);
    if (it == state.groups.end())
        return RK_ERR_VPSS_UNEXIST;
    if (out)
        *out = it->second.get();
    return RK_SUCCESS;
}

RK_S32 require_vpss_channel(VpssState &state,
                            VPSS_GRP group,
                            VPSS_CHN chn,
                            VpssGroupState **groupOut,
                            VpssChannelState **channelOut) {
    if (!valid_vpss_chn(chn))
        return RK_ERR_VPSS_INVALID_CHNID;
    VpssGroupState *grp = nullptr;
    RK_S32 result = require_vpss_group(state, group, &grp);
    if (result != RK_SUCCESS)
        return result;
    auto &channel = grp->channels[static_cast<std::size_t>(chn)];
    if (!channel.configured)
        return RK_ERR_VPSS_UNEXIST;
    if (groupOut)
        *groupOut = grp;
    if (channelOut)
        *channelOut = &channel;
    return RK_SUCCESS;
}

std::size_t vpss_group_capacity(const VpssGroupState &group) {
    return std::max<std::size_t>(1, group.attr.u32MaxQueue == 0 ? 4 : group.attr.u32MaxQueue);
}

std::size_t vpss_channel_capacity(const VpssChannelState &channel) {
    if (channel.attr.u32Depth != 0)
        return channel.attr.u32Depth;
    if (channel.attr.u32FrameBufCnt != 0)
        return channel.attr.u32FrameBufCnt;
    return 4;
}

VIDEO_FRAME_INFO_S vpss_transform_metadata(const VIDEO_FRAME_INFO_S &input,
                                           const VpssGroupState &group,
                                           const VpssChannelState &channel) {
    VIDEO_FRAME_INFO_S output = input;
    if (channel.attr.u32Width)
        output.stVFrame.u32Width = channel.attr.u32Width;
    if (channel.attr.u32Height)
        output.stVFrame.u32Height = channel.attr.u32Height;
    if (channel.attr.u32Width)
        output.stVFrame.u32VirWidth = channel.attr.u32Width;
    if (channel.attr.u32Height)
        output.stVFrame.u32VirHeight = channel.attr.u32Height;
    output.stVFrame.enPixelFormat = channel.attr.enPixelFormat;
    output.stVFrame.enVideoFormat = channel.attr.enVideoFormat;
    output.stVFrame.enCompressMode = channel.attr.enCompressMode;
    output.stVFrame.enDynamicRange = channel.attr.enDynamicRange;
    if (channel.rotation == ROTATION_90 || channel.rotation == ROTATION_270) {
        std::swap(output.stVFrame.u32Width, output.stVFrame.u32Height);
        std::swap(output.stVFrame.u32VirWidth, output.stVFrame.u32VirHeight);
    }
    if (group.rotation == ROTATION_90 || group.rotation == ROTATION_270) {
        std::swap(output.stVFrame.u32Width, output.stVFrame.u32Height);
        std::swap(output.stVFrame.u32VirWidth, output.stVFrame.u32VirHeight);
    }
    return output;
}

void clear_vpss_group(VpssGroupState &group) {
    release_queue(group.groupReady);
    release_outstanding(group.groupOutstanding);
    for (auto &channel : group.channels) {
        release_queue(channel.ready);
        release_outstanding(channel.outstanding);
        if (channel.hasLatestFrame) {
            release_mb(frame_block(channel.latestFrame));
            zero_struct(&channel.latestFrame);
            channel.hasLatestFrame = false;
        }
    }
}

// ---------------- VO ----------------

struct GraphicsBuffer {
    RK_U32 width{0};
    RK_U32 height{0};
    RK_U32 format{0};
    std::vector<RK_U8> bytes;
};

// Public VO graphics handles are opaque tokens, never C++ object pointers.
// Tokens are retained until process exit so an already-destroyed handle can
// never become valid again through allocator address reuse (ABA).  The large
// pixel storage is still released immediately on Destroy.
struct GraphicsHandleToken {
    std::uint64_t serial{0};
};

struct GraphicsBufferRegistry {
    std::mutex mutex;
    std::uint64_t nextSerial{1};
    std::unordered_map<RK_VOID *, std::shared_ptr<GraphicsBuffer>> live;
    std::vector<std::unique_ptr<GraphicsHandleToken>> tokens;
};

GraphicsBufferRegistry &graphics_buffer_registry() {
    static GraphicsBufferRegistry registry;
    return registry;
}

std::shared_ptr<GraphicsBuffer> lookup_graphics_buffer(RK_VOID *handle) {
    if (!handle)
        return {};
    auto &registry = graphics_buffer_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto iterator = registry.live.find(handle);
    return iterator == registry.live.end() ? std::shared_ptr<GraphicsBuffer>{} : iterator->second;
}

struct VoChannelState {
    bool configured{false};
    bool enabled{false};
    bool paused{false};
    bool shown{true};
    VO_CHN_ATTR_S attr{};
    VO_CHN_PARAM_S param{};
    POINT_S position{};
    VO_BORDER_S border{};
    RK_S32 frameRate{-1};
    RK_U32 recvThreshold{0};
    RK_U64 pts{0};
    std::deque<FrameItem> ready;
    std::unordered_map<MB_BLK, RK_U32> outstanding;
    VO_FRAME_INFO_S gfxFrame{};
    bool hasGfxFrame{false};
};

rockit_local::FrameTransform vo_cpu_transform(const VoChannelState &channel,
                                              const VIDEO_FRAME_INFO_S &frame) {
    rockit_local::FrameTransform transform{};
    transform.outputWidth =
        channel.attr.stRect.u32Width ? channel.attr.stRect.u32Width : frame.stVFrame.u32Width;
    transform.outputHeight =
        channel.attr.stRect.u32Height ? channel.attr.stRect.u32Height : frame.stVFrame.u32Height;
    transform.mirror =
        channel.attr.enMirror == MIRROR_HORIZONTAL || channel.attr.enMirror == MIRROR_BOTH
            ? RK_TRUE
            : RK_FALSE;
    transform.flip =
        channel.attr.enMirror == MIRROR_VERTICAL || channel.attr.enMirror == MIRROR_BOTH ? RK_TRUE
                                                                                         : RK_FALSE;
    transform.rotation = channel.attr.enRotation;
    return transform;
}

struct VoLayerState {
    bool bound{false};
    VO_DEV device{-1};
    VO_LAYER_MODE_E mode{VO_LAYER_MODE_VIDEO};
    bool configured{false};
    bool enabled{false};
    bool composerPaused{false};
    VO_VIDEO_LAYER_ATTR_S attr{};
    RK_U32 priority{0};
    VO_CSC_S csc{};
    RK_U32 dispBufLen{4};
    RK_U32 cursorX{0};
    RK_U32 cursorY{0};
    RK_U32 cursorMaxWidth{0};
    RK_U32 cursorMaxHeight{0};
    VO_SPLICE_MODE_E spliceMode{VO_SPLICE_MODE_GPU};
    RK_U32 lowDelayCh{0};
    std::deque<FrameItem> ready;
    std::unordered_map<MB_BLK, RK_U32> outstanding;
    std::unordered_map<VO_CHN, VoChannelState> channels;
};

struct VoDeviceState {
    bool configured{false};
    bool enabled{false};
    VO_PUB_ATTR_S attr{};
    VO_CSC_S postProcess{VO_CSC_MATRIX_IDENTITY, 50, 50, 50, 50};
    RK_U32 vcntTiming{0};
    RK_VO_VSYNC_CALLBACK_FUNC_S vsyncCallback{};
};

struct VoWbcState {
    bool configured{false};
    bool enabled{false};
    VO_WBC_SOURCE_S source{};
    VO_WBC_ATTR_S attr{};
    VO_WBC_MODE_E mode{VO_WBC_MODE_NORMAL};
    RK_S32 depth{4};
    std::deque<FrameItem> ready;
    std::unordered_map<MB_BLK, RK_U32> outstanding;
};

struct VoInterfaceKey {
    RK_U32 type{0};
    RK_U32 id{0};
    bool operator==(const VoInterfaceKey &other) const noexcept {
        return type == other.type && id == other.id;
    }
};

struct VoInterfaceKeyHash {
    std::size_t operator()(const VoInterfaceKey &key) const noexcept {
        return rockit_local::packed_pair_hash<std::size_t>(key.type, key.id);
    }
};

struct VoState {
    std::mutex mutex;
    std::condition_variable changed;
    std::array<VoDeviceState, VO_MAX_DEV_NUM> devices{};
    std::array<VoLayerState, VO_MAX_LAYER_NUM> layers{};
    std::array<VoWbcState, VO_MAX_WBC_NUM> wbc{};
    std::unordered_map<VoInterfaceKey, VO_HDMI_PARAM_S, VoInterfaceKeyHash> hdmi;
    std::unordered_map<VoInterfaceKey, RK_VO_CALLBACK_FUNC_S, VoInterfaceKeyHash> callbacks;
    VO_GFX_MODE_E gfxMode{VO_MODE_NORMAL};
};

VoState &vo_state() {
    static VoState state;
    return state;
}

bool valid_vo_dev(VO_DEV dev) {
    return dev >= 0 && dev < VO_MAX_DEV_NUM;
}
bool valid_vo_layer(VO_LAYER layer) {
    return layer >= 0 && layer < VO_MAX_LAYER_NUM;
}
bool valid_vo_chn(VO_CHN chn) {
    return chn >= 0 && chn < VO_MAX_CHN_NUM;
}
bool valid_vo_wbc(VO_WBC wbc) {
    return wbc >= 0 && wbc < VO_MAX_WBC_NUM;
}

RK_S32 require_vo_layer(VoState &state, VO_LAYER layer, VoLayerState **out) {
    if (!valid_vo_layer(layer))
        return RK_ERR_VO_INVALID_LAYERID;
    auto &value = state.layers[static_cast<std::size_t>(layer)];
    if (out)
        *out = &value;
    return RK_SUCCESS;
}

RK_S32 require_vo_channel(VoState &state,
                          VO_LAYER layer,
                          VO_CHN chn,
                          VoLayerState **layerOut,
                          VoChannelState **channelOut) {
    if (!valid_vo_chn(chn))
        return RK_ERR_VO_INVALID_CHNID;
    VoLayerState *layerState = nullptr;
    RK_S32 result = require_vo_layer(state, layer, &layerState);
    if (result != RK_SUCCESS)
        return result;
    auto it = layerState->channels.find(chn);
    if (it == layerState->channels.end() || !it->second.configured)
        return RK_ERR_VO_CHN_NOT_CONFIG;
    if (layerOut)
        *layerOut = layerState;
    if (channelOut)
        *channelOut = &it->second;
    return RK_SUCCESS;
}

std::size_t vo_channel_capacity(const VoChannelState &channel) {
    if (channel.attr.u32MaxChnQueue != 0)
        return channel.attr.u32MaxChnQueue;
    if (channel.recvThreshold != 0)
        return channel.recvThreshold;
    return 4;
}

void fanout_wbc(VoState &state, const VIDEO_FRAME_INFO_S &frame) {
    for (auto &wbc : state.wbc) {
        if (!wbc.enabled)
            continue;
        const std::size_t capacity = std::max<RK_S32>(1, wbc.depth);
        if (wbc.ready.size() >= capacity)
            continue;
        (void)queue_frame(wbc.ready, frame);
    }
}

} // namespace

// ================================ VI API ================================

extern "C" ROCKIT_API RK_S32 RK_MPI_VI_SetDevAttr(VI_DEV dev, const VI_DEV_ATTR_S *attr) {
    if (!attr)
        return RK_ERR_VI_INVALID_NULL_PTR;
    if (!valid_vi_dev(dev))
        return RK_ERR_VI_INVALID_DEVID;
    auto &s = vi_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto &d = s.devices[static_cast<std::size_t>(dev)];
    if (d.enabled)
        return RK_ERR_VI_BUSY;
    d.attr = *attr;
    d.configured = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetDevAttr(VI_DEV dev, VI_DEV_ATTR_S *attr) {
    if (!attr)
        return RK_ERR_VI_INVALID_NULL_PTR;
    if (!valid_vi_dev(dev))
        return RK_ERR_VI_INVALID_DEVID;
    auto &s = vi_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto &d = s.devices[dev];
    if (!d.configured)
        return RK_ERR_VI_NOT_CONFIG;
    *attr = d.attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_EnableDev(VI_DEV dev) {
    if (!valid_vi_dev(dev))
        return RK_ERR_VI_INVALID_DEVID;
    auto &s = vi_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto &d = s.devices[static_cast<std::size_t>(dev)];
    if (!d.configured)
        return RK_ERR_VI_NOT_CONFIG;
    d.enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_DisableDev(VI_DEV dev) {
    if (!valid_vi_dev(dev))
        return RK_ERR_VI_INVALID_DEVID;
    auto &s = vi_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto &d = s.devices[static_cast<std::size_t>(dev)];
    if (!d.enabled)
        return RK_ERR_VI_NOT_PERM;
    d.enabled = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_SetDevBindPipe(VI_DEV dev,
                                                      const VI_DEV_BIND_PIPE_S *binding) {
    if (dev < 0)
        return RK_ERR_VI_INVALID_DEVID;
    if (!binding)
        return RK_ERR_VI_INVALID_NULL_PTR;
    if (!valid_vi_dev(dev))
        return RK_ERR_VI_INVALID_DEVID;
    if (binding->u32Num > MAX_VI_BIND_PIPE_NUM)
        return RK_ERR_VI_INVALID_PARA;
    for (RK_U32 i = 0; i < binding->u32Num; ++i) {
        if (!valid_vi_pipe(binding->PipeId[i]))
            return RK_ERR_VI_INVALID_PIPEID;
    }
    auto &s = vi_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.devices[static_cast<std::size_t>(dev)].binding = *binding;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetDevBindPipe(VI_DEV dev, VI_DEV_BIND_PIPE_S *binding) {
    if (dev < 0)
        return RK_ERR_VI_INVALID_DEVID;
    if (!binding)
        return RK_ERR_VI_INVALID_NULL_PTR;
    if (!valid_vi_dev(dev))
        return RK_ERR_VI_INVALID_DEVID;
    auto &s = vi_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    *binding = s.devices[static_cast<std::size_t>(dev)].binding;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetDevIsEnable(VI_DEV dev) {
    if (!valid_vi_dev(dev))
        return RK_ERR_VI_INVALID_DEVID;
    auto &s = vi_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.devices[static_cast<std::size_t>(dev)].enabled ? RK_TRUE : RK_FALSE;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_QueryDevStatus(VI_DEV, VI_DEV_STATUS_S *) {
    return RK_ERR_VI_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_StartPipe(VI_PIPE) {
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_StopPipe(VI_PIPE) {
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_SetChnAttr(VI_PIPE pipe,
                                                  VI_CHN chn,
                                                  const VI_CHN_ATTR_S *attr) {
    if (pipe < 0)
        return RK_ERR_VI_INVALID_DEVID;
    if (!attr)
        return RK_ERR_VI_INVALID_NULL_PTR;
    if (!valid_vi_pipe(pipe))
        return RK_ERR_VI_INVALID_PIPEID;
    if (!valid_vi_chn(chn))
        return RK_ERR_VI_INVALID_CHNID;
    auto &s = vi_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto &c = s.pipes[pipe].channels[chn];
    if (c.enabled)
        return RK_ERR_VI_BUSY;
    c.attr = *attr;
    c.configured = true;
    c.fd = 3000 + pipe * VI_MAX_CHN_NUM + chn;
    return RK_SUCCESS;
}
#define ROCKIT_VI_VALIDATE_DEVICE_PIPE(pipe_)                                                      \
    if ((pipe_) < 0)                                                                               \
        return RK_ERR_VI_INVALID_DEVID
#define ROCKIT_VI_LOCK_EXISTING_CHANNEL(pipe_, channel_)                                           \
    auto &s = vi_state();                                                                          \
    std::lock_guard<std::mutex> lock(s.mutex);                                                     \
    ViChannelState *c = nullptr;                                                                   \
    RK_S32 r = require_vi_channel(s, (pipe_), (channel_), &c);                                     \
    if (r != RK_SUCCESS)                                                                           \
        return r

extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetChnAttr(VI_PIPE pipe, VI_CHN chn, VI_CHN_ATTR_S *attr) {
    ROCKIT_VI_VALIDATE_DEVICE_PIPE(pipe);
    if (!attr)
        return RK_ERR_VI_INVALID_NULL_PTR;
    ROCKIT_VI_LOCK_EXISTING_CHANNEL(pipe, chn);
    *attr = c->attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_EnableChn(VI_PIPE pipe, VI_CHN chn) {
    ROCKIT_VI_VALIDATE_DEVICE_PIPE(pipe);
    ROCKIT_VI_LOCK_EXISTING_CHANNEL(pipe, chn);
    c->enabled = true;
    c->paused = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_EnableChnExt(VI_PIPE, VI_CHN) {
    return RK_ERR_VI_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_DisableChn(VI_PIPE pipe, VI_CHN chn) {
    ROCKIT_VI_VALIDATE_DEVICE_PIPE(pipe);
    ROCKIT_VI_LOCK_EXISTING_CHANNEL(pipe, chn);
    if (!c->outstandingFrames.empty() || !c->outstandingStreams.empty())
        return RK_ERR_VI_BUSY;
    c->enabled = false;
    c->paused = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_DisableChnExt(VI_PIPE, VI_CHN) {
    return RK_ERR_VI_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_PauseChn(VI_PIPE pipe, VI_CHN chn) {
    ROCKIT_VI_VALIDATE_DEVICE_PIPE(pipe);
    ROCKIT_VI_LOCK_EXISTING_CHANNEL(pipe, chn);
    if (!c->enabled)
        return RK_ERR_VI_NOT_PERM;
    c->paused = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_ResumeChn(VI_PIPE pipe, VI_CHN chn) {
    ROCKIT_VI_VALIDATE_DEVICE_PIPE(pipe);
    ROCKIT_VI_LOCK_EXISTING_CHANNEL(pipe, chn);
    if (!c->enabled)
        return RK_ERR_VI_NOT_PERM;
    c->paused = false;
    s.changed.notify_all();
    return RK_SUCCESS;
}
#undef ROCKIT_VI_LOCK_EXISTING_CHANNEL
#undef ROCKIT_VI_VALIDATE_DEVICE_PIPE

extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetChnFrame(VI_PIPE pipe,
                                                   VI_CHN chn,
                                                   VIDEO_FRAME_INFO_S *frame,
                                                   RK_S32 timeout) {
    if (!rockit_local::system_initialized())
        return RK_ERR_VI_NOT_CONFIG;
    if (pipe < 0)
        return RK_ERR_VI_INVALID_PIPEID;
    if (!frame) {
        return RK_ERR_VI_INVALID_NULL_PTR;
    }
    auto &state = vi_state();
    std::unique_lock<std::mutex> lock(state.mutex);
    ViChannelState *channel = nullptr;
    RK_S32 result = require_vi_channel(state, pipe, chn, &channel);
    if (result != RK_SUCCESS) {
        return result;
    }
    auto ready = [&] {
        return channel->enabled && !channel->paused && channel->userPicEnabled &&
               channel->hasUserPic;
    };
    if (!wait_until(state.changed, lock, timeout, ready)) {
        return RK_ERR_VI_BUF_EMPTY;
    }

    VIDEO_FRAME_INFO_S produced{};
    bool ownsProducedBlock = false;
    if (channel->userPic.enUsrPicMode == VI_USERPIC_MODE_BGC) {
        produced.stVFrame.u32Width = channel->attr.stSize.u32Width;
        produced.stVFrame.u32Height = channel->attr.stSize.u32Height;
        produced.stVFrame.u32VirWidth = channel->attr.stSize.u32Width;
        produced.stVFrame.u32VirHeight = channel->attr.stSize.u32Height;
        produced.stVFrame.enPixelFormat = channel->attr.enPixelFormat;
        produced.stVFrame.enVideoFormat = channel->attr.enVideoFormat;
        produced.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
        produced.stVFrame.enDynamicRange = channel->attr.enDynamicRange;
        if (rockit_local::cpu_solid_frame(produced,
                                          channel->userPic.unUsrPic.stUsrPicBg.u32BgColor,
                                          &produced) != RK_SUCCESS) {
            return RK_ERR_VI_NOMEM;
        }
        ownsProducedBlock = true;
    } else if (channel->userPic.enUsrPicMode == VI_USERPIC_MODE_PIC) {
        const VIDEO_FRAME_INFO_S &source = channel->userPic.unUsrPic.stUsrPicFrm;
        if (rockit_local::cpu_frame_is_supported(source) &&
            channel->attr.enPixelFormat == source.stVFrame.enPixelFormat) {
            const auto transform = vi_cpu_transform(*channel, source);
            if (rockit_local::cpu_transform_frame(source, transform, nullptr, &produced) !=
                RK_SUCCESS) {
                return RK_ERR_VI_NOMEM;
            }
            ownsProducedBlock = true;
        } else {
            produced = source;
            if (!retain_mb(frame_block(produced))) {
                return RK_ERR_VI_UNEXIST;
            }
        }
    } else {
        return RK_ERR_VI_INVALID_PARA;
    }

    if (ownsProducedBlock) {
        const MPP_CHN_S mppChannel{RK_ID_VI, pipe, chn};
        RK_U32 applied = 0;
        (void)rockit_local::apply_regions_to_frame(mppChannel, &produced, &applied);
    }

    ++channel->sequence;
    ++channel->deliveredFrames;
    produced.stVFrame.u32TimeRef = channel->sequence;
    *frame = produced;
    const MB_BLK producedBlock = frame_block(*frame);
    ++channel->outstandingFrames[producedBlock];
    const MPP_CHN_S sourceChannel{RK_ID_VI, pipe, chn};
    lock.unlock();
    const RK_S32 routeResult = rockit_local::route_video_frame(sourceChannel, *frame, timeout);
    if (routeResult == RK_SUCCESS)
        return RK_SUCCESS;

    lock.lock();
    ViChannelState *current = nullptr;
    if (require_vi_channel(state, pipe, chn, &current) == RK_SUCCESS) {
        auto iterator = current->outstandingFrames.find(producedBlock);
        if (iterator != current->outstandingFrames.end()) {
            if (--iterator->second == 0)
                current->outstandingFrames.erase(iterator);
        }
    }
    release_mb(producedBlock);
    zero_struct(frame);
    state.changed.notify_all();
    return routeResult;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_ReleaseChnFrame(VI_PIPE pipe,
                                                       VI_CHN chn,
                                                       const VIDEO_FRAME_INFO_S *frame) {
    if (!rockit_local::system_initialized())
        return RK_ERR_VI_NOT_CONFIG;
    if (pipe < 0)
        return RK_ERR_VI_INVALID_PIPEID;
    if (!frame)
        return RK_ERR_VI_INVALID_NULL_PTR;
    auto &s = vi_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    ViChannelState *c = nullptr;
    RK_S32 r = require_vi_channel(s, pipe, chn, &c);
    if (r != RK_SUCCESS)
        return r;
    MB_BLK block = frame_block(*frame);
    auto it = c->outstandingFrames.find(block);
    if (it == c->outstandingFrames.end() || it->second == 0)
        return RK_ERR_VI_INVALID_PARA;
    if (--it->second == 0)
        c->outstandingFrames.erase(it);
    release_mb(block);
    s.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_ChnSaveFile(VI_PIPE pipe,
                                                   VI_CHN chn,
                                                   VI_SAVE_FILE_INFO_S *info) {
    if (!rockit_local::system_initialized())
        return RK_ERR_VI_NOT_CONFIG;
    if (pipe < 0)
        return RK_ERR_VI_INVALID_PIPEID;
    if (!info)
        return RK_ERR_VI_INVALID_NULL_PTR;
    auto &s = vi_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return require_vi_channel(s, pipe, chn, nullptr);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_QueryChnStatus(VI_PIPE pipe,
                                                      VI_CHN chn,
                                                      VI_CHN_STATUS_S *status) {
    if (!rockit_local::system_initialized())
        return RK_ERR_VI_NOT_CONFIG;
    if (pipe < 0)
        return RK_ERR_VI_INVALID_PIPEID;
    if (!status)
        return RK_ERR_VI_INVALID_NULL_PTR;
    auto &s = vi_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    ViChannelState *c = nullptr;
    RK_S32 r = require_vi_channel(s, pipe, chn, &c);
    if (r != RK_SUCCESS)
        return r;
    zero_struct(status);
    status->bEnable = c->enabled ? RK_TRUE : RK_FALSE;
    status->u32FrameRate = c->attr.stFrameRate.s32DstFrameRate > 0
                               ? static_cast<RK_U32>(c->attr.stFrameRate.s32DstFrameRate)
                               : 0;
    status->u32CurFrameID = c->sequence;
    status->stSize = c->attr.stSize;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetChnWrapBufAttr(VI_PIPE, VI_CHN, VI_CHN_BUF_WRAP_S *) {
    return RK_ERR_VI_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_SetChnWrapBufAttr(VI_PIPE,
                                                         VI_CHN,
                                                         const VI_CHN_BUF_WRAP_S *) {
    return RK_ERR_VI_NOT_SUPPORT;
}
// VI public entry points share two invariant preambles.  Keeping them in
// one place prevents validation/error-order drift across the large surface.
#define ROCKIT_VI_VALIDATE_PIPE(pipe)                                                              \
    if (!rockit_local::system_initialized())                                                       \
        return RK_ERR_VI_NOT_CONFIG;                                                               \
    if ((pipe) < 0)                                                                                \
    return RK_ERR_VI_INVALID_PIPEID

#define ROCKIT_VI_LOCK_CHANNEL(pipe, channel)                                                      \
    auto &s = vi_state();                                                                          \
    std::lock_guard<std::mutex> l(s.mutex);                                                        \
    ViChannelState *x = nullptr;                                                                   \
    RK_S32 r = require_vi_channel(s, (pipe), (channel), &x);                                       \
    if (r != RK_SUCCESS)                                                                           \
    return r

extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetChnFd(VI_PIPE p, VI_CHN c) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    auto &s = vi_state();
    std::lock_guard<std::mutex> l(s.mutex);
    ViChannelState *x = nullptr;
    RK_S32 r = require_vi_channel(s, p, c, &x);
    return r == RK_SUCCESS ? x->fd : r;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_CloseChnFd(VI_PIPE p, VI_CHN c) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    x->fd = -1;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_SetChnFreeze(VI_PIPE p, VI_CHN c, RK_BOOL v) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    x->freeze = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetChnFreeze(VI_PIPE p, VI_CHN c, RK_BOOL *v) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    if (!v)
        return RK_ERR_VI_INVALID_NULL_PTR;
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    *v = x->freeze;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_AttachMbPool(VI_PIPE, VI_CHN, MB_POOL) {
    return RK_ERR_VI_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_DetachMbPool(VI_PIPE, VI_CHN) {
    return RK_ERR_VI_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_SetUserPic(VI_PIPE p, VI_CHN c, const VI_USERPIC_ATTR_S *v) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    if (!v)
        return RK_ERR_VI_INVALID_NULL_PTR;
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    if (v->enUsrPicMode == VI_USERPIC_MODE_PIC &&
        !retain_mb(v->unUsrPic.stUsrPicFrm.stVFrame.pMbBlk))
        return RK_ERR_VI_INVALID_PARA;
    release_vi_user_pic(*x);
    x->userPic = *v;
    x->hasUserPic = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_EnableUserPic(VI_PIPE p, VI_CHN c) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    if (!x->hasUserPic)
        return RK_ERR_VI_NOT_CONFIG;
    x->userPicEnabled = true;
    s.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_DisableUserPic(VI_PIPE p, VI_CHN c) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    x->userPicEnabled = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetChnConnectInfo(VI_PIPE p,
                                                         VI_CHN c,
                                                         VI_CONNECT_INFO_S *v) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    if (!v)
        return RK_ERR_VI_INVALID_NULL_PTR;
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    zero_struct(v);
    v->u32Width = x->attr.stSize.u32Width;
    v->u32Height = x->attr.stSize.u32Height;
    v->f32FrameRate = static_cast<RK_FLOAT>(x->attr.stFrameRate.s32DstFrameRate);
    v->enPixFmt = x->attr.enPixelFormat;
    v->enConnect = x->enabled ? VI_CONNECT_STATE_CONNECT : VI_CONNECT_STATE_DISCONNECT;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_SetChnEdid(VI_PIPE p, VI_CHN c, const VI_EDID_S *v) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    if (!v)
        return RK_ERR_VI_INVALID_NULL_PTR;
    if (v->u32Blocks && !v->pu8Edid)
        return RK_ERR_VI_INVALID_NULL_PTR;
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    const std::size_t bytes = static_cast<std::size_t>(v->u32Blocks) * 128U;
    if (bytes == 0)
        x->edid.clear();
    else
        x->edid.assign(v->pu8Edid, v->pu8Edid + bytes);
    x->edidPad = v->u32Pad;
    x->edidStartBlock = v->u32StartBlock;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetChnEdid(VI_PIPE p, VI_CHN c, VI_EDID_S *v) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    if (!v || (!v->pu8Edid && v->u32Blocks))
        return RK_ERR_VI_INVALID_NULL_PTR;
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    const std::size_t cap = static_cast<std::size_t>(v->u32Blocks) * 128U;
    const std::size_t n = std::min(cap, x->edid.size());
    if (n)
        std::memcpy(v->pu8Edid, x->edid.data(), n);
    v->u32Pad = x->edidPad;
    v->u32StartBlock = x->edidStartBlock;
    v->u32Blocks = static_cast<RK_U32>((x->edid.size() + 127U) / 128U);
    return cap < x->edid.size() ? RK_ERR_VI_BUF_FULL : RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetChnStreamCodec(VI_PIPE p, VI_CHN c, RK_CODEC_ID_E *v) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    if (!v)
        return RK_ERR_VI_INVALID_NULL_PTR;
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    *v = x->codec;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_SetChnStreamCodec(VI_PIPE p,
                                                         VI_CHN c,
                                                         const RK_CODEC_ID_E v) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    x->codec = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetChnStream(VI_PIPE p,
                                                    VI_CHN c,
                                                    VI_STREAM_S *v,
                                                    RK_S32 timeout) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    if (!v)
        return RK_ERR_VI_INVALID_NULL_PTR;
    auto &s = vi_state();
    std::unique_lock<std::mutex> l(s.mutex);
    ViChannelState *x = nullptr;
    RK_S32 r = require_vi_channel(s, p, c, &x);
    if (r != RK_SUCCESS)
        return r;
    auto ready = [&] {
        return x->enabled && !x->paused && x->userPicEnabled && x->hasUserPic &&
               x->userPic.enUsrPicMode == VI_USERPIC_MODE_PIC;
    };
    if (!wait_until(s.changed, l, timeout, ready))
        return RK_ERR_VI_BUF_EMPTY;
    MB_BLK b = x->userPic.unUsrPic.stUsrPicFrm.stVFrame.pMbBlk;
    if (!retain_mb(b))
        return RK_ERR_VI_UNEXIST;
    v->pMbBlk = b;
    v->u32Len = static_cast<RK_U32>(std::min<RK_U64>(RK_MPI_MB_GetSize(b), UINT32_MAX));
    v->u32Seq = ++x->sequence;
    v->u64PTS = x->userPic.unUsrPic.stUsrPicFrm.stVFrame.u64PTS;
    ++x->outstandingStreams[b];
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_ReleaseChnStream(VI_PIPE p, VI_CHN c, const VI_STREAM_S *v) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    if (!v)
        return RK_ERR_VI_INVALID_NULL_PTR;
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    auto it = x->outstandingStreams.find(v->pMbBlk);
    if (it == x->outstandingStreams.end() || it->second == 0)
        return RK_ERR_VI_INVALID_PARA;
    if (--it->second == 0)
        x->outstandingStreams.erase(it);
    release_mb(v->pMbBlk);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_RegChnEventCallback(VI_PIPE p,
                                                           VI_CHN c,
                                                           const VI_EVENT_CALL_BACK_S *v) {

    ROCKIT_VI_VALIDATE_PIPE(p);
    if (!v)
        return RK_ERR_VI_INVALID_NULL_PTR;
    ROCKIT_VI_LOCK_CHANNEL(p, c);
    x->callback = *v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetEptz(VI_PIPE, VI_CHN, VI_CROP_INFO_S *) {
    return RK_ERR_VI_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_SetEptz(VI_PIPE, VI_CHN, VI_CROP_INFO_S) {
    return RK_ERR_VI_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_GetChnMirrorFlip(VI_PIPE, VI_CHN, VI_ISP_MIRROR_FLIP_S *) {
    return RK_ERR_VI_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VI_SetChnMirrorFlip(VI_PIPE, VI_CHN, VI_ISP_MIRROR_FLIP_S) {
    return RK_ERR_VI_NOT_SUPPORT;
}
#define VI_GETSET_PTR(name, type, field)                                                           \
    extern "C" ROCKIT_API RK_S32 RK_MPI_VI_Set##name(VI_PIPE p, VI_CHN c, const type *v) {         \
        ROCKIT_VI_VALIDATE_PIPE(p);                                                                \
        if (!v)                                                                                    \
            return RK_ERR_VI_INVALID_NULL_PTR;                                                     \
        ROCKIT_VI_LOCK_CHANNEL(p, c);                                                              \
        x->field = *v;                                                                             \
        return RK_SUCCESS;                                                                         \
    }                                                                                              \
    extern "C" ROCKIT_API RK_S32 RK_MPI_VI_Get##name(VI_PIPE p, VI_CHN c, type *v) {               \
        if (!rockit_local::system_initialized())                                                   \
            return RK_ERR_VI_NOT_CONFIG;                                                           \
        if (!v)                                                                                    \
            return RK_ERR_VI_INVALID_NULL_PTR;                                                     \
        ROCKIT_VI_LOCK_CHANNEL(p, c);                                                              \
        *v = x->field;                                                                             \
        return RK_SUCCESS;                                                                         \
    }
VI_GETSET_PTR(ChnDISConfig, DIS_CONFIG_S, disConfig)
VI_GETSET_PTR(ChnDISAttr, DIS_ATTR_S, disAttr)
VI_GETSET_PTR(SwcacConfig, SWCAC_CONFIG_S, swcac)
#undef VI_GETSET_PTR
#undef ROCKIT_VI_LOCK_CHANNEL
#undef ROCKIT_VI_VALIDATE_PIPE

// =============================== VPSS API ===============================

#define ROCKIT_VPSS_LOCK_GROUP(group_)                                                           \
    auto &s = vpss_state();                                                                       \
    std::lock_guard<std::mutex> l(s.mutex);                                                       \
    VpssGroupState *x = nullptr;                                                                  \
    RK_S32 r = require_vpss_group(s, (group_), &x);                                               \
    if (r != RK_SUCCESS)                                                                          \
        return r
#define ROCKIT_VPSS_LOCK_CHANNEL(group_, channel_)                                                \
    auto &s = vpss_state();                                                                       \
    std::lock_guard<std::mutex> l(s.mutex);                                                       \
    VpssChannelState *x = nullptr;                                                                \
    RK_S32 r = require_vpss_channel(s, (group_), (channel_), nullptr, &x);                        \
    if (r != RK_SUCCESS)                                                                          \
        return r


extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_CreateGrp(VPSS_GRP g, const VPSS_GRP_ATTR_S *a) {
    if (!valid_vpss_group(g))
        return RK_ERR_VPSS_INVALID_DEVID;
    if (!a)
        return RK_ERR_VPSS_NULL_PTR;
    if (!valid_vpss_group(g))
        return RK_ERR_VPSS_INVALID_DEVID;
    auto &s = vpss_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (s.groups.count(g))
        return RK_ERR_VPSS_EXIST;
    auto v = std::make_unique<VpssGroupState>();
    v->attr = *a;
    for (int c = 0; c < VPSS_MAX_CHN_NUM; ++c)
        v->channels[c].fd = 4000 + g * VPSS_MAX_CHN_NUM + c;
    s.groups.emplace(g, std::move(v));
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_DestroyGrp(VPSS_GRP g) {
    ROCKIT_VPSS_LOCK_GROUP(g);
    if (x->started)
        return RK_ERR_VPSS_BUSY;
    for (auto &c : x->channels)
        if (!c.outstanding.empty())
            return RK_ERR_VPSS_BUSY;
    if (!x->groupOutstanding.empty())
        return RK_ERR_VPSS_BUSY;
    clear_vpss_group(*x);
    s.groups.erase(g);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_StartGrp(VPSS_GRP g) {
    ROCKIT_VPSS_LOCK_GROUP(g);
    x->started = true;
    s.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_StopGrp(VPSS_GRP g) {
    ROCKIT_VPSS_LOCK_GROUP(g);
    x->started = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_ResetGrp(VPSS_GRP g) {
    ROCKIT_VPSS_LOCK_GROUP(g);
    if (!x->groupOutstanding.empty())
        return RK_ERR_VPSS_BUSY;
    for (auto &c : x->channels)
        if (!c.outstanding.empty())
            return RK_ERR_VPSS_BUSY;
    clear_vpss_group(*x);
    return RK_SUCCESS;
}
#define ROCKIT_VPSS_LOCK_GROUP_PTR(group_, pointer_)                                               \
    if (!valid_vpss_group(group_))                                                                 \
        return RK_ERR_VPSS_INVALID_DEVID;                                                          \
    if (!(pointer_))                                                                               \
        return RK_ERR_VPSS_NULL_PTR;                                                               \
    ROCKIT_VPSS_LOCK_GROUP(group_)
#define VPSS_GRP_GETSET_PTR(name, type, field)                                                     \
    extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_Set##name(VPSS_GRP g, const type *v) {                \
        ROCKIT_VPSS_LOCK_GROUP_PTR(g, v);                                                          \
        x->field = *v;                                                                             \
        return RK_SUCCESS;                                                                         \
    }                                                                                              \
    extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_Get##name(VPSS_GRP g, type *v) {                      \
        ROCKIT_VPSS_LOCK_GROUP_PTR(g, v);                                                          \
        *v = x->field;                                                                             \
        return RK_SUCCESS;                                                                         \
    }
VPSS_GRP_GETSET_PTR(GrpAttr, VPSS_GRP_ATTR_S, attr)
VPSS_GRP_GETSET_PTR(GrpCrop, VPSS_CROP_INFO_S, crop)
VPSS_GRP_GETSET_PTR(GrpAIISPAttr, AIISP_ATTR_S, aiisp)
#undef VPSS_GRP_GETSET_PTR
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_SetGrpRotation(VPSS_GRP g, ROTATION_E v) {
    ROCKIT_VPSS_LOCK_GROUP(g);
    x->rotation = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_GetGrpRotation(VPSS_GRP g, ROTATION_E *v) {
    ROCKIT_VPSS_LOCK_GROUP_PTR(g, v);
    *v = x->rotation;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_SetGrpMirror(VPSS_GRP g, RK_BOOL m, RK_BOOL f) {
    ROCKIT_VPSS_LOCK_GROUP(g);
    x->mirror = m;
    x->flip = f;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_GetGrpMirror(VPSS_GRP g, RK_BOOL *m, RK_BOOL *f) {
    if (!valid_vpss_group(g))
        return RK_ERR_VPSS_INVALID_DEVID;
    if (!m || !f)
        return RK_ERR_VPSS_NULL_PTR;
    ROCKIT_VPSS_LOCK_GROUP(g);
    *m = x->mirror;
    *f = x->flip;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_SendFrame(VPSS_GRP groupId,
                                                   VPSS_GRP_PIPE,
                                                   const VIDEO_FRAME_INFO_S *frame,
                                                   RK_S32 timeout) {
    if (!valid_vpss_group(groupId))
        return RK_ERR_VPSS_INVALID_DEVID;
    if (!frame) {
        return RK_ERR_VPSS_NULL_PTR;
    }
    auto &state = vpss_state();
    std::unique_lock<std::mutex> lock(state.mutex);
    VpssGroupState *group = nullptr;
    RK_S32 result = require_vpss_group(state, groupId, &group);
    if (result != RK_SUCCESS) {
        return result;
    }
    if (!group->started) {
        return RK_ERR_VPSS_NOTREADY;
    }
    auto room = [&] {
        if (group->groupReady.size() >= vpss_group_capacity(*group)) {
            return false;
        }
        for (const auto &channel : group->channels) {
            if (channel.enabled && channel.ready.size() >= vpss_channel_capacity(channel)) {
                return false;
            }
        }
        return true;
    };
    if (!wait_until(state.changed, lock, timeout, room)) {
        return RK_ERR_VPSS_BUF_FULL;
    }

    struct PreparedOutput {
        VpssChannelState *channel{nullptr};
        VPSS_CHN channelId{0};
        VIDEO_FRAME_INFO_S frame{};
        bool ownsInitialReference{false};
    };
    std::vector<PreparedOutput> outputs;
    outputs.reserve(VPSS_MAX_CHN_NUM);

    const auto groupTransform = vpss_group_cpu_transform(*group, *frame);
    RK_U32 groupWidth =
        groupTransform.cropEnabled ? groupTransform.crop.u32Width : frame->stVFrame.u32Width;
    RK_U32 groupHeight =
        groupTransform.cropEnabled ? groupTransform.crop.u32Height : frame->stVFrame.u32Height;
    if (groupTransform.rotation == ROTATION_90 || groupTransform.rotation == ROTATION_270) {
        std::swap(groupWidth, groupHeight);
    }

    for (VPSS_CHN channelId = 0; channelId < VPSS_MAX_CHN_NUM; ++channelId) {
        auto &channel = group->channels[static_cast<std::size_t>(channelId)];
        if (!channel.enabled) {
            continue;
        }
        PreparedOutput output{};
        output.channel = &channel;
        output.channelId = channelId;
        const bool sameFormat = channel.attr.enPixelFormat == frame->stVFrame.enPixelFormat;
        const bool cpuCandidate = sameFormat && channel.attr.enCompressMode == COMPRESS_MODE_NONE &&
                                  rockit_local::cpu_frame_is_supported(*frame);
        if (cpuCandidate) {
            const auto channelTransform =
                vpss_channel_cpu_transform(channel, groupWidth, groupHeight);
            if (rockit_local::cpu_transform_frame(
                    *frame, groupTransform, &channelTransform, &output.frame) != RK_SUCCESS) {
                for (auto &prepared : outputs) {
                    if (prepared.ownsInitialReference) {
                        release_mb(frame_block(prepared.frame));
                    }
                }
                return RK_ERR_VPSS_NOMEM;
            }
            output.ownsInitialReference = true;
            output.frame.stVFrame.enPixelFormat = channel.attr.enPixelFormat;
            output.frame.stVFrame.enVideoFormat = channel.attr.enVideoFormat;
            output.frame.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
            output.frame.stVFrame.enDynamicRange = channel.attr.enDynamicRange;
            const MPP_CHN_S mppChannel{RK_ID_VPSS, groupId, channelId};
            RK_U32 applied = 0;
            (void)rockit_local::apply_regions_to_frame(mppChannel, &output.frame, &applied);
        } else {
            output.frame = vpss_transform_metadata(*frame, *group, channel);
        }
        outputs.push_back(output);
    }

    if (!queue_frame(group->groupReady, *frame)) {
        for (auto &prepared : outputs) {
            if (prepared.ownsInitialReference) {
                release_mb(frame_block(prepared.frame));
            }
        }
        return RK_ERR_VPSS_ILLEGAL_PARAM;
    }

#define ROCKIT_ROLLBACK_VPSS_GROUP_AND_PREPARED()                                                \
    release_mb(frame_block(group->groupReady.back().frame));                                      \
    group->groupReady.pop_back();                                                                  \
    for (auto &prepared : outputs) {                                                               \
        if (prepared.ownsInitialReference) {                                                       \
            release_mb(frame_block(prepared.frame));                                               \
        }                                                                                          \
    }                                                                                              \
    return RK_ERR_VPSS_ILLEGAL_PARAM

    std::size_t queuedCount = 0;
    for (auto &output : outputs) {
        if (!queue_frame(output.channel->ready, output.frame)) {
            for (std::size_t index = 0; index < queuedCount; ++index) {
                auto &queued = outputs[index];
                release_mb(frame_block(queued.channel->ready.back().frame));
                queued.channel->ready.pop_back();
            }
            ROCKIT_ROLLBACK_VPSS_GROUP_AND_PREPARED();
        }
        ++queuedCount;
    }

    std::size_t latestRetainedCount = 0;
    for (auto &output : outputs) {
        if (!retain_mb(frame_block(output.frame))) {
            for (std::size_t index = 0; index < latestRetainedCount; ++index) {
                release_mb(frame_block(outputs[index].frame));
            }
            for (auto &queued : outputs) {
                release_mb(frame_block(queued.channel->ready.back().frame));
                queued.channel->ready.pop_back();
            }
            ROCKIT_ROLLBACK_VPSS_GROUP_AND_PREPARED();
        }
        ++latestRetainedCount;
    }
#undef ROCKIT_ROLLBACK_VPSS_GROUP_AND_PREPARED
    for (auto &output : outputs) {
        if (output.channel->hasLatestFrame) {
            release_mb(frame_block(output.channel->latestFrame));
        }
        output.channel->latestFrame = output.frame;
        output.channel->hasLatestFrame = true;
    }
    for (auto &output : outputs) {
        if (output.ownsInitialReference) {
            release_mb(frame_block(output.frame));
        }
    }
    std::vector<std::pair<MPP_CHN_S, VIDEO_FRAME_INFO_S>> routeFrames;
    routeFrames.reserve(outputs.size());
    for (const auto &output : outputs) {
        routeFrames.push_back({MPP_CHN_S{RK_ID_VPSS, groupId, output.channelId}, output.frame});
    }
    state.changed.notify_all();
    lock.unlock();
    for (const auto &route : routeFrames) {
        const RK_S32 routeResult =
            rockit_local::route_video_frame(route.first, route.second, timeout);
        if (routeResult != RK_SUCCESS)
            return routeResult;
    }
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_GetGrpFrame(VPSS_GRP g,
                                                     VPSS_GRP_PIPE,
                                                     VIDEO_FRAME_INFO_S *f) {
    ROCKIT_VPSS_LOCK_GROUP_PTR(g, f);
    if (x->groupReady.empty())
        return RK_ERR_VPSS_BUF_EMPTY;
    *f = x->groupReady.front().frame;
    x->groupReady.pop_front();
    ++x->groupOutstanding[frame_block(*f)];
    s.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_ReleaseGrpFrame(VPSS_GRP g,
                                                         VPSS_GRP_PIPE,
                                                         const VIDEO_FRAME_INFO_S *f) {
    ROCKIT_VPSS_LOCK_GROUP_PTR(g, f);
    MB_BLK b = frame_block(*f);
    auto it = x->groupOutstanding.find(b);
    if (it == x->groupOutstanding.end())
        return RK_ERR_VPSS_ILLEGAL_PARAM;
    if (--it->second == 0)
        x->groupOutstanding.erase(it);
    release_mb(b);
    s.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_EnableBackupFrame(VPSS_GRP g) {
    ROCKIT_VPSS_LOCK_GROUP(g);
    x->backupFrame = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_DisableBackupFrame(VPSS_GRP g) {
    ROCKIT_VPSS_LOCK_GROUP(g);
    x->backupFrame = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_SetGrpDelay(VPSS_GRP g, RK_U32 v) {
    ROCKIT_VPSS_LOCK_GROUP(g);
    x->delay = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_GetGrpDelay(VPSS_GRP g, RK_U32 *v) {
    if (!v)
        return RK_ERR_VPSS_NULL_PTR;
    ROCKIT_VPSS_LOCK_GROUP(g);
    *v = x->delay;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_EnableUserFrameRateCtrl(VPSS_GRP) {
    return RK_ERR_VPSS_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_DisableUserFrameRateCtrl(VPSS_GRP) {
    return RK_ERR_VPSS_NOT_SUPPORT;
}
#define ROCKIT_VPSS_LOCK_CHANNEL_PTR(group_, channel_, pointer_, pointer_error_)                   \
    if (!valid_vpss_chn(channel_))                                                                 \
        return RK_ERR_VPSS_INVALID_CHNID;                                                          \
    if (!(pointer_))                                                                               \
        return pointer_error_;                                                                     \
    ROCKIT_VPSS_LOCK_CHANNEL(group_, channel_)
#define VPSS_CHN_GETSET_PTR(name, type, field)                                                     \
    extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_Set##name(VPSS_GRP g, VPSS_CHN c, const type *v) {    \
        ROCKIT_VPSS_LOCK_CHANNEL_PTR(g, c, v, RK_ERR_VPSS_NULL_PTR);                               \
        x->field = *v;                                                                             \
        return RK_SUCCESS;                                                                         \
    }                                                                                              \
    extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_Get##name(VPSS_GRP g, VPSS_CHN c, type *v) {          \
        ROCKIT_VPSS_LOCK_CHANNEL_PTR(g, c, v, RK_ERR_VPSS_NULL_PTR);                               \
        *v = x->field;                                                                             \
        return RK_SUCCESS;                                                                         \
    }
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_SetChnAttr(VPSS_GRP g,
                                                    VPSS_CHN c,
                                                    const VPSS_CHN_ATTR_S *v) {
    if (!valid_vpss_chn(c))
        return RK_ERR_VPSS_INVALID_CHNID;
    if (!v)
        return RK_ERR_VPSS_NULL_PTR;
    if (!valid_vpss_chn(c))
        return RK_ERR_VPSS_INVALID_CHNID;
    auto &s = vpss_state();
    std::lock_guard<std::mutex> l(s.mutex);
    VpssGroupState *group = nullptr;
    RK_S32 r = require_vpss_group(s, g, &group);
    if (r != RK_SUCCESS)
        return r;
    auto &channel = group->channels[static_cast<std::size_t>(c)];
    if (channel.enabled)
        return RK_ERR_VPSS_BUSY;
    channel.attr = *v;
    channel.configured = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_GetChnAttr(VPSS_GRP g, VPSS_CHN c, VPSS_CHN_ATTR_S *v) {
    if (!valid_vpss_chn(c))
        return RK_ERR_VPSS_INVALID_CHNID;
    if (!v)
        return RK_ERR_VPSS_NULL_PTR;
    auto &s = vpss_state();
    std::lock_guard<std::mutex> l(s.mutex);
    VpssChannelState *channel = nullptr;
    RK_S32 r = require_vpss_channel(s, g, c, nullptr, &channel);
    if (r != RK_SUCCESS)
        return r;
    *v = channel->attr;
    return RK_SUCCESS;
}
VPSS_CHN_GETSET_PTR(ChnCrop, VPSS_CROP_INFO_S, crop)
#undef VPSS_CHN_GETSET_PTR
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_SetChnRotationEx(VPSS_GRP g,
                                                          VPSS_CHN c,
                                                          const VPSS_ROTATION_EX_ATTR_S *v) {
    ROCKIT_VPSS_LOCK_CHANNEL_PTR(g, c, v, RK_ERR_VPSS_ILLEGAL_PARAM);
    x->rotationEx = *v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_GetChnRotationEx(VPSS_GRP g,
                                                          VPSS_CHN c,
                                                          VPSS_ROTATION_EX_ATTR_S *v) {
    ROCKIT_VPSS_LOCK_CHANNEL_PTR(g, c, v, RK_ERR_VPSS_ILLEGAL_PARAM);
    *v = x->rotationEx;
    return RK_SUCCESS;
}
#undef ROCKIT_VPSS_LOCK_CHANNEL_PTR
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_EnableChn(VPSS_GRP g, VPSS_CHN c) {
    if (!valid_vpss_chn(c))
        return RK_ERR_VPSS_INVALID_CHNID;
    ROCKIT_VPSS_LOCK_GROUP(g);
    auto &ch = x->channels[c];
    if (!ch.configured)
        return RK_ERR_VPSS_UNEXIST;
    ch.enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_DisableChn(VPSS_GRP g, VPSS_CHN c) {
    auto &s = vpss_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    VpssChannelState *channel = nullptr;
    RK_S32 result = require_vpss_channel(s, g, c, nullptr, &channel);
    if (result != RK_SUCCESS)
        return result;
    if (!channel->outstanding.empty())
        return RK_ERR_VPSS_BUSY;
    channel->enabled = false;
    release_queue(channel->ready);
    if (channel->hasLatestFrame) {
        release_mb(frame_block(channel->latestFrame));
        zero_struct(&channel->latestFrame);
        channel->hasLatestFrame = false;
    }
    s.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_SetChnRotation(VPSS_GRP g, VPSS_CHN c, ROTATION_E v) {
    ROCKIT_VPSS_LOCK_CHANNEL(g, c);
    x->rotation = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_GetChnRotation(VPSS_GRP g, VPSS_CHN c, ROTATION_E *v) {
    if (!valid_vpss_chn(c))
        return RK_ERR_VPSS_INVALID_CHNID;
    if (!v)
        return RK_ERR_VPSS_NULL_PTR;
    ROCKIT_VPSS_LOCK_CHANNEL(g, c);
    *v = x->rotation;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_GetChnFrame(VPSS_GRP g,
                                                     VPSS_CHN c,
                                                     VIDEO_FRAME_INFO_S *f,
                                                     RK_S32 timeout) {
    if (!valid_vpss_chn(c))
        return RK_ERR_VPSS_INVALID_CHNID;
    if (!f)
        return RK_ERR_VPSS_NULL_PTR;
    auto &s = vpss_state();
    std::unique_lock<std::mutex> l(s.mutex);
    VpssChannelState *x = nullptr;
    RK_S32 r = require_vpss_channel(s, g, c, nullptr, &x);
    if (r != RK_SUCCESS)
        return r;
    if (!x->enabled)
        return RK_ERR_VPSS_NOT_PERM;
    auto ready = [&] {
        return !x->ready.empty();
    };
    if (!wait_until(s.changed, l, timeout, ready))
        return RK_ERR_VPSS_BUF_EMPTY;
    *f = x->ready.front().frame;
    x->ready.pop_front();
    ++x->outstanding[frame_block(*f)];
    s.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_ReleaseChnFrame(VPSS_GRP g,
                                                         VPSS_CHN c,
                                                         const VIDEO_FRAME_INFO_S *f) {
    if (!valid_vpss_chn(c))
        return RK_ERR_VPSS_INVALID_CHNID;
    if (!f)
        return RK_ERR_VPSS_NULL_PTR;
    ROCKIT_VPSS_LOCK_CHANNEL(g, c);
    MB_BLK b = frame_block(*f);
    auto it = x->outstanding.find(b);
    if (it == x->outstanding.end())
        return RK_ERR_VPSS_ILLEGAL_PARAM;
    if (--it->second == 0)
        x->outstanding.erase(it);
    release_mb(b);
    s.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32
RK_MPI_VPSS_GetRegionLuma(VPSS_GRP, VPSS_CHN, const VIDEO_REGION_INFO_S *, RK_U64 *, RK_S32) {
    return RK_ERR_VPSS_NOT_SUPPORT;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_AttachMbPool(VPSS_GRP g, VPSS_CHN c, MB_POOL p) {
    ROCKIT_VPSS_LOCK_CHANNEL(g, c);
    x->pool = p;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_DetachMbPool(VPSS_GRP g, VPSS_CHN c) {
    return RK_MPI_VPSS_AttachMbPool(g, c, MB_INVALID_POOLID);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_EnableBufferShare(VPSS_GRP, VPSS_CHN) {
    return RK_ERR_VPSS_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_DisableBufferShare(VPSS_GRP, VPSS_CHN) {
    return RK_ERR_VPSS_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_GetChnFd(VPSS_GRP g, VPSS_CHN c) {
    auto &s = vpss_state();
    std::lock_guard<std::mutex> l(s.mutex);
    VpssChannelState *x = nullptr;
    RK_S32 r = require_vpss_channel(s, g, c, nullptr, &x);
    return r == RK_SUCCESS ? x->fd : r;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_CloseFd(VPSS_GRP g, VPSS_CHN c) {
    ROCKIT_VPSS_LOCK_CHANNEL(g, c);
    x->fd = -1;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_SetVProcDev(VPSS_GRP g, VIDEO_PROC_DEV_TYPE_E v) {
    if (!valid_vpss_group(g))
        return RK_ERR_VPSS_ILLEGAL_PARAM;
    ROCKIT_VPSS_LOCK_GROUP(g);
    x->procDev = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_GetVProcDev(VPSS_GRP g, VIDEO_PROC_DEV_TYPE_E *v) {
    ROCKIT_VPSS_LOCK_GROUP_PTR(g, v);
    *v = x->procDev;
    return RK_SUCCESS;
}
#undef ROCKIT_VPSS_LOCK_GROUP_PTR
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_SetModParam(const VPSS_MOD_PARAM_S *v) {
    if (!v)
        return RK_ERR_VPSS_NULL_PTR;
    auto &s = vpss_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.modParam = *v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VPSS_GetModParam(VPSS_MOD_PARAM_S *v) {
    if (!v)
        return RK_ERR_VPSS_NULL_PTR;
    auto &s = vpss_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *v = s.modParam;
    return RK_SUCCESS;
}

#undef ROCKIT_VPSS_LOCK_CHANNEL
#undef ROCKIT_VPSS_LOCK_GROUP

// ================================ VO API ================================

extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetPubAttr(VO_DEV d, const VO_PUB_ATTR_S *a) {
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    if (!a)
        return RK_ERR_VO_NULL_PTR;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &v = s.devices[d];
    v.attr = *a;
    v.configured = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetPubAttr(VO_DEV d, VO_PUB_ATTR_S *a) {
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    if (!a)
        return RK_ERR_VO_NULL_PTR;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *a = s.devices[d].attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetPostProcessParam(VO_DEV d, VO_CSC_S *v) {
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    if (!v)
        return RK_ERR_VO_NULL_PTR;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *v = s.devices[d].postProcess;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetPostProcessParam(VO_DEV d, VO_CSC_S *v) {
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    if (!v)
        return RK_ERR_VO_NULL_PTR;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devices[d].postProcess = *v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetVcntTiming(VO_DEV d, RK_U32 v) {
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devices[d].vcntTiming = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_Enable(VO_DEV d) {
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &v = s.devices[d];
    if (!v.configured)
        return RK_ERR_VO_BUSY;
    v.enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_Disable(VO_DEV d) {
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &v = s.devices[d];
    if (!v.enabled)
        return RK_ERR_VO_DEV_NOT_ENABLE;
    for (auto &layer : s.layers)
        if (layer.bound && layer.device == d && layer.enabled)
            return RK_ERR_VO_BUSY;
    v.enabled = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SyncDevs(RK_U32 devices) {
    if (devices == UINT32_MAX)
        return RK_ERR_VO_SYS_NOTREADY;
    return RK_ERR_VO_NOT_SUPPORT;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetHdmiParam(RK_U32 t, RK_U32 i, VO_HDMI_PARAM_S *v) {
    if (!rockit_local::system_initialized())
        return RK_ERR_VO_SYS_NOTREADY;
    if (!v)
        return RK_ERR_VO_NULL_PTR;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto it = s.hdmi.find({t, i});
    if (it == s.hdmi.end()) {
        zero_struct(v);
        return RK_SUCCESS;
    }
    *v = it->second;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetHdmiParam(RK_U32 t, RK_U32 i, const VO_HDMI_PARAM_S *v) {
    if (!rockit_local::system_initialized())
        return RK_ERR_VO_SYS_NOTREADY;
    if (!v)
        return RK_ERR_VO_NULL_PTR;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.hdmi[{t, i}] = *v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_Get_Edid(RK_U32, RK_U32, VO_EDID_S *v) {
    if (!rockit_local::system_initialized())
        return RK_ERR_VO_SYS_NOTREADY;
    if (!v)
        return RK_ERR_VO_NULL_PTR;
    zero_struct(v);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetSinkCapability(RK_U32, RK_U32, VO_SINK_CAPABILITY_S *v) {
    if (!rockit_local::system_initialized())
        return RK_ERR_VO_SYS_NOTREADY;
    if (!v)
        return RK_ERR_VO_NULL_PTR;
    zero_struct(v);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_RegCallbackFunc(RK_U32 t,
                                                       RK_U32 i,
                                                       RK_VO_CALLBACK_FUNC_S *v) {
    if (!rockit_local::system_initialized())
        return RK_ERR_VO_SYS_NOTREADY;
    if (!v)
        return RK_ERR_VO_NULL_PTR;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.callbacks[{t, i}] = *v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_UnRegCallbackFunc(RK_U32 t,
                                                         RK_U32 i,
                                                         RK_VO_CALLBACK_FUNC_S *) {
    if (!rockit_local::system_initialized())
        return RK_ERR_VO_SYS_NOTREADY;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.callbacks.erase({t, i});
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_RegVsyncCallbackFunc(VO_DEV d,
                                                            RK_VO_VSYNC_CALLBACK_FUNC_S *v) {
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    if (!rockit_local::system_initialized())
        return RK_ERR_VO_SYS_NOTREADY;
    if (!v)
        return RK_ERR_VO_NULL_PTR;
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devices[d].vsyncCallback = *v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_UnRegVsyncCallbackFunc(VO_DEV d,
                                                              RK_VO_VSYNC_CALLBACK_FUNC_S *) {
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    if (!rockit_local::system_initialized())
        return RK_ERR_VO_SYS_NOTREADY;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    zero_struct(&s.devices[d].vsyncCallback);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_CloseFd(RK_VOID) {
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_CreateGraphicsFrameBuffer(RK_U32 width,
                                                                 RK_U32 height,
                                                                 RK_U32 format,
                                                                 RK_VOID **handle) {
    if (!handle)
        return RK_ERR_VO_NULL_PTR;
    *handle = nullptr;

    const std::uint64_t pixels =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    if (pixels > static_cast<std::uint64_t>(SIZE_MAX) / 4U) {
        return RK_ERR_VO_NO_MEM;
    }

    std::shared_ptr<GraphicsBuffer> buffer;
    try {
        buffer = std::make_shared<GraphicsBuffer>();
        buffer->width = width;
        buffer->height = height;
        buffer->format = format;
        buffer->bytes.resize(static_cast<std::size_t>(pixels * 4U));
    } catch (...) {
        return RK_ERR_VO_NO_MEM;
    }

    auto token = std::unique_ptr<GraphicsHandleToken>(new (std::nothrow) GraphicsHandleToken());
    if (!token)
        return RK_ERR_VO_NO_MEM;

    auto &registry = graphics_buffer_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (registry.nextSerial == 0)
        return RK_ERR_VO_NO_MEM;
    token->serial = registry.nextSerial++;
    RK_VOID *const publicHandle = token.get();

    try {
        const auto inserted = registry.live.emplace(publicHandle, std::move(buffer));
        if (!inserted.second)
            return RK_ERR_VO_NO_MEM;
        try {
            registry.tokens.push_back(std::move(token));
        } catch (...) {
            registry.live.erase(publicHandle);
            throw;
        }
    } catch (...) {
        return RK_ERR_VO_NO_MEM;
    }

    *handle = publicHandle;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VO_DestroyGraphicsFrameBuffer(RK_VOID *handle) {
    if (!handle)
        return RK_ERR_VO_ILLEGAL_PARAM;

    std::shared_ptr<GraphicsBuffer> released;
    auto &registry = graphics_buffer_registry();
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto iterator = registry.live.find(handle);
        if (iterator == registry.live.end())
            return RK_ERR_VO_ILLEGAL_PARAM;
        released = std::move(iterator->second);
        registry.live.erase(iterator);
    }
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_VOID *RK_MPI_VO_GetGraphicsFrameBuffer(RK_VOID *handle) {
    const auto buffer = lookup_graphics_buffer(handle);
    return buffer && !buffer->bytes.empty() ? buffer->bytes.data() : nullptr;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetFrameInfo(RK_VOID *handle, VO_FRAME_INFO_S *info) {
    if (!handle || !info)
        return RK_ERR_VO_ILLEGAL_PARAM;
    const auto buffer = lookup_graphics_buffer(handle);
    if (!buffer)
        return RK_ERR_VO_ILLEGAL_PARAM;

    zero_struct(info);
    info->u32Width = buffer->width;
    info->u32Height = buffer->height;
    info->u32VirWidth = buffer->width;
    info->u32VirHeight = buffer->height;
    info->enPixelFormat = static_cast<PIXEL_FORMAT_E>(buffer->format);
    info->pData = buffer->bytes.empty() ? nullptr : buffer->bytes.data();
    info->u32Size = static_cast<RK_U32>(std::min<std::size_t>(buffer->bytes.size(), UINT32_MAX));
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetGFxMode(VO_GFX_MODE_E v) {
    if (static_cast<RK_S32>(v) < 0 ||
        static_cast<RK_U32>(v) > static_cast<RK_U32>(VO_MODE_GFX_PRE_CREATED))
        return RK_ERR_VO_ILLEGAL_PARAM;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.gfxMode = v;
    return RK_SUCCESS;
}
#define ROCKIT_VO_VALIDATE_LAYER(layer_)                                                          \
    if (!valid_vo_layer(layer_))                                                                   \
        return RK_ERR_VO_INVALID_LAYERID
#define ROCKIT_VO_LOCK_CHANNEL_BODY(layer_, channel_, missing_)                                   \
    auto &s = vo_state();                                                                          \
    std::lock_guard<std::mutex> lock(s.mutex);                                                     \
    VoChannelState *x = nullptr;                                                                   \
    RK_S32 r = require_vo_channel(s, (layer_), (channel_), nullptr, &x);                           \
    if (r == RK_ERR_VO_CHN_NOT_CONFIG)                                                             \
        return missing_;                                                                           \
    if (r != RK_SUCCESS)                                                                           \
        return r
#define ROCKIT_VO_LOCK_CHANNEL_OR(layer_, channel_, missing_)                                     \
    ROCKIT_VO_VALIDATE_LAYER(layer_);                                                              \
    ROCKIT_VO_LOCK_CHANNEL_BODY(layer_, channel_, missing_)
#define ROCKIT_VO_LOCK_CHANNEL_PTR_OR(layer_, channel_, pointer_, missing_)                       \
    ROCKIT_VO_VALIDATE_LAYER(layer_);                                                              \
    if (!(pointer_))                                                                               \
        return RK_ERR_VO_NULL_PTR;                                                                 \
    ROCKIT_VO_LOCK_CHANNEL_BODY(layer_, channel_, missing_)
#define ROCKIT_VO_LOCK_LAYER_BODY(layer_)                                                         \
    auto &s = vo_state();                                                                          \
    std::lock_guard<std::mutex> l(s.mutex);                                                        \
    VoLayerState *x = nullptr;                                                                     \
    RK_S32 r = require_vo_layer(s, (layer_), &x);                                                  \
    if (r != RK_SUCCESS)                                                                           \
        return r
#define ROCKIT_VO_LOCK_LAYER(layer_)                                                              \
    ROCKIT_VO_VALIDATE_LAYER(layer_);                                                              \
    ROCKIT_VO_LOCK_LAYER_BODY(layer_)
#define ROCKIT_VO_LOCK_LAYER_PTR(layer_, pointer_)                                                \
    ROCKIT_VO_VALIDATE_LAYER(layer_);                                                              \
    if (!(pointer_))                                                                               \
        return RK_ERR_VO_NULL_PTR;                                                                 \
    ROCKIT_VO_LOCK_LAYER_BODY(layer_)

extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetGfxFrameBuffer(VO_LAYER l, VO_CHN c, VO_FRAME_INFO_S *v) {
    ROCKIT_VO_LOCK_CHANNEL_PTR_OR(l, c, v, RK_ERR_VO_NOT_PERMIT);
    if (!x->hasGfxFrame)
        return RK_ERR_VO_CHN_NOT_ALLOC;
    *v = x->gfxFrame;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_UpdateGfxFrameBuffer(VO_LAYER l,
                                                            VO_CHN c,
                                                            VO_FRAME_INFO_S *v) {
    ROCKIT_VO_LOCK_CHANNEL_PTR_OR(l, c, v, RK_ERR_VO_NOT_PERMIT);
    x->gfxFrame = *v;
    x->hasGfxFrame = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_ReleaseGfxFrameBuffer(VO_LAYER l, VO_CHN c) {
    ROCKIT_VO_LOCK_CHANNEL_OR(l, c, RK_ERR_VO_NOT_PERMIT);
    zero_struct(&x->gfxFrame);
    x->hasGfxFrame = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_ClearLayersBinding(RK_VOID) {
    auto &s = vo_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (auto &x : s.layers) {
        x.bound = false;
        x.device = -1;
        x.mode = VO_LAYER_MODE_VIDEO;
    }
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_BindLayer(VO_LAYER l, VO_DEV d, VO_LAYER_MODE_E m) {
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    if (!valid_vo_layer(l))
        return RK_ERR_VO_INVALID_LAYERID;
    if (static_cast<RK_S32>(m) < 0 ||
        static_cast<RK_U32>(m) > static_cast<RK_U32>(VO_LAYER_MODE_BUTT))
        return RK_ERR_VO_ILLEGAL_PARAM;
    auto &s = vo_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto &x = s.layers[l];
    if (x.bound)
        return RK_ERR_VO_LAYER_HAS_BINDED;
    x.bound = true;
    x.device = d;
    x.mode = m;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_UnBindLayer(VO_LAYER l, VO_DEV d) {
    if (!valid_vo_dev(d))
        return RK_ERR_VO_INVALID_DEVID;
    if (!valid_vo_layer(l))
        return RK_ERR_VO_INVALID_LAYERID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto &x = s.layers[l];
    if (!x.bound || x.device != d)
        return RK_ERR_VO_LAYER_NOT_BINDED;
    if (x.enabled)
        return RK_ERR_VO_LAYER_NOT_DISABLE;
    x.bound = false;
    x.device = -1;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetLayerAttr(VO_LAYER id, const VO_VIDEO_LAYER_ATTR_S *v) {
    ROCKIT_VO_LOCK_LAYER_PTR(id, v);
    if (!x->bound && v->stDispRect.u32Width == 0 && v->stDispRect.u32Height == 0 &&
        v->stImageSize.u32Width == 0 && v->stImageSize.u32Height == 0)
        return RK_ERR_VO_LAYER_NOT_BINDED;
    x->attr = *v;
    x->configured = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetLayerAttr(VO_LAYER id, VO_VIDEO_LAYER_ATTR_S *v) {
    ROCKIT_VO_LOCK_LAYER_PTR(id, v);
    if (!x->configured)
        return RK_SUCCESS;
    *v = x->attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetLayerCSC(VO_LAYER id, const VO_CSC_S *v) {
    ROCKIT_VO_LOCK_LAYER_PTR(id, v);
    x->csc = *v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetLayerCSC(VO_LAYER id, VO_CSC_S *v) {
    ROCKIT_VO_LOCK_LAYER_PTR(id, v);
    if (!x->configured)
        return RK_SUCCESS;
    *v = x->csc;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_EnableLayer(VO_LAYER id) {
    ROCKIT_VO_LOCK_LAYER(id);
    if (!x->bound && !x->configured && id == 0)
        return RK_ERR_VO_BUSY;
    if (!x->configured)
        return RK_ERR_VO_LAYER_NOT_CONFIG;
    if (x->bound && !s.devices[x->device].enabled)
        return RK_ERR_VO_DEV_NOT_ENABLE;
    x->enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_DisableLayer(VO_LAYER id) {
    ROCKIT_VO_LOCK_LAYER(id);
    for (auto &[_, c] : x->channels)
        if (c.enabled || !c.outstanding.empty())
            return RK_ERR_VO_BUSY;
    if (!x->outstanding.empty())
        return RK_ERR_VO_BUSY;
    x->enabled = false;
    release_queue(x->ready);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetLayerPriority(VO_LAYER id, RK_U32 v) {
    if (!valid_vo_layer(id))
        return RK_ERR_VO_INVALID_LAYERID;
    if (v > VO_MAX_PRIORITY)
        return RK_ERR_VO_ILLEGAL_PARAM;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    VoLayerState *x = nullptr;
    RK_S32 r = require_vo_layer(s, id, &x);
    if (r != RK_SUCCESS)
        return r;
    if (!x->bound)
        return RK_ERR_VO_DEV_NOT_ENABLE;
    x->priority = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetLayerPriority(VO_LAYER id, RK_U32 *v) {
    ROCKIT_VO_LOCK_LAYER_PTR(id, v);
    if (!x->bound)
        return RK_ERR_VO_DEV_NOT_ENABLE;
    *v = x->priority;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetLayerDispBufLen(VO_LAYER id, RK_U32 v) {
    if (!valid_vo_layer(id))
        return RK_ERR_VO_INVALID_LAYERID;
    if (v == 0 || v > VO_MAX_BUF_NUM)
        return RK_ERR_VO_ILLEGAL_PARAM;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    VoLayerState *x = nullptr;
    RK_S32 r = require_vo_layer(s, id, &x);
    if (r != RK_SUCCESS)
        return r;
    x->dispBufLen = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetLayerDispBufLen(VO_LAYER id, RK_U32 *v) {
    ROCKIT_VO_LOCK_LAYER_PTR(id, v);
    if (!x->configured)
        return RK_SUCCESS;
    *v = x->dispBufLen;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetLayerFrame(VO_LAYER id,
                                                     VIDEO_FRAME_INFO_S *f,
                                                     RK_S32 timeout) {
    if (!valid_vo_layer(id))
        return RK_ERR_VO_INVALID_LAYERID;
    if (!f)
        return RK_ERR_VO_NULL_PTR;
    auto &s = vo_state();
    std::unique_lock<std::mutex> l(s.mutex);
    VoLayerState *x = nullptr;
    RK_S32 r = require_vo_layer(s, id, &x);
    if (r != RK_SUCCESS)
        return r;
    if (!x->enabled)
        return RK_ERR_VO_LAYER_NOT_ENABLE;
    auto ready = [&] {
        return !x->ready.empty();
    };
    if (!wait_until(s.changed, l, timeout, ready))
        return RK_ERR_VO_WAIT_TIMEOUT;
    *f = x->ready.front().frame;
    x->ready.pop_front();
    ++x->outstanding[frame_block(*f)];
    s.changed.notify_all();
    return RK_SUCCESS;
}
#define ROCKIT_RELEASE_VO_OUTSTANDING_FRAME(block_)                                              \
    auto it = x->outstanding.find(block_);                                                        \
    if (it == x->outstanding.end())                                                               \
        return RK_ERR_VO_INVALID_VFRAME;                                                          \
    if (--it->second == 0)                                                                        \
        x->outstanding.erase(it);                                                                 \
    release_mb(block_);                                                                           \
    s.changed.notify_all();                                                                       \
    return RK_SUCCESS

extern "C" ROCKIT_API RK_S32 RK_MPI_VO_ReleaseLayerFrame(VO_LAYER id, VIDEO_FRAME_INFO_S *f) {
    ROCKIT_VO_LOCK_LAYER_PTR(id, f);
    MB_BLK b = frame_block(*f);
    ROCKIT_RELEASE_VO_OUTSTANDING_FRAME(b);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SendLayerFrame(VO_LAYER id, VIDEO_FRAME_INFO_S *f) {
    ROCKIT_VO_LOCK_LAYER_PTR(id, f);
    if (!x->enabled)
        return RK_ERR_VO_LAYER_NOT_ENABLE;
    if (x->ready.size() >= x->dispBufLen)
        return RK_ERR_VO_BUSY;
    if (!queue_frame(x->ready, *f))
        return RK_ERR_VO_INVALID_VFRAME;
    fanout_wbc(s, *f);
    s.changed.notify_all();
    return RK_SUCCESS;
}
#define ROCKIT_VO_LOCK_CURSOR_LAYER(layer_)                                                        \
    if (!valid_vo_layer(layer_))                                                                   \
        return RK_ERR_VO_INVALID_LAYERID;                                                          \
    auto &s = vo_state();                                                                          \
    std::lock_guard<std::mutex> l(s.mutex);                                                        \
    VoLayerState *v = nullptr;                                                                     \
    RK_S32 r = require_vo_layer(s, (layer_), &v);                                                  \
    if (r != RK_SUCCESS)                                                                           \
        return r

extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetCursorPostion(VO_LAYER id,
                                                        const RK_U32 x,
                                                        const RK_U32 y) {
    ROCKIT_VO_LOCK_CURSOR_LAYER(id);
    if (!v->configured)
        return RK_ERR_VO_LAYER_NOT_ENABLE;
    v->cursorX = x;
    v->cursorY = y;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetCursorLogicalRect(VO_LAYER id,
                                                            const RK_U32 w,
                                                            const RK_U32 h) {
    ROCKIT_VO_LOCK_CURSOR_LAYER(id);
    v->cursorMaxWidth = w;
    v->cursorMaxHeight = h;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_PauseComposer(VO_LAYER id) {
    ROCKIT_VO_LOCK_CURSOR_LAYER(id);
    v->composerPaused = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_ResumeComposer(VO_LAYER id) {
    ROCKIT_VO_LOCK_CURSOR_LAYER(id);
    v->composerPaused = false;
    return RK_SUCCESS;
}
#undef ROCKIT_VO_LOCK_CURSOR_LAYER
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetChnAttr(VO_LAYER l, VO_CHN c, const VO_CHN_ATTR_S *v) {
    if (!valid_vo_layer(l))
        return RK_ERR_VO_INVALID_LAYERID;
    if (!v)
        return RK_ERR_VO_NULL_PTR;
    if (!valid_vo_chn(c))
        return RK_ERR_VO_INVALID_CHNID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    VoLayerState *x = nullptr;
    RK_S32 r = require_vo_layer(s, l, &x);
    if (r != RK_SUCCESS)
        return r;
    auto &ch = x->channels[c];
    if (ch.enabled)
        return RK_ERR_VO_CHN_NOT_DISABLE;
    ch.attr = *v;
    ch.configured = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetChnAttr(VO_LAYER l, VO_CHN c, VO_CHN_ATTR_S *v) {
    ROCKIT_VO_LOCK_CHANNEL_PTR_OR(l, c, v, RK_ERR_VO_CHN_NOT_CONFIG);
    *v = x->attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_EnableChn(VO_LAYER l, VO_CHN c) {
    if (!valid_vo_layer(l))
        return RK_ERR_VO_INVALID_LAYERID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    VoChannelState *x = nullptr;
    VoLayerState *y = nullptr;
    RK_S32 r = require_vo_channel(s, l, c, &y, &x);
    if (r == RK_ERR_VO_CHN_NOT_CONFIG)
        return RK_ERR_VO_LAYER_NOT_ENABLE;
    if (r != RK_SUCCESS)
        return r;
    if (!y->enabled)
        return RK_ERR_VO_LAYER_NOT_ENABLE;
    x->enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_DisableChn(VO_LAYER l, VO_CHN c) {
    ROCKIT_VO_LOCK_CHANNEL_OR(l, c, RK_ERR_VO_LAYER_NOT_ENABLE);
    if (!x->outstanding.empty())
        return RK_ERR_VO_BUSY;
    x->enabled = false;
    release_queue(x->ready);
    return RK_SUCCESS;
}
#define VO_CHN_GETSET_PTR(name, type, field)                                                       \
    extern "C" ROCKIT_API RK_S32 RK_MPI_VO_Set##name(VO_LAYER l, VO_CHN c, const type *v) {        \
        ROCKIT_VO_LOCK_CHANNEL_PTR_OR(l, c, v, RK_ERR_VO_CHN_NOT_CONFIG);                          \
        x->field = *v;                                                                             \
        return RK_SUCCESS;                                                                         \
    }                                                                                              \
    extern "C" ROCKIT_API RK_S32 RK_MPI_VO_Get##name(VO_LAYER l, VO_CHN c, type *v) {              \
        ROCKIT_VO_LOCK_CHANNEL_PTR_OR(l, c, v, RK_ERR_VO_CHN_NOT_CONFIG);                          \
        *v = x->field;                                                                             \
        return RK_SUCCESS;                                                                         \
    }
VO_CHN_GETSET_PTR(ChnParam, VO_CHN_PARAM_S, param)
VO_CHN_GETSET_PTR(ChnDispPos, POINT_S, position)
#undef VO_CHN_GETSET_PTR
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetChnBorder(VO_LAYER l, VO_CHN c, const VO_BORDER_S *v) {
    ROCKIT_VO_LOCK_CHANNEL_PTR_OR(l, c, v, RK_SUCCESS);
    x->border = *v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetChnBorder(VO_LAYER l, VO_CHN c, VO_BORDER_S *v) {
    ROCKIT_VO_LOCK_CHANNEL_PTR_OR(l, c, v, RK_SUCCESS);
    *v = x->border;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetChnFrame(VO_LAYER l,
                                                   VO_CHN c,
                                                   VIDEO_FRAME_INFO_S *f,
                                                   RK_S32 timeout) {
    if (!valid_vo_layer(l))
        return RK_ERR_VO_INVALID_LAYERID;
    if (!f)
        return RK_ERR_VO_NULL_PTR;
    auto &s = vo_state();
    std::unique_lock<std::mutex> lock(s.mutex);
    VoChannelState *x = nullptr;
    RK_S32 r = require_vo_channel(s, l, c, nullptr, &x);
    if (r == RK_ERR_VO_CHN_NOT_CONFIG)
        return RK_ERR_VO_LAYER_NOT_ENABLE;
    if (r != RK_SUCCESS)
        return r;
    auto ready = [&] {
        return !x->ready.empty();
    };
    if (!wait_until(s.changed, lock, timeout, ready))
        return RK_ERR_VO_WAIT_TIMEOUT;
    *f = x->ready.front().frame;
    x->ready.pop_front();
    ++x->outstanding[frame_block(*f)];
    s.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_ReleaseChnFrame(VO_LAYER l,
                                                       VO_CHN c,
                                                       const VIDEO_FRAME_INFO_S *f) {
    if (!f)
        return RK_ERR_VO_NULL_PTR;
    if (!valid_vo_layer(l))
        return RK_ERR_VO_INVALID_LAYERID;
    MB_BLK b = frame_block(*f);
    if (!b)
        return RK_SUCCESS;
    auto &s = vo_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    VoChannelState *x = nullptr;
    RK_S32 r = require_vo_channel(s, l, c, nullptr, &x);
    if (r != RK_SUCCESS)
        return r;
    ROCKIT_RELEASE_VO_OUTSTANDING_FRAME(b);
}
#undef ROCKIT_RELEASE_VO_OUTSTANDING_FRAME
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetChnFrameRate(VO_LAYER l, VO_CHN c, RK_S32 v) {
    ROCKIT_VO_LOCK_CHANNEL_OR(l, c, RK_SUCCESS);
    x->frameRate = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetChnFrameRate(VO_LAYER l, VO_CHN c, RK_S32 *v) {
    ROCKIT_VO_LOCK_CHANNEL_PTR_OR(l, c, v, RK_SUCCESS);
    *v = x->frameRate;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_PauseChn(VO_LAYER l, VO_CHN c) {
    ROCKIT_VO_LOCK_CHANNEL_OR(l, c, RK_SUCCESS);
    x->paused = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_ResumeChn(VO_LAYER l, VO_CHN c) {
    ROCKIT_VO_LOCK_CHANNEL_OR(l, c, RK_SUCCESS);
    x->paused = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_StepChn(VO_LAYER l, VO_CHN c) {
    if (!valid_vo_layer(l))
        return RK_ERR_VO_INVALID_LAYERID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    RK_S32 r = require_vo_channel(s, l, c, nullptr, nullptr);
    return r == RK_ERR_VO_CHN_NOT_CONFIG ? RK_ERR_VO_LAYER_NOT_ENABLE : r;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_RefreshChn(VO_LAYER l, VO_CHN c) {
    if (!valid_vo_layer(l))
        return RK_ERR_VO_INVALID_LAYERID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    RK_S32 r = require_vo_channel(s, l, c, nullptr, nullptr);
    return r == RK_ERR_VO_CHN_NOT_CONFIG ? RK_SUCCESS : r;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_ShowChn(VO_LAYER l, VO_CHN c) {
    ROCKIT_VO_LOCK_CHANNEL_OR(l, c, RK_SUCCESS);
    x->shown = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_HideChn(VO_LAYER l, VO_CHN c) {
    ROCKIT_VO_LOCK_CHANNEL_OR(l, c, RK_SUCCESS);
    x->shown = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SendFrame(VO_LAYER layerId,
                                                 VO_CHN channelId,
                                                 VIDEO_FRAME_INFO_S *frame,
                                                 RK_S32 timeout) {
    if (!frame) {
        return RK_ERR_VO_NULL_PTR;
    }
    if (!valid_vo_layer(layerId))
        return RK_ERR_VO_INVALID_LAYERID;
    if (!frame_block(*frame))
        return RK_ERR_VO_INVALID_VFRAME;
    auto &state = vo_state();
    std::unique_lock<std::mutex> lock(state.mutex);
    VoChannelState *channel = nullptr;
    VoLayerState *layer = nullptr;
    RK_S32 result = require_vo_channel(state, layerId, channelId, &layer, &channel);
    if (result != RK_SUCCESS) {
        return result;
    }
    if (!channel->enabled) {
        return RK_ERR_VO_CHN_NOT_ENABLE;
    }
    if (channel->paused) {
        return RK_ERR_VO_NOT_PERMIT;
    }
    auto room = [&] {
        return channel->ready.size() < vo_channel_capacity(*channel);
    };
    if (!wait_until(state.changed, lock, timeout, room)) {
        return RK_ERR_VO_BUSY;
    }

    VIDEO_FRAME_INFO_S presented = *frame;
    bool ownsInitialReference = false;
    const bool layerFormatCompatible =
        !layer->configured || layer->attr.enPixFormat == frame->stVFrame.enPixelFormat;
    if (layerFormatCompatible && layer->attr.enCompressMode == COMPRESS_MODE_NONE &&
        rockit_local::cpu_frame_is_supported(*frame)) {
        const auto transform = vo_cpu_transform(*channel, *frame);
        if (rockit_local::cpu_transform_frame(*frame, transform, nullptr, &presented) !=
            RK_SUCCESS) {
            return RK_ERR_VO_NO_MEM;
        }
        ownsInitialReference = true;
        const MPP_CHN_S mppChannel{RK_ID_VO, layerId, channelId};
        RK_U32 applied = 0;
        (void)rockit_local::apply_regions_to_frame(mppChannel, &presented, &applied);
    }

    if (!queue_frame(channel->ready, presented)) {
        if (ownsInitialReference) {
            release_mb(frame_block(presented));
        }
        return RK_ERR_VO_INVALID_VFRAME;
    }
    channel->pts = presented.stVFrame.u64PTS;
    if (layer->enabled && layer->ready.size() < layer->dispBufLen) {
        (void)queue_frame(layer->ready, presented);
    }
    fanout_wbc(state, presented);
    if (ownsInitialReference) {
        release_mb(frame_block(presented));
    }
    state.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_ClearChnBuffer(VO_LAYER l, VO_CHN c, RK_BOOL) {
    ROCKIT_VO_LOCK_CHANNEL_OR(l, c, RK_ERR_VO_LAYER_NOT_ENABLE);
    release_queue(x->ready);
    s.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetChnRecvThreshold(VO_LAYER l, VO_CHN c, RK_U32 v) {
    if (!valid_vo_layer(l))
        return RK_ERR_VO_INVALID_LAYERID;
    if (v == 0)
        return RK_ERR_VO_ILLEGAL_PARAM;
    auto &s = vo_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    VoChannelState *x = nullptr;
    RK_S32 r = require_vo_channel(s, l, c, nullptr, &x);
    if (r != RK_SUCCESS)
        return r;
    x->recvThreshold = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetChnRecvThreshold(VO_LAYER l, VO_CHN c, RK_U32 *v) {
    ROCKIT_VO_LOCK_CHANNEL_PTR_OR(l, c, v, RK_ERR_VO_CHN_NOT_CONFIG);
    *v = x->recvThreshold;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetChnPts(VO_LAYER l, VO_CHN c, RK_U64 *v) {
    ROCKIT_VO_LOCK_CHANNEL_PTR_OR(l, c, v, RK_ERR_VO_LAYER_NOT_ENABLE);
    *v = x->pts;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_QueryChnStat(VO_LAYER l, VO_CHN c, VO_QUERY_STATUS_S *v) {
    ROCKIT_VO_LOCK_CHANNEL_PTR_OR(l, c, v, RK_ERR_VO_LAYER_NOT_ENABLE);
    zero_struct(v);
    v->u32ChnBufUsed = static_cast<RK_U32>(x->ready.size() + x->outstanding.size());
    return RK_SUCCESS;
}
#define ROCKIT_VO_LOCK_WBC_PTR(wbc_, pointer_)                                                     \
    if (!valid_vo_wbc(wbc_))                                                                       \
        return RK_ERR_VO_INVALID_WBCID;                                                            \
    if (!(pointer_))                                                                               \
        return RK_ERR_VO_NULL_PTR;                                                                 \
    if (!valid_vo_wbc(wbc_))                                                                       \
        return RK_ERR_VO_INVALID_WBCID;                                                            \
    auto &s = vo_state();                                                                          \
    std::lock_guard<std::mutex> l(s.mutex)

extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetWbcSource(VO_WBC w, const VO_WBC_SOURCE_S *v) {
    ROCKIT_VO_LOCK_WBC_PTR(w, v);
    s.wbc[w].source = *v;
    s.wbc[w].configured = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetWbcSource(VO_WBC w, VO_WBC_SOURCE_S *v) {
    ROCKIT_VO_LOCK_WBC_PTR(w, v);
    *v = s.wbc[w].source;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_EnableWbc(VO_WBC w) {
    if (!valid_vo_wbc(w))
        return RK_ERR_VO_INVALID_WBCID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &x = s.wbc[w];
    if (!x.configured)
        return RK_ERR_VO_WBC_NOT_CONFIG;
    x.enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_DisableWbc(VO_WBC w) {
    if (!valid_vo_wbc(w))
        return RK_ERR_VO_INVALID_WBCID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &x = s.wbc[w];
    if (!x.outstanding.empty())
        return RK_ERR_VO_BUSY;
    x.enabled = false;
    release_queue(x.ready);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetWbcAttr(VO_WBC w, const VO_WBC_ATTR_S *v) {
    ROCKIT_VO_LOCK_WBC_PTR(w, v);
    s.wbc[w].attr = *v;
    s.wbc[w].configured = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetWbcAttr(VO_WBC w, VO_WBC_ATTR_S *v) {
    ROCKIT_VO_LOCK_WBC_PTR(w, v);
    *v = s.wbc[w].attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetWbcMode(VO_WBC w, VO_WBC_MODE_E v) {
    if (!valid_vo_wbc(w))
        return RK_ERR_VO_INVALID_WBCID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.wbc[w].mode = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetWbcMode(VO_WBC w, VO_WBC_MODE_E *v) {
    ROCKIT_VO_LOCK_WBC_PTR(w, v);
    *v = s.wbc[w].mode;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetWbcDepth(VO_WBC w, RK_S32 v) {
    if (!valid_vo_wbc(w))
        return RK_ERR_VO_INVALID_WBCID;
    if (v < 0)
        return RK_ERR_VO_ILLEGAL_PARAM;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.wbc[w].depth = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetWbcDepth(VO_WBC w, RK_S32 *v) {
    if (!valid_vo_wbc(w))
        return RK_ERR_VO_INVALID_WBCID;
    if (!v)
        return RK_ERR_VO_NULL_PTR;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (!s.wbc[w].configured)
        return RK_SUCCESS;
    *v = s.wbc[w].depth;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetWbcFrame(VO_WBC w,
                                                   VIDEO_FRAME_INFO_S *f,
                                                   RK_S32 timeout) {
    if (!valid_vo_wbc(w))
        return RK_ERR_VO_INVALID_WBCID;
    if (!f)
        return RK_ERR_VO_NULL_PTR;
    auto &s = vo_state();
    std::unique_lock<std::mutex> l(s.mutex);
    auto &x = s.wbc[w];
    if (!x.configured)
        return RK_ERR_VO_WBC_NOT_CONFIG;
    if (!x.enabled)
        return RK_ERR_VO_WBC_NOT_DISABLE;
    auto ready = [&] {
        return !x.ready.empty();
    };
    if (!wait_until(s.changed, l, timeout, ready))
        return RK_ERR_VO_WAIT_TIMEOUT;
    *f = x.ready.front().frame;
    x.ready.pop_front();
    ++x.outstanding[frame_block(*f)];
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_ReleaseWbcFrame(VO_WBC w, VIDEO_FRAME_INFO_S *f) {
    if (!valid_vo_wbc(w))
        return RK_ERR_VO_INVALID_WBCID;
    if (!f)
        return RK_ERR_VO_NULL_PTR;
    if (!valid_vo_wbc(w))
        return RK_ERR_VO_INVALID_WBCID;
    auto &s = vo_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &x = s.wbc[w];
    MB_BLK b = frame_block(*f);
    auto it = x.outstanding.find(b);
    if (it == x.outstanding.end())
        return RK_ERR_VO_INVALID_VFRAME;
    if (--it->second == 0)
        x.outstanding.erase(it);
    release_mb(b);
    s.changed.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetLayerSpliceMode(VO_LAYER id, VO_SPLICE_MODE_E v) {
    ROCKIT_VO_LOCK_LAYER(id);
    x->spliceMode = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetLayerSpliceMode(VO_LAYER id, VO_SPLICE_MODE_E *v) {
    ROCKIT_VO_LOCK_LAYER_PTR(id, v);
    *v = x->spliceMode;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_SetLowDelayCh(VO_LAYER id, RK_U32 v) {
    ROCKIT_VO_LOCK_LAYER(id);
    x->lowDelayCh = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VO_GetLowDelayCh(VO_LAYER id, RK_U32 *v) {
    ROCKIT_VO_LOCK_LAYER_PTR(id, v);
    *v = x->lowDelayCh;
    return RK_SUCCESS;
}

#undef ROCKIT_VO_LOCK_WBC_PTR
#undef ROCKIT_VO_LOCK_LAYER_PTR
#undef ROCKIT_VO_LOCK_LAYER
#undef ROCKIT_VO_LOCK_LAYER_BODY
#undef ROCKIT_VO_LOCK_CHANNEL_PTR_OR
#undef ROCKIT_VO_LOCK_CHANNEL_OR
#undef ROCKIT_VO_LOCK_CHANNEL_BODY
#undef ROCKIT_VO_VALIDATE_LAYER
