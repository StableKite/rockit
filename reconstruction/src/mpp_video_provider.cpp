#include "mpp_video_provider.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include "rk_mpi_mb.h"
#include "rk_mpi_mmz.h"
#include "rk_mpi_sys.h"
#include "mpp_abi_shim.hpp"

namespace rockit_mpp {
namespace {

namespace abi = rockit_mpp_abi;
using MPP_RET = abi::Ret;
using MppCtx = abi::Context;
using MppFrame = abi::Frame;
using MppPacket = abi::Packet;
using MppBuffer = abi::Buffer;
using MppBufferGroup = abi::BufferGroup;
using MppEncCfg = abi::EncCfg;
using MppApi = abi::Api;
using MppCodingType = abi::CodingType;
using MppFrameFormat = abi::FrameFormat;
using MppBufferInfo = abi::BufferInfo;
using MppBufferType = std::int32_t;
using MppEncHeaderMode = std::int32_t;

constexpr MPP_RET MPP_OK = abi::kOk;
constexpr MPP_RET MPP_ERR_TIMEOUT = abi::kErrorTimeout;
constexpr MPP_RET MPP_ERR_BUFFER_FULL = abi::kErrorBufferFull;
constexpr MPP_RET MPP_ERR_DISPLAY_FULL = abi::kErrorDisplayFull;
constexpr MPP_RET MPP_ERR_NULL_PTR = abi::kErrorNullPointer;
constexpr MPP_RET MPP_ERR_VALUE = abi::kErrorValue;
constexpr MPP_RET MPP_ERR_MALLOC = abi::kErrorMalloc;
constexpr MPP_RET MPP_ERR_NOMEM = abi::kErrorNoMemory;
constexpr MPP_RET MPP_ERR_STREAM = abi::kErrorStream;
constexpr MPP_RET MPP_ERR_PROTOL = abi::kErrorProtocol;
constexpr MPP_RET MPP_FAIL_SPLIT_FRAME = abi::kErrorSplitFrame;
constexpr std::int32_t MPP_CTX_DEC = abi::kContextDecoder;
constexpr std::int32_t MPP_CTX_ENC = abi::kContextEncoder;
constexpr MppCodingType MPP_VIDEO_CodingUnused = abi::kCodingUnused;
constexpr MppCodingType MPP_VIDEO_CodingMPEG2 = abi::kCodingMpeg2;
constexpr MppCodingType MPP_VIDEO_CodingMPEG4 = abi::kCodingMpeg4;
constexpr MppCodingType MPP_VIDEO_CodingAVC = abi::kCodingAvc;
constexpr MppCodingType MPP_VIDEO_CodingMJPEG = abi::kCodingMjpeg;
constexpr MppCodingType MPP_VIDEO_CodingVP8 = abi::kCodingVp8;
constexpr MppCodingType MPP_VIDEO_CodingVP9 = abi::kCodingVp9;
constexpr MppCodingType MPP_VIDEO_CodingHEVC = abi::kCodingHevc;
constexpr MppCodingType MPP_VIDEO_CodingAV1 = abi::kCodingAv1;
constexpr MppFrameFormat MPP_FMT_YUV420SP = abi::kFormatYuv420Sp;
constexpr MppFrameFormat MPP_FMT_YUV422SP = abi::kFormatYuv422Sp;
constexpr MppFrameFormat MPP_FMT_YUV420P = abi::kFormatYuv420P;
constexpr MppFrameFormat MPP_FMT_YUV420SP_VU = abi::kFormatYuv420SpVu;
constexpr MppFrameFormat MPP_FMT_YUV422SP_VU = abi::kFormatYuv422SpVu;
constexpr MppFrameFormat MPP_FMT_YUV422_YUYV = abi::kFormatYuv422Yuyv;
constexpr MppFrameFormat MPP_FMT_YUV422_UYVY = abi::kFormatYuv422Uyvy;
constexpr MppFrameFormat MPP_FMT_RGB888 = abi::kFormatRgb888;
constexpr MppFrameFormat MPP_FMT_BGR888 = abi::kFormatBgr888;
constexpr MppFrameFormat MPP_FMT_ARGB8888 = abi::kFormatArgb8888;
constexpr MppFrameFormat MPP_FMT_ABGR8888 = abi::kFormatAbgr8888;
constexpr MppFrameFormat MPP_FMT_BGRA8888 = abi::kFormatBgra8888;
constexpr MppFrameFormat MPP_FMT_RGBA8888 = abi::kFormatRgba8888;
constexpr MppFrameFormat MPP_FMT_BUTT = abi::kFormatButt;
constexpr MppFrameFormat MPP_FRAME_FMT_MASK = abi::kFrameFormatMask;
constexpr std::int64_t MPP_POLL_BLOCK = abi::kPollBlock;
constexpr std::int64_t MPP_POLL_NON_BLOCK = abi::kPollNonBlock;
constexpr abi::Command MPP_SET_INPUT_TIMEOUT = abi::kSetInputTimeout;
constexpr abi::Command MPP_SET_OUTPUT_TIMEOUT = abi::kSetOutputTimeout;
constexpr abi::Command MPP_DEC_SET_EXT_BUF_GROUP = abi::kDecSetExternalBufferGroup;
constexpr abi::Command MPP_DEC_SET_INFO_CHANGE_READY = abi::kDecSetInfoChangeReady;
constexpr abi::Command MPP_DEC_SET_PARSER_SPLIT_MODE = abi::kDecSetParserSplitMode;
constexpr abi::Command MPP_ENC_SET_CFG = abi::kEncSetConfig;
constexpr abi::Command MPP_ENC_GET_CFG = abi::kEncGetConfig;
constexpr abi::Command MPP_ENC_SET_IDR_FRAME = abi::kEncSetIdrFrame;
constexpr abi::Command MPP_ENC_SET_HEADER_MODE = abi::kEncSetHeaderMode;
constexpr std::int32_t MPP_BUFFER_INTERNAL = abi::kBufferInternal;
constexpr std::int32_t MPP_BUFFER_TYPE_ION = abi::kBufferTypeIon;
constexpr std::int32_t MPP_BUFFER_TYPE_DRM = abi::kBufferTypeDrm;
constexpr std::int32_t MPP_BUFFER_TYPE_DMA_HEAP = abi::kBufferTypeDmaHeap;
constexpr std::int32_t MPP_BUFFER_FLAGS_CACHABLE = abi::kBufferFlagCacheable;
constexpr std::int32_t MPP_ENC_RC_MODE_VBR = abi::kRcModeVbr;
constexpr std::int32_t MPP_ENC_RC_MODE_CBR = abi::kRcModeCbr;
constexpr std::int32_t MPP_ENC_RC_MODE_FIXQP = abi::kRcModeFixQp;
constexpr std::int32_t MPP_ENC_RC_MODE_AVBR = abi::kRcModeAvbr;
constexpr std::uint32_t MPP_ENC_RC_DROP_FRM_DISABLED = abi::kRcDropDisabled;
constexpr MppEncHeaderMode MPP_ENC_HEADER_MODE_EACH_IDR = abi::kHeaderEachIdr;
constexpr std::int32_t MPP_FRAME_RANGE_JPEG = abi::kFrameRangeJpeg;
constexpr abi::MetaKey KEY_OUTPUT_FRAME = abi::kMetaKeyOutputFrame;
constexpr bool MPP_FRAME_FMT_IS_FBC(MppFrameFormat format) noexcept {
    return (format & abi::kFrameFbcMask) != 0;
}

constexpr const char *kProviderEnvironment = "ROCKIT_VIDEO_PROVIDER";
constexpr const char *kMppLibraryPrimary = "librockchip_mpp.so.1";
constexpr const char *kMppLibraryFallback = "librockchip_mpp.so";
constexpr const char *kMppTag = "rockit-r0.34";
constexpr const char *kMppCaller = "rockit_public_video_provider";
constexpr RK_U32 kDefaultFrameCount = 12;

bool equals(const char *value, const char *expected) noexcept {
    return value && std::strcmp(value, expected) == 0;
}

Status status_from_mpp(MPP_RET result) noexcept {
    if (result == MPP_OK)
        return Status::Ok;
    if (result == MPP_ERR_TIMEOUT)
        return Status::Timeout;
    if (result == MPP_ERR_BUFFER_FULL || result == MPP_ERR_DISPLAY_FULL)
        return Status::Again;
    if (result == MPP_ERR_NULL_PTR || result == MPP_ERR_VALUE)
        return Status::Invalid;
    if (result == MPP_ERR_MALLOC || result == MPP_ERR_NOMEM)
        return Status::NoMemory;
    if (result == MPP_ERR_STREAM || result == MPP_ERR_PROTOL || result == MPP_FAIL_SPLIT_FRAME)
        return Status::Corrupt;
    return Status::Failed;
}

MppCodingType map_codec(RK_CODEC_ID_E codec) noexcept {
    switch (codec) {
    case RK_VIDEO_ID_AVC:
        return MPP_VIDEO_CodingAVC;
    case RK_VIDEO_ID_HEVC:
        return MPP_VIDEO_CodingHEVC;
    case RK_VIDEO_ID_JPEG:
    case RK_VIDEO_ID_MJPEG:
        return MPP_VIDEO_CodingMJPEG;
    case RK_VIDEO_ID_VP8:
        return MPP_VIDEO_CodingVP8;
    case RK_VIDEO_ID_VP9:
        return MPP_VIDEO_CodingVP9;
    case RK_VIDEO_ID_AV1:
        return MPP_VIDEO_CodingAV1;
    case RK_VIDEO_ID_MPEG2VIDEO:
        return MPP_VIDEO_CodingMPEG2;
    case RK_VIDEO_ID_MPEG4:
        return MPP_VIDEO_CodingMPEG4;
    default:
        return MPP_VIDEO_CodingUnused;
    }
}

MppFrameFormat map_pixel_format(PIXEL_FORMAT_E format) noexcept {
    switch (format) {
    case RK_FMT_YUV420SP:
        return MPP_FMT_YUV420SP;
    case RK_FMT_YUV420SP_VU:
        return MPP_FMT_YUV420SP_VU;
    case RK_FMT_YUV420P:
        return MPP_FMT_YUV420P;
    case RK_FMT_YUV422SP:
        return MPP_FMT_YUV422SP;
    case RK_FMT_YUV422SP_VU:
        return MPP_FMT_YUV422SP_VU;
    case RK_FMT_YUV422_YUYV:
        return MPP_FMT_YUV422_YUYV;
    case RK_FMT_YUV422_UYVY:
        return MPP_FMT_YUV422_UYVY;
    case RK_FMT_RGB888:
        return MPP_FMT_RGB888;
    case RK_FMT_BGR888:
        return MPP_FMT_BGR888;
    case RK_FMT_ARGB8888:
        return MPP_FMT_ARGB8888;
    case RK_FMT_ABGR8888:
        return MPP_FMT_ABGR8888;
    case RK_FMT_RGBA8888:
        return MPP_FMT_RGBA8888;
    case RK_FMT_BGRA8888:
        return MPP_FMT_BGRA8888;
    default:
        return MPP_FMT_BUTT;
    }
}

PIXEL_FORMAT_E map_mpp_format(MppFrameFormat format) noexcept {
    switch (format & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:
        return RK_FMT_YUV420SP;
    case MPP_FMT_YUV420SP_VU:
        return RK_FMT_YUV420SP_VU;
    case MPP_FMT_YUV420P:
        return RK_FMT_YUV420P;
    case MPP_FMT_YUV422SP:
        return RK_FMT_YUV422SP;
    case MPP_FMT_YUV422SP_VU:
        return RK_FMT_YUV422SP_VU;
    case MPP_FMT_YUV422_YUYV:
        return RK_FMT_YUV422_YUYV;
    case MPP_FMT_YUV422_UYVY:
        return RK_FMT_YUV422_UYVY;
    case MPP_FMT_RGB888:
        return RK_FMT_RGB888;
    case MPP_FMT_BGR888:
        return RK_FMT_BGR888;
    case MPP_FMT_ARGB8888:
        return RK_FMT_ARGB8888;
    case MPP_FMT_ABGR8888:
        return RK_FMT_ABGR8888;
    case MPP_FMT_RGBA8888:
        return RK_FMT_RGBA8888;
    case MPP_FMT_BGRA8888:
        return RK_FMT_BGRA8888;
    default:
        return RK_FMT_BUTT;
    }
}

RK_S64 mpp_timeout(RK_S32 timeout_ms) noexcept {
    if (timeout_ms < 0)
        return MPP_POLL_BLOCK;
    return static_cast<RK_S64>(timeout_ms);
}

#if defined(__linux__)
template <typename Function>
bool load_function(void *library, const char *name, Function *target) noexcept {
    if (!library || !name || !target || sizeof(Function) != sizeof(void *))
        return false;
    dlerror();
    void *address = dlsym(library, name);
    const char *error = dlerror();
    if (error || !address)
        return false;
    std::memcpy(target, &address, sizeof(address));
    return true;
}
#endif

struct Runtime {
#if defined(__linux__)
    void *library{nullptr};
#endif
    abi::CreateFunction create{nullptr};
    abi::InitFunction init{nullptr};
    abi::DestroyFunction destroy{nullptr};

    abi::BufferImportFunction buffer_import{nullptr};
    abi::BufferGetFunction buffer_get{nullptr};
    abi::BufferPutFunction buffer_put{nullptr};
    abi::BufferGetPointerFunction buffer_get_ptr{nullptr};
    abi::BufferGetFdFunction buffer_get_fd{nullptr};
    abi::BufferGetSizeFunction buffer_get_size{nullptr};
    abi::BufferGroupGetFunction buffer_group_get{nullptr};
    abi::BufferGroupPutFunction buffer_group_put{nullptr};
    abi::BufferGroupLimitFunction buffer_group_limit{nullptr};

    abi::FrameInitFunction frame_init{nullptr};
    abi::FrameDeinitFunction frame_deinit{nullptr};
    abi::FrameSetU32Function frame_set_width{nullptr};
    abi::FrameSetU32Function frame_set_height{nullptr};
    abi::FrameSetU32Function frame_set_hor_stride{nullptr};
    abi::FrameSetU32Function frame_set_ver_stride{nullptr};
    abi::FrameSetU32Function frame_set_fmt{nullptr};
    abi::FrameSetU32Function frame_set_eos{nullptr};
    abi::FrameSetS64Function frame_set_pts{nullptr};
    abi::FrameSetBufferFunction frame_set_buffer{nullptr};
    abi::FrameGetU32Function frame_get_width{nullptr};
    abi::FrameGetU32Function frame_get_height{nullptr};
    abi::FrameGetU32Function frame_get_hor_stride{nullptr};
    abi::FrameGetU32Function frame_get_ver_stride{nullptr};
    abi::FrameGetU32Function frame_get_fmt{nullptr};
    abi::FrameGetU32Function frame_get_eos{nullptr};
    abi::FrameGetS64Function frame_get_pts{nullptr};
    abi::FrameGetBufferFunction frame_get_buffer{nullptr};
    abi::FrameGetU32Function frame_get_info_change{nullptr};
    abi::FrameGetSizeFunction frame_get_buf_size{nullptr};
    abi::FrameGetU32Function frame_get_errinfo{nullptr};
    abi::FrameGetU32Function frame_get_discard{nullptr};

    abi::PacketInitFunction packet_init{nullptr};
    abi::PacketInitWithBufferFunction packet_init_with_buffer{nullptr};
    abi::PacketDeinitFunction packet_deinit{nullptr};
    abi::PacketSetPointerFunction packet_set_pos{nullptr};
    abi::PacketSetSizeFunction packet_set_length{nullptr};
    abi::PacketSetS64Function packet_set_pts{nullptr};
    abi::PacketSetEosFunction packet_set_eos{nullptr};
    abi::PacketGetPointerFunction packet_get_pos{nullptr};
    abi::PacketGetSizeFunction packet_get_length{nullptr};
    abi::PacketGetS64Function packet_get_pts{nullptr};
    abi::PacketGetU32Function packet_get_eos{nullptr};
    abi::PacketGetBufferFunction packet_get_buffer{nullptr};
    abi::PacketGetMetaFunction packet_get_meta{nullptr};
    abi::MetaSetFrameFunction meta_set_frame{nullptr};
    abi::PacketGetU32Function packet_get_flag{nullptr};
    abi::PacketGetU32Function packet_is_partition{nullptr};
    abi::PacketGetU32Function packet_is_eoi{nullptr};

    abi::EncCfgInitFunction enc_cfg_init{nullptr};
    abi::EncCfgDeinitFunction enc_cfg_deinit{nullptr};
    abi::EncCfgSetS32Function enc_cfg_set_s32{nullptr};
    abi::EncCfgSetU32Function enc_cfg_set_u32{nullptr};

    ~Runtime() {
#if defined(__linux__)
        if (library)
            dlclose(library);
#endif
    }

    bool complete() const noexcept {
        return create && init && destroy && buffer_import && buffer_get && buffer_put &&
               buffer_get_ptr && buffer_get_fd && buffer_get_size && buffer_group_get &&
               buffer_group_put && buffer_group_limit && frame_init && frame_deinit &&
               frame_set_width && frame_set_height && frame_set_hor_stride &&
               frame_set_ver_stride && frame_set_fmt && frame_set_eos && frame_set_pts &&
               frame_set_buffer && frame_get_width && frame_get_height && frame_get_hor_stride &&
               frame_get_ver_stride && frame_get_fmt && frame_get_eos && frame_get_pts &&
               frame_get_buffer && frame_get_info_change && frame_get_buf_size &&
               frame_get_errinfo && frame_get_discard && packet_init && packet_init_with_buffer &&
               packet_deinit && packet_set_pos && packet_set_length && packet_set_pts &&
               packet_set_eos && packet_get_pos && packet_get_length && packet_get_pts &&
               packet_get_eos && packet_get_buffer && packet_get_meta && meta_set_frame &&
               packet_get_flag && packet_is_partition && packet_is_eoi && enc_cfg_init &&
               enc_cfg_deinit && enc_cfg_set_s32 && enc_cfg_set_u32;
    }
};

std::shared_ptr<Runtime> load_runtime() noexcept {
#if !defined(__linux__)
    return {};
#else
    static std::mutex mutex;
    static std::weak_ptr<Runtime> cached;
    std::lock_guard<std::mutex> lock(mutex);
    if (auto existing = cached.lock())
        return existing;

    try {
        auto runtime = std::make_shared<Runtime>();
        runtime->library = dlopen(kMppLibraryPrimary, RTLD_NOW | RTLD_LOCAL);
        if (!runtime->library)
            runtime->library = dlopen(kMppLibraryFallback, RTLD_NOW | RTLD_LOCAL);
        if (!runtime->library)
            return {};

#define ROCKIT_LOAD(member, symbol)                                                                \
    if (!load_function(runtime->library, symbol, &runtime->member))                                \
        return {}
        ROCKIT_LOAD(create, "mpp_create");
        ROCKIT_LOAD(init, "mpp_init");
        ROCKIT_LOAD(destroy, "mpp_destroy");
        ROCKIT_LOAD(buffer_import, "mpp_buffer_import_with_tag");
        ROCKIT_LOAD(buffer_get, "mpp_buffer_get_with_tag");
        ROCKIT_LOAD(buffer_put, "mpp_buffer_put_with_caller");
        ROCKIT_LOAD(buffer_get_ptr, "mpp_buffer_get_ptr_with_caller");
        ROCKIT_LOAD(buffer_get_fd, "mpp_buffer_get_fd_with_caller");
        ROCKIT_LOAD(buffer_get_size, "mpp_buffer_get_size_with_caller");
        ROCKIT_LOAD(buffer_group_get, "mpp_buffer_group_get");
        ROCKIT_LOAD(buffer_group_put, "mpp_buffer_group_put");
        ROCKIT_LOAD(buffer_group_limit, "mpp_buffer_group_limit_config");
        ROCKIT_LOAD(frame_init, "mpp_frame_init");
        ROCKIT_LOAD(frame_deinit, "mpp_frame_deinit");
        ROCKIT_LOAD(frame_set_width, "mpp_frame_set_width");
        ROCKIT_LOAD(frame_set_height, "mpp_frame_set_height");
        ROCKIT_LOAD(frame_set_hor_stride, "mpp_frame_set_hor_stride");
        ROCKIT_LOAD(frame_set_ver_stride, "mpp_frame_set_ver_stride");
        ROCKIT_LOAD(frame_set_fmt, "mpp_frame_set_fmt");
        ROCKIT_LOAD(frame_set_eos, "mpp_frame_set_eos");
        ROCKIT_LOAD(frame_set_pts, "mpp_frame_set_pts");
        ROCKIT_LOAD(frame_set_buffer, "mpp_frame_set_buffer");
        ROCKIT_LOAD(frame_get_width, "mpp_frame_get_width");
        ROCKIT_LOAD(frame_get_height, "mpp_frame_get_height");
        ROCKIT_LOAD(frame_get_hor_stride, "mpp_frame_get_hor_stride");
        ROCKIT_LOAD(frame_get_ver_stride, "mpp_frame_get_ver_stride");
        ROCKIT_LOAD(frame_get_fmt, "mpp_frame_get_fmt");
        ROCKIT_LOAD(frame_get_eos, "mpp_frame_get_eos");
        ROCKIT_LOAD(frame_get_pts, "mpp_frame_get_pts");
        ROCKIT_LOAD(frame_get_buffer, "mpp_frame_get_buffer");
        ROCKIT_LOAD(frame_get_info_change, "mpp_frame_get_info_change");
        ROCKIT_LOAD(frame_get_buf_size, "mpp_frame_get_buf_size");
        ROCKIT_LOAD(frame_get_errinfo, "mpp_frame_get_errinfo");
        ROCKIT_LOAD(frame_get_discard, "mpp_frame_get_discard");
        ROCKIT_LOAD(packet_init, "mpp_packet_init");
        ROCKIT_LOAD(packet_init_with_buffer, "mpp_packet_init_with_buffer");
        ROCKIT_LOAD(packet_deinit, "mpp_packet_deinit");
        ROCKIT_LOAD(packet_set_pos, "mpp_packet_set_pos");
        ROCKIT_LOAD(packet_set_length, "mpp_packet_set_length");
        ROCKIT_LOAD(packet_set_pts, "mpp_packet_set_pts");
        ROCKIT_LOAD(packet_set_eos, "mpp_packet_set_eos");
        ROCKIT_LOAD(packet_get_pos, "mpp_packet_get_pos");
        ROCKIT_LOAD(packet_get_length, "mpp_packet_get_length");
        ROCKIT_LOAD(packet_get_pts, "mpp_packet_get_pts");
        ROCKIT_LOAD(packet_get_eos, "mpp_packet_get_eos");
        ROCKIT_LOAD(packet_get_buffer, "mpp_packet_get_buffer");
        ROCKIT_LOAD(packet_get_meta, "mpp_packet_get_meta");
        ROCKIT_LOAD(meta_set_frame, "mpp_meta_set_frame");
        ROCKIT_LOAD(packet_get_flag, "mpp_packet_get_flag");
        ROCKIT_LOAD(packet_is_partition, "mpp_packet_is_partition");
        ROCKIT_LOAD(packet_is_eoi, "mpp_packet_is_eoi");
        ROCKIT_LOAD(enc_cfg_init, "mpp_enc_cfg_init");
        ROCKIT_LOAD(enc_cfg_deinit, "mpp_enc_cfg_deinit");
        ROCKIT_LOAD(enc_cfg_set_s32, "mpp_enc_cfg_set_s32");
        ROCKIT_LOAD(enc_cfg_set_u32, "mpp_enc_cfg_set_u32");
#undef ROCKIT_LOAD
        if (!runtime->complete())
            return {};
        cached = runtime;
        return runtime;
    } catch (...) {
        return {};
    }
#endif
}

struct ImportedBuffer {
    std::shared_ptr<Runtime> runtime;
    MppBuffer buffer{nullptr};

    ~ImportedBuffer() {
        if (runtime && buffer)
            (void)runtime->buffer_put(buffer, kMppCaller);
    }
};

Status import_mb(const std::shared_ptr<Runtime> &runtime,
                 MB_BLK block,
                 RK_U32 offset,
                 RK_U32 length,
                 ImportedBuffer *result) noexcept {
    if (!runtime || !block || !result)
        return Status::Invalid;
    const RK_U64 block_size = RK_MPI_MB_GetSize(block);
    const RK_U64 block_offset = RK_MPI_MB_GetOffset(block);
    const RK_U64 effective_offset = block_offset + offset;
    if (effective_offset > block_size || length > block_size - effective_offset)
        return Status::Invalid;

    const int fd = RK_MPI_MB_Handle2Fd(block);
    if (fd < 0)
        return Status::Unsupported;

    MppBufferInfo info{};
    // Match the vendor Rockit AArch64 path exactly: it imports an external
    // media-buffer fd through the legacy logical ION type (value 1), with no
    // user pointer.  On modern MPP this compatibility type selects the
    // available dma-heap allocator internally while retaining the logical
    // buffer type expected by the encoder/HAL path.
    info.type = MPP_BUFFER_TYPE_ION;
    info.size = static_cast<std::size_t>(block_size);
    info.ptr = nullptr;
    info.fd = fd;
    info.index = -1;
    MppBuffer imported = nullptr;
    const MPP_RET ret = runtime->buffer_import(nullptr, &info, &imported, kMppTag, kMppCaller);
    if (ret != MPP_OK || !imported)
        return status_from_mpp(ret);
    result->runtime = runtime;
    result->buffer = imported;
    return Status::Ok;
}

struct PacketOwner {
    std::shared_ptr<Runtime> runtime;
    MppPacket packet{nullptr};
};

RK_S32 free_packet_owner(void *opaque) noexcept {
    auto *owner = static_cast<PacketOwner *>(opaque);
    if (!owner)
        return RK_SUCCESS;
    if (owner->runtime && owner->packet)
        (void)owner->runtime->packet_deinit(&owner->packet);
    delete owner;
    return RK_SUCCESS;
}

struct FrameOwner {
    std::shared_ptr<Runtime> runtime;
    MppFrame frame{nullptr};
};

RK_S32 free_frame_owner(void *opaque) noexcept {
    auto *owner = static_cast<FrameOwner *>(opaque);
    if (!owner)
        return RK_SUCCESS;
    if (owner->runtime && owner->frame)
        (void)owner->runtime->frame_deinit(&owner->frame);
    delete owner;
    return RK_SUCCESS;
}

Status wrap_packet(const std::shared_ptr<Runtime> &runtime,
                   MppPacket packet,
                   EncodedPacket *output) noexcept {
    if (!runtime || !packet || !output)
        return Status::Invalid;
    void *position = runtime->packet_get_pos(packet);
    const std::size_t length = runtime->packet_get_length(packet);
    MppBuffer buffer = runtime->packet_get_buffer(packet);
    void *base = position;
    std::size_t size = length;
    int fd = -1;
    std::size_t offset = 0;
    if (buffer) {
        void *buffer_base = runtime->buffer_get_ptr(buffer, kMppCaller);
        const std::size_t buffer_size = runtime->buffer_get_size(buffer, kMppCaller);
        const int buffer_fd = runtime->buffer_get_fd(buffer, kMppCaller);
        if (buffer_base && position &&
            static_cast<const std::uint8_t *>(position) >=
                static_cast<const std::uint8_t *>(buffer_base)) {
            offset = static_cast<std::size_t>(static_cast<const std::uint8_t *>(position) -
                                              static_cast<const std::uint8_t *>(buffer_base));
            if (offset <= buffer_size && length <= buffer_size - offset) {
                base = buffer_base;
                size = buffer_size;
                fd = buffer_fd;
            } else {
                offset = 0;
            }
        }
    }
    if (!base || length == 0 || length > UINT32_MAX || offset > UINT32_MAX || size < length ||
        size > UINT64_MAX)
        return Status::Failed;

    auto *owner = new (std::nothrow) PacketOwner{runtime, packet};
    if (!owner)
        return Status::NoMemory;
    MB_EXT_CONFIG_S config{};
    config.pu8VirAddr = static_cast<RK_U8 *>(base);
    config.u64PhyAddr = 0;
    config.s32Fd = fd;
    config.u64Size = size;
    config.pFreeCB = free_packet_owner;
    config.pOpaque = owner;
    MB_BLK block = MB_INVALID_HANDLE;
    const RK_S32 create_result = RK_MPI_SYS_CreateMB(&block, &config);
    if (create_result != RK_SUCCESS || !block) {
        delete owner;
        return create_result == RK_ERR_SYS_NOMEM ? Status::NoMemory : Status::Failed;
    }
    (void)RK_MPI_MB_SetOffset(block, static_cast<RK_U64>(offset));

    output->block = block;
    output->length = static_cast<RK_U32>(length);
    output->offset = static_cast<RK_U32>(offset);
    output->pts = static_cast<RK_U64>(runtime->packet_get_pts(packet));
    output->frame_end =
        runtime->packet_is_partition(packet) == 0 || runtime->packet_is_eoi(packet) != 0;
    output->stream_end = runtime->packet_get_eos(packet) != 0;
    output->idr = (runtime->packet_get_flag(packet) & abi::kPacketFlagIntra) != 0;
    return Status::Ok;
}

Status
wrap_frame(const std::shared_ptr<Runtime> &runtime, MppFrame frame, DecodedFrame *output) noexcept {
    if (!runtime || !frame || !output)
        return Status::Invalid;
    MppBuffer buffer = runtime->frame_get_buffer(frame);
    if (!buffer)
        return Status::Failed;
    void *data = runtime->buffer_get_ptr(buffer, kMppCaller);
    const std::size_t size = runtime->buffer_get_size(buffer, kMppCaller);
    const int fd = runtime->buffer_get_fd(buffer, kMppCaller);
    if (!data || size == 0)
        return Status::Failed;

    const MppFrameFormat mpp_format = runtime->frame_get_fmt(frame);
    const PIXEL_FORMAT_E format = map_mpp_format(mpp_format);
    if (format == RK_FMT_BUTT)
        return Status::Unsupported;

    auto *owner = new (std::nothrow) FrameOwner{runtime, frame};
    if (!owner)
        return Status::NoMemory;
    MB_EXT_CONFIG_S config{};
    config.pu8VirAddr = static_cast<RK_U8 *>(data);
    config.u64PhyAddr = 0;
    config.s32Fd = fd;
    config.u64Size = size;
    config.pFreeCB = free_frame_owner;
    config.pOpaque = owner;
    MB_BLK block = MB_INVALID_HANDLE;
    const RK_S32 create_result = RK_MPI_SYS_CreateMB(&block, &config);
    if (create_result != RK_SUCCESS || !block) {
        (void)free_frame_owner(owner);
        return create_result == RK_ERR_SYS_NOMEM ? Status::NoMemory : Status::Failed;
    }

    VIDEO_FRAME_INFO_S result{};
    result.stVFrame.pMbBlk = block;
    result.stVFrame.u32Width = runtime->frame_get_width(frame);
    result.stVFrame.u32Height = runtime->frame_get_height(frame);
    result.stVFrame.u32VirWidth = runtime->frame_get_hor_stride(frame);
    result.stVFrame.u32VirHeight = runtime->frame_get_ver_stride(frame);
    result.stVFrame.enPixelFormat = format;
    result.stVFrame.enCompressMode =
        MPP_FRAME_FMT_IS_FBC(mpp_format) ? COMPRESS_AFBC_16x16 : COMPRESS_MODE_NONE;
    result.stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;
    result.stVFrame.enField = VIDEO_FIELD_FRAME;
    result.stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
    result.stVFrame.pVirAddr[0] = data;
    result.stVFrame.u64PTS = static_cast<RK_U64>(runtime->frame_get_pts(frame));
    output->frame = result;
    output->end_of_stream = runtime->frame_get_eos(frame) != 0;
    output->corrupt =
        runtime->frame_get_errinfo(frame) != 0 || runtime->frame_get_discard(frame) != 0;
    owner->frame = frame;
    return Status::Ok;
}

struct EncoderConfig {
    MppCodingType codec{MPP_VIDEO_CodingUnused};
    MppFrameFormat format{MPP_FMT_BUTT};
    RK_U32 width{0};
    RK_U32 height{0};
    RK_U32 horizontal_stride{0};
    RK_U32 vertical_stride{0};
    RK_S32 rc_mode{MPP_ENC_RC_MODE_FIXQP};
    RK_S32 fps_in_num{30};
    RK_S32 fps_in_den{1};
    RK_S32 fps_out_num{30};
    RK_S32 fps_out_den{1};
    RK_S32 gop{60};
    RK_S32 target_bps{1500000};
    RK_S32 max_bps{2500000};
    RK_S32 min_bps{46080};
    RK_S32 qp_i{26};
    RK_S32 qp_p{26};
    RK_S32 jpeg_quality{80};
    RK_S32 profile{100};
};

RK_S32 positive_or(RK_U32 value, RK_S32 fallback) noexcept {
    return value ? static_cast<RK_S32>(std::min<RK_U32>(value, INT32_MAX)) : fallback;
}

#define ROCKIT_APPLY_RC_FRAME_RATES(rc)                                                            \
    config.fps_in_num = positive_or((rc).u32SrcFrameRateNum, 30);                                  \
    config.fps_in_den = positive_or((rc).u32SrcFrameRateDen, 1);                                   \
    config.fps_out_num = positive_or((rc).fr32DstFrameRateNum, 30);                                \
    config.fps_out_den = positive_or((rc).fr32DstFrameRateDen, 1)

#define ROCKIT_APPLY_RC_FRAME_RATES_AND_GOP(rc)                                                    \
    ROCKIT_APPLY_RC_FRAME_RATES(rc);                                                               \
    config.gop = positive_or((rc).u32Gop, 60)

#define ROCKIT_APPLY_RC_CBR(rc)                                                                    \
    config.rc_mode = MPP_ENC_RC_MODE_CBR;                                                          \
    ROCKIT_APPLY_RC_FRAME_RATES_AND_GOP(rc);                                                       \
    config.target_bps = positive_or((rc).u32BitRate, 1500) * 1000;                                 \
    config.max_bps = config.target_bps * 17 / 16;                                                  \
    config.min_bps = config.target_bps * 15 / 16

#define ROCKIT_APPLY_RC_VBR(rc)                                                                    \
    ROCKIT_APPLY_RC_FRAME_RATES_AND_GOP(rc);                                                       \
    config.target_bps = positive_or((rc).u32BitRate, 1500) * 1000;                                 \
    config.max_bps = positive_or((rc).u32MaxBitRate, 2500) * 1000;                                 \
    config.min_bps = positive_or((rc).u32MinBitRate, 46) * 1000

#define ROCKIT_APPLY_RC_FIXQP(rc)                                                                  \
    config.rc_mode = MPP_ENC_RC_MODE_FIXQP;                                                        \
    ROCKIT_APPLY_RC_FRAME_RATES_AND_GOP(rc);                                                       \
    config.qp_i = positive_or((rc).u32IQp, 26);                                                    \
    config.qp_p = positive_or((rc).u32PQp, 26)

EncoderConfig build_encoder_config(const VENC_CHN_ATTR_S &attribute) noexcept {
    EncoderConfig config{};
    const VENC_ATTR_S &video = attribute.stVencAttr;
    config.codec = map_codec(video.enType);
    config.format = map_pixel_format(video.enPixelFormat);
    config.width = video.u32PicWidth;
    config.height = video.u32PicHeight;
    config.horizontal_stride = video.u32VirWidth ? video.u32VirWidth : video.u32PicWidth;
    config.vertical_stride = video.u32VirHeight ? video.u32VirHeight : video.u32PicHeight;
    config.profile = video.u32Profile ? static_cast<RK_S32>(video.u32Profile) : 100;

    switch (attribute.stRcAttr.enRcMode) {
    case VENC_RC_MODE_H264CBR: {
        const auto &rc = attribute.stRcAttr.stH264Cbr;
        ROCKIT_APPLY_RC_CBR(rc);
    } break;
    case VENC_RC_MODE_H264VBR:
    case VENC_RC_MODE_H264AVBR: {
        const auto &rc = attribute.stRcAttr.stH264Vbr;
        config.rc_mode = attribute.stRcAttr.enRcMode == VENC_RC_MODE_H264AVBR ? MPP_ENC_RC_MODE_AVBR
                                                                              : MPP_ENC_RC_MODE_VBR;
        ROCKIT_APPLY_RC_VBR(rc);
    } break;
    case VENC_RC_MODE_H264FIXQP: {
        const auto &rc = attribute.stRcAttr.stH264FixQp;
        ROCKIT_APPLY_RC_FIXQP(rc);
    } break;
    case VENC_RC_MODE_H265CBR: {
        const auto &rc = attribute.stRcAttr.stH265Cbr;
        ROCKIT_APPLY_RC_CBR(rc);
    } break;
    case VENC_RC_MODE_H265VBR:
    case VENC_RC_MODE_H265AVBR: {
        const auto &rc = attribute.stRcAttr.stH265Vbr;
        config.rc_mode = attribute.stRcAttr.enRcMode == VENC_RC_MODE_H265AVBR ? MPP_ENC_RC_MODE_AVBR
                                                                              : MPP_ENC_RC_MODE_VBR;
        ROCKIT_APPLY_RC_VBR(rc);
    } break;
    case VENC_RC_MODE_H265FIXQP: {
        const auto &rc = attribute.stRcAttr.stH265FixQp;
        ROCKIT_APPLY_RC_FIXQP(rc);
    } break;
    case VENC_RC_MODE_MJPEGFIXQP: {
        const auto &rc = attribute.stRcAttr.stMjpegFixQp;
        config.rc_mode = MPP_ENC_RC_MODE_FIXQP;
        ROCKIT_APPLY_RC_FRAME_RATES(rc);
        config.jpeg_quality = positive_or(rc.u32Qfactor, 80);
    } break;
    default:
        config.rc_mode = MPP_ENC_RC_MODE_FIXQP;
        break;
    }
    return config;
}

#undef ROCKIT_APPLY_RC_FIXQP
#undef ROCKIT_APPLY_RC_VBR
#undef ROCKIT_APPLY_RC_CBR
#undef ROCKIT_APPLY_RC_FRAME_RATES_AND_GOP
#undef ROCKIT_APPLY_RC_FRAME_RATES

Status configure_encoder(const std::shared_ptr<Runtime> &runtime,
                         MppCtx context,
                         MppApi *api,
                         const EncoderConfig &config) noexcept {
    MppEncCfg cfg = nullptr;
    MPP_RET ret = runtime->enc_cfg_init(&cfg);
    if (ret != MPP_OK || !cfg)
        return status_from_mpp(ret);
    struct Guard {
        std::shared_ptr<Runtime> runtime;
        MppEncCfg cfg;
        ~Guard() {
            if (runtime && cfg)
                (void)runtime->enc_cfg_deinit(cfg);
        }
    } guard{runtime, cfg};

    ret = api->control(context, MPP_ENC_GET_CFG, cfg);
    if (ret != MPP_OK)
        return status_from_mpp(ret);

    auto set_s32 = [&](const char *name, RK_S32 value) -> bool {
        return runtime->enc_cfg_set_s32(cfg, name, value) == MPP_OK;
    };
    auto set_u32 = [&](const char *name, RK_U32 value) -> bool {
        return runtime->enc_cfg_set_u32(cfg, name, value) == MPP_OK;
    };

    bool ok = true;
    ok = set_s32("codec:type", config.codec) && ok;
    ok = set_s32("prep:width", static_cast<RK_S32>(config.width)) && ok;
    ok = set_s32("prep:height", static_cast<RK_S32>(config.height)) && ok;
    ok = set_s32("prep:hor_stride", static_cast<RK_S32>(config.horizontal_stride)) && ok;
    ok = set_s32("prep:ver_stride", static_cast<RK_S32>(config.vertical_stride)) && ok;
    ok = set_s32("prep:format", config.format) && ok;
    ok = set_s32("prep:range", MPP_FRAME_RANGE_JPEG) && ok;
    ok = set_s32("rc:mode", config.rc_mode) && ok;
    ok = set_u32("rc:max_reenc_times", 0) && ok;
    ok = set_u32("rc:drop_mode", MPP_ENC_RC_DROP_FRM_DISABLED) && ok;
    ok = set_s32("rc:fps_in_flex", 0) && ok;
    ok = set_s32("rc:fps_in_num", config.fps_in_num) && ok;
    ok = set_s32("rc:fps_in_denom", config.fps_in_den) && ok;
    ok = set_s32("rc:fps_out_flex", 0) && ok;
    ok = set_s32("rc:fps_out_num", config.fps_out_num) && ok;
    ok = set_s32("rc:fps_out_denom", config.fps_out_den) && ok;
    ok = set_s32("rc:gop", config.gop) && ok;
    ok = set_s32("rc:bps_target", config.target_bps) && ok;
    if (config.rc_mode != MPP_ENC_RC_MODE_FIXQP) {
        ok = set_s32("rc:bps_max", config.max_bps) && ok;
        ok = set_s32("rc:bps_min", config.min_bps) && ok;
    }

    if (config.codec == MPP_VIDEO_CodingAVC || config.codec == MPP_VIDEO_CodingHEVC) {
        if (config.rc_mode == MPP_ENC_RC_MODE_FIXQP) {
            const RK_S32 qp = std::max<RK_S32>(1, std::min<RK_S32>(51, config.qp_p));
            const RK_S32 i_qp = std::max<RK_S32>(1, std::min<RK_S32>(51, config.qp_i));
            ok = set_s32("rc:qp_init", qp) && ok;
            ok = set_s32("rc:qp_max", qp) && ok;
            ok = set_s32("rc:qp_min", qp) && ok;
            ok = set_s32("rc:qp_max_i", i_qp) && ok;
            ok = set_s32("rc:qp_min_i", i_qp) && ok;
            ok = set_s32("rc:qp_ip", i_qp - qp) && ok;
        } else {
            ok = set_s32("rc:qp_init", -1) && ok;
            ok = set_s32("rc:qp_max", 51) && ok;
            ok = set_s32("rc:qp_min", 10) && ok;
            ok = set_s32("rc:qp_max_i", 51) && ok;
            ok = set_s32("rc:qp_min_i", 10) && ok;
            ok = set_s32("rc:qp_ip", 2) && ok;
        }
    } else if (config.codec == MPP_VIDEO_CodingMJPEG) {
        const RK_S32 quality = std::max<RK_S32>(1, std::min<RK_S32>(99, config.jpeg_quality));
        ok = set_s32("jpeg:q_factor", quality) && ok;
        ok = set_s32("jpeg:qf_max", 99) && ok;
        ok = set_s32("jpeg:qf_min", 1) && ok;
    }

    if (config.codec == MPP_VIDEO_CodingAVC) {
        ok = set_s32("h264:profile", config.profile ? config.profile : 100) && ok;
        ok = set_s32("h264:level", 40) && ok;
        ok = set_s32("h264:cabac_en", config.profile == 66 ? 0 : 1) && ok;
        ok = set_s32("h264:cabac_idc", 0) && ok;
        ok = set_s32("h264:trans8x8", config.profile == 100 ? 1 : 0) && ok;
    }

    if (!ok)
        return Status::Failed;
    ret = api->control(context, MPP_ENC_SET_CFG, cfg);
    if (ret != MPP_OK)
        return status_from_mpp(ret);
    if (config.codec == MPP_VIDEO_CodingAVC || config.codec == MPP_VIDEO_CodingHEVC) {
        MppEncHeaderMode mode = MPP_ENC_HEADER_MODE_EACH_IDR;
        ret = api->control(context, MPP_ENC_SET_HEADER_MODE, &mode);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
    }
    return Status::Ok;
}

} // namespace

struct Encoder {
    std::shared_ptr<Runtime> runtime;
    MppCtx context{nullptr};
    MppApi *api{nullptr};
    EncoderConfig config{};
    bool initialized{false};
    std::mutex mutex;

    ~Encoder() {
        std::lock_guard<std::mutex> lock(mutex);
        if (runtime && context) {
            if (initialized && api && api->reset)
                (void)api->reset(context);
            (void)runtime->destroy(context);
            context = nullptr;
            api = nullptr;
        }
    }
};

struct Decoder {
    std::shared_ptr<Runtime> runtime;
    MppCtx context{nullptr};
    MppApi *api{nullptr};
    MppCodingType codec{MPP_VIDEO_CodingUnused};
    RK_U32 frame_count{kDefaultFrameCount};
    std::size_t jpeg_output_size{0};
    MppBufferGroup frame_group{nullptr};
    bool initialized{false};
    std::mutex mutex;

    ~Decoder() {
        std::lock_guard<std::mutex> lock(mutex);
        if (runtime && context) {
            if (initialized && api && api->reset)
                (void)api->reset(context);
            (void)runtime->destroy(context);
            context = nullptr;
            api = nullptr;
        }
        if (runtime && frame_group) {
            (void)runtime->buffer_group_put(frame_group);
            frame_group = nullptr;
        }
    }
};

Policy policy() noexcept {
    const char *value = std::getenv(kProviderEnvironment);
    if (equals(value, "model") || equals(value, "host") || equals(value, "disabled") ||
        equals(value, "off"))
        return Policy::Disabled;
    if (equals(value, "mpp") || equals(value, "required-mpp") || equals(value, "required"))
        return Policy::Required;
    if (equals(value, "auto"))
        return Policy::Auto;
#if defined(__aarch64__)
    return Policy::Auto;
#else
    return Policy::Disabled;
#endif
}

bool policy_is_required() noexcept {
    return policy() == Policy::Required;
}

const char *status_name(Status status) noexcept {
    switch (status) {
    case Status::Ok:
        return "ok";
    case Status::Again:
        return "again";
    case Status::Timeout:
        return "timeout";
    case Status::Unavailable:
        return "unavailable";
    case Status::Unsupported:
        return "unsupported";
    case Status::Invalid:
        return "invalid";
    case Status::NoMemory:
        return "no-memory";
    case Status::Corrupt:
        return "corrupt";
    case Status::Failed:
        return "failed";
    }
    return "unknown";
}

Status create_encoder(const VENC_CHN_ATTR_S &attribute,
                      std::shared_ptr<Encoder> *encoder) noexcept {
    if (!encoder)
        return Status::Invalid;
    encoder->reset();
    if (policy() == Policy::Disabled)
        return Status::Unavailable;
    const EncoderConfig config = build_encoder_config(attribute);
    if (config.codec == MPP_VIDEO_CodingUnused || config.format == MPP_FMT_BUTT)
        return Status::Unsupported;
    if (config.width == 0 || config.height == 0 || config.horizontal_stride < config.width ||
        config.vertical_stride < config.height)
        return Status::Invalid;

    try {
        auto runtime = load_runtime();
        if (!runtime)
            return Status::Unavailable;
        auto result = std::make_shared<Encoder>();
        result->runtime = runtime;
        result->config = config;
        MPP_RET ret = runtime->create(&result->context, &result->api);
        if (ret != MPP_OK || !result->context || !result->api)
            return status_from_mpp(ret);
        RK_S64 timeout = MPP_POLL_BLOCK;
        ret = result->api->control(result->context, MPP_SET_INPUT_TIMEOUT, &timeout);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        ret = result->api->control(result->context, MPP_SET_OUTPUT_TIMEOUT, &timeout);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        ret = runtime->init(result->context, MPP_CTX_ENC, config.codec);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        result->initialized = true;
        const Status configured = configure_encoder(runtime, result->context, result->api, config);
        if (configured != Status::Ok)
            return configured;
        *encoder = std::move(result);
        return Status::Ok;
    } catch (const std::bad_alloc &) {
        return Status::NoMemory;
    } catch (...) {
        return Status::Failed;
    }
}

#define ROCKIT_SET_ENCODER_TIMEOUTS(encoder_, timeout_ms_, result_)                               \
    RK_S64 timeout = mpp_timeout(timeout_ms_);                                                     \
    result_ = (encoder_)->api->control((encoder_)->context, MPP_SET_INPUT_TIMEOUT, &timeout);      \
    if ((result_) != MPP_OK)                                                                       \
        return status_from_mpp(result_);                                                           \
    result_ = (encoder_)->api->control((encoder_)->context, MPP_SET_OUTPUT_TIMEOUT, &timeout);     \
    if ((result_) != MPP_OK)                                                                       \
        return status_from_mpp(result_)

Status encoder_send(const std::shared_ptr<Encoder> &encoder,
                    const VIDEO_FRAME_INFO_S &frame,
                    RK_S32 timeout_ms,
                    EncodedPacket *packet) noexcept {
    if (!encoder || !packet || !frame.stVFrame.pMbBlk)
        return Status::Invalid;
    *packet = EncodedPacket{};
    try {
        std::lock_guard<std::mutex> lock(encoder->mutex);
        RK_U32 width = frame.stVFrame.u32Width ? frame.stVFrame.u32Width : encoder->config.width;
        RK_U32 height =
            frame.stVFrame.u32Height ? frame.stVFrame.u32Height : encoder->config.height;
        RK_U32 hstride = frame.stVFrame.u32VirWidth ? frame.stVFrame.u32VirWidth
                                                    : encoder->config.horizontal_stride;
        RK_U32 vstride = frame.stVFrame.u32VirHeight ? frame.stVFrame.u32VirHeight
                                                     : encoder->config.vertical_stride;
        const MppFrameFormat format = map_pixel_format(frame.stVFrame.enPixelFormat);
        if (format == MPP_FMT_BUTT || width == 0 || height == 0 || hstride < width ||
            vstride < height)
            return Status::Invalid;

        ImportedBuffer imported{};
        const RK_U64 available =
            RK_MPI_MB_GetSize(frame.stVFrame.pMbBlk) - RK_MPI_MB_GetOffset(frame.stVFrame.pMbBlk);
        if (available == 0 || available > UINT32_MAX)
            return Status::Invalid;
        Status status = import_mb(
            encoder->runtime, frame.stVFrame.pMbBlk, 0, static_cast<RK_U32>(available), &imported);
        if (status != Status::Ok)
            return status;

        MppFrame mpp_frame = nullptr;
        MPP_RET ret = encoder->runtime->frame_init(&mpp_frame);
        if (ret != MPP_OK || !mpp_frame)
            return status_from_mpp(ret);
        struct FrameGuard {
            std::shared_ptr<Runtime> runtime;
            MppFrame *frame;
            ~FrameGuard() {
                if (runtime && frame && *frame)
                    (void)runtime->frame_deinit(frame);
            }
        } frame_guard{encoder->runtime, &mpp_frame};

        encoder->runtime->frame_set_width(mpp_frame, width);
        encoder->runtime->frame_set_height(mpp_frame, height);
        encoder->runtime->frame_set_hor_stride(mpp_frame, hstride);
        encoder->runtime->frame_set_ver_stride(mpp_frame, vstride);
        encoder->runtime->frame_set_fmt(mpp_frame, format);
        encoder->runtime->frame_set_pts(mpp_frame, static_cast<RK_S64>(frame.stVFrame.u64PTS));
        encoder->runtime->frame_set_eos(mpp_frame, 0);
        encoder->runtime->frame_set_buffer(mpp_frame, imported.buffer);

        ROCKIT_SET_ENCODER_TIMEOUTS(encoder, timeout_ms, ret);
        ret = encoder->api->encode_put_frame(encoder->context, mpp_frame);
        if (ret != MPP_OK)
            return status_from_mpp(ret);

        MppPacket mpp_packet = nullptr;
        ret = encoder->api->encode_get_packet(encoder->context, &mpp_packet);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        if (!mpp_packet)
            return Status::Again;
        status = wrap_packet(encoder->runtime, mpp_packet, packet);
        if (status != Status::Ok)
            (void)encoder->runtime->packet_deinit(&mpp_packet);
        return status;
    } catch (const std::bad_alloc &) {
        return Status::NoMemory;
    } catch (...) {
        return Status::Failed;
    }
}

Status encoder_request_idr(const std::shared_ptr<Encoder> &encoder) noexcept {
    if (!encoder)
        return Status::Invalid;
    try {
        std::lock_guard<std::mutex> lock(encoder->mutex);
        const MPP_RET ret = encoder->api->control(encoder->context, MPP_ENC_SET_IDR_FRAME, nullptr);
        return status_from_mpp(ret);
    } catch (...) {
        return Status::Failed;
    }
}

Status encoder_reset(const std::shared_ptr<Encoder> &encoder) noexcept {
    if (!encoder)
        return Status::Invalid;
    try {
        std::lock_guard<std::mutex> lock(encoder->mutex);
        if (!encoder->initialized)
            return Status::Failed;
        return status_from_mpp(encoder->api->reset(encoder->context));
    } catch (...) {
        return Status::Failed;
    }
}

Status encoder_send_eos(const std::shared_ptr<Encoder> &encoder,
                        RK_S32 timeout_ms,
                        EncodedPacket *packet) noexcept {
    if (!encoder || !packet)
        return Status::Invalid;
    *packet = EncodedPacket{};
    try {
        std::lock_guard<std::mutex> lock(encoder->mutex);
        MppFrame frame = nullptr;
        MPP_RET ret = encoder->runtime->frame_init(&frame);
        if (ret != MPP_OK || !frame)
            return status_from_mpp(ret);
        struct Guard {
            std::shared_ptr<Runtime> runtime;
            MppFrame *frame;
            ~Guard() {
                if (runtime && frame && *frame)
                    (void)runtime->frame_deinit(frame);
            }
        } guard{encoder->runtime, &frame};
        encoder->runtime->frame_set_width(frame, encoder->config.width);
        encoder->runtime->frame_set_height(frame, encoder->config.height);
        encoder->runtime->frame_set_hor_stride(frame, encoder->config.horizontal_stride);
        encoder->runtime->frame_set_ver_stride(frame, encoder->config.vertical_stride);
        encoder->runtime->frame_set_fmt(frame, encoder->config.format);
        encoder->runtime->frame_set_eos(frame, 1);
        encoder->runtime->frame_set_buffer(frame, nullptr);
        ROCKIT_SET_ENCODER_TIMEOUTS(encoder, timeout_ms, ret);
        ret = encoder->api->encode_put_frame(encoder->context, frame);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        MppPacket output = nullptr;
        ret = encoder->api->encode_get_packet(encoder->context, &output);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        if (!output)
            return Status::Again;
        if (encoder->runtime->packet_get_length(output) == 0 &&
            encoder->runtime->packet_get_eos(output) != 0) {
            packet->stream_end = true;
            packet->frame_end = true;
            (void)encoder->runtime->packet_deinit(&output);
            return Status::Ok;
        }
        const Status status = wrap_packet(encoder->runtime, output, packet);
        if (status != Status::Ok)
            (void)encoder->runtime->packet_deinit(&output);
        return status;
    } catch (...) {
        return Status::Failed;
    }
}

Status create_decoder(const VDEC_CHN_ATTR_S &attribute,
                      std::shared_ptr<Decoder> *decoder) noexcept {
    if (!decoder)
        return Status::Invalid;
    decoder->reset();
    if (policy() == Policy::Disabled)
        return Status::Unavailable;
    const MppCodingType codec = map_codec(attribute.enType);
    if (codec == MPP_VIDEO_CodingUnused)
        return Status::Unsupported;
    try {
        auto runtime = load_runtime();
        if (!runtime)
            return Status::Unavailable;
        auto result = std::make_shared<Decoder>();
        result->runtime = runtime;
        result->codec = codec;
        result->frame_count =
            attribute.u32FrameBufCnt ? attribute.u32FrameBufCnt : kDefaultFrameCount;
        if (codec == MPP_VIDEO_CodingMJPEG) {
            const std::uint64_t width =
                attribute.u32PicVirWidth ? attribute.u32PicVirWidth : attribute.u32PicWidth;
            const std::uint64_t height =
                attribute.u32PicVirHeight ? attribute.u32PicVirHeight : attribute.u32PicHeight;
            if (width == 0 || height == 0 || width > UINT32_MAX - 15U || height > UINT32_MAX - 15U)
                return Status::Invalid;
            const std::uint64_t aligned_width = (width + 15U) & ~UINT64_C(15);
            const std::uint64_t aligned_height = (height + 15U) & ~UINT64_C(15);
            const std::uint64_t conservative_size = aligned_width * aligned_height * UINT64_C(4);
            const std::uint64_t requested_size =
                std::max<std::uint64_t>(conservative_size, attribute.u32FrameBufSize);
            if (requested_size == 0 || requested_size > std::numeric_limits<std::size_t>::max())
                return Status::Invalid;
            result->jpeg_output_size = static_cast<std::size_t>(requested_size);
        }
        MPP_RET ret = runtime->create(&result->context, &result->api);
        if (ret != MPP_OK || !result->context || !result->api)
            return status_from_mpp(ret);
        RK_U32 split = attribute.enMode == VIDEO_MODE_STREAM ? 1U : 0U;
        ret = result->api->control(result->context, MPP_DEC_SET_PARSER_SPLIT_MODE, &split);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        RK_S64 input_timeout = MPP_POLL_BLOCK;
        RK_S64 output_timeout = MPP_POLL_NON_BLOCK;
        ret = result->api->control(result->context, MPP_SET_INPUT_TIMEOUT, &input_timeout);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        ret = result->api->control(result->context, MPP_SET_OUTPUT_TIMEOUT, &output_timeout);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        ret = runtime->init(result->context, MPP_CTX_DEC, codec);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        result->initialized = true;
        *decoder = std::move(result);
        return Status::Ok;
    } catch (const std::bad_alloc &) {
        return Status::NoMemory;
    } catch (...) {
        return Status::Failed;
    }
}

Status ensure_decoder_group_locked(const std::shared_ptr<Decoder> &decoder,
                                   std::size_t size) noexcept {
    if (!decoder || size == 0)
        return Status::Invalid;
    if (!decoder->frame_group) {
        MppBufferGroup group = nullptr;
        MPP_RET ret = decoder->runtime->buffer_group_get(
            &group,
            static_cast<MppBufferType>(MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CACHABLE),
            MPP_BUFFER_INTERNAL,
            kMppTag,
            kMppCaller);
        if (ret != MPP_OK || !group) {
            ret = decoder->runtime->buffer_group_get(
                &group, MPP_BUFFER_TYPE_DRM, MPP_BUFFER_INTERNAL, kMppTag, kMppCaller);
        }
        if (ret != MPP_OK || !group)
            return status_from_mpp(ret);
        decoder->frame_group = group;
    }
    const MPP_RET ret = decoder->runtime->buffer_group_limit(
        decoder->frame_group,
        size,
        static_cast<RK_S32>(std::max<RK_U32>(decoder->frame_count, 4U)));
    return status_from_mpp(ret);
}

Status configure_decoder_group_locked(const std::shared_ptr<Decoder> &decoder,
                                      MppFrame frame) noexcept {
    const std::size_t size = decoder->runtime->frame_get_buf_size(frame);
    if (size == 0)
        return Status::Invalid;
    Status group_status = ensure_decoder_group_locked(decoder, size);
    if (group_status != Status::Ok)
        return group_status;
    MPP_RET ret =
        decoder->api->control(decoder->context, MPP_DEC_SET_EXT_BUF_GROUP, decoder->frame_group);
    if (ret != MPP_OK)
        return status_from_mpp(ret);
    ret = decoder->api->control(decoder->context, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
    return status_from_mpp(ret);
}

Status drain_decoder_locked(const std::shared_ptr<Decoder> &decoder,
                            RK_S32 timeout_ms,
                            std::vector<DecodedFrame> *frames) noexcept {
    if (!frames)
        return Status::Invalid;
    bool produced = false;
    bool first = true;
    for (std::size_t iteration = 0; iteration < 64; ++iteration) {
        RK_S64 timeout = first ? mpp_timeout(timeout_ms) : static_cast<RK_S64>(MPP_POLL_NON_BLOCK);
        MPP_RET ret = decoder->api->control(decoder->context, MPP_SET_OUTPUT_TIMEOUT, &timeout);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        MppFrame frame = nullptr;
        ret = decoder->api->decode_get_frame(decoder->context, &frame);
        first = false;
        if (ret == MPP_ERR_TIMEOUT)
            return produced ? Status::Ok : Status::Timeout;
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        if (!frame)
            return produced ? Status::Ok : Status::Again;

        if (decoder->runtime->frame_get_info_change(frame)) {
            const Status group_status = configure_decoder_group_locked(decoder, frame);
            (void)decoder->runtime->frame_deinit(&frame);
            if (group_status != Status::Ok)
                return group_status;
            first = true;
            continue;
        }

        DecodedFrame output{};
        const Status wrapped = wrap_frame(decoder->runtime, frame, &output);
        if (wrapped != Status::Ok) {
            (void)decoder->runtime->frame_deinit(&frame);
            return wrapped;
        }
        frames->push_back(output);
        produced = true;
        if (output.end_of_stream)
            return Status::Ok;
    }
    return produced ? Status::Ok : Status::Again;
}

Status decoder_send_mjpeg_locked(const std::shared_ptr<Decoder> &decoder,
                                 const VDEC_STREAM_S &stream,
                                 RK_S32 timeout_ms,
                                 std::vector<DecodedFrame> *frames) noexcept {
    if (!decoder || !frames || decoder->jpeg_output_size == 0)
        return Status::Invalid;

    const Status group_status = ensure_decoder_group_locked(decoder, decoder->jpeg_output_size);
    if (group_status != Status::Ok)
        return group_status;

    ImportedBuffer imported_input{};
    MppBuffer owned_input = nullptr;
    struct BufferGuard {
        std::shared_ptr<Runtime> runtime;
        MppBuffer *buffer;
        ~BufferGuard() {
            if (runtime && buffer && *buffer)
                (void)runtime->buffer_put(*buffer, kMppCaller);
        }
    } owned_input_guard{decoder->runtime, &owned_input};

    MppBuffer input_buffer = nullptr;
    void *input_position = nullptr;
    if (stream.u32Len != 0) {
        if (!stream.pMbBlk)
            return Status::Invalid;
        const RK_U64 block_size = RK_MPI_MB_GetSize(stream.pMbBlk);
        const RK_U64 block_offset = RK_MPI_MB_GetOffset(stream.pMbBlk);
        if (block_offset > block_size || stream.u32Len > block_size - block_offset)
            return Status::Invalid;
        const Status imported =
            import_mb(decoder->runtime, stream.pMbBlk, 0, stream.u32Len, &imported_input);
        if (imported != Status::Ok)
            return imported;
        input_buffer = imported_input.buffer;
        void *base = decoder->runtime->buffer_get_ptr(input_buffer, kMppCaller);
        if (!base)
            return Status::Unsupported;
        input_position = static_cast<std::uint8_t *>(base) + block_offset;
    } else {
        MPP_RET ret = decoder->runtime->buffer_get(nullptr, &owned_input, 1U, kMppTag, kMppCaller);
        if (ret != MPP_OK || !owned_input)
            return status_from_mpp(ret);
        input_buffer = owned_input;
        input_position = decoder->runtime->buffer_get_ptr(input_buffer, kMppCaller);
        if (!input_position)
            return Status::Failed;
    }

    MppPacket packet = nullptr;
    MPP_RET ret = decoder->runtime->packet_init_with_buffer(&packet, input_buffer);
    if (ret != MPP_OK || !packet)
        return status_from_mpp(ret);
    struct PacketGuard {
        std::shared_ptr<Runtime> runtime;
        MppPacket *packet;
        ~PacketGuard() {
            if (runtime && packet && *packet)
                (void)runtime->packet_deinit(packet);
        }
    } packet_guard{decoder->runtime, &packet};

    decoder->runtime->packet_set_pos(packet, input_position);
    decoder->runtime->packet_set_length(packet, stream.u32Len);
    decoder->runtime->packet_set_pts(packet, static_cast<RK_S64>(stream.u64PTS));
    if (stream.bEndOfStream)
        (void)decoder->runtime->packet_set_eos(packet);

    MppBuffer output_buffer = nullptr;
    ret = decoder->runtime->buffer_get(
        decoder->frame_group, &output_buffer, decoder->jpeg_output_size, kMppTag, kMppCaller);
    if (ret != MPP_OK || !output_buffer)
        return status_from_mpp(ret);
    BufferGuard output_buffer_guard{decoder->runtime, &output_buffer};

    MppFrame output_frame = nullptr;
    ret = decoder->runtime->frame_init(&output_frame);
    if (ret != MPP_OK || !output_frame)
        return status_from_mpp(ret);
    struct FrameGuard {
        std::shared_ptr<Runtime> runtime;
        MppFrame *frame;
        ~FrameGuard() {
            if (runtime && frame && *frame)
                (void)runtime->frame_deinit(frame);
        }
    } output_frame_guard{decoder->runtime, &output_frame};
    decoder->runtime->frame_set_buffer(output_frame, output_buffer);

    abi::Meta meta = decoder->runtime->packet_get_meta(packet);
    if (!meta)
        return Status::Failed;
    ret = decoder->runtime->meta_set_frame(meta, KEY_OUTPUT_FRAME, output_frame);
    if (ret != MPP_OK)
        return status_from_mpp(ret);

    RK_S64 timeout = mpp_timeout(timeout_ms);
    ret = decoder->api->control(decoder->context, MPP_SET_INPUT_TIMEOUT, &timeout);
    if (ret != MPP_OK)
        return status_from_mpp(ret);
    ret = decoder->api->control(decoder->context, MPP_SET_OUTPUT_TIMEOUT, &timeout);
    if (ret != MPP_OK)
        return status_from_mpp(ret);
    ret = decoder->api->decode_put_packet(decoder->context, packet);
    if (ret != MPP_OK)
        return status_from_mpp(ret);

    MppFrame returned_frame = nullptr;
    ret = decoder->api->decode_get_frame(decoder->context, &returned_frame);
    if (ret != MPP_OK)
        return status_from_mpp(ret);
    if (!returned_frame)
        return Status::Again;

    if (returned_frame == output_frame)
        output_frame = nullptr;

    DecodedFrame decoded{};
    const Status wrapped = wrap_frame(decoder->runtime, returned_frame, &decoded);
    if (wrapped != Status::Ok) {
        (void)decoder->runtime->frame_deinit(&returned_frame);
        return wrapped;
    }
    frames->push_back(decoded);
    return Status::Ok;
}

#undef ROCKIT_SET_ENCODER_TIMEOUTS

Status decoder_send(const std::shared_ptr<Decoder> &decoder,
                    const VDEC_STREAM_S &stream,
                    RK_S32 timeout_ms,
                    std::vector<DecodedFrame> *frames) noexcept {
    if (!decoder || !frames)
        return Status::Invalid;
    try {
        std::lock_guard<std::mutex> lock(decoder->mutex);
        if (decoder->codec == MPP_VIDEO_CodingMJPEG)
            return decoder_send_mjpeg_locked(decoder, stream, timeout_ms, frames);
        MppPacket packet = nullptr;
        std::uint8_t eos_byte = 0;
        if (stream.u32Len != 0) {
            if (!stream.pMbBlk)
                return Status::Invalid;
            const RK_U64 block_size = RK_MPI_MB_GetSize(stream.pMbBlk);
            const RK_U64 offset = RK_MPI_MB_GetOffset(stream.pMbBlk);
            if (offset > block_size || stream.u32Len > block_size - offset)
                return Status::Invalid;
            void *base = RK_MPI_MB_Handle2VirAddr(stream.pMbBlk);
            if (!base)
                return Status::Unsupported;
            void *position = static_cast<std::uint8_t *>(base) + offset;
            const MPP_RET ret = decoder->runtime->packet_init(&packet, position, stream.u32Len);
            if (ret != MPP_OK || !packet)
                return status_from_mpp(ret);
            decoder->runtime->packet_set_pos(packet, position);
            decoder->runtime->packet_set_length(packet, stream.u32Len);
        } else {
            const MPP_RET ret = decoder->runtime->packet_init(&packet, &eos_byte, sizeof(eos_byte));
            if (ret != MPP_OK || !packet)
                return status_from_mpp(ret);
            decoder->runtime->packet_set_length(packet, 0);
        }
        struct PacketGuard {
            std::shared_ptr<Runtime> runtime;
            MppPacket *packet;
            ~PacketGuard() {
                if (runtime && packet && *packet)
                    (void)runtime->packet_deinit(packet);
            }
        } guard{decoder->runtime, &packet};

        decoder->runtime->packet_set_pts(packet, static_cast<RK_S64>(stream.u64PTS));
        if (stream.bEndOfStream)
            (void)decoder->runtime->packet_set_eos(packet);
        RK_S64 input_timeout = mpp_timeout(timeout_ms);
        MPP_RET ret =
            decoder->api->control(decoder->context, MPP_SET_INPUT_TIMEOUT, &input_timeout);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        ret = decoder->api->decode_put_packet(decoder->context, packet);
        if (ret != MPP_OK)
            return status_from_mpp(ret);
        return drain_decoder_locked(decoder, 0, frames);
    } catch (const std::bad_alloc &) {
        return Status::NoMemory;
    } catch (...) {
        return Status::Failed;
    }
}

Status decoder_drain(const std::shared_ptr<Decoder> &decoder,
                     RK_S32 timeout_ms,
                     std::vector<DecodedFrame> *frames) noexcept {
    if (!decoder || !frames)
        return Status::Invalid;
    try {
        std::lock_guard<std::mutex> lock(decoder->mutex);
        return drain_decoder_locked(decoder, timeout_ms, frames);
    } catch (const std::bad_alloc &) {
        return Status::NoMemory;
    } catch (...) {
        return Status::Failed;
    }
}

Status decoder_reset(const std::shared_ptr<Decoder> &decoder) noexcept {
    if (!decoder)
        return Status::Invalid;
    try {
        std::lock_guard<std::mutex> lock(decoder->mutex);
        if (!decoder->initialized)
            return Status::Failed;
        return status_from_mpp(decoder->api->reset(decoder->context));
    } catch (...) {
        return Status::Failed;
    }
}

} // namespace rockit_mpp
