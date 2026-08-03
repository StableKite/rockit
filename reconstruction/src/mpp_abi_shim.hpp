#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rockit_mpp_abi {

using Ret = std::int32_t;
using Context = void *;
using Frame = void *;
using Packet = void *;
using Buffer = void *;
using BufferGroup = void *;
using Meta = void *;
using EncCfg = void *;
using CodingType = std::int32_t;
using FrameFormat = std::uint32_t;
using Command = std::uint32_t;
using MetaKey = std::uint32_t;

constexpr Ret kOk = 0;
constexpr Ret kErrorTimeout = -8;
constexpr Ret kErrorBufferFull = -1012;
constexpr Ret kErrorDisplayFull = -1013;
constexpr Ret kErrorNullPointer = -3;
constexpr Ret kErrorValue = -6;
constexpr Ret kErrorMalloc = -4;
constexpr Ret kErrorNoMemory = -1006;
constexpr Ret kErrorStream = -1004;
constexpr Ret kErrorProtocol = -1007;
constexpr Ret kErrorSplitFrame = -1008;

constexpr std::int32_t kContextDecoder = 0;
constexpr std::int32_t kContextEncoder = 1;

constexpr CodingType kCodingUnused = 0;
constexpr CodingType kCodingMpeg2 = 2;
constexpr CodingType kCodingMpeg4 = 4;
constexpr CodingType kCodingAvc = 7;
constexpr CodingType kCodingMjpeg = 8;
constexpr CodingType kCodingVp8 = 9;
constexpr CodingType kCodingVp9 = 10;
constexpr CodingType kCodingHevc = 16777220;
constexpr CodingType kCodingAv1 = 16777224;

constexpr FrameFormat kFormatYuv420Sp = 0;
constexpr FrameFormat kFormatYuv420Sp10Bit = 1;
constexpr FrameFormat kFormatYuv422Sp = 2;
constexpr FrameFormat kFormatYuv420P = 4;
constexpr FrameFormat kFormatYuv420SpVu = 5;
constexpr FrameFormat kFormatYuv422SpVu = 7;
constexpr FrameFormat kFormatYuv422Yuyv = 8;
constexpr FrameFormat kFormatYuv422Uyvy = 10;
constexpr FrameFormat kFormatRgb888 = 65542;
constexpr FrameFormat kFormatBgr888 = 65543;
constexpr FrameFormat kFormatArgb8888 = 65546;
constexpr FrameFormat kFormatAbgr8888 = 65547;
constexpr FrameFormat kFormatBgra8888 = 65548;
constexpr FrameFormat kFormatRgba8888 = 65549;
constexpr FrameFormat kFormatButt = 65553;
constexpr FrameFormat kFrameFormatMask = 0x000fffffU;
constexpr FrameFormat kFrameFbcMask = 0x00f00000U;

constexpr std::int64_t kPollBlock = -1;
constexpr std::int64_t kPollNonBlock = 0;

constexpr Command kSetInputTimeout = 2097158U;
constexpr Command kSetOutputTimeout = 2097159U;
constexpr Command kDecSetExternalBufferGroup = 3211266U;
constexpr Command kDecSetInfoChangeReady = 3211267U;
constexpr Command kDecSetParserSplitMode = 3211269U;
constexpr Command kEncSetConfig = 3276801U;
constexpr Command kEncGetConfig = 3276802U;
constexpr Command kEncSetIdrFrame = 3276809U;
constexpr Command kEncSetHeaderMode = 3309569U;

constexpr std::int32_t kBufferInternal = 0;
constexpr std::int32_t kBufferExternal = 1;
constexpr std::int32_t kBufferTypeNormal = 0;
constexpr std::int32_t kBufferTypeIon = 1;
constexpr std::int32_t kBufferTypeExternalDma = 2;
constexpr std::int32_t kBufferTypeDrm = 3;
constexpr std::int32_t kBufferTypeDmaHeap = 4;
constexpr std::int32_t kBufferFlagCacheable = 0x00020000;

constexpr std::int32_t kRcModeVbr = 0;
constexpr std::int32_t kRcModeCbr = 1;
constexpr std::int32_t kRcModeFixQp = 2;
constexpr std::int32_t kRcModeAvbr = 3;
constexpr std::uint32_t kRcDropDisabled = 0;
constexpr std::int32_t kHeaderEachIdr = 1;
constexpr std::uint32_t kPacketFlagIntra = 0x10U;
constexpr std::int32_t kFrameRangeJpeg = 2;
constexpr MetaKey kMetaKeyOutputFrame = 0x6f66726dU;

struct BufferInfo {
    std::int32_t type;
    std::size_t size;
    void *ptr;
    void *hnd;
    std::int32_t fd;
    std::int32_t index;
};

static_assert(std::is_standard_layout<BufferInfo>::value, "MPP BufferInfo must be standard layout");
static_assert(offsetof(BufferInfo, type) == 0, "MPP BufferInfo type offset");
static_assert(offsetof(BufferInfo, size) == (sizeof(void *) == 8 ? 8U : 4U),
              "MPP BufferInfo size offset");
static_assert(offsetof(BufferInfo, ptr) == (sizeof(void *) == 8 ? 16U : 8U),
              "MPP BufferInfo pointer offset");
static_assert(offsetof(BufferInfo, hnd) == (sizeof(void *) == 8 ? 24U : 12U),
              "MPP BufferInfo handle offset");
static_assert(offsetof(BufferInfo, fd) == (sizeof(void *) == 8 ? 32U : 16U),
              "MPP BufferInfo fd offset");
static_assert(offsetof(BufferInfo, index) == (sizeof(void *) == 8 ? 36U : 20U),
              "MPP BufferInfo index offset");
static_assert(sizeof(BufferInfo) == (sizeof(void *) == 8 ? 40U : 24U), "MPP BufferInfo size");

using DecodeFunction = Ret (*)(Context, Packet, Frame *);
using DecodePutPacketFunction = Ret (*)(Context, Packet);
using DecodeGetFrameFunction = Ret (*)(Context, Frame *);
using EncodeFunction = Ret (*)(Context, Frame, Packet *);
using EncodePutFrameFunction = Ret (*)(Context, Frame);
using EncodeGetPacketFunction = Ret (*)(Context, Packet *);
using GenericContextFunction = Ret (*)(Context, void *, void *);
using PollFunction = Ret (*)(Context, std::int32_t, std::int64_t);
using QueueFunction = Ret (*)(Context, std::int32_t, void **);
using ResetFunction = Ret (*)(Context);
using ControlFunction = Ret (*)(Context, Command, void *);

struct Api {
    std::uint32_t size;
    std::uint32_t version;
    DecodeFunction decode;
    DecodePutPacketFunction decode_put_packet;
    DecodeGetFrameFunction decode_get_frame;
    EncodeFunction encode;
    EncodePutFrameFunction encode_put_frame;
    EncodeGetPacketFunction encode_get_packet;
    GenericContextFunction isp;
    GenericContextFunction isp_put_frame;
    GenericContextFunction isp_get_frame;
    PollFunction poll;
    QueueFunction dequeue;
    QueueFunction enqueue;
    ResetFunction reset;
    ControlFunction control;
};

static_assert(std::is_standard_layout<Api>::value, "MPP Api must be standard layout");
static_assert(offsetof(Api, decode) == 8, "MPP Api decode offset");
static_assert(offsetof(Api, encode_put_frame) == (sizeof(void *) == 8 ? 40U : 24U),
              "MPP Api encode_put_frame offset");
static_assert(offsetof(Api, reset) == (sizeof(void *) == 8 ? 104U : 56U), "MPP Api reset offset");
static_assert(offsetof(Api, control) == (sizeof(void *) == 8 ? 112U : 60U),
              "MPP Api control offset");
static_assert(sizeof(Api) == (sizeof(void *) == 8 ? 120U : 64U), "MPP Api prefix size");

using CreateFunction = Ret (*)(Context *, Api **);
using InitFunction = Ret (*)(Context, std::int32_t, CodingType);
using DestroyFunction = Ret (*)(Context);
using BufferImportFunction =
    Ret (*)(BufferGroup, BufferInfo *, Buffer *, const char *, const char *);
using BufferGetFunction = Ret (*)(BufferGroup, Buffer *, std::size_t, const char *, const char *);
using BufferPutFunction = Ret (*)(Buffer, const char *);
using BufferGetPointerFunction = void *(*)(Buffer, const char *);
using BufferGetFdFunction = std::int32_t (*)(Buffer, const char *);
using BufferGetSizeFunction = std::size_t (*)(Buffer, const char *);
using BufferGroupGetFunction =
    Ret (*)(BufferGroup *, std::int32_t, std::int32_t, const char *, const char *);
using BufferGroupPutFunction = Ret (*)(BufferGroup);
using BufferGroupLimitFunction = Ret (*)(BufferGroup, std::size_t, std::int32_t);

using FrameInitFunction = Ret (*)(Frame *);
using FrameDeinitFunction = Ret (*)(Frame *);
using FrameSetU32Function = void (*)(Frame, std::uint32_t);
using FrameSetS64Function = void (*)(Frame, std::int64_t);
using FrameSetBufferFunction = void (*)(Frame, Buffer);
using FrameGetU32Function = std::uint32_t (*)(Frame);
using FrameGetSizeFunction = std::size_t (*)(Frame);
using FrameGetS64Function = std::int64_t (*)(Frame);
using FrameGetBufferFunction = Buffer (*)(Frame);

using PacketInitFunction = Ret (*)(Packet *, void *, std::size_t);
using PacketInitWithBufferFunction = Ret (*)(Packet *, Buffer);
using PacketDeinitFunction = Ret (*)(Packet *);
using PacketSetPointerFunction = void (*)(Packet, void *);
using PacketSetSizeFunction = void (*)(Packet, std::size_t);
using PacketSetS64Function = void (*)(Packet, std::int64_t);
using PacketSetEosFunction = Ret (*)(Packet);
using PacketGetPointerFunction = void *(*)(Packet);
using PacketGetSizeFunction = std::size_t (*)(Packet);
using PacketGetS64Function = std::int64_t (*)(Packet);
using PacketGetU32Function = std::uint32_t (*)(Packet);
using PacketGetBufferFunction = Buffer (*)(Packet);
using PacketGetMetaFunction = Meta (*)(Packet);
using MetaSetFrameFunction = Ret (*)(Meta, MetaKey, Frame);

using EncCfgInitFunction = Ret (*)(EncCfg *);
using EncCfgDeinitFunction = Ret (*)(EncCfg);
using EncCfgSetS32Function = Ret (*)(EncCfg, const char *, std::int32_t);
using EncCfgSetU32Function = Ret (*)(EncCfg, const char *, std::uint32_t);

} // namespace rockit_mpp_abi
